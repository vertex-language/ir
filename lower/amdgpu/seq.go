package amdgpu

// The rows that are not one instruction. Each is LLVM's expansion for
// the same operation on gfx9, read off llc's output for gfx942 and
// written here in vregs; where LLVM took a scalar path because its
// operands happened to be uniform, this is the vector path it takes
// otherwise.
//
// # Traps
//
// A trap is a wave-wide event: the program stops. So a per-lane
// condition that should trap — a zero divisor in any lane, a float
// out of range in any lane — is a lane mask tested against exec, and a
// scalar branch to one s_trap at the end of the function. That is what
// makes the §A and §C2 traps lowerable before divergent control flow
// is: the branch is uniform by construction, since the trap either
// happens or does not.

import (
	"fmt"
	"math"

	"github.com/vertex-language/ir"
	"github.com/vertex-language/ir/lower/mir"
)

// The modifier and float-constant operands.
func neg(i int) opnd              { return opnd{kind: oUse, i: i, neg: true} }
func fimm(f float32) opnd         { return opnd{kind: oFImm, imm: int64(math.Float32bits(f))} }
func rs(v ...mir.VReg) []mir.VReg { return v }

// constV32 is a dword in a fresh VGPR.
func (x *fnState) constV32(c *cursor, bits uint32) mir.VReg {
	t := x.vr.temp(v32)
	x.emit(c, "v_mov_b32", rs(t), nil, def(0), imm(int64(int32(bits))))
	return t
}

// constV64 is a quadword in a fresh VGPR pair.
func (x *fnState) constV64(c *cursor, bits uint64) mir.VReg {
	t := x.vr.temp(v64)
	x.emit(c, "v_mov_b32", rs(t), nil, defLo(0), imm(int64(int32(uint32(bits)))))
	x.emit(c, "v_mov_b32", rs(t), rs(t), defHi(0), imm(int64(int32(uint32(bits>>32)))))
	return t
}

// trapIf traps the wave when any active lane's bit in mask is set.
func (x *fnState) trapIf(c *cursor, mask mir.VReg) {
	c.Emit(mir.Instr{Op: trapIfOp{}, Uses: rs(mask)})
}

// —— §A integer division ——

// divRem32 is LLVM's expandDivRem32: a reciprocal from v_rcp_iflag_f32,
// one Newton step, a quotient estimate and two correction rounds. The
// unsigned core; the signed rows take absolute values around it.
func (x *fnState) divRem32(c *cursor, in *ir.Inst) error {
	verb := in.Op().Verb
	signed := verb == ir.VSDiv || verb == ir.VSRem
	wantRem := verb == ir.VSRem || verb == ir.VURem
	d, err := x.result(in)
	if err != nil {
		return err
	}
	a, err := x.args(in)
	if err != nil {
		return err
	}
	num, den := a[0], a[1]
	t := func() mir.VReg { return x.vr.temp(v32) }
	m := func() mir.VReg { return x.vr.temp(s64) }

	// The traps: a zero divisor, and INT_MIN / -1.
	zero := m()
	x.emit(c, "v_cmp_eq_u32", rs(zero), rs(den), def(0), imm(0), use(0))
	x.trapIf(c, zero)
	if signed {
		minInt := x.constV32(c, 0x80000000)
		m1, m2, both := m(), m(), m()
		x.emit(c, "v_cmp_eq_u32", rs(m1), rs(minInt, num), def(0), use(0), use(1))
		x.emit(c, "v_cmp_eq_u32", rs(m2), rs(den), def(0), imm(-1), use(0))
		x.emit(c, "s_and_b64", rs(both), rs(m1, m2), def(0), use(0), use(1))
		x.trapIf(c, both)
	}

	// Absolute values, and the signs to put back.
	var sn, sd mir.VReg
	if signed {
		sn, sd = t(), t()
		x.emit(c, "v_ashrrev_i32", rs(sn), rs(num), def(0), imm(31), use(0))
		x.emit(c, "v_ashrrev_i32", rs(sd), rs(den), def(0), imm(31), use(0))
		an, ad := t(), t()
		x.emit(c, "v_xor_b32", rs(an), rs(num, sn), def(0), use(0), use(1))
		x.emit(c, "v_sub_u32", rs(an), rs(an, sn), def(0), use(0), use(1))
		x.emit(c, "v_xor_b32", rs(ad), rs(den, sd), def(0), use(0), use(1))
		x.emit(c, "v_sub_u32", rs(ad), rs(ad, sd), def(0), use(0), use(1))
		num, den = an, ad
	}

	f, r, tt := t(), t(), t()
	x.emit(c, "v_cvt_f32_u32", rs(f), rs(den), def(0), use(0))
	x.emit(c, "v_rcp_iflag_f32", rs(f), rs(f), def(0), use(0))
	x.emit(c, "v_mul_f32", rs(f), rs(f), def(0), imm(0x4f7ffffe), use(0))
	x.emit(c, "v_cvt_u32_f32", rs(r), rs(f), def(0), use(0))
	x.emit(c, "v_sub_u32", rs(tt), rs(den), def(0), imm(0), use(0))
	x.emit(c, "v_mul_lo_u32", rs(tt), rs(tt, r), def(0), use(0), use(1))
	x.emit(c, "v_mul_hi_u32", rs(tt), rs(r, tt), def(0), use(0), use(1))
	x.emit(c, "v_add_u32", rs(r), rs(r, tt), def(0), use(0), use(1))
	q, rem := t(), t()
	x.emit(c, "v_mul_hi_u32", rs(q), rs(num, r), def(0), use(0), use(1))
	x.emit(c, "v_mul_lo_u32", rs(tt), rs(q, den), def(0), use(0), use(1))
	x.emit(c, "v_sub_u32", rs(rem), rs(num, tt), def(0), use(0), use(1))
	for round := 0; round < 2; round++ {
		q1, rem1, ge := t(), t(), m()
		x.emit(c, "v_add_u32", rs(q1), rs(q), def(0), imm(1), use(0))
		x.emit(c, "v_sub_u32", rs(rem1), rs(rem, den), def(0), use(0), use(1))
		x.emit(c, "v_cmp_ge_u32", rs(ge), rs(rem, den), def(0), use(0), use(1))
		nq, nrem := t(), t()
		x.emit(c, "v_cndmask_b32", rs(nq), rs(q, q1, ge), def(0), use(0), use(1), use(2))
		x.emit(c, "v_cndmask_b32", rs(nrem), rs(rem, rem1, ge), def(0), use(0), use(1), use(2))
		q, rem = nq, nrem
	}

	res, sign := q, sn
	if wantRem {
		res = rem
	} else if signed {
		sign = t()
		x.emit(c, "v_xor_b32", rs(sign), rs(sn, sd), def(0), use(0), use(1))
	}
	if !signed {
		emitCopy(c, d, res, v32)
		return nil
	}
	x.emit(c, "v_xor_b32", rs(d), rs(res, sign), def(0), use(0), use(1))
	x.emit(c, "v_sub_u32", rs(d), rs(d, sign), def(0), use(0), use(1))
	return nil
}

// —— §A3 correctly rounded division and square root ——

// fdiv32 is the v_div_scale sequence: the IEEE quotient, since v_rcp_f32
// alone is a ULP off. Denormals are on in the descriptor's float mode,
// so no mode switch surrounds it.
func (x *fnState) fdiv32(c *cursor, d, num, den mir.VReg) {
	t := func() mir.VReg { return x.vr.temp(v32) }
	ds, r, ns := t(), t(), t()
	x.emit(c, "v_div_scale_f32", rs(ds), rs(den, num), def(0), vcc(), use(0), use(0), use(1))
	x.emit(c, "v_rcp_f32", rs(r), rs(ds), def(0), use(0))
	x.emit(c, "v_div_scale_f32", rs(ns), rs(num, den), def(0), vcc(), use(0), use(1), use(0))
	e := t()
	x.emit(c, "v_fma_f32", rs(e), rs(ds, r), def(0), neg(0), use(1), fimm(1))
	r2 := t()
	x.emit(c, "v_fma_f32", rs(r2), rs(e, r), def(0), use(0), use(1), use(1))
	qq := t()
	x.emit(c, "v_mul_f32", rs(qq), rs(ns, r2), def(0), use(0), use(1))
	e2 := t()
	x.emit(c, "v_fma_f32", rs(e2), rs(ds, qq, ns), def(0), neg(0), use(1), use(2))
	q2 := t()
	x.emit(c, "v_fma_f32", rs(q2), rs(e2, r2, qq), def(0), use(0), use(1), use(2))
	e3 := t()
	x.emit(c, "v_fma_f32", rs(e3), rs(ds, q2, ns), def(0), neg(0), use(1), use(2))
	// v_div_fmas reads the VCC the second div_scale wrote; nothing between
	// them touches it.
	fm := t()
	x.emit(c, "v_div_fmas_f32", rs(fm), rs(e3, r2, q2), def(0), use(0), use(1), use(2))
	x.emit(c, "v_div_fixup_f32", rs(d), rs(fm, den, num), def(0), use(0), use(1), use(2))
}

func (x *fnState) fdiv64(c *cursor, d, num, den mir.VReg) {
	t := func() mir.VReg { return x.vr.temp(v64) }
	ds, r, ns := t(), t(), t()
	x.emit(c, "v_div_scale_f64", rs(ds), rs(den, num), def(0), vcc(), use(0), use(0), use(1))
	x.emit(c, "v_rcp_f64", rs(r), rs(ds), def(0), use(0))
	x.emit(c, "v_div_scale_f64", rs(ns), rs(num, den), def(0), vcc(), use(0), use(1), use(0))
	for i := 0; i < 2; i++ {
		e, r2 := t(), t()
		x.emit(c, "v_fma_f64", rs(e), rs(ds, r), def(0), neg(0), use(1), fimm(1))
		x.emit(c, "v_fma_f64", rs(r2), rs(r, e), def(0), use(0), use(1), use(0))
		r = r2
	}
	qq, e, fm := t(), t(), t()
	x.emit(c, "v_mul_f64", rs(qq), rs(ns, r), def(0), use(0), use(1))
	x.emit(c, "v_fma_f64", rs(e), rs(ds, qq, ns), def(0), neg(0), use(1), use(2))
	x.emit(c, "v_div_fmas_f64", rs(fm), rs(e, r, qq), def(0), use(0), use(1), use(2))
	x.emit(c, "v_div_fixup_f64", rs(d), rs(fm, den, num), def(0), use(0), use(1), use(2))
}

// fsqrt32 is v_sqrt_f32 and a one-ULP correction either way, with the
// input scaled up by 2^32 when it is below 2^-96 so the correction's
// products do not lose bits, and zero and infinity passed through.
func (x *fnState) fsqrt32(c *cursor, d, a mir.VReg) {
	t := func() mir.VReg { return x.vr.temp(v32) }
	m := func() mir.VReg { return x.vr.temp(s64) }
	small := m()
	tiny := x.constV32(c, 0x0f800000)
	x.emit(c, "v_cmp_lt_f32", rs(small), rs(a, tiny), def(0), use(0), use(1))
	scaled, in := t(), t()
	x.emit(c, "v_mul_f32", rs(scaled), rs(a), def(0), imm(0x4f800000), use(0))
	x.emit(c, "v_cndmask_b32", rs(in), rs(a, scaled, small), def(0), use(0), use(1), use(2))
	s := t()
	x.emit(c, "v_sqrt_f32", rs(s), rs(in), def(0), use(0))
	down, up := t(), t()
	x.emit(c, "v_add_u32", rs(down), rs(s), def(0), imm(-1), use(0))
	x.emit(c, "v_add_u32", rs(up), rs(s), def(0), imm(1), use(0))
	rd, le := t(), m()
	x.emit(c, "v_fma_f32", rs(rd), rs(down, s, in), def(0), neg(0), use(1), use(2))
	x.emit(c, "v_cmp_ge_f32", rs(le), rs(rd), def(0), imm(0), use(0))
	s1 := t()
	x.emit(c, "v_cndmask_b32", rs(s1), rs(s, down, le), def(0), use(0), use(1), use(2))
	ru, gt := t(), m()
	x.emit(c, "v_fma_f32", rs(ru), rs(up, s, in), def(0), neg(0), use(1), use(2))
	x.emit(c, "v_cmp_lt_f32", rs(gt), rs(ru), def(0), imm(0), use(0))
	s2 := t()
	x.emit(c, "v_cndmask_b32", rs(s2), rs(s1, up, gt), def(0), use(0), use(1), use(2))
	unscaled, s3 := t(), t()
	x.emit(c, "v_mul_f32", rs(unscaled), rs(s2), def(0), imm(0x37800000), use(0))
	x.emit(c, "v_cndmask_b32", rs(s3), rs(s2, unscaled, small), def(0), use(0), use(1), use(2))
	// Zero and infinity are their own square roots: class mask 0x260 is
	// -0, +0 and +inf.
	class := x.constV32(c, 0x260)
	special := m()
	x.emit(c, "v_cmp_class_f32", rs(special), rs(in, class), def(0), use(0), use(1))
	x.emit(c, "v_cndmask_b32", rs(d), rs(s3, in, special), def(0), use(0), use(1), use(2))
}

// fsqrt64 is v_rsq_f64 refined by two Goldschmidt steps, the input
// scaled by 2^256 when below 2^-767 and the result by 2^-128 after.
func (x *fnState) fsqrt64(c *cursor, d, a mir.VReg) {
	t := func() mir.VReg { return x.vr.temp(v64) }
	t32 := func() mir.VReg { return x.vr.temp(v32) }
	m := func() mir.VReg { return x.vr.temp(s64) }
	small := m()
	tiny := x.constV64(c, 0x1000000000000000)
	x.emit(c, "v_cmp_lt_f64", rs(small), rs(a, tiny), def(0), use(0), use(1))
	one, e := t32(), t32()
	x.emit(c, "v_cndmask_b32", rs(one), rs(small), def(0), imm(0), imm(1), use(0))
	x.emit(c, "v_lshlrev_b32", rs(e), rs(one), def(0), imm(8), use(0))
	in := t()
	x.emit(c, "v_ldexp_f64", rs(in), rs(a, e), def(0), use(0), use(1))
	r := t()
	x.emit(c, "v_rsq_f64", rs(r), rs(in), def(0), use(0))
	y, h := t(), t()
	x.emit(c, "v_mul_f64", rs(y), rs(in, r), def(0), use(0), use(1))
	x.emit(c, "v_mul_f64", rs(h), rs(r), def(0), use(0), fimm(0.5))
	e1 := t()
	x.emit(c, "v_fma_f64", rs(e1), rs(h, y), def(0), neg(0), use(1), fimm(0.5))
	y2, h2 := t(), t()
	x.emit(c, "v_fma_f64", rs(y2), rs(y, e1), def(0), use(0), use(1), use(0))
	x.emit(c, "v_fma_f64", rs(h2), rs(h, e1), def(0), use(0), use(1), use(0))
	for i := 0; i < 2; i++ {
		e2, y3 := t(), t()
		x.emit(c, "v_fma_f64", rs(e2), rs(y2, in), def(0), neg(0), use(0), use(1))
		x.emit(c, "v_fma_f64", rs(y3), rs(e2, h2, y2), def(0), use(0), use(1), use(2))
		y2 = y3
	}
	back, out := t32(), t()
	neg128 := x.constV32(c, 0xffffff80)
	x.emit(c, "v_cndmask_b32", rs(back), rs(neg128, small), def(0), imm(0), use(0), use(1))
	x.emit(c, "v_ldexp_f64", rs(out), rs(y2, back), def(0), use(0), use(1))
	class := x.constV32(c, 0x260)
	special := m()
	x.emit(c, "v_cmp_class_f64", rs(special), rs(in, class), def(0), use(0), use(1))
	x.emit(c, "v_cndmask_b32", rs(d), rs(out, in, special), defLo(0), useLo(0), useLo(1), use(2))
	x.emit(c, "v_cndmask_b32", rs(d), rs(out, in, special, d), defHi(0), useHi(0), useHi(1), use(2))
}

// minMax is §A3's four: minnum and maxnum after quieting each operand,
// since v_min under IEEE mode answers a signalling NaN with a NaN where
// the row says the other operand; minimum and maximum as the plain
// instruction with NaN put back when either side is one.
func (x *fnState) minMax(c *cursor, in *ir.Inst) error {
	t := in.Op().Type
	d, err := x.result(in)
	if err != nil {
		return err
	}
	a, err := x.args(in)
	if err != nil {
		return err
	}
	w := v32
	if t == ir.TypeF64 {
		w = v64
	}
	base := "v_min"
	if in.Op().Verb == ir.VMaxNum || in.Op().Verb == ir.VMaximum {
		base = "v_max"
	}
	mn := floatMn(t, base)
	switch in.Op().Verb {
	case ir.VMinNum, ir.VMaxNum:
		qa, qb := x.vr.temp(w), x.vr.temp(w)
		x.emit(c, floatMn(t, "v_max"), rs(qa), rs(a[0]), def(0), use(0), use(0))
		x.emit(c, floatMn(t, "v_max"), rs(qb), rs(a[1]), def(0), use(0), use(0))
		x.emit(c, mn, rs(d), rs(qa, qb), def(0), use(0), use(1))
	default:
		r, ord := x.vr.temp(w), x.vr.temp(s64)
		x.emit(c, mn, rs(r), a, def(0), use(0), use(1))
		x.emit(c, floatMn(t, "v_cmp_o"), rs(ord), a, def(0), use(0), use(1))
		if w == v32 {
			nan := x.constV32(c, 0x7fc00000)
			x.emit(c, "v_cndmask_b32", rs(d), rs(nan, r, ord), def(0), use(0), use(1), use(2))
		} else {
			nan := x.constV64(c, 0x7ff8000000000000)
			x.emit(c, "v_cndmask_b32", rs(d), rs(nan, r, ord), defLo(0), useLo(0), useLo(1), use(2))
			x.emit(c, "v_cndmask_b32", rs(d), rs(nan, r, ord, d), defHi(0), useHi(0), useHi(1), use(2))
		}
	}
	return nil
}

// —— §C2 conversions ——

// floatToInt is the trapping and saturating rows together: the range
// test first where the row traps, then the conversion. v_cvt_i32_f32
// clamps and sends NaN to zero, which is the saturating row for 32
// bits; 64 bits is two conversions of the halves.
func (x *fnState) floatToInt(c *cursor, in *ir.Inst) error {
	t := in.Op().Type
	src := in.Arg(0).Type()
	verb := in.Op().Verb
	signed := verb == ir.VSCvtF32 || verb == ir.VSCvtF64 || verb == ir.VSCvtSatF32 || verb == ir.VSCvtSatF64
	trapping := verb == ir.VSCvtF32 || verb == ir.VSCvtF64 || verb == ir.VUCvtF32 || verb == ir.VUCvtF64
	d, err := x.result(in)
	if err != nil {
		return err
	}
	a, err := x.args(in)
	if err != nil {
		return err
	}
	v := a[0]
	if trapping {
		// trunc(v) is in range exactly when v is in (lo-1, hi+1); the
		// bounds are the source type's nearest representable ones, as
		// lower/ptx chooses them. Both compares are ordered, so a NaN
		// fails and traps.
		var lo, hi float64
		loCmp := "ge"
		switch {
		case signed && t == ir.TypeI32:
			lo, hi = -2147483648, 2147483648
			if src == ir.TypeF64 {
				lo, loCmp = -2147483649, "gt"
			}
		case signed:
			lo, hi = -9223372036854775808, 9223372036854775808
		case t == ir.TypeI32:
			lo, hi, loCmp = -1, 4294967296, "gt"
		default:
			lo, hi, loCmp = -1, 18446744073709551616, "gt"
		}
		var lor, hir mir.VReg
		if src == ir.TypeF32 {
			lor, hir = x.constV32(c, math.Float32bits(float32(lo))), x.constV32(c, math.Float32bits(float32(hi)))
		} else {
			lor, hir = x.constV64(c, math.Float64bits(lo)), x.constV64(c, math.Float64bits(hi))
		}
		m1, m2, ok, bad := x.vr.temp(s64), x.vr.temp(s64), x.vr.temp(s64), x.vr.temp(s64)
		x.emit(c, floatMn(src, "v_cmp_"+loCmp), rs(m1), rs(v, lor), def(0), use(0), use(1))
		x.emit(c, floatMn(src, "v_cmp_lt"), rs(m2), rs(v, hir), def(0), use(0), use(1))
		x.emit(c, "s_and_b64", rs(ok), rs(m1, m2), def(0), use(0), use(1))
		x.emit(c, "s_not_b64", rs(bad), rs(ok), def(0), use(0))
		x.trapIf(c, bad)
	}
	if t == ir.TypeI32 {
		mn := "v_cvt_i32"
		if !signed {
			mn = "v_cvt_u32"
		}
		x.emit(c, floatMn(src, mn), rs(d), rs(v), def(0), use(0))
		return nil
	}
	// 64 bits: the truncated magnitude split at 2^32 — the high half is
	// floor(|t| * 2^-32), the low half is what fma leaves — and the sign
	// put back over both halves.
	tr := x.vr.temp(widthOfFloat(src))
	x.emit(c, floatMn(src, "v_trunc"), rs(tr), rs(v), def(0), use(0))
	mag := tr
	var sign mir.VReg
	if signed {
		sign = x.vr.temp(v32)
		if src == ir.TypeF32 {
			x.emit(c, "v_ashrrev_i32", rs(sign), rs(tr), def(0), imm(31), use(0))
			mag = x.vr.temp(v32)
			x.emit(c, "v_and_b32", rs(mag), rs(tr), def(0), imm(0x7fffffff), use(0))
		} else {
			x.emit(c, "v_ashrrev_i32", rs(sign), rs(tr), def(0), imm(31), useHi(0))
			mag = x.vr.temp(v64)
			x.emit(c, "v_mov_b32", rs(mag), rs(tr), defLo(0), useLo(0))
			x.emit(c, "v_and_b32", rs(mag), rs(tr, mag), defHi(0), imm(0x7fffffff), useHi(0))
		}
	}
	if src == ir.TypeF32 {
		hiF, lof := x.vr.temp(v32), x.vr.temp(v32)
		x.emit(c, "v_mul_f32", rs(hiF), rs(mag), def(0), imm(0x2f800000), use(0))
		x.emit(c, "v_floor_f32", rs(hiF), rs(hiF), def(0), use(0))
		big := x.constV32(c, 0xcf800000)
		x.emit(c, "v_fma_f32", rs(lof), rs(hiF, big, mag), def(0), use(0), use(1), use(2))
		x.emit(c, "v_cvt_u32_f32", rs(d), rs(hiF), defHi(0), use(0))
		x.emit(c, "v_cvt_u32_f32", rs(d), rs(lof, d), defLo(0), use(0))
	} else {
		hiF, lof := x.vr.temp(v64), x.vr.temp(v64)
		down := x.constV32(c, 0xffffffe0) // -32 is past the inline constants
		x.emit(c, "v_ldexp_f64", rs(hiF), rs(mag, down), def(0), use(0), use(1))
		x.emit(c, "v_floor_f64", rs(hiF), rs(hiF), def(0), use(0))
		big := x.constV64(c, 0xc1f0000000000000)
		x.emit(c, "v_fma_f64", rs(lof), rs(hiF, big, mag), def(0), use(0), use(1), use(2))
		x.emit(c, "v_cvt_u32_f64", rs(d), rs(hiF), defHi(0), use(0))
		x.emit(c, "v_cvt_u32_f64", rs(d), rs(lof, d), defLo(0), use(0))
	}
	if signed {
		x.emit(c, "v_xor_b32", rs(d), rs(d, sign), defLo(0), useLo(0), use(1))
		x.emit(c, "v_xor_b32", rs(d), rs(d, sign), defHi(0), useHi(0), use(1))
		x.emit(c, "v_sub_co_u32", rs(d), rs(d, sign), defLo(0), vcc(), useLo(0), use(1))
		x.emit(c, "v_subb_co_u32", rs(d), rs(d, sign), defHi(0), vcc(), useHi(0), use(1), vcc())
	}
	return nil
}

func widthOfFloat(t ir.RegType) width {
	if t == ir.TypeF64 {
		return v64
	}
	return v32
}

// int64ToFloat is §C2's i64 rows. To f64, the halves convert exactly
// and one add rounds once. To f32 there is no such split — a double
// rounding through f64 is a ULP off — so the value is normalised: shifted
// left by its leading zeros, its top dword converted with the rest
// folded into a sticky bit, and scaled back with v_ldexp.
func (x *fnState) int64ToFloat(c *cursor, in *ir.Inst) error {
	t := in.Op().Type
	signed := in.Op().Verb == ir.VSCvtI64
	d, err := x.result(in)
	if err != nil {
		return err
	}
	a, err := x.args(in)
	if err != nil {
		return err
	}
	v := a[0]
	if t == ir.TypeF64 {
		hiF, loF := x.vr.temp(v64), x.vr.temp(v64)
		mn := "v_cvt_f64_u32"
		if signed {
			mn = "v_cvt_f64_i32"
		}
		x.emit(c, mn, rs(hiF), rs(v), def(0), useHi(0))
		x.emit(c, "v_ldexp_f64", rs(hiF), rs(hiF), def(0), use(0), imm(32))
		x.emit(c, "v_cvt_f64_u32", rs(loF), rs(v), def(0), useLo(0))
		x.emit(c, "v_add_f64", rs(d), rs(hiF, loF), def(0), use(0), use(1))
		return nil
	}
	mag := v
	var sign mir.VReg
	if signed {
		sign = x.vr.temp(v32)
		x.emit(c, "v_ashrrev_i32", rs(sign), rs(v), def(0), imm(31), useHi(0))
		mag = x.vr.temp(v64)
		x.emit(c, "v_xor_b32", rs(mag), rs(v, sign), defLo(0), useLo(0), use(1))
		x.emit(c, "v_xor_b32", rs(mag), rs(v, sign, mag), defHi(0), useHi(0), use(1))
		x.emit(c, "v_sub_co_u32", rs(mag), rs(mag, sign), defLo(0), vcc(), useLo(0), use(1))
		x.emit(c, "v_subb_co_u32", rs(mag), rs(mag, sign), defHi(0), vcc(), useHi(0), use(1), vcc())
	}
	lz := x.clz64(c, mag)
	n := x.vr.temp(v64)
	x.emit(c, "v_lshlrev_b64", rs(n), rs(lz, mag), def(0), use(0), use(1))
	rest, sticky, top := x.vr.temp(s64), x.vr.temp(v32), x.vr.temp(v32)
	x.emit(c, "v_cmp_ne_u32", rs(rest), rs(n), def(0), imm(0), useLo(0))
	x.emit(c, "v_cndmask_b32", rs(sticky), rs(rest), def(0), imm(0), imm(1), use(0))
	x.emit(c, "v_or_b32", rs(top), rs(n, sticky), def(0), useHi(0), use(1))
	f, e := x.vr.temp(v32), x.vr.temp(v32)
	x.emit(c, "v_cvt_f32_u32", rs(f), rs(top), def(0), use(0))
	x.emit(c, "v_sub_u32", rs(e), rs(lz), def(0), imm(32), use(0))
	if !signed {
		x.emit(c, "v_ldexp_f32", rs(d), rs(f, e), def(0), use(0), use(1))
		return nil
	}
	r, sb := x.vr.temp(v32), x.vr.temp(v32)
	x.emit(c, "v_ldexp_f32", rs(r), rs(f, e), def(0), use(0), use(1))
	x.emit(c, "v_and_b32", rs(sb), rs(sign), def(0), imm(-0x80000000), use(0))
	x.emit(c, "v_xor_b32", rs(d), rs(r, sb), def(0), use(0), use(1))
	return nil
}

// —— §A6 at 64 bits, and bswap ——

// clz64 is the count of leading zeros of a pair as a dword: the high
// half's where it is non-zero, else 32 past the low half's. v_ffbh says
// -1 for zero, so each half is clamped to 32 first.
func (x *fnState) clz64(c *cursor, v mir.VReg) mir.VReg {
	t := func() mir.VReg { return x.vr.temp(v32) }
	ch, cl, cl32, hiNZ, r := t(), t(), t(), x.vr.temp(s64), t()
	x.emit(c, "v_ffbh_u32", rs(ch), rs(v), def(0), useHi(0))
	x.emit(c, "v_ffbh_u32", rs(cl), rs(v), def(0), useLo(0))
	x.emit(c, "v_min_u32", rs(cl), rs(cl), def(0), imm(32), use(0))
	x.emit(c, "v_add_u32", rs(cl32), rs(cl), def(0), imm(32), use(0))
	x.emit(c, "v_cmp_ne_u32", rs(hiNZ), rs(v), def(0), imm(0), useHi(0))
	x.emit(c, "v_cndmask_b32", rs(r), rs(cl32, ch, hiNZ), def(0), use(0), use(1), use(2))
	return r
}

func (x *fnState) bits64(c *cursor, in *ir.Inst) error {
	d, err := x.result(in)
	if err != nil {
		return err
	}
	a, err := x.args(in)
	if err != nil {
		return err
	}
	v := a[0]
	t := func() mir.VReg { return x.vr.temp(v32) }
	switch in.Op().Verb {
	case ir.VClz:
		r := x.clz64(c, v)
		x.emit(c, "v_mov_b32", rs(d), rs(r), defLo(0), use(0))
	case ir.VCtz:
		cl, ch, ch32, loNZ := t(), t(), t(), x.vr.temp(s64)
		x.emit(c, "v_ffbl_b32", rs(cl), rs(v), def(0), useLo(0))
		x.emit(c, "v_ffbl_b32", rs(ch), rs(v), def(0), useHi(0))
		x.emit(c, "v_min_u32", rs(ch), rs(ch), def(0), imm(32), use(0))
		x.emit(c, "v_add_u32", rs(ch32), rs(ch), def(0), imm(32), use(0))
		x.emit(c, "v_cmp_ne_u32", rs(loNZ), rs(v), def(0), imm(0), useLo(0))
		x.emit(c, "v_cndmask_b32", rs(d), rs(ch32, cl, loNZ), defLo(0), use(0), use(1), use(2))
	case ir.VPopcnt:
		lo := t()
		x.emit(c, "v_bcnt_u32_b32", rs(lo), rs(v), def(0), useLo(0), imm(0))
		x.emit(c, "v_bcnt_u32_b32", rs(d), rs(v, lo), defLo(0), useHi(0), use(1))
	}
	x.emit(c, "v_mov_b32", rs(d), rs(d), defHi(0), imm(0))
	return nil
}

// bswap is v_perm_b32 with the selector that reads the bytes backwards;
// at 64 bits each half is permuted into the other's place.
func (x *fnState) bswap(c *cursor, in *ir.Inst) error {
	d, err := x.result(in)
	if err != nil {
		return err
	}
	a, err := x.args(in)
	if err != nil {
		return err
	}
	sel := x.constV32(c, 0x00010203)
	if in.Op().Type == ir.TypeI32 {
		x.emit(c, "v_perm_b32", rs(d), rs(a[0], sel), def(0), imm(0), use(0), use(1))
		return nil
	}
	x.emit(c, "v_perm_b32", rs(d), rs(a[0], sel), defLo(0), imm(0), useHi(0), use(1))
	x.emit(c, "v_perm_b32", rs(d), rs(a[0], sel, d), defHi(0), imm(0), useLo(0), use(1))
	return nil
}

// rotate64 is two funnel shifts: the halves shifted into each other by
// the count and its complement, through a 64-bit shift each way.
func (x *fnState) rotate64(c *cursor, in *ir.Inst) error {
	d, err := x.result(in)
	if err != nil {
		return err
	}
	a, err := x.args(in)
	if err != nil {
		return err
	}
	// rotl(v, n) = (v << n) | (v >> (64-n)); rotr the other way. The
	// shift count is taken modulo 64 by the hardware, and 64-0 = 64 is
	// a shift by zero, which would double v — so the second shift is by
	// (63-n) then 1, the way LLVM does it.
	n, nn, t1, t2 := x.vr.temp(v32), x.vr.temp(v32), x.vr.temp(v64), x.vr.temp(v64)
	x.emit(c, "v_and_b32", rs(n), rs(a[1]), def(0), imm(63), useLo(0))
	x.emit(c, "v_sub_u32", rs(nn), rs(n), def(0), imm(63), use(0))
	first, second := "v_lshlrev_b64", "v_lshrrev_b64"
	if in.Op().Verb == ir.VRotR {
		first, second = second, first
	}
	x.emit(c, first, rs(t1), rs(n, a[0]), def(0), use(0), use(1))
	x.emit(c, second, rs(t2), rs(a[0]), def(0), imm(1), use(0))
	x.emit(c, second, rs(t2), rs(nn, t2), def(0), use(0), use(1))
	x.emit(c, "v_or_b32", rs(d), rs(t1, t2), defLo(0), useLo(0), useLo(1))
	x.emit(c, "v_or_b32", rs(d), rs(t1, t2, d), defHi(0), useHi(0), useHi(1))
	return nil
}

// mulHi64 is the high quadword of a 128-bit product: four v_mad_u64_u32
// partial products, each accumulating into the next, and for the signed
// row the two corrections a two's-complement operand owes.
func (x *fnState) mulHi64(c *cursor, in *ir.Inst) error {
	d, err := x.result(in)
	if err != nil {
		return err
	}
	a, err := x.args(in)
	if err != nil {
		return err
	}
	l, r := a[0], a[1]
	t := func() mir.VReg { return x.vr.temp(v64) }
	zero := x.constV64(c, 0)
	p00, acc1, acc2, p01, p10, p11 := t(), t(), t(), t(), t(), t()
	// p00 = l0*r0; the carry-out pair is scratch.
	x.emit(c, "v_mad_u64_u32", rs(p00), rs(l, r, zero), def(0), vcc(), useLo(0), useLo(1), use(2))
	// acc1 = hi(p00) as a quadword.
	x.emit(c, "v_mov_b32", rs(acc1), rs(p00), defLo(0), useHi(0))
	x.emit(c, "v_mov_b32", rs(acc1), rs(acc1), defHi(0), imm(0))
	// p01 = l0*r1 + hi(p00)
	x.emit(c, "v_mad_u64_u32", rs(p01), rs(l, r, acc1), def(0), vcc(), useLo(0), useHi(1), use(2))
	// acc2 = lo(p01) as a quadword; p10 = l1*r0 + lo(p01)
	x.emit(c, "v_mov_b32", rs(acc2), rs(p01), defLo(0), useLo(0))
	x.emit(c, "v_mov_b32", rs(acc2), rs(acc2), defHi(0), imm(0))
	x.emit(c, "v_mad_u64_u32", rs(p10), rs(l, r, acc2), def(0), vcc(), useHi(0), useLo(1), use(2))
	// p11 = l1*r1 + hi(p01) + hi(p10)
	acc3 := t()
	x.emit(c, "v_mov_b32", rs(acc3), rs(p01), defLo(0), useHi(0))
	x.emit(c, "v_mov_b32", rs(acc3), rs(acc3), defHi(0), imm(0))
	x.emit(c, "v_mad_u64_u32", rs(p11), rs(l, r, acc3), def(0), vcc(), useHi(0), useHi(1), use(2))
	hi := t()
	x.emit(c, "v_add_co_u32", rs(hi), rs(p11, p10), defLo(0), vcc(), useLo(0), useHi(1))
	x.emit(c, "v_addc_co_u32", rs(hi), rs(p11, hi), defHi(0), vcc(), useHi(0), imm(0), vcc())
	if in.Op().Verb == ir.VUMulHi {
		emitCopy(c, d, hi, v64)
		return nil
	}
	// Signed: subtract r where l is negative and l where r is, since
	// each was read 2^64 too large.
	sl, sr := x.vr.temp(v32), x.vr.temp(v32)
	x.emit(c, "v_ashrrev_i32", rs(sl), rs(l), def(0), imm(31), useHi(0))
	x.emit(c, "v_ashrrev_i32", rs(sr), rs(r), def(0), imm(31), useHi(0))
	c1, c2 := t(), t()
	x.emit(c, "v_and_b32", rs(c1), rs(r, sl), defLo(0), useLo(0), use(1))
	x.emit(c, "v_and_b32", rs(c1), rs(r, sl, c1), defHi(0), useHi(0), use(1))
	x.emit(c, "v_and_b32", rs(c2), rs(l, sr), defLo(0), useLo(0), use(1))
	x.emit(c, "v_and_b32", rs(c2), rs(l, sr, c2), defHi(0), useHi(0), use(1))
	s1 := t()
	x.emit(c, "v_sub_co_u32", rs(s1), rs(hi, c1), defLo(0), vcc(), useLo(0), useLo(1))
	x.emit(c, "v_subb_co_u32", rs(s1), rs(hi, c1, s1), defHi(0), vcc(), useHi(0), useHi(1), vcc())
	x.emit(c, "v_sub_co_u32", rs(d), rs(s1, c2), defLo(0), vcc(), useLo(0), useLo(1))
	x.emit(c, "v_subb_co_u32", rs(d), rs(s1, c2, d), defHi(0), vcc(), useHi(0), useHi(1), vcc())
	return nil
}

// copySign64 is bfi on the high halves.
func (x *fnState) copySign64(c *cursor, in *ir.Inst) error {
	d, err := x.result(in)
	if err != nil {
		return err
	}
	a, err := x.args(in)
	if err != nil {
		return err
	}
	m := x.constV32(c, 0x7fffffff)
	x.emit(c, "v_mov_b32", rs(d), rs(a[0]), defLo(0), useLo(0))
	x.emit(c, "v_bfi_b32", rs(d), rs(m, a[0], a[1], d), defHi(0), use(0), useHi(1), useHi(2))
	return nil
}

// overflow is §A2: the predicate rows, each the wrapping operation and
// the test LLVM uses for the same intrinsic, keeping only the flag.
func (x *fnState) overflow(c *cursor, in *ir.Inst) error {
	if in.Op().Type != ir.TypeI32 {
		return fmt.Errorf("a 64-bit overflow predicate is not lowered yet")
	}
	a, err := x.args(in)
	if err != nil {
		return err
	}
	o, err := x.result(in)
	if err != nil {
		return err
	}
	l, r := a[0], a[1]
	d := x.vr.temp(v32)
	switch in.Op().Verb {
	case ir.VUAddO:
		x.emit(c, "v_add_co_u32", rs(d, o), rs(l, r), def(0), def(1), use(0), use(1))
	case ir.VSAddO:
		// Signed overflow: the operands agree in sign and the sum does
		// not. (l ^ d) & (r ^ d) has the top bit set exactly then.
		x.emit(c, "v_add_u32", rs(d), rs(l, r), def(0), use(0), use(1))
		t1, t2, t3 := x.vr.temp(v32), x.vr.temp(v32), x.vr.temp(v32)
		x.emit(c, "v_xor_b32", rs(t1), rs(l, d), def(0), use(0), use(1))
		x.emit(c, "v_xor_b32", rs(t2), rs(r, d), def(0), use(0), use(1))
		x.emit(c, "v_and_b32", rs(t3), rs(t1, t2), def(0), use(0), use(1))
		x.emit(c, "v_cmp_lt_i32", rs(o), rs(t3), def(0), use(0), imm(0))
	case ir.VSSubO:
		x.emit(c, "v_sub_u32", rs(d), rs(l, r), def(0), use(0), use(1))
		t1, t2, t3 := x.vr.temp(v32), x.vr.temp(v32), x.vr.temp(v32)
		x.emit(c, "v_xor_b32", rs(t1), rs(l, r), def(0), use(0), use(1))
		x.emit(c, "v_xor_b32", rs(t2), rs(l, d), def(0), use(0), use(1))
		x.emit(c, "v_and_b32", rs(t3), rs(t1, t2), def(0), use(0), use(1))
		x.emit(c, "v_cmp_lt_i32", rs(o), rs(t3), def(0), use(0), imm(0))
	case ir.VUMulO:
		x.emit(c, "v_mul_hi_u32", rs(d), rs(l, r), def(0), use(0), use(1))
		x.emit(c, "v_cmp_ne_u32", rs(o), rs(d), def(0), imm(0), use(0))
	case ir.VSMulO:
		// Overflow unless the high half is the sign extension of the low.
		hi, ext := x.vr.temp(v32), x.vr.temp(v32)
		x.emit(c, "v_mul_lo_u32", rs(d), rs(l, r), def(0), use(0), use(1))
		x.emit(c, "v_mul_hi_i32", rs(hi), rs(l, r), def(0), use(0), use(1))
		x.emit(c, "v_ashrrev_i32", rs(ext), rs(d), def(0), imm(31), use(0))
		x.emit(c, "v_cmp_ne_u32", rs(o), rs(hi, ext), def(0), use(0), use(1))
	}
	return nil
}
