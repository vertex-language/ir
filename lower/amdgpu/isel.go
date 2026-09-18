package amdgpu

// Instruction selection: one VIR instruction to one or several machine
// instructions, in terms of vregs.
//
// The rows that take more than one are the 64-bit ones, which the VALU
// does in halves — an add is v_add_co_u32 and v_addc_co_u32 through VCC —
// and the rows where the hardware's answer at the edge is not §0's: ffbh
// says -1 for zero where clz says 32, and v_sin takes revolutions rather
// than radians.

import (
	"fmt"
	"math"

	"github.com/vertex-language/ir"
	"github.com/vertex-language/ir/lower/globals"
	"github.com/vertex-language/ir/lower/mir"
)

// The operand template constructors.
func def(i int) opnd   { return opnd{kind: oDef, i: i} }
func use(i int) opnd   { return opnd{kind: oUse, i: i} }
func defLo(i int) opnd { return opnd{kind: oDefLo, i: i} }
func defHi(i int) opnd { return opnd{kind: oDefHi, i: i} }
func useLo(i int) opnd { return opnd{kind: oUseLo, i: i} }
func useHi(i int) opnd { return opnd{kind: oUseHi, i: i} }
func imm(v int64) opnd { return opnd{kind: oImm, imm: v} }
func vcc() opnd        { return opnd{kind: oVCC} }
func flat(i int) opnd  { return opnd{kind: oFlat, i: i} }

// emit is one machine instruction.
func (x *fnState) emit(c *cursor, mn string, defs, uses []mir.VReg, ops ...opnd) {
	c.Emit(mir.Instr{Op: amdOp{mn: mn, ops: ops}, Defs: defs, Uses: uses})
}

// emitMem is one memory instruction, followed by a wait.
func (x *fnState) emitMem(c *cursor, mn string, defs, uses []mir.VReg, ops ...opnd) {
	c.Emit(mir.Instr{Op: amdOp{mn: mn, ops: ops, wait: true}, Defs: defs, Uses: uses})
}

// result defines the instruction's single result.
func (x *fnState) result(in *ir.Inst) (mir.VReg, error) { return x.vr.define(in.Result(0)) }

// args are the instruction's register operands.
func (x *fnState) args(in *ir.Inst) ([]mir.VReg, error) {
	out := make([]mir.VReg, len(in.Args()))
	for i, a := range in.Args() {
		r, err := x.vr.use(a)
		if err != nil {
			return nil, fmt.Errorf("operand %d: %w", i, err)
		}
		out[i] = r
	}
	return out, nil
}

func (x *fnState) selectInst(c *cursor, in *ir.Inst) error {
	op := in.Op()
	if op.IsBare() {
		return x.selectBare(c, in)
	}
	t := op.Type
	switch op.Verb {
	case ir.VConst:
		return x.constant(c, in)

	// —— §A ——
	case ir.VAdd, ir.VSub:
		return x.addSub(c, in)
	case ir.VMul:
		return x.mul(c, in)
	case ir.VSMulHi, ir.VUMulHi:
		if t != ir.TypeI32 {
			return x.mulHi64(c, in)
		}
		mn := "v_mul_hi_i32"
		if op.Verb == ir.VUMulHi {
			mn = "v_mul_hi_u32"
		}
		return x.binary(c, in, mn)
	case ir.VSDiv, ir.VUDiv, ir.VSRem, ir.VURem:
		if t != ir.TypeI32 {
			return fmt.Errorf("the 64-bit divide expansion is not lowered yet")
		}
		return x.divRem32(c, in)
	case ir.VNeg:
		return x.neg(c, in)

	// —— §A3 ——
	case ir.VDiv:
		d, err := x.result(in)
		if err != nil {
			return err
		}
		a, err := x.args(in)
		if err != nil {
			return err
		}
		if t == ir.TypeF64 {
			x.fdiv64(c, d, a[0], a[1])
		} else {
			x.fdiv32(c, d, a[0], a[1])
		}
		return nil
	case ir.VSqrt:
		d, err := x.result(in)
		if err != nil {
			return err
		}
		a, err := x.args(in)
		if err != nil {
			return err
		}
		if t == ir.TypeF64 {
			x.fsqrt64(c, d, a[0])
		} else {
			x.fsqrt32(c, d, a[0])
		}
		return nil
	case ir.VFMA:
		if t == ir.TypeF64 {
			return x.ternary(c, in, "v_fma_f64")
		}
		return x.ternary(c, in, "v_fma_f32")
	case ir.VAbs:
		return x.signBit(c, in, "v_and_b32", 0x7fffffff)
	case ir.VMinNum, ir.VMaxNum, ir.VMinimum, ir.VMaximum:
		return x.minMax(c, in)
	case ir.VCopySign:
		if t != ir.TypeF32 {
			return x.copySign64(c, in)
		}
		// bfi: (mask & a) | (~mask & b), with the mask clearing the
		// sign. VOP3 takes no literal on GFX9, so the mask is a move.
		d, err := x.result(in)
		if err != nil {
			return err
		}
		a, err := x.args(in)
		if err != nil {
			return err
		}
		m := x.vr.temp(v32)
		x.emit(c, "v_mov_b32", []mir.VReg{m}, nil, def(0), imm(0x7fffffff))
		x.emit(c, "v_bfi_b32", []mir.VReg{d}, []mir.VReg{m, a[0], a[1]}, def(0), use(0), use(1), use(2))
		return nil
	case ir.VCeil:
		return x.unary(c, in, floatMn(t, "v_ceil"))
	case ir.VFloor:
		return x.unary(c, in, floatMn(t, "v_floor"))
	case ir.VTrunc:
		return x.unary(c, in, floatMn(t, "v_trunc"))
	case ir.VNearest:
		return x.unary(c, in, floatMn(t, "v_rndne"))
	case ir.VRcpApprox:
		return x.unary(c, in, "v_rcp_f32")
	case ir.VRsqrtApprox:
		return x.unary(c, in, "v_rsq_f32")
	case ir.VExp2Approx:
		return x.unary(c, in, "v_exp_f32")
	case ir.VLog2Approx:
		return x.unary(c, in, "v_log_f32")
	case ir.VSinApprox, ir.VCosApprox:
		// v_sin and v_cos take revolutions: radians over 2π first.
		d, err := x.result(in)
		if err != nil {
			return err
		}
		a, err := x.args(in)
		if err != nil {
			return err
		}
		rev := x.vr.temp(v32)
		x.emit(c, "v_mul_f32", []mir.VReg{rev}, a, def(0), imm(int64(math.Float32bits(float32(1/(2*math.Pi))))), use(0))
		mn := "v_sin_f32"
		if op.Verb == ir.VCosApprox {
			mn = "v_cos_f32"
		}
		x.emit(c, mn, []mir.VReg{d}, []mir.VReg{rev}, def(0), use(0))
		return nil

	// —— §A2 ——
	case ir.VSAddO, ir.VUAddO, ir.VSSubO, ir.VSMulO, ir.VUMulO:
		return x.overflow(c, in)

	// —— §A4 ——
	case ir.VNot:
		return x.not(c, in)
	case ir.VAnd:
		return x.bitwise(c, in, "and")
	case ir.VOr:
		return x.bitwise(c, in, "or")
	case ir.VXor:
		return x.bitwise(c, in, "xor")

	// —— §A5 ——
	case ir.VShl:
		return x.shift(c, in, "v_lshlrev")
	case ir.VUShr:
		return x.shift(c, in, "v_lshrrev")
	case ir.VSShr:
		return x.shift(c, in, "v_ashrrev")
	case ir.VRotL, ir.VRotR:
		if t == ir.TypeI64 {
			return x.rotate64(c, in)
		}
		return x.rotate(c, in)

	// —— §A6 ——
	case ir.VClz, ir.VCtz, ir.VPopcnt:
		if t == ir.TypeI64 {
			return x.bits64(c, in)
		}
		return x.bits(c, in)
	case ir.VBswap:
		return x.bswap(c, in)

	// —— §B ——
	case ir.VEq, ir.VNe, ir.VSLt, ir.VULt, ir.VSLe, ir.VULe, ir.VLt, ir.VLe, ir.VUno:
		return x.compare(c, in)

	// —— §C ——
	case ir.VWrapI64:
		d, err := x.result(in)
		if err != nil {
			return err
		}
		a, err := x.args(in)
		if err != nil {
			return err
		}
		x.emit(c, "v_mov_b32", []mir.VReg{d}, a, def(0), useLo(0))
		return nil
	case ir.VSExtI32, ir.VZExtI32:
		d, err := x.result(in)
		if err != nil {
			return err
		}
		a, err := x.args(in)
		if err != nil {
			return err
		}
		x.emit(c, "v_mov_b32", []mir.VReg{d}, a, defLo(0), use(0))
		if op.Verb == ir.VSExtI32 {
			x.emit(c, "v_ashrrev_i32", []mir.VReg{d}, []mir.VReg{a[0], d}, defHi(0), imm(31), use(0))
		} else {
			x.emit(c, "v_mov_b32", []mir.VReg{d}, []mir.VReg{d}, defHi(0), imm(0))
		}
		return nil
	case ir.VZExtI1:
		d, err := x.result(in)
		if err != nil {
			return err
		}
		a, err := x.args(in)
		if err != nil {
			return err
		}
		if t == ir.TypeI64 {
			x.emit(c, "v_cndmask_b32", []mir.VReg{d}, a, defLo(0), imm(0), imm(1), use(0))
			x.emit(c, "v_mov_b32", []mir.VReg{d}, []mir.VReg{d}, defHi(0), imm(0))
			return nil
		}
		x.emit(c, "v_cndmask_b32", []mir.VReg{d}, a, def(0), imm(0), imm(1), use(0))
		return nil

	// —— §C2, §C3, §C4 ——
	case ir.VSCvtI32, ir.VUCvtI32:
		mn := "v_cvt_f32_i32"
		if op.Verb == ir.VUCvtI32 {
			mn = "v_cvt_f32_u32"
		}
		if t == ir.TypeF64 {
			mn = "v_cvt_f64_i32"
			if op.Verb == ir.VUCvtI32 {
				mn = "v_cvt_f64_u32"
			}
		}
		return x.unary(c, in, mn)
	case ir.VSCvtI64, ir.VUCvtI64:
		return x.int64ToFloat(c, in)
	case ir.VSCvtSatF32, ir.VUCvtSatF32, ir.VSCvtSatF64, ir.VUCvtSatF64,
		ir.VSCvtF32, ir.VUCvtF32, ir.VSCvtF64, ir.VUCvtF64:
		return x.floatToInt(c, in)
	case ir.VFCvtF32:
		return x.unary(c, in, "v_cvt_f64_f32")
	case ir.VFCvtF64:
		return x.unary(c, in, "v_cvt_f32_f64")
	case ir.VBitcastF32, ir.VBitcastI32, ir.VBitcastF64, ir.VBitcastI64, ir.VFromI64, ir.VFromPtr:
		d, err := x.result(in)
		if err != nil {
			return err
		}
		a, err := x.args(in)
		if err != nil {
			return err
		}
		emitCopy(c, d, a[0], x.vr.widthOfVReg(d))
		return nil

	// —— §D ——
	case ir.VLoad, ir.VSLoad8, ir.VSLoad16, ir.VSLoad32, ir.VULoad8, ir.VULoad16, ir.VULoad32:
		return x.load(c, in)
	case ir.VStore, ir.VStore8, ir.VStore16, ir.VStore32:
		return x.store(c, in)

	// —— §D3 ——
	case ir.VGetAddr:
		return x.getaddr(c, in)
	case ir.VDiff:
		return x.addSub(c, in)
	case ir.VAlloc:
		return x.alloc(c, in)
	case ir.VAlloca, ir.VStackSave, ir.VStackRestore:
		return fmt.Errorf("a dynamic alloca needs a stack pointer, which a kernel has no convention for yet")
	case ir.VTLSAddr, ir.VBlockAddr, ir.VFrameAddr, ir.VReturnAddr:
		return fmt.Errorf("no lowering on the device")

	// —— §F ——
	case ir.VSelect:
		return x.selectV(c, in)

	// —— §E ——
	case ir.VMemCmp:
		return x.bulk(c, in)

	// —— §W ——
	case ir.VWorkitemID, ir.VWorkgroupID, ir.VWorkgroupSize, ir.VNumWorkgroups, ir.VLaneID, ir.VWaveSize:
		return x.workitem(c, in)
	case ir.VWaveShflIdx, ir.VWaveShflUp, ir.VWaveShflDown, ir.VWaveShflXor,
		ir.VWaveReadFirstLane, ir.VWaveBallot, ir.VWaveAny, ir.VWaveAll:
		return x.wave(c, in)

	// —— §H ——
	case ir.VAtomicLoad, ir.VAtomicULoad8, ir.VAtomicULoad16,
		ir.VAtomicStore, ir.VAtomicStore8, ir.VAtomicStore16,
		ir.VAtomicCas, ir.VAtomicCas8, ir.VAtomicCas16,
		ir.VAtomicRmwAdd, ir.VAtomicRmwSub, ir.VAtomicRmwAnd, ir.VAtomicRmwOr,
		ir.VAtomicRmwXor, ir.VAtomicRmwXchg,
		ir.VAtomicRmwSMin, ir.VAtomicRmwSMax, ir.VAtomicRmwUMin, ir.VAtomicRmwUMax,
		ir.VAtomicRmwAdd8, ir.VAtomicRmwSub8, ir.VAtomicRmwAnd8, ir.VAtomicRmwOr8, ir.VAtomicRmwXor8, ir.VAtomicRmwXchg8,
		ir.VAtomicRmwAdd16, ir.VAtomicRmwSub16, ir.VAtomicRmwAnd16, ir.VAtomicRmwOr16, ir.VAtomicRmwXor16, ir.VAtomicRmwXchg16:
		return x.atomic(c, in)
	}
	return fmt.Errorf("not lowered")
}

func (x *fnState) selectBare(c *cursor, in *ir.Inst) error {
	switch in.Op().Verb {
	case ir.VBarrier:
		c.Emit(mir.Instr{Op: amdOp{mn: "s_waitcnt", ops: []opnd{{kind: oWait}}}})
		x.emit(c, "s_barrier", nil, nil)
		return nil
	case ir.VFence:
		return x.fence(c, in)
	case ir.VCall, ir.VCallInd:
		return fmt.Errorf("calls are not lowered yet")
	case ir.VMemCpy, ir.VMemMove, ir.VMemSet, ir.VMemCmp:
		return x.bulk(c, in)
	}
	return fmt.Errorf("not lowered")
}

// floatMn is a float mnemonic at the namespace's width.
func floatMn(t ir.RegType, base string) string {
	if t == ir.TypeF64 {
		return base + "_f64"
	}
	return base + "_f32"
}

// —— constants ——

func (x *fnState) constant(c *cursor, in *ir.Inst) error {
	d, err := x.result(in)
	if err != nil {
		return err
	}
	lit, _ := in.Lit()
	t := in.Op().Type
	switch t {
	case ir.TypeF32:
		if lit.Kind() != ir.ConstFloat {
			return fmt.Errorf("a float constant needs a float literal")
		}
		x.emit(c, "v_mov_b32", []mir.VReg{d}, nil, def(0), imm(int64(math.Float32bits(float32(lit.Float())))))
		return nil
	case ir.TypeF64:
		if lit.Kind() != ir.ConstFloat {
			return fmt.Errorf("a float constant needs a float literal")
		}
		bits := math.Float64bits(lit.Float())
		x.emit(c, "v_mov_b32", []mir.VReg{d}, nil, defLo(0), imm(int64(uint32(bits))))
		x.emit(c, "v_mov_b32", []mir.VReg{d}, []mir.VReg{d}, defHi(0), imm(int64(uint32(bits>>32))))
		return nil
	}
	v, err := globals.ConstInt(layout{}, lit)
	if err != nil {
		return err
	}
	switch t {
	case ir.TypeI1:
		mask := int64(0)
		if v != 0 {
			mask = -1
		}
		x.emit(c, "s_mov_b64", []mir.VReg{d}, nil, def(0), imm(mask))
	case ir.TypeI32:
		x.emit(c, "v_mov_b32", []mir.VReg{d}, nil, def(0), imm(int64(int32(v))))
	default:
		x.emit(c, "v_mov_b32", []mir.VReg{d}, nil, defLo(0), imm(int64(uint32(v))))
		x.emit(c, "v_mov_b32", []mir.VReg{d}, []mir.VReg{d}, defHi(0), imm(int64(uint32(uint64(v)>>32))))
	}
	return nil
}

// —— the shapes ——

func (x *fnState) unary(c *cursor, in *ir.Inst, mn string) error {
	d, err := x.result(in)
	if err != nil {
		return err
	}
	a, err := x.args(in)
	if err != nil {
		return err
	}
	x.emit(c, mn, []mir.VReg{d}, a, def(0), use(0))
	return nil
}

func (x *fnState) binary(c *cursor, in *ir.Inst, mn string) error {
	d, err := x.result(in)
	if err != nil {
		return err
	}
	a, err := x.args(in)
	if err != nil {
		return err
	}
	x.emit(c, mn, []mir.VReg{d}, a, def(0), use(0), use(1))
	return nil
}

func (x *fnState) ternary(c *cursor, in *ir.Inst, mn string) error {
	d, err := x.result(in)
	if err != nil {
		return err
	}
	a, err := x.args(in)
	if err != nil {
		return err
	}
	x.emit(c, mn, []mir.VReg{d}, a, def(0), use(0), use(1), use(2))
	return nil
}

// addSub is §A's add and sub, ptr.add, ptr.sub and ptr.diff.
func (x *fnState) addSub(c *cursor, in *ir.Inst) error {
	t := in.Op().Type
	sub := in.Op().Verb == ir.VSub || in.Op().Verb == ir.VDiff
	d, err := x.result(in)
	if err != nil {
		return err
	}
	a, err := x.args(in)
	if err != nil {
		return err
	}
	switch t {
	case ir.TypeF32, ir.TypeF64:
		if t == ir.TypeF64 && sub {
			// v_add_f64 with the second operand's sign flipped: the
			// encoder has no neg modifier, so flip it in a copy.
			n := x.vr.temp(v64)
			x.emit(c, "v_mov_b32", []mir.VReg{n}, a, defLo(0), useLo(1))
			x.emit(c, "v_xor_b32", []mir.VReg{n}, []mir.VReg{a[1], n}, defHi(0), imm(-0x80000000), useHi(0))
			x.emit(c, "v_add_f64", []mir.VReg{d}, []mir.VReg{a[0], n}, def(0), use(0), use(1))
			return nil
		}
		mn := floatMn(t, "v_add")
		if sub {
			mn = "v_sub_f32"
		}
		x.emit(c, mn, []mir.VReg{d}, a, def(0), use(0), use(1))
	case ir.TypeI32:
		mn := "v_add_u32"
		if sub {
			mn = "v_sub_u32"
		}
		x.emit(c, mn, []mir.VReg{d}, a, def(0), use(0), use(1))
	default:
		// 64-bit, in halves through VCC. ptr.add's second operand is an
		// i64 and ptr.diff's operands are pointers; both are pairs.
		lo, hi := "v_add_co_u32", "v_addc_co_u32"
		if sub {
			lo, hi = "v_sub_co_u32", "v_subb_co_u32"
		}
		x.emit(c, lo, []mir.VReg{d}, a, defLo(0), vcc(), useLo(0), useLo(1))
		x.emit(c, hi, []mir.VReg{d}, []mir.VReg{a[0], a[1], d}, defHi(0), vcc(), useHi(0), useHi(1), vcc())
	}
	return nil
}

func (x *fnState) mul(c *cursor, in *ir.Inst) error {
	t := in.Op().Type
	d, err := x.result(in)
	if err != nil {
		return err
	}
	a, err := x.args(in)
	if err != nil {
		return err
	}
	switch t {
	case ir.TypeF32, ir.TypeF64:
		x.emit(c, floatMn(t, "v_mul"), []mir.VReg{d}, a, def(0), use(0), use(1))
	case ir.TypeI32:
		x.emit(c, "v_mul_lo_u32", []mir.VReg{d}, a, def(0), use(0), use(1))
	default:
		// lo = lo(a)*lo(b); hi = mulhi(lo(a), lo(b)) + hi(a)*lo(b) + lo(a)*hi(b).
		t1, t2, t3 := x.vr.temp(v32), x.vr.temp(v32), x.vr.temp(v32)
		x.emit(c, "v_mul_lo_u32", []mir.VReg{d}, a, defLo(0), useLo(0), useLo(1))
		x.emit(c, "v_mul_hi_u32", []mir.VReg{t1}, a, def(0), useLo(0), useLo(1))
		x.emit(c, "v_mul_lo_u32", []mir.VReg{t2}, a, def(0), useHi(0), useLo(1))
		x.emit(c, "v_mul_lo_u32", []mir.VReg{t3}, a, def(0), useLo(0), useHi(1))
		x.emit(c, "v_add_u32", []mir.VReg{t1}, []mir.VReg{t1, t2}, def(0), use(0), use(1))
		x.emit(c, "v_add_u32", []mir.VReg{d}, []mir.VReg{t1, t3, d}, defHi(0), use(0), use(1))
	}
	return nil
}

func (x *fnState) neg(c *cursor, in *ir.Inst) error {
	t := in.Op().Type
	d, err := x.result(in)
	if err != nil {
		return err
	}
	a, err := x.args(in)
	if err != nil {
		return err
	}
	switch t {
	case ir.TypeF32:
		x.emit(c, "v_xor_b32", []mir.VReg{d}, a, def(0), imm(-0x80000000), use(0))
	case ir.TypeF64:
		x.emit(c, "v_mov_b32", []mir.VReg{d}, a, defLo(0), useLo(0))
		x.emit(c, "v_xor_b32", []mir.VReg{d}, []mir.VReg{a[0], d}, defHi(0), imm(-0x80000000), useHi(0))
	case ir.TypeI32:
		x.emit(c, "v_sub_u32", []mir.VReg{d}, a, def(0), imm(0), use(0))
	default:
		x.emit(c, "v_sub_co_u32", []mir.VReg{d}, a, defLo(0), vcc(), imm(0), useLo(0))
		x.emit(c, "v_subb_co_u32", []mir.VReg{d}, []mir.VReg{a[0], d}, defHi(0), vcc(), imm(0), useHi(0), vcc())
	}
	return nil
}

// signBit is abs: the sign cleared, in the high half for an f64.
func (x *fnState) signBit(c *cursor, in *ir.Inst, mn string, mask int64) error {
	d, err := x.result(in)
	if err != nil {
		return err
	}
	a, err := x.args(in)
	if err != nil {
		return err
	}
	if in.Op().Type == ir.TypeF64 {
		x.emit(c, "v_mov_b32", []mir.VReg{d}, a, defLo(0), useLo(0))
		x.emit(c, mn, []mir.VReg{d}, []mir.VReg{a[0], d}, defHi(0), imm(mask), useHi(0))
		return nil
	}
	x.emit(c, mn, []mir.VReg{d}, a, def(0), imm(mask), use(0))
	return nil
}

func (x *fnState) not(c *cursor, in *ir.Inst) error {
	d, err := x.result(in)
	if err != nil {
		return err
	}
	a, err := x.args(in)
	if err != nil {
		return err
	}
	switch in.Op().Type {
	case ir.TypeI1:
		x.emit(c, "s_not_b64", []mir.VReg{d}, a, def(0), use(0))
	case ir.TypeI32:
		x.emit(c, "v_not_b32", []mir.VReg{d}, a, def(0), use(0))
	default:
		x.emit(c, "v_not_b32", []mir.VReg{d}, a, defLo(0), useLo(0))
		x.emit(c, "v_not_b32", []mir.VReg{d}, []mir.VReg{a[0], d}, defHi(0), useHi(0))
	}
	return nil
}

func (x *fnState) bitwise(c *cursor, in *ir.Inst, verb string) error {
	d, err := x.result(in)
	if err != nil {
		return err
	}
	a, err := x.args(in)
	if err != nil {
		return err
	}
	switch in.Op().Type {
	case ir.TypeI1:
		x.emit(c, "s_"+verb+"_b64", []mir.VReg{d}, a, def(0), use(0), use(1))
	case ir.TypeI32:
		x.emit(c, "v_"+verb+"_b32", []mir.VReg{d}, a, def(0), use(0), use(1))
	default:
		x.emit(c, "v_"+verb+"_b32", []mir.VReg{d}, a, defLo(0), useLo(0), useLo(1))
		x.emit(c, "v_"+verb+"_b32", []mir.VReg{d}, []mir.VReg{a[0], a[1], d}, defHi(0), useHi(0), useHi(1))
	}
	return nil
}

// shift is §A5: the count comes first in the mnemonic's operand order,
// and the hardware masks it to the width as §0 asks.
func (x *fnState) shift(c *cursor, in *ir.Inst, mn string) error {
	d, err := x.result(in)
	if err != nil {
		return err
	}
	a, err := x.args(in)
	if err != nil {
		return err
	}
	suffix := "_b"
	if mn == "v_ashrrev" {
		suffix = "_i"
	}
	if in.Op().Type == ir.TypeI32 {
		x.emit(c, mn+suffix+"32", []mir.VReg{d}, a, def(0), use(1), use(0))
		return nil
	}
	name := mn + suffix + "64"
	// The count of an i64 shift is an i64; the low half is what the
	// instruction reads.
	x.emit(c, name, []mir.VReg{d}, a, def(0), useLo(1), use(0))
	return nil
}

// rotate is v_alignbit_b32 of a value with itself: a funnel shift, which
// rotates right by its count; left is right by the count's negation.
func (x *fnState) rotate(c *cursor, in *ir.Inst) error {
	if in.Op().Type != ir.TypeI32 {
		return fmt.Errorf("a 64-bit rotate is not lowered yet")
	}
	d, err := x.result(in)
	if err != nil {
		return err
	}
	a, err := x.args(in)
	if err != nil {
		return err
	}
	n := a[1]
	if in.Op().Verb == ir.VRotL {
		n = x.vr.temp(v32)
		x.emit(c, "v_sub_u32", []mir.VReg{n}, []mir.VReg{a[1]}, def(0), imm(0), use(0))
	}
	x.emit(c, "v_alignbit_b32", []mir.VReg{d}, []mir.VReg{a[0], n}, def(0), use(0), use(0), use(1))
	return nil
}

// bits is §A6: ffbh and ffbl answer -1 for zero where the IR says the
// width, so a min against the width follows them.
func (x *fnState) bits(c *cursor, in *ir.Inst) error {
	if in.Op().Type != ir.TypeI32 {
		return fmt.Errorf("64-bit bit counting is not lowered yet")
	}
	d, err := x.result(in)
	if err != nil {
		return err
	}
	a, err := x.args(in)
	if err != nil {
		return err
	}
	switch in.Op().Verb {
	case ir.VPopcnt:
		x.emit(c, "v_bcnt_u32_b32", []mir.VReg{d}, a, def(0), use(0), imm(0))
	default:
		mn := "v_ffbh_u32"
		if in.Op().Verb == ir.VCtz {
			mn = "v_ffbl_b32"
		}
		t := x.vr.temp(v32)
		x.emit(c, mn, []mir.VReg{t}, a, def(0), use(0))
		x.emit(c, "v_min_u32", []mir.VReg{d}, []mir.VReg{t}, def(0), imm(32), use(0))
	}
	return nil
}

// compare is §B: a VOPC compare in its VOP3 form, writing a lane mask.
func (x *fnState) compare(c *cursor, in *ir.Inst) error {
	t := in.Op().Type
	d, err := x.result(in)
	if err != nil {
		return err
	}
	a, err := x.args(in)
	if err != nil {
		return err
	}
	var suffix, cmp string
	switch t {
	case ir.TypeI32:
		suffix = "i32"
	case ir.TypeI64:
		suffix = "i64"
	case ir.TypePtr:
		suffix = "u64"
	case ir.TypeF32:
		suffix = "f32"
	case ir.TypeF64:
		suffix = "f64"
	default:
		return fmt.Errorf("no compare at %s", t)
	}
	switch in.Op().Verb {
	case ir.VEq:
		cmp = "eq"
	case ir.VNe:
		cmp = "ne"
		if t.IsFloat() {
			cmp = "neq" // unordered not-equal: true on a NaN, as §B says
		}
	case ir.VSLt, ir.VLt:
		cmp = "lt"
	case ir.VSLe, ir.VLe:
		cmp = "le"
	case ir.VULt:
		cmp, suffix = "lt", "u"+suffix[1:]
	case ir.VULe:
		cmp, suffix = "le", "u"+suffix[1:]
	case ir.VUno:
		cmp = "u"
	}
	x.emit(c, "v_cmp_"+cmp+"_"+suffix, []mir.VReg{d}, a, def(0), use(0), use(1))
	return nil
}

// selectV is §F: v_cndmask takes the false operand first.
func (x *fnState) selectV(c *cursor, in *ir.Inst) error {
	t := in.Op().Type
	d, err := x.result(in)
	if err != nil {
		return err
	}
	a, err := x.args(in)
	if err != nil {
		return err
	}
	cond, yes, no := a[0], a[1], a[2]
	switch t {
	case ir.TypeI1:
		t1, t2 := x.vr.temp(s64), x.vr.temp(s64)
		x.emit(c, "s_and_b64", []mir.VReg{t1}, []mir.VReg{cond, yes}, def(0), use(0), use(1))
		x.emit(c, "s_andn2_b64", []mir.VReg{t2}, []mir.VReg{no, cond}, def(0), use(0), use(1))
		x.emit(c, "s_or_b64", []mir.VReg{d}, []mir.VReg{t1, t2}, def(0), use(0), use(1))
	case ir.TypeI32, ir.TypeF32:
		x.emit(c, "v_cndmask_b32", []mir.VReg{d}, []mir.VReg{no, yes, cond}, def(0), use(0), use(1), use(2))
	default:
		x.emit(c, "v_cndmask_b32", []mir.VReg{d}, []mir.VReg{no, yes, cond}, defLo(0), useLo(0), useLo(1), use(2))
		x.emit(c, "v_cndmask_b32", []mir.VReg{d}, []mir.VReg{no, yes, cond, d}, defHi(0), useHi(0), useHi(1), use(2))
	}
	return nil
}

// —— memory ——

func (x *fnState) load(c *cursor, in *ir.Inst) error {
	t := in.Op().Type
	if t == ir.TypeI1 {
		return fmt.Errorf("i1 has no storage width")
	}
	if a, ok := in.Align(); ok && a < uint64(accessWidth(t, in.Op().Verb)) {
		return fmt.Errorf("an access below natural alignment is not lowered yet")
	}
	d, err := x.result(in)
	if err != nil {
		return err
	}
	a, err := x.args(in)
	if err != nil {
		return err
	}
	var mn string
	shared := sharedPtr(in.Arg(0))
	switch in.Op().Verb {
	case ir.VSLoad8:
		mn = pick(shared, "ds_read_i8", "flat_load_sbyte")
	case ir.VULoad8:
		mn = pick(shared, "ds_read_u8", "flat_load_ubyte")
	case ir.VSLoad16:
		mn = pick(shared, "ds_read_i16", "flat_load_sshort")
	case ir.VULoad16:
		mn = pick(shared, "ds_read_u16", "flat_load_ushort")
	case ir.VSLoad32, ir.VULoad32:
		mn = pick(shared, "ds_read_b32", "flat_load_dword")
	default:
		mn = pick(shared, "ds_read_b32", "flat_load_dword")
		if t == ir.TypeI64 || t == ir.TypePtr || t == ir.TypeF64 {
			mn = pick(shared, "ds_read_b64", "flat_load_dwordx2")
		}
	}
	addr := flat(0)
	if shared {
		addr = ds(0)
	}
	if t == ir.TypeI64 && in.Op().Verb != ir.VLoad {
		// A sub-width load into an i64 fills the low half; the high
		// half is the sign, or zero.
		x.emitMem(c, mn, []mir.VReg{d}, a, defLo(0), addr)
		if in.Op().Verb == ir.VSLoad8 || in.Op().Verb == ir.VSLoad16 || in.Op().Verb == ir.VSLoad32 {
			x.emit(c, "v_ashrrev_i32", []mir.VReg{d}, []mir.VReg{d}, defHi(0), imm(31), useLo(0))
		} else {
			x.emit(c, "v_mov_b32", []mir.VReg{d}, []mir.VReg{d}, defHi(0), imm(0))
		}
		return nil
	}
	x.emitMem(c, mn, []mir.VReg{d}, a, def(0), addr)
	return nil
}

// pick is the LDS or the flat spelling.
func pick(shared bool, ds, flat string) string {
	if shared {
		return ds
	}
	return flat
}

func (x *fnState) store(c *cursor, in *ir.Inst) error {
	t := in.Op().Type
	if t == ir.TypeI1 {
		return fmt.Errorf("i1 has no storage width")
	}
	if a, ok := in.Align(); ok && a < uint64(accessWidth(t, in.Op().Verb)) {
		return fmt.Errorf("an access below natural alignment is not lowered yet")
	}
	a, err := x.args(in)
	if err != nil {
		return err
	}
	var mn string
	val := use(0)
	shared := sharedPtr(in.Arg(1))
	switch in.Op().Verb {
	case ir.VStore8:
		mn = pick(shared, "ds_write_b8", "flat_store_byte")
	case ir.VStore16:
		mn = pick(shared, "ds_write_b16", "flat_store_short")
	case ir.VStore32:
		mn = pick(shared, "ds_write_b32", "flat_store_dword")
	default:
		mn = pick(shared, "ds_write_b32", "flat_store_dword")
		if t == ir.TypeI64 || t == ir.TypePtr || t == ir.TypeF64 {
			mn = pick(shared, "ds_write_b64", "flat_store_dwordx2")
		}
	}
	if t == ir.TypeI64 && in.Op().Verb != ir.VStore {
		val = useLo(0)
	}
	addr := flat(1)
	if shared {
		addr = ds(1)
	}
	x.emitMem(c, mn, nil, a, addr, val)
	return nil
}

// accessWidth is a §D or §D2 access's natural width in bytes.
func accessWidth(t ir.RegType, v ir.Verb) int {
	switch v {
	case ir.VSLoad8, ir.VULoad8, ir.VStore8:
		return 1
	case ir.VSLoad16, ir.VULoad16, ir.VStore16:
		return 2
	case ir.VSLoad32, ir.VULoad32, ir.VStore32:
		return 4
	}
	switch t {
	case ir.TypeI64, ir.TypePtr, ir.TypeF64:
		return 8
	}
	return 4
}

// getaddr is a global's address: the PC, plus the symbol's distance from
// it in two rel32 halves, into an SGPR pair and then into the VGPRs.
func (x *fnState) getaddr(c *cursor, in *ir.Inst) error {
	sym := in.Symbol()
	if f, ok := sym.(*ir.Func); ok {
		return fmt.Errorf("the address of @%s: functions are not lowered yet", f.Name())
	}
	d, err := x.result(in)
	if err != nil {
		return err
	}
	if g, ok := sym.(*ir.Global); ok && g.Domain() == ir.Shared {
		return x.getaddrShared(c, d, g)
	}
	pc, addr := x.vr.temp(s64), x.vr.temp(s64)
	x.emit(c, "s_getpc_b64", []mir.VReg{pc}, nil, def(0))
	c.Emit(mir.Instr{Op: amdOp{mn: "s_add_u32", ops: []opnd{defLo(0), useLo(0), {kind: oSymLo, sym: sym.Name(), imm: 4}}},
		Defs: []mir.VReg{addr}, Uses: []mir.VReg{pc}})
	c.Emit(mir.Instr{Op: amdOp{mn: "s_addc_u32", ops: []opnd{defHi(0), useHi(0), {kind: oSymHi, sym: sym.Name(), imm: 12}}},
		Defs: []mir.VReg{addr}, Uses: []mir.VReg{pc, addr}})
	x.movPair(c, d, addr)
	return nil
}
