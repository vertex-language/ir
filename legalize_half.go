package ir

// Legalizing f16 and bf16 into i32.
//
// The half-float namespaces are admitted on every stock target, and most
// of those targets have no half-precision arithmetic to lower them to.
// This pass runs before instruction selection, as LegalizeI128 does, and
// takes both types out of the module: every half value becomes an i32
// whose low sixteen bits are its encoding and whose high sixteen are
// zero, and every half instruction becomes the i32 and f32 instructions
// that compute the same bits. What reaches a backend afterwards is a
// module in the reg-types it already had.
//
// The arithmetic is done in f32 and rounded back, which is exact: f32's
// 24 bits of significand are at least twice either half format's plus
// two (2*11+2 for f16, 2*8+2 for bf16), and for +, -, *, / and sqrt that
// is the condition under which rounding to f32 and then to the half is
// rounding to the half once. The comparisons, min/max and the rounding
// verbs are exact in f32 outright. neg, abs and copysign are bit
// operations, which is what keeps a NaN's payload.
//
// fma and the narrowings from f64 are the cases the double-rounding
// argument does not cover, and each rounds to odd first: f64 to f32 is
// truncated towards zero with the lowest bit set when anything was lost,
// and a value rounded to odd with two bits to spare rounds to nearest
// correctly afterwards. fma computes a*b exactly in f64 and recovers the
// addition's error with TwoSum to do the same.
//
// The conversions are Giesen's branch-free ones written as selects: a
// narrowing to f16 computes the normal, subnormal and overflow answers
// and chooses, a widening from f16 re-biases the exponent and fixes up
// the subnormals with one f32 subtraction, and bf16 is a shift either
// way. A narrowed NaN is quieted and keeps the top of its payload.
//
// One conversion is not correctly rounded: i64 to bf16 goes through f64,
// so an integer above 2^53 may round twice. i64 to f16 does not care --
// everything that large overflows f16 -- and i32 goes through f64
// exactly.
//
// A half parameter or result is an i32 in the signature afterwards. That
// is this IR's convention for a legalized half and not the C ABI's for
// _Float16, which passes it in a float register on x86-64 and AArch64; a
// module calling C across a half-typed boundary needs a target that
// keeps the type.

// LegalizeHalf rewrites every f16 and bf16 value in the module into an
// i32 carrier, and reports whether it changed anything. It is safe to run
// on a module with no half floats, and on one already legalized.
func (m *Module) LegalizeHalf() bool {
	if m == nil || m.err != nil {
		return false
	}
	changed := false
	for _, it := range m.items {
		switch x := it.(type) {
		case *Func:
			if x.legalizeHalf() {
				changed = true
			}
			if retypeSig(x.sig) {
				changed = true
			}
		case *FuncImport:
			if retypeSig(x.sig) {
				changed = true
			}
		case *Type:
			if retypeSig(x.sig) {
				changed = true
			}
		}
	}
	return changed
}

// retypeSig makes a signature's half parameters and results i32s.
func retypeSig(s *Sig) bool {
	if s == nil {
		return false
	}
	changed := false
	for i := range s.params {
		if s.params[i].Type.IsHalfFloat() {
			s.params[i].Type = TypeI32
			changed = true
		}
	}
	for i := range s.rets {
		if s.rets[i].Type.IsHalfFloat() {
			s.rets[i].Type = TypeI32
			changed = true
		}
	}
	return changed
}

func (f *Func) legalizeHalf() bool {
	if f == nil || f.m == nil || f.m.err != nil || len(f.blocks) == 0 || !f.usesHalf() {
		return false
	}
	p := &halfPass{f: f, repl: map[*Def]*Def{}}
	for _, b := range f.blocks {
		p.blk = b
		p.out = make([]*Inst, 0, len(b.insts)+8)
		for _, in := range b.insts {
			if !p.expand(in) {
				p.out = append(p.out, in)
			}
		}
		b.insts = p.out
	}
	if len(p.repl) > 0 {
		f.WalkUses(func(u Use) bool {
			if d := u.Def(); d != nil {
				if r, ok := p.repl[d]; ok {
					u.Set(r)
				}
			}
			return true
		})
	}
	// Every half definition left -- a parameter, a block parameter, the
	// result of a load, select, const or call this pass kept -- is now
	// the i32 that carries it.
	retype := func(d *Def) {
		if d != nil && d.typ.IsHalfFloat() {
			d.typ = TypeI32
		}
	}
	for _, d := range f.params {
		retype(d)
	}
	for _, b := range f.blocks {
		for _, d := range b.params {
			retype(d)
		}
		for _, in := range b.All() {
			for _, d := range in.results {
				retype(d)
			}
		}
	}
	return true
}

// usesHalf reports whether any definition or instruction in f is a half.
func (f *Func) usesHalf() bool {
	for _, d := range f.params {
		if d.typ.IsHalfFloat() {
			return true
		}
	}
	for _, b := range f.blocks {
		for _, d := range b.params {
			if d.typ.IsHalfFloat() {
				return true
			}
		}
		for _, in := range b.All() {
			if in.op.Type.IsHalfFloat() || halfSource(in.op.Verb) != TypeNone {
				return true
			}
			for _, r := range in.results {
				if r.typ.IsHalfFloat() {
					return true
				}
			}
		}
	}
	return false
}

// halfSource is the half type a conversion verb reads, or TypeNone.
func halfSource(v Verb) RegType {
	switch v {
	case VFCvtF16, VSCvtF16, VUCvtF16, VSCvtSatF16, VUCvtSatF16, VBitcastF16:
		return TypeF16
	case VFCvtBF16, VSCvtBF16, VUCvtBF16, VSCvtSatBF16, VUCvtSatBF16, VBitcastBF16:
		return TypeBF16
	}
	return TypeNone
}

// f32Source is the f32-sourced verb a half-sourced conversion becomes
// once its operand has been widened.
var f32Source = map[Verb]Verb{
	VSCvtF16: VSCvtF32, VSCvtBF16: VSCvtF32,
	VUCvtF16: VUCvtF32, VUCvtBF16: VUCvtF32,
	VSCvtSatF16: VSCvtSatF32, VSCvtSatBF16: VSCvtSatF32,
	VUCvtSatF16: VUCvtSatF32, VUCvtSatBF16: VUCvtSatF32,
}

type halfPass struct {
	f    *Func
	blk  *Block
	out  []*Inst
	repl map[*Def]*Def // a result this pass replaced, and what its uses read now
}

// --- emission ---------------------------------------------------------

func (p *halfPass) emit(t RegType, v Verb, args ...*Def) *Def {
	return p.emitImm(t, t, v, nil, args...)
}

// emitAs is an instruction in namespace ns whose result is of type rt,
// which differs for the comparisons and the conversions.
func (p *halfPass) emitAs(ns, rt RegType, v Verb, args ...*Def) *Def {
	return p.emitImm(ns, rt, v, nil, args...)
}

func (p *halfPass) emitImm(ns, rt RegType, v Verb, im *imm, args ...*Def) *Def {
	in := &Inst{op: Op{ns, v}, blk: p.blk, args: args, im: im}
	d := p.f.newDef(rt, "", in, 0)
	in.results = []*Def{d}
	p.out = append(p.out, in)
	return d
}

func (p *halfPass) i32(v int64) *Def {
	return p.emitImm(TypeI32, TypeI32, VConst, &imm{lit: Int(v), hasLit: true})
}
func (p *halfPass) i64(v int64) *Def {
	return p.emitImm(TypeI64, TypeI64, VConst, &imm{lit: Int(v), hasLit: true})
}
func (p *halfPass) f32(v float64) *Def {
	return p.emitImm(TypeF32, TypeF32, VConst, &imm{lit: Float(v), hasLit: true})
}
func (p *halfPass) f64(v float64) *Def {
	return p.emitImm(TypeF64, TypeF64, VConst, &imm{lit: Float(v), hasLit: true})
}

func (p *halfPass) and(a, b *Def) *Def  { return p.emit(TypeI32, VAnd, a, b) }
func (p *halfPass) or(a, b *Def) *Def   { return p.emit(TypeI32, VOr, a, b) }
func (p *halfPass) xor(a, b *Def) *Def  { return p.emit(TypeI32, VXor, a, b) }
func (p *halfPass) add(a, b *Def) *Def  { return p.emit(TypeI32, VAdd, a, b) }
func (p *halfPass) sub(a, b *Def) *Def  { return p.emit(TypeI32, VSub, a, b) }
func (p *halfPass) shl(a, b *Def) *Def  { return p.emit(TypeI32, VShl, a, b) }
func (p *halfPass) ushr(a, b *Def) *Def { return p.emit(TypeI32, VUShr, a, b) }
func (p *halfPass) eq(a, b *Def) *Def   { return p.emitAs(TypeI32, TypeI1, VEq, a, b) }
func (p *halfPass) ult(a, b *Def) *Def  { return p.emitAs(TypeI32, TypeI1, VULt, a, b) }
func (p *halfPass) sel(c, a, b *Def) *Def {
	return p.emit(TypeI32, VSelect, c, a, b)
}

func (p *halfPass) bitsOf(x *Def) *Def { return p.emitAs(TypeI32, TypeI32, VBitcastF32, x) }
func (p *halfPass) floatOf(b *Def) *Def {
	return p.emitAs(TypeF32, TypeF32, VBitcastI32, b)
}

// --- the conversions --------------------------------------------------

// widen is a half's bits as the f32 of the same value, exactly.
func (p *halfPass) widen(t RegType, h *Def) *Def {
	if t == TypeBF16 {
		return p.floatOf(p.shl(h, p.i32(16)))
	}
	o := p.shl(p.and(h, p.i32(0x7fff)), p.i32(13)) // exponent and fraction, in place
	exp := p.and(o, p.i32(0x0f800000))             // the half's exponent field, shifted
	o = p.add(o, p.i32(0x38000000))                // rebias: (127-15) << 23
	inf := p.add(o, p.i32(0x38000000))             // Inf and NaN: the exponent to all ones
	// A subnormal half is a normal f32: give it the smallest exponent
	// and subtract that exponent's implicit one back out.
	den := p.bitsOf(p.emit(TypeF32, VSub,
		p.floatOf(p.add(o, p.i32(0x00800000))), p.floatOf(p.i32(0x38800000))))
	o = p.sel(p.eq(exp, p.i32(0x0f800000)), inf, p.sel(p.eq(exp, p.i32(0)), den, o))
	o = p.or(o, p.shl(p.and(h, p.i32(0x8000)), p.i32(16)))
	return p.floatOf(o)
}

// narrow is an f32 rounded to the nearest even half, as that half's bits.
func (p *halfPass) narrow(t RegType, x *Def) *Def {
	b := p.bitsOf(x)
	if t == TypeBF16 {
		hi := p.ushr(b, p.i32(16))
		rnd := p.ushr(p.add(p.add(b, p.i32(0x7fff)), p.and(hi, p.i32(1))), p.i32(16))
		nan := p.or(hi, p.i32(0x40))
		isNaN := p.emitAs(TypeF32, TypeI1, VUno, x, x)
		return p.sel(isNaN, nan, rnd)
	}
	sign := p.and(b, p.i32(-0x80000000))
	f := p.xor(b, sign)
	// Overflow and Inf go to Inf; a NaN stays one, quieted, with the top
	// of its payload.
	nan := p.or(p.i32(0x7e00), p.and(p.ushr(f, p.i32(13)), p.i32(0x1ff)))
	big := p.sel(p.ult(p.i32(0x7f800000), f), nan, p.i32(0x7c00))
	// Below f16's smallest normal, adding one half lets the f32 adder do
	// the rounding into the subnormal's bits.
	den := p.sub(p.bitsOf(p.emit(TypeF32, VAdd, p.floatOf(f), p.f32(0.5))), p.i32(0x3f000000))
	// A normal: rebias, and round to nearest even by adding just under
	// half an ulp and the bit that breaks the tie.
	odd := p.and(p.ushr(f, p.i32(13)), p.i32(1))
	nrm := p.ushr(p.add(p.add(f, p.i32(-0x37fff001)), odd), p.i32(13)) // ((15-127)<<23) + 0xfff
	o := p.sel(p.ult(f, p.i32(0x47800000)), p.sel(p.ult(f, p.i32(0x38800000)), den, nrm), big)
	return p.or(o, p.ushr(sign, p.i32(16)))
}

// toOdd rounds an f64 to f32 towards zero and sets the lowest bit when
// the result is inexact. Rounding that to a half to nearest is rounding
// the f64 there once.
func (p *halfPass) toOdd(x *Def) *Def {
	r := p.emitAs(TypeF32, TypeF32, VFCvtF64, x)
	back := p.emitAs(TypeF64, TypeF64, VFCvtF32, r)
	away := p.emitAs(TypeF64, TypeI1, VLt, p.emit(TypeF64, VAbs, x), p.emit(TypeF64, VAbs, back))
	inexact := p.emit(TypeI1, VAnd,
		p.emitAs(TypeF64, TypeI1, VNe, back, x),
		p.emit(TypeI1, VNot, p.emitAs(TypeF64, TypeI1, VUno, x, x)))
	rb := p.sub(p.bitsOf(r), p.emitAs(TypeI32, TypeI32, VZExtI1, away))
	rb = p.or(rb, p.emitAs(TypeI32, TypeI32, VZExtI1, inexact))
	return p.floatOf(rb)
}

// fma is a*b+c over widened halves, rounded to odd in f64 so that the
// narrowing after it rounds once. a*b is exact in f64; TwoSum recovers
// what the addition lost.
func (p *halfPass) fma(a, b, c *Def) *Def {
	wide := func(x *Def) *Def { return p.emitAs(TypeF64, TypeF64, VFCvtF32, x) }
	pr := p.emit(TypeF64, VMul, wide(a), wide(b))
	cc := wide(c)
	s := p.emit(TypeF64, VAdd, pr, cc)
	bb := p.emit(TypeF64, VSub, s, pr)
	e := p.emit(TypeF64, VAdd,
		p.emit(TypeF64, VSub, pr, p.emit(TypeF64, VSub, s, bb)),
		p.emit(TypeF64, VSub, cc, bb))
	// e is NaN when s is Inf or NaN, and then nothing is to be adjusted.
	finite := p.emitAs(TypeF64, TypeI1, VLt, p.emit(TypeF64, VAbs, s), p.f64(inf64()))
	inexact := p.emit(TypeI1, VAnd, finite, p.emitAs(TypeF64, TypeI1, VNe, e, p.f64(0)))
	sb := p.emitAs(TypeI64, TypeI64, VBitcastF64, s)
	eb := p.emitAs(TypeI64, TypeI64, VBitcastF64, e)
	opposite := p.emitAs(TypeI64, TypeI1, VSLt, p.emit(TypeI64, VXor, sb, eb), p.i64(0))
	away := p.emit(TypeI1, VAnd, inexact, opposite)
	sb = p.emit(TypeI64, VSub, sb, p.emitAs(TypeI64, TypeI64, VZExtI1, away))
	sb = p.emit(TypeI64, VOr, sb, p.emitAs(TypeI64, TypeI64, VZExtI1, inexact))
	return p.toOdd(p.emitAs(TypeF64, TypeF64, VBitcastI64, sb))
}

// fromF64 is an f64 rounded once to a half.
func (p *halfPass) fromF64(t RegType, x *Def) *Def { return p.narrow(t, p.toOdd(x)) }

// --- expansion --------------------------------------------------------

// replace records that in's result now reads as d.
func (p *halfPass) replace(in *Inst, d *Def) bool {
	if len(in.results) == 1 {
		p.repl[in.results[0]] = d
	}
	return true
}

// arg is an operand as it reads now.
func (p *halfPass) arg(d *Def) *Def {
	if r, ok := p.repl[d]; ok {
		return r
	}
	return d
}

// expand rewrites one instruction, and reports whether it replaced it.
// One it keeps may still have been changed in place.
func (p *halfPass) expand(in *Inst) bool {
	for i, a := range in.args {
		if a != nil {
			in.args[i] = p.arg(a)
		}
	}
	t := in.op.Type
	if src := halfSource(in.op.Verb); src != TypeNone && !t.IsHalfFloat() {
		return p.fromHalf(in, src)
	}
	if !t.IsHalfFloat() {
		return false
	}
	a := in.args
	switch in.op.Verb {
	case VConst:
		if in.im != nil && in.im.hasLit && in.im.lit.kind == ConstFloat {
			in.im.lit = Int(int64(HalfBits(t, in.im.lit.f)))
		}
		in.op.Type = TypeI32
		return false
	case VLoad:
		in.op = Op{TypeI32, VULoad16}
		return false
	case VStore:
		in.op = Op{TypeI32, VStore16}
		return false
	case VSelect:
		in.op.Type = TypeI32
		return false

	case VBitcastI32:
		// The carrier is the encoding: only the low sixteen bits count.
		return p.replace(in, p.and(a[0], p.i32(0xffff)))
	case VNeg:
		return p.replace(in, p.xor(a[0], p.i32(0x8000)))
	case VAbs:
		return p.replace(in, p.and(a[0], p.i32(0x7fff)))
	case VCopySign:
		return p.replace(in, p.or(p.and(a[0], p.i32(0x7fff)), p.and(a[1], p.i32(0x8000))))

	case VSqrt, VCeil, VFloor, VTrunc, VNearest:
		return p.replace(in, p.narrow(t, p.emit(TypeF32, in.op.Verb, p.widen(t, a[0]))))
	case VAdd, VSub, VMul, VDiv, VMinimum, VMaximum, VMinNum, VMaxNum:
		r := p.emit(TypeF32, in.op.Verb, p.widen(t, a[0]), p.widen(t, a[1]))
		return p.replace(in, p.narrow(t, r))
	case VFMA:
		return p.replace(in, p.narrow(t, p.fma(p.widen(t, a[0]), p.widen(t, a[1]), p.widen(t, a[2]))))
	case VEq, VNe, VLt, VLe, VUno:
		return p.replace(in, p.emitAs(TypeF32, TypeI1, in.op.Verb, p.widen(t, a[0]), p.widen(t, a[1])))

	case VFCvtF32:
		return p.replace(in, p.narrow(t, a[0]))
	case VFCvtF64:
		return p.replace(in, p.fromF64(t, a[0]))
	case VFCvtF16:
		return p.replace(in, p.narrow(t, p.widen(TypeF16, a[0])))
	case VFCvtBF16:
		return p.replace(in, p.narrow(t, p.widen(TypeBF16, a[0])))
	case VSCvtI32, VUCvtI32, VSCvtI64, VUCvtI64:
		return p.replace(in, p.fromF64(t, p.emitAs(TypeF64, TypeF64, in.op.Verb, a[0])))
	}
	p.f.m.fail(p.f.name, p.blk.Label(), in.op, ErrType, "legalizing %s: no expansion for %s", t, in.op)
	return false
}

// fromHalf rewrites a conversion out of a half into the same conversion
// out of the f32 it widens to exactly.
func (p *halfPass) fromHalf(in *Inst, src RegType) bool {
	if in.op.Verb == VBitcastF16 || in.op.Verb == VBitcastBF16 {
		// The carrier already is the encoding, zero above it.
		return p.replace(in, in.args[0])
	}
	w := p.widen(src, in.args[0])
	switch in.op.Type {
	case TypeF32:
		return p.replace(in, w)
	case TypeF64:
		return p.replace(in, p.emitAs(TypeF64, TypeF64, VFCvtF32, w))
	}
	if v, ok := f32Source[in.op.Verb]; ok {
		in.op.Verb = v
		in.args[0] = w
		return false
	}
	p.f.m.fail(p.f.name, p.blk.Label(), in.op, ErrType, "legalizing %s: no expansion for %s", src, in.op)
	return false
}
