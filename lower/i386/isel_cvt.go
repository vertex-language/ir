package i386

// §C's float conversions, and the §A3 verbs SSE2 has no instruction for.
//
// The shape of this file is the shape of what the extension leaves out. SSE2
// converts between a float and a *32-bit* integer and nothing wider: the
// 64-bit forms are REX.W, which is x86-64. And it rounds to an integer only
// with ROUNDSS, which is SSE4.1. So everything at 64 bits and every rounding
// verb is a call to the helper a C compiler would call, and everything at 32
// bits is an instruction and a fixup.

import (
	"fmt"
	"math"

	"github.com/vertex-language/ir"
	"github.com/vertex-language/ir/lower/mir"
)

// floatHelpers is the C library name for each §A3 verb with no instruction.
//
// The rounding four are libm's, and fma is libm's too. A compiler targeting
// SSE4.1 would emit ROUNDSS for the first four and one targeting FMA3 would
// emit VFMADD for the last; at SSE2 they are calls, which is what every
// compiler does at this baseline.
var floatHelpers = map[ir.Verb][2]string{
	ir.VCeil:    {"ceilf", "ceil"},
	ir.VFloor:   {"floorf", "floor"},
	ir.VTrunc:   {"truncf", "trunc"},
	ir.VNearest: {"rintf", "rint"},
	ir.VFMA:     {"fmaf", "fma"},
}

// helperFor is the name of the helper one verb becomes at one width.
func helperFor(verb ir.Verb, w width) (string, bool) {
	names, ok := floatHelpers[verb]
	if !ok {
		return "", false
	}
	if w == wf32 {
		return names[0], true
	}
	return names[1], true
}

// iselFloatLibcall lowers a §A3 verb into the call it is.
func iselFloatLibcall(c *cursor, vr *vregs, fr *frame, in *ir.Inst) error {
	op := in.Op()
	dst, err := vr.define(in.Result(0))
	if err != nil {
		return fmt.Errorf("%s: %w", op, err)
	}
	name, ok := helperFor(op.Verb, dst.w)
	if !ok {
		return fmt.Errorf("%s: no helper for this verb", op)
	}
	args := make([]value, 0, 3)
	for i := range in.Args() {
		v, found := vr.lookup(in.Arg(i))
		if !found {
			return fmt.Errorf("%s: operand %d defined outside the function", op, i)
		}
		args = append(args, v)
	}
	return emitLibcall(c, vr, fr, name, args, dst, true)
}

// iselFloatWidth lowers §C3's fcvt between the two float widths.
func iselFloatWidth(c *cursor, vr *vregs, in *ir.Inst) error {
	ops, err := operands(vr, in, 1)
	if err != nil {
		return err
	}
	dst, err := vr.define(in.Result(0))
	if err != nil {
		return fmt.Errorf("%s: %w", in.Op(), err)
	}
	if !dst.w.isFloat() || !ops[0].w.isFloat() {
		return fmt.Errorf("%s: fcvt is between two floats, and f80 is not a type this package has", in.Op())
	}
	c.Emit(mir.Instr{
		Op:   cvtFloatOp{w: dst.w},
		Defs: []mir.VReg{dst.lo}, Uses: []mir.VReg{ops[0].lo},
	})
	return nil
}

// iselBitcast lowers §C3's bitcast: the same bits read as the other kind of
// thing, which is MOVD at 32 bits and a frame slot at 64.
func iselBitcast(c *cursor, vr *vregs, fr *frame, in *ir.Inst) error {
	ops, err := operands(vr, in, 1)
	if err != nil {
		return err
	}
	dst, err := vr.define(in.Result(0))
	if err != nil {
		return fmt.Errorf("%s: %w", in.Op(), err)
	}

	switch in.Op().Verb {
	case ir.VBitcastI32:
		c.Emit(mir.Instr{Op: movdToXmmOp{}, Defs: []mir.VReg{dst.lo}, Uses: []mir.VReg{ops[0].lo}})
	case ir.VBitcastF32:
		c.Emit(mir.Instr{Op: movdToGPOp{}, Defs: []mir.VReg{dst.lo}, Uses: []mir.VReg{ops[0].lo}})
	case ir.VBitcastI64:
		c.Emit(mir.Instr{
			Op:   pairToFloatOp{off: fr.scratch()},
			Defs: []mir.VReg{dst.lo}, Uses: []mir.VReg{ops[0].lo, ops[0].hi},
		})
	case ir.VBitcastF64:
		c.Emit(mir.Instr{
			Op:   floatToPairOp{off: fr.scratch()},
			Defs: []mir.VReg{dst.lo, dst.hi}, Uses: []mir.VReg{ops[0].lo},
		})
	}
	return nil
}

// intToFloatHelpers is the C library name for a 64-bit integer conversion,
// which SSE2 cannot do: CVTSI2SD from a 64-bit source is REX.W.
var intToFloatHelpers = map[[2]bool]string{
	// {signed, double}
	{true, true}:   "__floatdidf",
	{true, false}:  "__floatdisf",
	{false, true}:  "__floatundidf",
	{false, false}: "__floatundisf",
}

var floatToIntHelpers = map[[2]bool]string{
	{true, true}:   "__fixdfdi",
	{true, false}:  "__fixsfdi",
	{false, true}:  "__fixunsdfdi",
	{false, false}: "__fixunssfdi",
}

// iselIntToFloat lowers §C2's int-to-float conversions.
//
// From a 32-bit signed integer it is one instruction. From an unsigned one it
// is that instruction and a correction: CVTSI2SD reads its source as signed,
// so a value with the top bit set converts to a negative double, and adding
// 2^32 puts it back. That is exact — every uint32 is exactly a double — and
// the single-precision case goes the same way and narrows afterwards, so the
// rounding happens once.
//
// From a 64-bit integer there is no instruction at all.
func iselIntToFloat(c *cursor, vr *vregs, fr *frame, in *ir.Inst, opts Options) error {
	op := in.Op()
	ops, err := operands(vr, in, 1)
	if err != nil {
		return err
	}
	dst, err := vr.define(in.Result(0))
	if err != nil {
		return fmt.Errorf("%s: %w", op, err)
	}
	signed := op.Verb == ir.VSCvtI32 || op.Verb == ir.VSCvtI64
	wide := op.Verb == ir.VSCvtI64 || op.Verb == ir.VUCvtI64

	if wide {
		name := intToFloatHelpers[[2]bool{signed, dst.w == wf64}]
		return emitLibcall(c, vr, fr, name, []value{ops[0]}, dst, true)
	}

	if signed {
		c.Emit(mir.Instr{
			Op:   cvtIntToFloatOp{w: dst.w},
			Defs: []mir.VReg{dst.lo}, Uses: []mir.VReg{ops[0].lo},
		})
		return nil
	}

	// Unsigned. Convert to a double whatever the destination is, correct
	// it, and narrow at the end if the answer is a float — which keeps the
	// rounding to one step.
	wide64 := vr.vec()
	c.Emit(mir.Instr{
		Op:   cvtIntToFloatOp{w: wf64},
		Defs: []mir.VReg{wide64}, Uses: []mir.VReg{ops[0].lo},
	})

	done := c.open("done")
	fix := c.open("fixup")
	c.Emit(mir.Instr{Op: testOp{}, Uses: []mir.VReg{ops[0].lo}})
	c.branch(condL, fix, done)

	// The sign test is on the integer: S set means the top bit was set,
	// which is the case CVTSI2SD read as negative.
	fix.Emit(mir.Instr{
		Op:   fbinOp{verb: ir.VAdd, w: wf64},
		Defs: []mir.VReg{wide64},
		Uses: []mir.VReg{wide64, floatConstIn(fix, vr, fr, wf64, 0x41f0000000000000)},
	})
	fix.Emit(mir.Instr{Op: jmpOp{target: done.Label}})
	c.mf.Succ(fix, done.Label)

	c.resume(done)
	if dst.w == wf64 {
		c.Emit(mir.Instr{Op: fmovOp{}, Defs: []mir.VReg{dst.lo}, Uses: []mir.VReg{wide64}, Copy: true})
		return nil
	}
	c.Emit(mir.Instr{
		Op:   cvtFloatOp{w: wf32},
		Defs: []mir.VReg{dst.lo}, Uses: []mir.VReg{wide64},
	})
	return nil
}

// iselFloatToInt lowers §C2's float-to-integer conversions.
//
// Four destinations — signed and unsigned, at thirty-two bits and at
// sixty-four — each of which exists in a trapping form and a saturating one.
// What all eight share is the interval: §C2 admits every source value whose
// truncation the destination can hold, so the trapping forms are a range
// check in front of the conversion and the saturating forms answer the bound
// itself outside it. See f2iRange.
//
// The conversion the check wraps is one of three things, which is why the two
// halves are worth separating. CVTTSS2SI answers the signed 32-bit case
// outright. It answers the unsigned one after a bias, because its result is
// signed and half the range does not fit. And it answers neither 64-bit case
// at all — those are REX.W, which is x86-64 — so each of those is a call to
// the helper a C compiler would call.
func iselFloatToInt(c *cursor, vr *vregs, fr *frame, in *ir.Inst, opts Options) error {
	op := in.Op()
	ops, err := operands(vr, in, 1)
	if err != nil {
		return err
	}
	dst, err := vr.define(in.Result(0))
	if err != nil {
		return fmt.Errorf("%s: %w", op, err)
	}
	src := ops[0]
	if !src.w.isFloat() {
		return fmt.Errorf("%s: the source of a float-to-integer conversion is a float, and f80 and f128 are types this package does not have", op)
	}

	signed := signedConversions[op.Verb]
	r := f2iRange(src.w, dst.w, signed)

	// Below the interval, and the NaN with it: UCOMIS sets CF for "below"
	// and sets both CF and ZF for an unordered pair, so B and BE are each
	// true of a NaN. Which of the two the bound wants is r's own business.
	below := condB
	if r.loStrict {
		below = condBE
	}

	if !satConversions[op.Verb] {
		lo := floatConst(c, vr, fr, src.w, floatBits(src.w, r.lo))
		trap := c.open("trap")
		inLo := c.open("inlo")
		c.Emit(mir.Instr{Op: fcmpOp{w: src.w}, Uses: []mir.VReg{src.lo, lo}})
		c.branch(below, trap, inLo)

		// Above it. A NaN has already been caught, so this test has
		// only ordered values left to separate.
		hi := floatConst(c, vr, fr, src.w, floatBits(src.w, r.hi))
		conv := c.open("conv")
		c.Emit(mir.Instr{Op: fcmpOp{w: src.w}, Uses: []mir.VReg{src.lo, hi}})
		c.branch(condAE, trap, conv)

		trap.Emit(mir.Instr{Op: trapOp{}})

		return emitConvertInRange(c, vr, fr, src, dst, signed)
	}

	// Saturating, which is the same three tests answering constants rather
	// than trapping. The bound is the destination's own extreme and not the
	// truncation of the interval's edge: at f32 into an i32 those differ,
	// the largest float below 2^31 being 2147483520 where §C2 asks for
	// 2147483647.
	minBits, maxBits := satBounds(dst.w, signed)
	done := c.open("done")

	// The NaN first, because it satisfies both of the tests below and its
	// answer is neither bound.
	nan := c.open("nan")
	num := c.open("num")
	c.Emit(mir.Instr{Op: fcmpOp{w: src.w}, Uses: []mir.VReg{src.lo, src.lo}})
	c.branch(condP, nan, num)
	emitIntConst(nan, dst, 0)
	nan.Emit(mir.Instr{Op: jmpOp{target: done.Label}})
	c.mf.Succ(nan, done.Label)

	lo := floatConst(c, vr, fr, src.w, floatBits(src.w, r.lo))
	under := c.open("under")
	inLo := c.open("inlo")
	c.Emit(mir.Instr{Op: fcmpOp{w: src.w}, Uses: []mir.VReg{src.lo, lo}})
	c.branch(below, under, inLo)
	emitIntConst(under, dst, minBits)
	under.Emit(mir.Instr{Op: jmpOp{target: done.Label}})
	c.mf.Succ(under, done.Label)

	hi := floatConst(c, vr, fr, src.w, floatBits(src.w, r.hi))
	over := c.open("over")
	conv := c.open("conv")
	c.Emit(mir.Instr{Op: fcmpOp{w: src.w}, Uses: []mir.VReg{src.lo, hi}})
	c.branch(condAE, over, conv)
	emitIntConst(over, dst, maxBits)
	over.Emit(mir.Instr{Op: jmpOp{target: done.Label}})
	c.mf.Succ(over, done.Label)

	if err := emitConvertInRange(c, vr, fr, src, dst, signed); err != nil {
		return err
	}
	c.to(done)
	return nil
}

// emitConvertInRange converts a value already known to be one the destination
// can hold, which is what both of iselFloatToInt's halves end with.
func emitConvertInRange(c *cursor, vr *vregs, fr *frame, src, dst value, signed bool) error {
	if dst.w.pairs() {
		name := floatToIntHelpers[[2]bool{signed, src.w == wf64}]
		return emitLibcall(c, vr, fr, name, []value{src}, dst, true)
	}
	if signed {
		c.Emit(mir.Instr{
			Op:   cvtFloatToIntOp{from: src.w},
			Defs: []mir.VReg{dst.lo}, Uses: []mir.VReg{src.lo},
		})
		return nil
	}

	// Unsigned, where CVTTSS2SI's signed result is half the range short.
	// Below 2^31 it is the answer; at or above it, the same instruction on
	// the value biased down by 2^31 gives the low thirty-one bits and the
	// bit that was subtracted is put back. The subtraction is exact — both
	// operands and the difference are representable at either width — so
	// nothing is rounded on the way.
	bias := floatConst(c, vr, fr, src.w, floatBits(src.w, 2147483648))
	high := c.open("high")
	low := c.open("low")
	done := c.open("done")
	c.Emit(mir.Instr{Op: fcmpOp{w: src.w}, Uses: []mir.VReg{src.lo, bias}})
	c.branch(condAE, high, low)

	low.Emit(mir.Instr{
		Op:   cvtFloatToIntOp{from: src.w},
		Defs: []mir.VReg{dst.lo}, Uses: []mir.VReg{src.lo},
	})
	low.Emit(mir.Instr{Op: jmpOp{target: done.Label}})
	c.mf.Succ(low, done.Label)

	biased := vr.vec()
	high.Emit(mir.Instr{Op: fmovOp{}, Defs: []mir.VReg{biased}, Uses: []mir.VReg{src.lo}})
	high.Emit(mir.Instr{
		Op:   fbinOp{verb: ir.VSub, w: src.w},
		Defs: []mir.VReg{biased}, Uses: []mir.VReg{biased, bias},
	})
	high.Emit(mir.Instr{
		Op:   cvtFloatToIntOp{from: src.w},
		Defs: []mir.VReg{dst.lo}, Uses: []mir.VReg{biased},
	})
	top := vr.reg32()
	high.Emit(mir.Instr{Op: constOp{imm: int64(int32(-2147483648))}, Defs: []mir.VReg{top}})
	high.Emit(mir.Instr{
		Op:   aluOp{verb: ir.VOr},
		Defs: []mir.VReg{dst.lo}, Uses: []mir.VReg{dst.lo, top},
	})
	high.Emit(mir.Instr{Op: jmpOp{target: done.Label}})
	c.mf.Succ(high, done.Label)

	c.resume(done)
	return nil
}

// emitIntConst writes a literal into a destination that may be a pair.
func emitIntConst(b *mir.Block, dst value, bits uint64) {
	b.Emit(mir.Instr{Op: constOp{imm: int64(int32(uint32(bits)))}, Defs: []mir.VReg{dst.lo}})
	if dst.w.pairs() {
		b.Emit(mir.Instr{Op: constOp{imm: int64(int32(uint32(bits >> 32)))}, Defs: []mir.VReg{dst.hi}})
	}
}

// satBounds is what a saturating conversion answers below and above the
// interval, as the destination's bit patterns.
func satBounds(dst width, signed bool) (min, max uint64) {
	switch {
	case dst.pairs() && signed:
		return 1 << 63, 1<<63 - 1
	case dst.pairs():
		return 0, ^uint64(0)
	case signed:
		return 1 << 31, 1<<31 - 1
	}
	return 0, 1<<32 - 1
}

// satConversions and signedConversions are which of §C2's eight float-to-int
// verbs a mnemonic is.
var satConversions = map[ir.Verb]bool{
	ir.VSCvtSatF32: true, ir.VSCvtSatF64: true,
	ir.VUCvtSatF32: true, ir.VUCvtSatF64: true,
}

var signedConversions = map[ir.Verb]bool{
	ir.VSCvtF32: true, ir.VSCvtF64: true,
	ir.VSCvtSatF32: true, ir.VSCvtSatF64: true,
}

// f64Conversions is which of them read a double rather than a float, which is
// the other half of the helper's name at sixty-four bits.
var f64Conversions = map[ir.Verb]bool{
	ir.VSCvtF64: true, ir.VUCvtF64: true,
	ir.VSCvtSatF64: true, ir.VUCvtSatF64: true,
}

// floatToIntCalls is every §C2 float-to-int verb; isFloatToInt is the
// membership test libcall.go walks a function with.
var floatToIntCalls = []ir.Verb{
	ir.VSCvtF32, ir.VSCvtF64, ir.VUCvtF32, ir.VUCvtF64,
	ir.VSCvtSatF32, ir.VSCvtSatF64, ir.VUCvtSatF32, ir.VUCvtSatF64,
}

func isFloatToInt(verb ir.Verb) bool {
	for _, v := range floatToIntCalls {
		if v == verb {
			return true
		}
	}
	return false
}

// A cvtRange is the interval of source values one conversion admits: those
// whose truncation toward zero the destination can hold, and no others.
type cvtRange struct {
	lo, hi   float64
	loStrict bool // x > lo rather than x >= lo
}

// f2iRange is that interval, for one source width and one destination.
//
// The high bound is always exclusive and always a power of two, which is
// exactly representable at both source widths — 2^31 is not an int32, so the
// check is AE and not A. The low bound is where the two widths stop agreeing,
// because it depends on what the source can name just under the destination's
// most negative value:
//
//   - Unsigned. Truncation toward zero carries everything above −1 to zero,
//     so the interval opens at −1 rather than at zero and −0.5 converts
//     rather than trapping.
//
//   - Signed, from a double, into an i32. A double names values between
//     −2^31−1 and −2^31, all of which truncate to −2^31 — so the bound is
//     −2^31−1 and it is strict.
//
//   - Signed, otherwise. A float's neighbours around −2^31 are 256 apart and
//     a double's around −2^63 are 1024, so there is nothing between the bound
//     and the first value that does not fit: the bound itself is admitted and
//     nothing below it is.
//
// The same table serves the saturating forms, whose two bounds are the two
// answers rather than two refusals.
func f2iRange(from, dst width, signed bool) cvtRange {
	if !signed {
		hi := 4294967296.0
		if dst.pairs() {
			hi = 18446744073709551616.0
		}
		return cvtRange{lo: -1, hi: hi, loStrict: true}
	}
	if dst.pairs() {
		return cvtRange{lo: -9223372036854775808, hi: 9223372036854775808}
	}
	if from == wf64 {
		return cvtRange{lo: -2147483649, hi: 2147483648, loStrict: true}
	}
	return cvtRange{lo: -2147483648, hi: 2147483648}
}

// floatBits is a value's bit pattern at one float width.
func floatBits(w width, v float64) uint64 {
	if w == wf32 {
		return uint64(math.Float32bits(float32(v)))
	}
	return math.Float64bits(v)
}

// floatConstIn is floatConst emitting into a block other than the cursor's.
func floatConstIn(b *mir.Block, vr *vregs, fr *frame, w width, bits uint64) mir.VReg {
	out := vr.vec()
	lo := vr.reg32()
	hi := vr.reg32()
	b.Emit(mir.Instr{Op: constOp{imm: int64(int32(uint32(bits)))}, Defs: []mir.VReg{lo}})
	b.Emit(mir.Instr{Op: constOp{imm: int64(int32(uint32(bits >> 32)))}, Defs: []mir.VReg{hi}})
	b.Emit(mir.Instr{
		Op:   pairToFloatOp{off: fr.scratch()},
		Defs: []mir.VReg{out}, Uses: []mir.VReg{lo, hi},
	})
	return out
}
