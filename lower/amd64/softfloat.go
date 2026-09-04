package amd64

// f128, whose arithmetic is compiler-rt rather than silicon: §0 says a
// namespace the layout block admits is usable either way, and that
// lowering supplies the call where the target has no instruction.
//
// The value needs no call. §3.2.3 classifies __float128 SSE and SSEUP —
// two eightbytes in one XMM — so moving, spilling and passing one is an
// f64's work done sixteen bytes wide.

import (
	"fmt"
	"math"
	"math/bits"

	"github.com/vertex-language/ir"
	"github.com/vertex-language/ir/lower/mir"
)

// softArith is §A3's arithmetic on f128. The names are compiler-rt's,
// where tf is the 128-bit binary format and the trailing digit is the
// operand count.
var softArith = map[ir.Verb]string{
	ir.VAdd: "__addtf3",
	ir.VSub: "__subtf3",
	ir.VMul: "__multf3",
	ir.VDiv: "__divtf3",
}

// softCompare is §B's comparisons on f128. Each returns an int whose
// sign is the ordering — and non-zero for an unordered pair, which is
// what makes a single test enough — so a row is the call and a condition
// on its result against zero.
var softCompare = map[ir.Verb]struct {
	sym  string
	cond condCode
}{
	ir.VEq:  {"__eqtf2", condE},
	ir.VNe:  {"__netf2", condNE},
	ir.VLt:  {"__lttf2", condL},
	ir.VLe:  {"__letf2", condLE},
	ir.VUno: {"__unordtf2", condNE},
}

// softConvert is §C's conversions into and out of f128, keyed by the
// whole opcode: a verb names the source and the opcode's type is the
// destination, so f128.scvt_i32 and i32.scvt_f128 are opposite rows and
// either half alone is half of two.
//
// compiler-rt's names are positional — operation, source, destination —
// where si is i32, di is i64, sf is f32, df is f64 and tf is f128.
var softConvert = map[ir.Op]string{
	// Into f128, from a narrower float.
	{Type: ir.TypeF128, Verb: ir.VFCvtF32}: "__extendsftf2",
	{Type: ir.TypeF128, Verb: ir.VFCvtF64}: "__extenddftf2",

	// Out of f128, to a narrower float.
	{Type: ir.TypeF32, Verb: ir.VFCvtF128}: "__trunctfsf2",
	{Type: ir.TypeF64, Verb: ir.VFCvtF128}: "__trunctfdf2",

	// Into f128, from an integer. These never round and never trap:
	// binary128 holds every value of both integer widths exactly.
	{Type: ir.TypeF128, Verb: ir.VSCvtI32}: "__floatsitf",
	{Type: ir.TypeF128, Verb: ir.VSCvtI64}: "__floatditf",
	{Type: ir.TypeF128, Verb: ir.VUCvtI32}: "__floatunsitf",
	{Type: ir.TypeF128, Verb: ir.VUCvtI64}: "__floatunditf",
}

// softFixOut is §C2's conversions out of f128 into an integer, which are
// not in softConvert because they are not one call.
//
// compiler-rt's fix rounds toward zero and is undefined out of range,
// where §C2 traps and §C2's saturating forms clamp. So each is the range
// check iselFloatToInt builds around the hardware instruction, built
// around a call instead — the same shape one level out, with the
// comparisons themselves calls too. See iselSoftFixOut.
var softFixOut = map[ir.Op]string{
	{Type: ir.TypeI32, Verb: ir.VSCvtF128}:    "__fixtfsi",
	{Type: ir.TypeI64, Verb: ir.VSCvtF128}:    "__fixtfdi",
	{Type: ir.TypeI32, Verb: ir.VUCvtF128}:    "__fixunstfsi",
	{Type: ir.TypeI64, Verb: ir.VUCvtF128}:    "__fixunstfdi",
	{Type: ir.TypeI32, Verb: ir.VSCvtSatF128}: "__fixtfsi",
	{Type: ir.TypeI64, Verb: ir.VSCvtSatF128}: "__fixtfdi",
	{Type: ir.TypeI32, Verb: ir.VUCvtSatF128}: "__fixunstfsi",
	{Type: ir.TypeI64, Verb: ir.VUCvtSatF128}: "__fixunstfdi",
}

// softFloatSyms is every soft-float symbol a module's f128 operations
// need declared.
func softFloatSyms(m *ir.Module, prefix string) []string {
	seen := map[string]bool{}
	var out []string
	add := func(s string) {
		if s != "" && !seen[s] {
			seen[s] = true
			out = append(out, prefix+s)
		}
	}
	for _, fn := range m.Funcs() {
		fn.WalkInsts(func(in *ir.Inst) bool {
			op := in.Op()
			if op.Type == ir.TypeF128 {
				add(softArith[op.Verb])
				add(softCompare[op.Verb].sym)
			}
			add(softConvert[op])
			if sym, ok := softFixOut[op]; ok {
				// The conversion, and the two comparisons the
				// range check around it is made of.
				add(sym)
				add(softCompare[ir.VUno].sym)
				add(softCompare[ir.VLt].sym)
			}
			return true
		})
	}
	return out
}

// softFloatCalls reports whether an instruction's lowering makes a call,
// which decides whether it needs a frame. Not every f128 one does: a
// literal is a MOVAPS and the sign verbs are one logical instruction.
func softFloatCalls(in *ir.Inst) bool {
	op := in.Op()
	if op.Type == ir.TypeF128 {
		if _, ok := softArith[op.Verb]; ok {
			return true
		}
		if _, ok := softCompare[op.Verb]; ok {
			return true
		}
	}
	if _, ok := softConvert[op]; ok {
		return true
	}
	// A conversion out of f128 is three calls: the two comparisons the
	// range check is, and the conversion itself.
	_, ok := softFixOut[op]
	return ok
}

// isSoftFloat reports whether an instruction is one this file lowers:
// anything in the f128 namespace, and any conversion whose other end is
// one.
func isSoftFloat(in *ir.Inst) bool {
	op := in.Op()
	if op.Type == ir.TypeF128 {
		return true
	}
	if _, ok := softConvert[op]; ok {
		return true
	}
	_, ok := softFixOut[op]
	return ok
}

// iselSoftFloat lowers everything f128. One door rather than rows in the
// main dispatch: most width switches here end in a default arm meaning
// "the 64-bit integer one", so an unrouted f128 would be quietly
// mis-emitted rather than refused.
func iselSoftFloat(c *cursor, vr *vregs, in *ir.Inst) error {
	op := in.Op()

	if sym, ok := softArith[op.Verb]; ok && op.Type == ir.TypeF128 {
		return softCall(c, vr, in, sym)
	}
	if cmp, ok := softCompare[op.Verb]; ok && op.Type == ir.TypeF128 {
		return softCmp(c, vr, in, cmp.sym, cmp.cond)
	}
	if sym, ok := softConvert[op]; ok {
		return softCall(c, vr, in, sym)
	}
	if sym, ok := softFixOut[op]; ok {
		return iselSoftFixOut(c, vr, in, sym)
	}

	switch op.Verb {
	case ir.VConst:
		return iselF128Const(c, vr, in)
	case ir.VNeg, ir.VAbs, ir.VCopySign:
		return iselF128Sign(c, vr, in)
	}
	return fmt.Errorf("%s: no soft-float sequence for this verb yet", op)
}

// softCall is one soft-float operation: its operands into the argument
// registers of whichever file each belongs to, the call, and the result
// out of the register its own class names.
func softCall(c *cursor, vr *vregs, in *ir.Inst, sym string) error {
	op := in.Op()
	args := in.Args()
	vals := make([]libArg, 0, len(args))
	for i, a := range args {
		v, ok := vr.lookup(a)
		if !ok {
			return fmt.Errorf("%s: operand %d defined outside the function", op, i)
		}
		vals = append(vals, libArg{v: v, w: vr.widthOfVReg(v)})
	}

	var result mir.VReg
	var wantResult bool
	if len(in.Results()) == 1 {
		r, err := vr.define(in.Result(0))
		if err != nil {
			return fmt.Errorf("%s: %w", op, err)
		}
		result, wantResult = r, true
	}
	return emitLibcall(c, vr, sym, vals, result, wantResult)
}

// softCmp is §B on f128: the call, then the condition its integer answer
// has to satisfy.
func softCmp(c *cursor, vr *vregs, in *ir.Inst, sym string, cond condCode) error {
	op := in.Op()
	answer := vr.temp(w32)
	if err := softCallInto(c, vr, in, sym, answer); err != nil {
		return err
	}

	result, err := vr.define(in.Result(0))
	if err != nil {
		return fmt.Errorf("%s: %w", op, err)
	}
	// Against the whole 32-bit answer, not the low byte of it. These
	// functions return a signed int whose *sign* is the ordering, and
	// compiler-rt returning exactly -1, 0 and 1 today is not something
	// the ABI promises: a negative answer whose low byte happened to be
	// zero would read as equal.
	c.Emit(mir.Instr{Op: cmpImmOp{imm: 0, w: w32}, Uses: []mir.VReg{answer}})
	c.Emit(mir.Instr{Op: setccOp{cond: cond}, Defs: []mir.VReg{result}})
	return nil
}

// softCallInto is softCall with the result going somewhere this package
// chose rather than to the instruction's own.
func softCallInto(c *cursor, vr *vregs, in *ir.Inst, sym string, into mir.VReg) error {
	op := in.Op()
	vals := make([]libArg, 0, len(in.Args()))
	for i, a := range in.Args() {
		v, ok := vr.lookup(a)
		if !ok {
			return fmt.Errorf("%s: operand %d defined outside the function", op, i)
		}
		vals = append(vals, libArg{v: v, w: vr.widthOfVReg(v)})
	}
	return emitLibcall(c, vr, sym, vals, into, true)
}

// iselF128Sign lowers §A3's neg, abs and copysign, the one part of the
// namespace needing no call: ANDPD and XORPD already operate on all
// sixteen bytes. Only the mask had to change — no general-purpose
// register reaches bit 127, so it comes out of .rodata.
func iselF128Sign(c *cursor, vr *vregs, in *ir.Inst) error {
	op := in.Op()
	a, ok := vr.lookup(in.Arg(0))
	if !ok {
		return fmt.Errorf("%s: operand defined outside the function", op)
	}
	result, err := vr.define(in.Result(0))
	if err != nil {
		return fmt.Errorf("%s: %w", op, err)
	}

	switch op.Verb {
	case ir.VNeg:
		// The sign flipped, which is an XOR with the sign bit alone.
		// Not a subtraction from zero: §A3's neg flips the sign of a
		// zero and preserves a NaN's payload, and 0 - x does neither.
		mask := f128Mask(c, vr, 0, 1<<63)
		c.Emit(mir.Instr{
			Op:   fLogicOp{op: fXor, w: wf64},
			Defs: []mir.VReg{result},
			Uses: []mir.VReg{a, mask},
		})
		return nil

	case ir.VAbs:
		// The sign cleared, which is an AND with everything else.
		mask := f128Mask(c, vr, ^uint64(0), ^uint64(0)>>1)
		c.Emit(mir.Instr{
			Op:   fLogicOp{op: fAnd, w: wf64},
			Defs: []mir.VReg{result},
			Uses: []mir.VReg{a, mask},
		})
		return nil

	case ir.VCopySign:
		b, ok := vr.lookup(in.Arg(1))
		if !ok {
			return fmt.Errorf("%s: operand 1 defined outside the function", op)
		}
		// The magnitude of one and the sign of the other, which is two
		// masks and an OR: everything but the sign from a, and nothing
		// but the sign from b.
		body := vr.temp(wv128)
		c.Emit(mir.Instr{
			Op:   fLogicOp{op: fAnd, w: wf64},
			Defs: []mir.VReg{body},
			Uses: []mir.VReg{a, f128Mask(c, vr, ^uint64(0), ^uint64(0)>>1)},
		})
		sign := vr.temp(wv128)
		c.Emit(mir.Instr{
			Op:   fLogicOp{op: fAnd, w: wf64},
			Defs: []mir.VReg{sign},
			Uses: []mir.VReg{b, f128Mask(c, vr, 0, 1<<63)},
		})
		c.Emit(mir.Instr{
			Op:   fLogicOp{op: fOr, w: wf64},
			Defs: []mir.VReg{result},
			Uses: []mir.VReg{body, sign},
		})
		return nil
	}
	return fmt.Errorf("%s: not a sign-bit verb", op)
}

// f128Mask materializes a 128-bit constant into a vector register, which
// is the same instruction an f128 literal is: there is no difference
// between a mask and a literal once both are sixteen bytes in .rodata.
func f128Mask(c *cursor, vr *vregs, lo, hi uint64) mir.VReg {
	mask := vr.temp(wv128)
	c.Emit(mir.Instr{Op: wideConstOp{lo: lo, hi: hi}, Defs: []mir.VReg{mask}})
	return mask
}

// iselF128Const materializes an f128 literal. ir.Const carries a
// float64, so the value is that double widened — done here rather than
// by a run-time __extenddftf2, a constant that costs a call being no
// constant at all.
func iselF128Const(c *cursor, vr *vregs, in *ir.Inst) error {
	op := in.Op()
	lit, ok := in.Lit()
	if !ok {
		return fmt.Errorf("%s: only a plain literal is supported", op)
	}
	if lit.Kind() != ir.ConstFloat {
		return fmt.Errorf("%s: a float constant needs a float literal", op)
	}
	result, err := vr.define(in.Result(0))
	if err != nil {
		return fmt.Errorf("%s: %w", op, err)
	}
	lo, hi := f64ToF128Bits(lit.Float())
	c.Emit(mir.Instr{Op: wideConstOp{lo: lo, hi: hi}, Defs: []mir.VReg{result}})
	return nil
}

// The two exponent biases, which every widening and every constant here is
// a re-bias between.
const (
	f64Bias  = 1023
	f128Bias = 16383
)

// f64ToF128Bits is one double's value in binary128. Every double is
// exactly representable, so this is a re-bias and a shift that never
// rounds — except where the exponent field is not a number, which is the
// three cases below.
func f64ToF128Bits(f float64) (lo, hi uint64) {
	b := math.Float64bits(f)
	sign := b >> 63
	exp := (b >> 52) & 0x7ff
	frac := b & (1<<52 - 1)

	switch {
	case exp == 0 && frac == 0:
		// Zero, of whichever sign, which is the sign bit and nothing else.
		return 0, sign << 63

	case exp == 0:
		// Subnormal as a double and normal as an f128, the wider
		// exponent having room: shift the significand up to its
		// implicit bit and charge the exponent for each shift.
		e := int64(1 - f64Bias)
		for frac&(1<<52) == 0 {
			frac <<= 1
			e--
		}
		frac &^= 1 << 52
		return frac << 60, sign<<63 | uint64(e+f128Bias)<<48 | frac>>4

	case exp == 0x7ff:
		// Infinity and NaN. The payload moves with the significand, and
		// the quiet bit is the top of it in both formats, so a quiet
		// NaN stays quiet.
		return frac << 60, sign<<63 | 0x7fff<<48 | frac>>4
	}

	e := uint64(int64(exp) - f64Bias + f128Bias)
	return frac << 60, sign<<63 | e<<48 | frac>>4
}

// iselSoftFixOut lowers §C2's conversions out of f128 into an integer.
//
// Milestone 27's shape, one level out. There the conversion was an
// instruction that does not trap and the range check around it was two
// compares and a branch; here the conversion is a call that is undefined out
// of range, and the compares are calls too — compiler-rt's, whose answer is
// an int whose sign is the ordering.
//
// Which leaves the interval, and it is the same interval every other §C2 row
// in this tree admits: the source values whose truncation toward zero the
// destination can hold. Both ends are exclusive here, which they are not at
// f64 — binary128 keeps 113 significant bits, so it can name values between
// −2^63−1 and −2^63, all of which truncate to −2^63 and none of which a
// double can express. Trapping on those would be trapping on valid programs.
func iselSoftFixOut(c *cursor, vr *vregs, in *ir.Inst, sym string) error {
	op := in.Op()
	src, ok := vr.lookup(in.Arg(0))
	if !ok {
		return fmt.Errorf("%s: operand defined outside the function", op)
	}
	dst, err := vr.define(in.Result(0))
	if err != nil {
		return fmt.Errorf("%s: %w", op, err)
	}
	to := vr.widthOfVReg(dst)

	signed := op.Verb == ir.VSCvtF128 || op.Verb == ir.VSCvtSatF128
	sat := op.Verb == ir.VSCvtSatF128 || op.Verb == ir.VUCvtSatF128

	loLo, loHi, hiLo, hiHi := f128FixRange(to, signed)
	unord, lt := softCompare[ir.VUno].sym, softCompare[ir.VLt].sym

	// Out of range: the trap for one form, the bound for the other. Which
	// block each arm goes to is the only difference between them.
	bad := func(name string, bits func(*mir.Block)) *mir.Block {
		b := c.open(name)
		if !sat {
			b.Emit(mir.Instr{Op: trapOp{}})
			return b
		}
		bits(b)
		return b
	}

	done := c.open("fixdone")
	toDone := func(b *mir.Block) {
		b.Emit(mir.Instr{Op: jmpOp{target: done.Label}})
		c.mf.Succ(b, done.Label)
	}

	minBits, maxBits := satBoundsFor(to, signed)
	nan := bad("fixnan", func(b *mir.Block) {
		b.Emit(mir.Instr{Op: constOp{imm: 0, w: to}, Defs: []mir.VReg{dst}})
		toDone(b)
	})
	under := bad("fixunder", func(b *mir.Block) {
		b.Emit(mir.Instr{Op: constOp{imm: minBits, w: to}, Defs: []mir.VReg{dst}})
		toDone(b)
	})
	over := bad("fixover", func(b *mir.Block) {
		b.Emit(mir.Instr{Op: constOp{imm: maxBits, w: to}, Defs: []mir.VReg{dst}})
		toDone(b)
	})

	// A NaN first: it is unordered with both bounds, so neither comparison
	// below would catch it and __lttf2 answers positive for it either way.
	num := c.open("fixnum")
	if err := softCmpVRegs(c, vr, unord, src, src, condNE, nan, num); err != nil {
		return err
	}

	// At or below the low bound. __lttf2 answers negative for less and
	// zero for equal, so LE is both — and the bound is exclusive.
	aboveLo := c.open("fixabovelo")
	lo := f128Mask(c, vr, loLo, loHi)
	if err := softCmpVRegs(c, vr, lt, src, lo, condLE, under, aboveLo); err != nil {
		return err
	}

	// And at or above the high bound, which is the same test read the
	// other way: strictly less than it is the arm that converts.
	conv := c.open("fixconv")
	hi := f128Mask(c, vr, hiLo, hiHi)
	if err := softCmpVRegs(c, vr, lt, src, hi, condL, conv, over); err != nil {
		return err
	}

	c.resume(conv)
	if err := emitLibcall(c, vr, sym, []libArg{{v: src, w: wv128}}, dst, true); err != nil {
		return err
	}
	c.to(done)
	return nil
}

// softCmpVRegs calls one of compiler-rt's comparisons on two registers and
// branches on the sign of its answer.
//
// Against the whole 32-bit answer for the reason softCmp gives: these
// functions promise a sign and not a value, and a negative answer whose low
// byte happened to be zero would read as equal.
func softCmpVRegs(c *cursor, vr *vregs, sym string, a, b mir.VReg,
	cond condCode, then, els *mir.Block) error {

	answer := vr.temp(w32)
	if err := emitLibcall(c, vr, sym,
		[]libArg{{v: a, w: wv128}, {v: b, w: wv128}}, answer, true); err != nil {
		return err
	}
	c.Emit(mir.Instr{Op: cmpImmOp{imm: 0, w: w32}, Uses: []mir.VReg{answer}})
	c.branch(cond, then, els)
	return nil
}

// f128FixRange is the interval one destination admits, as the two binary128
// bit patterns its bounds are. Both are exclusive.
//
// The high bound is a power of two in every row and the low bound is one more
// than a power of two, or −1 — none of which rounds, since binary128 names
// every integer up to 2^113 exactly.
func f128FixRange(to width, signed bool) (loLo, loHi, hiLo, hiHi uint64) {
	if !signed {
		// Truncation toward zero carries everything above −1 to zero,
		// so the interval opens at −1 rather than at zero.
		loLo, loHi = f128FromUint64(true, 1)
		if to == w64 {
			hiLo, hiHi = f128Pow2(64)
		} else {
			hiLo, hiHi = f128Pow2(32)
		}
		return loLo, loHi, hiLo, hiHi
	}
	n := 31
	if to == w64 {
		n = 63
	}
	// One past the destination's most negative value, which is the first
	// source value whose truncation does not fit.
	loLo, loHi = f128FromUint64(true, 1<<uint(n)+1)
	hiLo, hiHi = f128Pow2(n)
	return loLo, loHi, hiLo, hiHi
}

// satBoundsFor is what a saturating conversion answers below and above the
// interval. The unsigned maxima are the all-ones patterns, which constOp
// carries as the negative int64 they are read as.
func satBoundsFor(to width, signed bool) (min, max int64) {
	switch {
	case to == w64 && signed:
		return -1 << 63, 1<<63 - 1
	case to == w64:
		return 0, -1
	case signed:
		return -1 << 31, 1<<31 - 1
	}
	return 0, int64(int32(-1))
}

// f128Pow2 is 2^n in binary128: the biased exponent, and a mantissa of zero.
func f128Pow2(n int) (lo, hi uint64) { return 0, uint64(f128Bias+n) << 48 }

// f128FromUint64 is an integer magnitude exactly, as binary128.
//
// Every uint64 fits with room to spare: the format keeps 113 significant
// bits. The leading one becomes the exponent and what is under it becomes the
// top of the 112-bit mantissa field, which straddles the two words.
func f128FromUint64(neg bool, v uint64) (lo, hi uint64) {
	var sign uint64
	if neg {
		sign = 1 << 63
	}
	if v == 0 {
		return 0, sign
	}

	n := uint(bits.Len64(v) - 1)
	frac := v &^ (1 << n)
	shift := 112 - n

	var mantLo, mantHi uint64
	if shift >= 64 {
		mantHi = frac << (shift - 64)
	} else {
		mantLo = frac << shift
		mantHi = frac >> (64 - shift)
	}
	return mantLo, sign | uint64(f128Bias+int(n))<<48 | mantHi
}
