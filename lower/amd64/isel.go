package amd64

import (
	"fmt"
	"math"

	"github.com/vertex-language/amd64/reg"

	"github.com/vertex-language/ir"
	"github.com/vertex-language/ir/lower/globals"
	"github.com/vertex-language/ir/lower/mir"
)

// binOps is every §A binary verb this package lowers, which is every one
// that is a single two-address instruction on this architecture.
var binOps = map[ir.Verb]bool{
	ir.VAdd: true,
	ir.VSub: true,
	ir.VMul: true,
	ir.VAnd: true,
	ir.VOr:  true,
	ir.VXor: true,
}

// fBinOps is §A3's arithmetic: the four operations that are one scalar
// SSE instruction each.
var fBinOps = map[ir.Verb]bool{
	ir.VAdd: true,
	ir.VSub: true,
	ir.VMul: true,
	ir.VDiv: true,
}

// unOps is §A's two in-place unary verbs. Both read and write one
// register, which is why they are their own shape rather than a binary
// op with a constant.
var unOps = map[ir.Verb]bool{
	ir.VNeg: true,
	ir.VNot: true,
}

// divOps is §A's four division verbs, as the two facts that tell them
// apart: whether the division is signed, and which of the two registers
// idiv leaves its answers in holds the one wanted.
var divOps = map[ir.Verb]struct {
	signed bool
	rem    bool
}{
	ir.VSDiv: {true, false},
	ir.VUDiv: {false, false},
	ir.VSRem: {true, true},
	ir.VURem: {false, true},
}

// shiftOps is §A5, which is five verbs and one shape.
var shiftOps = map[ir.Verb]bool{
	ir.VShl:  true,
	ir.VSShr: true,
	ir.VUShr: true,
	ir.VRotL: true,
	ir.VRotR: true,
}

// ovfOps is §A2, as the arithmetic that sets the flags and the flag that
// answers.
var ovfOps = map[ir.Verb]struct {
	verb ir.Verb
	cond condCode
	wide bool // the one-operand multiply, whose flags mean the high half
}{
	ir.VSAddO: {ir.VAdd, condO, false},
	ir.VUAddO: {ir.VAdd, condB, false},
	ir.VSSubO: {ir.VSub, condO, false},
	ir.VSMulO: {ir.VMul, condO, false},
	ir.VUMulO: {ir.VMul, condB, true},
}

// intConds is the whole integer compare table, shared by i32 and i64.
var intConds = map[ir.Verb]condCode{
	ir.VEq:  condE,
	ir.VNe:  condNE,
	ir.VSLt: condL,
	ir.VSLe: condLE,
	ir.VULt: condB,
	ir.VULe: condBE,
}

// ptrConds is §B's four pointer comparisons.
var ptrConds = map[ir.Verb]condCode{
	ir.VEq: condE,
	ir.VNe: condNE,
	ir.VLt: condB,
	ir.VLe: condBE,
}

// floatConds is the §B float rows that one condition answers, and the
// order the instruction has to compare in to answer them.
var floatConds = map[ir.Verb]struct {
	cond condCode
	swap bool
}{
	ir.VLt:  {condA, true},
	ir.VLe:  {condAE, true},
	ir.VUno: {condP, false},
}

// floatPairConds is §B's eq and ne on floats: two conditions and the
// operation that combines them.
var floatPairConds = map[ir.Verb]struct {
	ordered, value condCode
	combine        ir.Verb
}{
	ir.VEq: {condNP, condE, ir.VAnd},
	ir.VNe: {condP, condNE, ir.VOr},
}

// subLoads is §D2's load half, as the two facts that tell its members
// apart: how many bytes come out of memory, and whether what sits above
// them is the sign or zero.
var subLoads = map[ir.Verb]struct {
	from   access
	signed bool
}{
	ir.VSLoad8:  {a8, true},
	ir.VULoad8:  {a8, false},
	ir.VSLoad16: {a16, true},
	ir.VULoad16: {a16, false},
	ir.VSLoad32: {a32, true},
	ir.VULoad32: {a32, false},
}

// subStores is §D2's store half, which needs no sign.
var subStores = map[ir.Verb]access{
	ir.VStore8:  a8,
	ir.VStore16: a16,
	ir.VStore32: a32,
}

// iselInst pattern-matches one non-terminator, non-fused instruction into
// MIR: const and add, in whichever of the three namespaces the op names.
func iselInst(mf *mir.Func, c *cursor, vr *vregs, fr *frame, in *ir.Inst) error {
	// f128 first, before any of the width-switched rules below can see
	// it. Most of them end in a default arm meaning "the 64-bit integer
	// one", so an unrouted f128 would be quietly mis-emitted rather than
	// refused; see iselSoftFloat.
	if isSoftFloat(in) {
		return iselSoftFloat(c, vr, in)
	}

	// §V next, for the same reason and one more: its verbs are shared
	// with the scalar namespaces at and, or, xor and not, so the
	// namespace has to decide before the verb does.
	if in.Op().Type == ir.TypeV128 && vecVerb(in.Op().Verb) {
		return iselVec(c, vr, in)
	}

	verb := in.Op().Verb
	if binOps[verb] || (verb == ir.VDiv && in.Op().Type.IsFloat()) {
		return iselBinary(mf, c, vr, in)
	}
	if d, ok := divOps[verb]; ok {
		return iselDivide(c, vr, in, d.signed, d.rem)
	}
	if o, ok := ovfOps[verb]; ok {
		return iselOverflow(c, vr, in, o.verb, o.cond, o.wide)
	}
	if verb == ir.VSMulHi || verb == ir.VUMulHi {
		return iselWideMul(c, vr, in, verb == ir.VSMulHi)
	}
	if shiftOps[verb] {
		return iselShift(c, vr, in)
	}
	if unOps[verb] {
		return iselUnary(mf, c, vr, in)
	}

	if cc, ok := condFor(in.Op()); ok {
		return iselCompare(c, vr, in, cc)
	}
	if _, ok := floatPairConds[verb]; ok && in.Op().Type.IsFloat() {
		return iselFloatPairCompare(c, vr, in)
	}

	if a, ok := atomicLoads[verb]; ok {
		return iselAtomicLoad(c, vr, in, a)
	}
	if a, ok := atomicStores[verb]; ok {
		return iselAtomicStore(c, vr, in, a)
	}
	if r, ok := atomicRmws[verb]; ok {
		return iselAtomicRmw(c, vr, in, r.kind, r.a, r.alu)
	}
	if a, ok := atomicCases[verb]; ok {
		return iselAtomicCas(c, vr, in, a)
	}

	switch verb {
	case ir.VPopcnt, ir.VClz, ir.VCtz, ir.VBswap:
		return iselBitCount(c, vr, in)
	case ir.VFence:
		return iselFence(c, in)
	case ir.VAbs, ir.VSqrt:
		return iselFloatUnary(c, vr, in)
	case ir.VCopySign:
		return iselCopySign(c, vr, in)
	case ir.VWrapI64, ir.VZExtI1, ir.VZExtI32, ir.VSExtI32:
		return iselConvert(c, vr, in)
	case ir.VSCvtI32, ir.VSCvtI64, ir.VUCvtI32, ir.VUCvtI64,
		ir.VFCvtF32, ir.VFCvtF64,
		ir.VBitcastF32, ir.VBitcastI32, ir.VBitcastF64, ir.VBitcastI64:
		return iselFloatConvert(c, vr, in)

	case ir.VFMA:
		return iselFma(c, vr, in)
	case ir.VCeil, ir.VFloor, ir.VTrunc, ir.VNearest:
		return iselFloatRound(c, vr, in)
	case ir.VMinimum, ir.VMaximum, ir.VMinNum, ir.VMaxNum:
		return iselFloatMinMax(c, vr, in)
	case ir.VUCvtF32, ir.VUCvtF64, ir.VSCvtSatF32, ir.VSCvtSatF64, ir.VUCvtSatF32, ir.VUCvtSatF64:
		return iselFloatToIntExt(c, vr, in)
	case ir.VSCvtF32:
		return iselFloatToInt(c, vr, in, wf32)
	case ir.VSCvtF64:
		return iselFloatToInt(c, vr, in, wf64)
	case ir.VAlloc:
		return iselAlloc(mf, c, vr, fr, in)
	case ir.VGetAddr:
		return iselGetAddr(mf, c, vr, in)
	case ir.VTLSAddr:
		return iselTLSAddr(c, vr, in)
	case ir.VBlockAddr:
		return iselBlockAddr(c, vr, in)
	case ir.VFrameAddr:
		return iselFrameAddr(c, vr, in)
	case ir.VReturnAddr:
		return iselReturnAddr(c, vr, in)
	case ir.VDiff:
		return iselDiff(c, vr, in)
	case ir.VFromI64, ir.VFromPtr:
		return iselPtrIntCast(c, vr, in)
	case ir.VAlloca:
		return iselAlloca(c, vr, fr, in)
	case ir.VStackSave:
		return iselStackSave(c, vr, in)
	case ir.VStackRestore:
		return iselStackRestore(c, vr, in)
	case ir.VVaStart:
		return iselVaStart(c, vr, fr, in)
	case ir.VVaEnd:
		return iselVaEnd(in)
	case ir.VVaCopy:
		return iselVaCopy(c, vr, in)
	case ir.VVaArg:
		return iselVaArg(c, vr, in)
	case ir.VVaArgRef:
		return iselVaArgRef(c, vr, in)
	case ir.VMemCpy, ir.VMemMove, ir.VMemSet, ir.VMemCmp:
		return iselLibcall(c, vr, in)
	case ir.VAsm:
		return iselAsm(c.fn, c, vr, in, vr.nextAsmID(), nil)
	case ir.VCallInd:
		return iselCallInd(c, vr, in)
	case ir.VCall:
		return iselCall(c, vr, in)
	case ir.VSelect:
		return iselSelect(c, vr, in)
	case ir.VConst:
		return iselConst(mf, c, vr, in)

	case ir.VLoad:
		return iselLoad(mf, c, vr, in)
	case ir.VStore:
		return iselStore(c, vr, in)
	}
	if ext, ok := subLoads[in.Op().Verb]; ok {
		return iselExtLoad(mf, c, vr, in, ext.from, ext.signed)
	}
	if to, ok := subStores[in.Op().Verb]; ok {
		return iselSubStore(c, vr, in, to)
	}
	return fmt.Errorf("unsupported instruction %s", in.Op())
}

// iselConst materializes a literal into a fresh register.
func iselConst(mf *mir.Func, c *cursor, vr *vregs, in *ir.Inst) error {
	op := in.Op()
	lit, ok := in.Lit()
	if !ok {
		return fmt.Errorf("%s: only a plain literal is supported", op)
	}
	if op.Type == ir.TypeV128 {
		if lit.Kind() != ir.ConstBytes {
			return fmt.Errorf("%s: a v128 constant needs a bytes literal", op)
		}
		return iselVecConst(c, vr, in, lit.Bytes())
	}
	result, err := vr.define(in.Result(0))
	if err != nil {
		return fmt.Errorf("%s: %w", op, err)
	}
	w := vr.widthOfVReg(result)

	if w.isFloat() {
		if lit.Kind() != ir.ConstFloat {
			return fmt.Errorf("%s: a float constant needs a float literal", op)
		}
		return iselFloatConst(c, vr, result, w, lit.Float())
	}
	// An integer literal, or one of §2's three symbolic constants, which
	// are integers this target has to be asked for.
	v, err := globals.ConstInt(layout{}, lit)
	if err != nil {
		return fmt.Errorf("%s: %w", op, err)
	}
	c.Emit(mir.Instr{Op: constOp{imm: v, w: w}, Defs: []mir.VReg{result}})
	return nil
}

// iselFloatConst materializes a float literal as its own bit pattern.
func iselFloatConst(c *cursor, vr *vregs, dst mir.VReg, w width, f float64) error {
	bits, iw := uint64(math.Float64bits(f)), w64
	if w == wf32 {
		// float32(f) and not a reinterpretation: an f32 literal is the
		// nearest f32 to the value the module wrote, and rounding it is
		// what makes 0.1 an f32 rather than a truncated f64.
		bits, iw = uint64(math.Float32bits(float32(f))), w32
	}

	tmp := vr.temp(iw)
	c.Emit(mir.Instr{Op: constOp{imm: int64(bits), w: iw}, Defs: []mir.VReg{tmp}})
	c.Emit(mir.Instr{Op: bitsToFloatOp{w: w}, Defs: []mir.VReg{dst}, Uses: []mir.VReg{tmp}})
	return nil
}

// iselBinary lowers one of binOps: two registers in, one out.
func iselBinary(mf *mir.Func, c *cursor, vr *vregs, in *ir.Inst) error {
	op := in.Op()
	a, ok := vr.lookup(in.Arg(0))
	if !ok {
		return fmt.Errorf("%s: operand defined outside the function", op)
	}
	b, ok := vr.lookup(in.Arg(1))
	if !ok {
		return fmt.Errorf("%s: operand defined outside the function", op)
	}

	result, err := vr.define(in.Result(0))
	if err != nil {
		return fmt.Errorf("%s: %w", op, err)
	}
	w := vr.widthOfVReg(result)

	if w.isFloat() {
		if !fBinOps[op.Verb] {
			return fmt.Errorf("%s: not a float operation this package emits", op)
		}
		c.Emit(mir.Instr{
			Op:   fAluOp{verb: op.Verb, w: w},
			Defs: []mir.VReg{result},
			Uses: []mir.VReg{a, b},
		})
		return nil
	}

	c.Emit(mir.Instr{
		Op:   aluOp{verb: op.Verb, w: w},
		Defs: []mir.VReg{result},
		Uses: []mir.VReg{a, b},
	})
	return nil
}

// iselCompare materializes a compare that something other than a branch
// or a select reads — the case fusesEveryUse answered no to.
func iselCompare(c *cursor, vr *vregs, in *ir.Inst, cc compare) error {
	op := in.Op()
	a, b, err := compareOperands(vr, in, cc)
	if err != nil {
		return err
	}
	dst, err := vr.define(in.Result(0))
	if err != nil {
		return fmt.Errorf("%s: %w", op, err)
	}

	// Nothing between the two: SETcc reads the flags the compare set,
	// and most of what this package emits would write them.
	c.Emit(mir.Instr{Op: cmpOp{w: cc.w}, Uses: []mir.VReg{a, b}})
	c.Emit(mir.Instr{Op: setccOp{cond: cc.cond}, Defs: []mir.VReg{dst}})
	return nil
}

// iselFloatPairCompare lowers §B's float eq and ne, the two rows no
// single condition answers.
func iselFloatPairCompare(c *cursor, vr *vregs, in *ir.Inst) error {
	op := in.Op()
	pair, ok := floatPairConds[op.Verb]
	if !ok {
		return fmt.Errorf("%s: not a float comparison this package emits", op)
	}
	w, ok := widthOf(op.Type)
	if !ok || !w.isFloat() {
		return fmt.Errorf("%s: not a float comparison", op)
	}
	a, ok := vr.lookup(in.Arg(0))
	if !ok {
		return fmt.Errorf("%s: operand defined outside the function", op)
	}
	b, ok := vr.lookup(in.Arg(1))
	if !ok {
		return fmt.Errorf("%s: operand defined outside the function", op)
	}
	dst, err := vr.define(in.Result(0))
	if err != nil {
		return fmt.Errorf("%s: %w", op, err)
	}

	// Both setccs read the same compare's flags, which is why nothing
	// stands between them: SETcc writes a byte and the MOVZX that
	// follows it writes a register, and neither touches EFLAGS.
	ord := vr.temp(w32)
	c.Emit(mir.Instr{Op: cmpOp{w: w}, Uses: []mir.VReg{a, b}})
	c.Emit(mir.Instr{Op: setccOp{cond: pair.ordered}, Defs: []mir.VReg{ord}})
	c.Emit(mir.Instr{Op: setccOp{cond: pair.value}, Defs: []mir.VReg{dst}})
	c.Emit(mir.Instr{
		Op:   aluOp{verb: pair.combine, w: w32},
		Defs: []mir.VReg{dst},
		Uses: []mir.VReg{dst, ord},
	})
	return nil
}

// iselConvert lowers §C's register-to-register integer conversions.
//
// §C has four verbs and this emits two instructions, which is the whole
// shape of the section on this architecture.
func iselConvert(c *cursor, vr *vregs, in *ir.Inst) error {
	op := in.Op()
	src, ok := vr.lookup(in.Arg(0))
	if !ok {
		return fmt.Errorf("%s: operand defined outside the function", op)
	}
	dst, err := vr.define(in.Result(0))
	if err != nil {
		return fmt.Errorf("%s: %w", op, err)
	}

	switch {
	case op.Verb == ir.VWrapI64,
		op.Verb == ir.VZExtI1 && vr.widthOfVReg(dst) == w32:
		emitCopy(c, dst, src, w32)
	case op.Verb == ir.VSExtI32:
		c.Emit(mir.Instr{Op: sextOp{}, Defs: []mir.VReg{dst}, Uses: []mir.VReg{src}})
	default:
		c.Emit(mir.Instr{Op: zextOp{}, Defs: []mir.VReg{dst}, Uses: []mir.VReg{src}})
	}
	return nil
}

// f2iRange is the interval of source values one §C2 conversion admits,
// as the two comparisons that check it.
var f2iRange = map[[2]width]struct {
	lo, hi   float64
	loStrict bool // x > lo rather than x >= lo
}{
	{wf32, w32}: {lo: -2147483648, hi: 2147483648},
	{wf64, w32}: {lo: -2147483649, hi: 2147483648, loStrict: true},
	{wf32, w64}: {lo: -9223372036854775808, hi: 9223372036854775808},
	{wf64, w64}: {lo: -9223372036854775808, hi: 9223372036854775808},
}

// iselFloatToInt lowers §C2's trapping float-to-integer conversions.
//
// CVTTSD2SI does not trap. Given a NaN or a value out of range it writes
// the integer indefinite value, so this conversion is a range check.
func iselFloatToInt(c *cursor, vr *vregs, in *ir.Inst, from width) error {
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

	r, ok := f2iRange[[2]width{from, to}]
	if !ok {
		return fmt.Errorf("%s: no range check for this pair of widths", op)
	}

	trap := c.open("trap")

	// Below the interval, or unordered.
	lo := vr.temp(from)
	if err := iselFloatConst(c, vr, lo, from, r.lo); err != nil {
		return err
	}
	belowCond := condB
	if r.loStrict {
		belowCond = condBE
	}
	inRange := c.open("inrange")
	c.Emit(mir.Instr{Op: cmpOp{w: from}, Uses: []mir.VReg{src, lo}})
	c.branch(belowCond, trap, inRange)

	// Above the interval, or unordered. Comparing the bound against the
	// value rather than the other way round, so that the condition is
	// again one the unordered case satisfies.
	hi := vr.temp(from)
	if err := iselFloatConst(c, vr, hi, from, r.hi); err != nil {
		return err
	}
	convert := c.open("cvt")
	c.Emit(mir.Instr{Op: cmpOp{w: from}, Uses: []mir.VReg{hi, src}})
	c.branch(condBE, trap, convert)

	// ud2, and nothing after it. A trap is a terminator with no
	// successors: control does not leave the instruction, so there is no
	// frame to tear down and nothing to fall through to.
	trap.Emit(mir.Instr{Op: trapOp{}})

	c.Emit(mir.Instr{
		Op:   cvtFloatToIntOp{from: from, to: to},
		Defs: []mir.VReg{dst}, Uses: []mir.VReg{src},
	})
	return nil
}

// iselFloatConvert lowers the §C verbs that cross between the two
// register files, and the one that crosses between the two float widths.
func iselFloatConvert(c *cursor, vr *vregs, in *ir.Inst) error {
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

	switch op.Verb {
	case ir.VBitcastI32, ir.VBitcastI64:
		// The same bits read as a float. One instruction and no
		// rounding: this is the move across the files that iselFloatConst
		// ends with, reached from §C instead of from a literal.
		c.Emit(mir.Instr{Op: bitsToFloatOp{w: to}, Defs: []mir.VReg{dst}, Uses: []mir.VReg{src}})
		return nil

	case ir.VBitcastF32, ir.VBitcastF64:
		c.Emit(mir.Instr{Op: floatToBitsOp{w: vr.widthOfVReg(src)}, Defs: []mir.VReg{dst}, Uses: []mir.VReg{src}})
		return nil

	case ir.VFCvtF32, ir.VFCvtF64:
		c.Emit(mir.Instr{Op: cvtFloatOp{to: to}, Defs: []mir.VReg{dst}, Uses: []mir.VReg{src}})
		return nil

	case ir.VSCvtI32:
		c.Emit(mir.Instr{Op: cvtIntToFloatOp{from: w32, to: to}, Defs: []mir.VReg{dst}, Uses: []mir.VReg{src}})
		return nil

	case ir.VSCvtI64:
		c.Emit(mir.Instr{Op: cvtIntToFloatOp{from: w64, to: to}, Defs: []mir.VReg{dst}, Uses: []mir.VReg{src}})
		return nil

	case ir.VUCvtI32:
		// Zero-extended into a 64-bit register first, where it is an
		// exact signed value, and then converted as one.
		wide := vr.temp(w64)
		c.Emit(mir.Instr{Op: zextOp{}, Defs: []mir.VReg{wide}, Uses: []mir.VReg{src}})
		c.Emit(mir.Instr{Op: cvtIntToFloatOp{from: w64, to: to}, Defs: []mir.VReg{dst}, Uses: []mir.VReg{wide}})
		return nil

	case ir.VUCvtI64:
		return iselUCvtI64(c, vr, src, dst, to)
	}
	return fmt.Errorf("%s: not a conversion this package lowers", op)
}

// iselUCvtI64 lowers §C2's unsigned 64-bit source, the one row in the
// section with no instruction behind it.
//
// CVTSI2SD reads its source as signed, so every value with the top bit set
// converts to a negative float: the upper half of the range is not slightly
// wrong but wrong by 2^64. Below 2^63 the signed instruction is already the
// answer, and the branch is between those two cases.
//
// At or above it, the value is halved to bring it into the signed range and
// the result doubled — and doubling a float is exact, so the conversion is
// still the only rounding. What makes that rounding the right one is the bit
// the halving drops. It is OR'd back into the low bit rather than discarded,
// so a value that sits exactly halfway between two representable results
// still looks halfway after the shift instead of looking exact; without it
// the sequence rounds twice and answers a ulp low on the ties. The low bit
// cannot itself affect the result, since a value at or above 2^63 needs
// sixty-four bits and the destination keeps at most fifty-three of them.
func iselUCvtI64(c *cursor, vr *vregs, src, dst mir.VReg, to width) error {
	big := c.open("ucvtbig")
	small := c.open("ucvtsmall")
	done := c.open("ucvtdone")

	// Against zero at the full width, and not testOp: that one is the
	// byte-wide test milestone 20 built for an i1, and the bit this needs
	// is the sixty-fourth. L against zero is that bit on its own.
	zero := vr.temp(w64)
	c.Emit(mir.Instr{Op: zeroOp{}, Defs: []mir.VReg{zero}})
	c.Emit(mir.Instr{Op: cmpOp{w: w64}, Uses: []mir.VReg{src, zero}})
	c.branch(condL, big, small)

	c.resume(small)
	c.Emit(mir.Instr{
		Op:   cvtIntToFloatOp{from: w64, to: to},
		Defs: []mir.VReg{dst}, Uses: []mir.VReg{src},
	})
	c.to(done)

	c.resume(big)
	one := vr.temp(w64)
	c.Emit(mir.Instr{Op: constOp{imm: 1, w: w64}, Defs: []mir.VReg{one}})

	// The count is CL's, like every other shift here.
	cl := vr.physical(reg.RCX, w64)
	emitCopy(c, cl, one, w64)
	half := vr.temp(w64)
	c.Emit(mir.Instr{
		Op:   shiftOp{verb: ir.VUShr, w: w64},
		Defs: []mir.VReg{half}, Uses: []mir.VReg{src, cl},
	})

	low := vr.temp(w64)
	c.Emit(mir.Instr{
		Op:   aluOp{verb: ir.VAnd, w: w64},
		Defs: []mir.VReg{low}, Uses: []mir.VReg{src, one},
	})
	odd := vr.temp(w64)
	c.Emit(mir.Instr{
		Op:   aluOp{verb: ir.VOr, w: w64},
		Defs: []mir.VReg{odd}, Uses: []mir.VReg{half, low},
	})

	c.Emit(mir.Instr{
		Op:   cvtIntToFloatOp{from: w64, to: to},
		Defs: []mir.VReg{dst}, Uses: []mir.VReg{odd},
	})
	c.Emit(mir.Instr{
		Op:   fAluOp{verb: ir.VAdd, w: to},
		Defs: []mir.VReg{dst}, Uses: []mir.VReg{dst, dst},
	})
	c.to(done)

	c.resume(done)
	return nil
}

// iselBitCount lowers the bit counting instructions.
func iselBitCount(c *cursor, vr *vregs, in *ir.Inst) error {
	op := in.Op()
	src, ok := vr.lookup(in.Arg(0))
	if !ok {
		return fmt.Errorf("%s: operand defined outside the function", op)
	}
	dst, err := vr.define(in.Result(0))
	if err != nil {
		return fmt.Errorf("%s: %w", op, err)
	}
	w := vr.widthOfVReg(dst)
	if w.isFloat() {
		return fmt.Errorf("%s: not an integer operation", op)
	}

	if op.Verb == ir.VBswap {
		c.Emit(mir.Instr{Op: bswapOp{w: w}, Defs: []mir.VReg{dst}, Uses: []mir.VReg{src}})
		return nil
	}
	c.Emit(mir.Instr{
		Op:   bitCountOp{verb: op.Verb, w: w},
		Defs: []mir.VReg{dst}, Uses: []mir.VReg{src},
	})
	return nil
}

// iselSelect lowers §F: both arms are already evaluated, and one of them
// is kept.
func iselSelect(c *cursor, vr *vregs, in *ir.Inst) error {
	op := in.Op()
	cmp, sh, fusable := fusedCompare(in)
	cc := sh.cond

	var x, y mir.VReg
	if fusable {
		var err error
		x, y, err = compareOperands(vr, cmp, sh)
		if err != nil {
			return fmt.Errorf("%s: %w", op, err)
		}
	} else {
		v, ok := vr.lookup(in.Arg(0))
		if !ok {
			return fmt.Errorf("%s: condition %s is not a value this package produced", op, in.Arg(0))
		}
		x, cc = v, condNE
	}

	yes, ok := vr.lookup(in.Arg(1))
	if !ok {
		return fmt.Errorf("%s: operand defined outside the function", op)
	}
	no, ok := vr.lookup(in.Arg(2))
	if !ok {
		return fmt.Errorf("%s: operand defined outside the function", op)
	}

	result, err := vr.define(in.Result(0))
	if err != nil {
		return fmt.Errorf("%s: %w", op, err)
	}
	w := vr.widthOfVReg(result)

	// A vector select has no conditional move to reach for — there is no
	// CMOV in that register file — so it goes through a mask instead, and
	// wants the condition as a value rather than as flags.
	if w == wv128 {
		cond, ok := vr.lookup(in.Arg(0))
		if !ok {
			return fmt.Errorf("%s: condition %s is not a value this package produced", op, in.Arg(0))
		}
		iselVecSelect(c, vr, cond, yes, no, result)
		return nil
	}

	// The compare first, then the move, then the conditional move. The
	// move is a mov, which writes no flags, so the compare's answer is
	// still there when the cmov reads it.
	if fusable {
		c.Emit(mir.Instr{Op: cmpOp{w: sh.w}, Uses: []mir.VReg{x, y}})
	} else {
		c.Emit(mir.Instr{Op: testOp{}, Uses: []mir.VReg{x}})
	}
	c.Emit(mir.Instr{
		Op:   cmovOp{cond: cc, w: w},
		Defs: []mir.VReg{result},
		Uses: []mir.VReg{yes, no},
	})
	return nil
}

// A callSite is the physical registers one call names, each mapped to
// the single vreg that stands for it.
//
// A call names one register twice more often than it looks: RDX is the
// third argument and the second return, XMM0 the first of each, RAX the
// variadic count and the first return, and all of them are clobbers too.
// Two vregs pinned to one register and live at one instruction is
// regalloc.ErrPinConflict, so every reason for naming it gets one vreg.
// The width it is first asked for is the width it keeps; that only picks
// the register file, and the moves at either end carry their own.
type callSite struct {
	vr   *vregs
	ints map[reg.R64]mir.VReg
	xmms map[reg.Xmm]mir.VReg
}

func newCallSite(vr *vregs) *callSite {
	return &callSite{vr: vr, ints: map[reg.R64]mir.VReg{}, xmms: map[reg.Xmm]mir.VReg{}}
}

// intReg is the vreg pinned to r, made on first ask.
func (s *callSite) intReg(r reg.R64, w width) mir.VReg {
	if v, ok := s.ints[r]; ok {
		return v
	}
	v := s.vr.physical(r, w)
	s.ints[r] = v
	return v
}

// xmmReg is intReg for the other register file.
func (s *callSite) xmmReg(r reg.Xmm, w width) mir.VReg {
	if v, ok := s.xmms[r]; ok {
		return v
	}
	v := s.vr.physicalXmm(r, w)
	s.xmms[r] = v
	return v
}

// namedInt reports whether r already has a vreg at this site.
func (s *callSite) namedInt(r reg.R64) bool { _, ok := s.ints[r]; return ok }

// namedXmm is namedInt for the other register file.
func (s *callSite) namedXmm(r reg.Xmm) bool { _, ok := s.xmms[r]; return ok }

// iselCallInd lowers §G2's callind: a call through an address in a value
// rather than to a symbol.
func iselCallInd(c *cursor, vr *vregs, in *ir.Inst) error {
	op := in.Op()
	addr, ok := vr.lookup(in.Arg(0))
	if !ok {
		return fmt.Errorf("%s: callee defined outside the function", op)
	}

	// The address into a vreg of this call's own, rather than using the
	// one the value already lives in. Every caller-saved register is a
	// destination of the call, so the address interferes with all of
	// them and the allocator has to put it somewhere the call does not
	// write — which it can only do if the address is a value the call
	// uses and not one of the pinned vregs it also defines.
	target := vr.temp(w64)
	emitCopy(c, target, addr, w64)

	var sig *ir.Sig
	if t := in.NamedType(); t != nil {
		sig = t.Sig()
	}
	return iselCallSeq(c, vr, "callind", callArgSpec(in), sig, in.Args()[1:], in.Results(), []mir.VReg{target}, callIndOp{})
}

// iselCall lowers a direct call.
func iselCall(c *cursor, vr *vregs, in *ir.Inst) error {
	op := in.Op()
	sym := in.Symbol()
	if sym == nil {
		return fmt.Errorf("%s: no callee named", op)
	}
	var sig *ir.Sig
	if callee := in.Callee(); callee != nil {
		sig = callee.Signature()
	}
	return iselCallSeq(c, vr, "call @"+sym.Name(), callArgSpec(in), sig, in.Args(), in.Results(), nil, callOp{sym: sym.Name()})
}

// iselCallSeq is the calling sequence both call verbs share: arguments
// into the places SysV names, the clobbers, the instruction, the results
// back out. They differ only in how the callee is named, so op is the
// instruction and leading is what precedes the arguments in Uses — the
// address for an indirect call, where emit looks for it. A slice rather
// than a vreg and a flag: vreg zero is a vreg like any other.
func iselCallSeq(c *cursor, vr *vregs, what string, spec []abiArg, sig *ir.Sig, args, results []*ir.Def, leading []mir.VReg, op any) error {
	abi := c.fn.Module().Layout().ABI
	places, err := classify(abi, spec)
	if err != nil {
		return fmt.Errorf("%s: %w", what, err)
	}
	rets, err := classifyRet(c.fn.Module().Layout().ABI, typesOf(results))
	if err != nil {
		return fmt.Errorf("%s: %w", what, err)
	}

	// The by-reference copies first of all. Each is a memcpy, which is a
	// call, so nothing else can be in a register yet — and each lands
	// above every argument slot, so no later store can reach one.
	for i, pl := range places {
		if !pl.indirect {
			continue
		}
		src, ok := vr.lookup(args[i])
		if !ok {
			return fmt.Errorf("%s: argument %d defined outside the function", what, i)
		}
		if err := copyToOutgoingAt(c, vr, what, pl.copyOff, pl.size, src); err != nil {
			return err
		}
	}

	// The stack arguments next, into the outgoing area at the bottom of
	// the frame — before the register copies below, not after. A stack
	// argument's value may be living in any register at all, including
	// one a copy below is about to write, and a store that has already
	// happened is a value that no longer cares where it was. Ordered the
	// other way, every stack argument would be live across all the
	// copies and the allocator would have to put it somewhere they
	// cannot reach.
	for i, pl := range places {
		if pl.kind != placeStack {
			continue
		}
		src, ok := vr.lookup(args[i])
		if !ok {
			return fmt.Errorf("%s: argument %d defined outside the function", what, i)
		}
		if pl.indirect {
			// The pointer, not the bytes: the copy is already in the
			// outgoing area and what the callee receives is its address.
			addr := vr.temp(w64)
			c.Emit(mir.Instr{Op: leaOutOp{off: pl.copyOff}, Defs: []mir.VReg{addr}})
			c.Emit(mir.Instr{Op: argStoreOp{off: pl.off, w: w64}, Uses: []mir.VReg{addr}})
			continue
		}
		if pl.isAggregate() {
			// The bytes, not the pointer. byval says the callee receives
			// what the pointer names, and a MEMORY-class aggregate is
			// received as a copy in the outgoing area — so the caller is
			// what makes that copy.
			if err := copyToOutgoing(c, vr, what, pl, src); err != nil {
				return err
			}
			continue
		}
		c.Emit(mir.Instr{Op: argStoreOp{off: pl.off, w: pl.w()}, Uses: []mir.VReg{src}})
	}

	// Then the register arguments. Every one is copied into its register
	// before any of them is read, which is what makes the copies a
	// parallel assignment and not a sequence — a value already in RSI
	// being copied to RDI while another goes to RSI would lose it. They
	// cannot collide, because each destination is a distinct pinned vreg
	// and the allocator has to keep them apart; the copies are what may
	// be elided, never reordered into each other.
	site := newCallSite(vr)
	var inRegs []mir.VReg
	var floats int
	for i, pl := range places {
		if pl.kind == placeStack {
			continue
		}
		src, ok := vr.lookup(args[i])
		if !ok {
			return fmt.Errorf("%s: argument %d defined outside the function", what, i)
		}
		for k, slot := range pl.regs {
			var dst mir.VReg
			if slot.kind == placeFloat {
				dst = site.xmmReg(floatArgReg(abi, slot.i), slot.w)
				floats++
			} else {
				dst = site.intReg(intArgReg(abi, slot.i), slot.w)
			}
			inRegs = append(inRegs, dst)

			if pl.indirect {
				// As above: the address of the copy made before any of
				// these registers was written.
				addr := vr.temp(w64)
				c.Emit(mir.Instr{Op: leaOutOp{off: pl.copyOff}, Defs: []mir.VReg{addr}})
				emitCopy(c, dst, addr, w64)
				continue
			}
			if !pl.isAggregate() {
				emitCopy(c, dst, src, slot.w)
				if pl.dupInt {
					// The same bits in the integer register at the same
					// position, for a callee that homed it and reads it
					// back through va_arg. See place.dupInt.
					also := site.intReg(intArgReg(abi, slot.i), w64)
					c.Emit(mir.Instr{
						Op:   floatToBitsOp{w: slot.w},
						Defs: []mir.VReg{also},
						Uses: []mir.VReg{dst},
					})
					inRegs = append(inRegs, also)
				}
				continue
			}
			// An aggregate small enough for registers travels as its
			// eightbytes, read out of the storage the pointer names.
			// One load each, into the file its own class named: the
			// same bytes reach an integer register or a vector one
			// depending on what the fields in them were.
			c.Emit(mir.Instr{
				Op:   loadAtOp{off: int32(k * 8), w: slot.w},
				Defs: []mir.VReg{dst},
				Uses: []mir.VReg{src},
			})
		}
	}

	// Uses[0] is the address for an indirect call, because that is where
	// emit looks for it. It is not one of the pinned vregs and not one
	// of the defs: the call reads the address, it does not write it.
	uses := append(append([]mir.VReg{}, leading...), inRegs...)

	// RAX, whether or not anything returns in it: it is clobbered either
	// way, and naming it first is what makes it the head of the clobber
	// list below.
	rax := site.intReg(reg.RAX, w64)

	if sig != nil && sig.IsVariadic() && abi != abiMS {
		// AL carries the number of vector registers used, which is what
		// a variadic SysV callee reads to decide how much of its save
		// area to write; too small and va_arg reads a register nobody
		// saved. The Microsoft ABI has no such register: its callee
		// homes four fixed registers unconditionally.
		//
		// A 32-bit move, not a byte one: writing the 32-bit view zeroes
		// the rest of RAX rather than merging into it, and the upper
		// bytes are unspecified anyway.
		c.Emit(mir.Instr{Op: constOp{imm: int64(floats), w: w32}, Defs: []mir.VReg{rax}})
		uses = append(uses, rax)
	}

	// Every caller-saved register is a destination whether or not the
	// call names it: that is the list of places a value live across it
	// cannot be. Every vector register this package allocates too — SysV
	// has no callee-saved one, and under the Microsoft ABI the ones it
	// does have are not in the allocatable set. Either way a float live
	// across a call is spilled, which is the ABI's answer and not this
	// package's choice.
	defs := []mir.VReg{rax}
	defs = append(defs, inRegs...)
	for _, r := range regsFor(abi).callerSaved {
		if site.namedInt(r) {
			continue
		}
		defs = append(defs, site.intReg(r, w64))
	}
	for _, r := range regsFor(abi).xmm {
		if site.namedXmm(r) {
			continue
		}
		defs = append(defs, site.xmmReg(r, wf64))
	}

	c.Emit(mir.Instr{Op: op, Defs: defs, Uses: uses})

	// A result §3.2.3 brought back in registers, into the storage the
	// caller set aside for it. There is no ir.Def to copy into — the call
	// returns nothing, and the front end said so by writing an sret
	// parameter — so what the caller gets is the bytes, written through the
	// address it passed as that parameter.
	if len(spec) > 0 && !spec[0].sret.IsZero() {
		agg, inRegs, err := sretRegs(abi, spec[0].sret)
		if err != nil {
			return fmt.Errorf("%s: %w", what, err)
		}
		if inRegs {
			dst, ok := vr.lookup(args[0])
			if !ok {
				return fmt.Errorf("%s: the sret argument is defined outside the function", what)
			}
			for k, slot := range sretSlots(abi, agg) {
				var src mir.VReg
				if slot.kind == placeFloat {
					src = site.xmmReg(floatRetReg(abi, slot.i), slot.w)
				} else {
					src = site.intReg(intRetReg(abi, slot.i), slot.w)
				}
				// Address first, value second, which is this op's order.
				c.Emit(mir.Instr{
					Op:   storeAtOp{off: int32(k * 8), w: slot.w},
					Uses: []mir.VReg{dst, src},
				})
			}
		}
	}

	// And the results back out of the registers they came back in. Each
	// source comes from the site, so a result in a register an argument
	// arrived in is the vreg that argument used — the call redefined it,
	// which is exactly what happened.
	for i, pl := range rets {
		result, err := vr.define(results[i])
		if err != nil {
			return fmt.Errorf("%s: %w", what, err)
		}
		w := vr.widthOfVReg(result)
		slot := pl.regs[0]
		var src mir.VReg
		if slot.kind == placeFloat {
			src = site.xmmReg(floatRetReg(abi, slot.i), w)
		} else {
			src = site.intReg(intRetReg(abi, slot.i), w)
		}
		emitCopy(c, result, src, w)
	}
	return nil
}

// copyToOutgoing copies a MEMORY-class byval aggregate into the
// outgoing area, which is the caller's half of what byval means.
//
// A memcpy rather than an open-coded run of loads and stores, for the
// reason §E's verbs are calls: which sequence is right depends on the
// size, and the library function is what knows. It is emitted in the
// stack pass, before any argument register is set up, so the call it
// makes cannot disturb them.
func copyToOutgoing(c *cursor, vr *vregs, what string, pl place, src mir.VReg) error {
	return copyToOutgoingAt(c, vr, what, pl.off, pl.size, src)
}

// copyToOutgoingAt is copyToOutgoing to a chosen offset, which is what
// the Microsoft ABI's by-reference copies need: they do not sit at the
// argument's own slot, because that slot holds the pointer.
func copyToOutgoingAt(c *cursor, vr *vregs, what string, off int32, size uint64, src mir.VReg) error {
	dst := vr.temp(w64)
	c.Emit(mir.Instr{Op: leaOutOp{off: off}, Defs: []mir.VReg{dst}})

	n := vr.temp(w64)
	c.Emit(mir.Instr{Op: constOp{imm: int64(size), w: w64}, Defs: []mir.VReg{n}})

	if err := emitLibcall(c, vr, memcpySym, []libArg{
		{v: dst, w: w64},
		{v: src, w: w64},
		{v: n, w: w64},
	}, 0, false); err != nil {
		return fmt.Errorf("%s: %w", what, err)
	}
	return nil
}

// iselWideMul lowers §A's smul_hi and umul_hi.
func iselWideMul(c *cursor, vr *vregs, in *ir.Inst, signed bool) error {
	op := in.Op()
	hi, err := emitWideMul(c, vr, in, signed)
	if err != nil {
		return err
	}

	result, err := vr.define(in.Result(0))
	if err != nil {
		return fmt.Errorf("%s: %w", op, err)
	}
	emitCopy(c, result, hi, vr.widthOfVReg(result))
	return nil
}

// iselOverflow lowers §A2's five predicates.
//
// The arithmetic's own result is discarded.
func iselOverflow(c *cursor, vr *vregs, in *ir.Inst, arith ir.Verb, cc condCode, wide bool) error {
	op := in.Op()
	if wide {
		if _, err := emitWideMul(c, vr, in, false); err != nil {
			return err
		}
		return emitSetcc(c, vr, in, cc)
	}

	a, ok := vr.lookup(in.Arg(0))
	if !ok {
		return fmt.Errorf("%s: operand defined outside the function", op)
	}
	b, ok := vr.lookup(in.Arg(1))
	if !ok {
		return fmt.Errorf("%s: operand defined outside the function", op)
	}
	w, ok := widthOf(in.Arg(0).Type())
	if !ok {
		return fmt.Errorf("%s: %s is not a value this package holds in a general-purpose register", op, in.Arg(0).Type())
	}

	// Into a temporary, because the sum is not the answer. Nothing reads
	// it, and the allocator is free to put it wherever it likes and to
	// let the next value have that register immediately.
	sum := vr.temp(w)
	emitCopy(c, sum, a, w)
	c.Emit(mir.Instr{Op: aluOp{verb: arith, w: w}, Defs: []mir.VReg{sum}, Uses: []mir.VReg{sum, b}})
	return emitSetcc(c, vr, in, cc)
}

// emitWideMul is the one-operand multiply both of its callers need.
func emitWideMul(c *cursor, vr *vregs, in *ir.Inst, signed bool) (hi mir.VReg, err error) {
	op := in.Op()
	a, ok := vr.lookup(in.Arg(0))
	if !ok {
		return 0, fmt.Errorf("%s: operand defined outside the function", op)
	}
	b, ok := vr.lookup(in.Arg(1))
	if !ok {
		return 0, fmt.Errorf("%s: operand defined outside the function", op)
	}
	w, ok := widthOf(in.Arg(0).Type())
	if !ok {
		return 0, fmt.Errorf("%s: %s is not a value this package holds in a general-purpose register", op, in.Arg(0).Type())
	}

	lo := vr.physical(reg.RAX, w)
	hi = vr.physical(reg.RDX, w)

	// The multiplier is copied for the reason the divisor is: it may
	// already be in RAX or RDX, and this instruction writes both.
	mul := vr.temp(w)
	emitCopy(c, mul, b, w)
	emitCopy(c, lo, a, w)

	c.Emit(mir.Instr{
		Op:   mulOp{signed: signed, w: w},
		Defs: []mir.VReg{lo, hi},
		Uses: []mir.VReg{lo, mul},
	})
	return hi, nil
}

// emitSetcc gives in's i1 result the flag cc names.
func emitSetcc(c *cursor, vr *vregs, in *ir.Inst, cc condCode) error {
	dst, err := vr.define(in.Result(0))
	if err != nil {
		return fmt.Errorf("%s: %w", in.Op(), err)
	}
	c.Emit(mir.Instr{Op: setccOp{cond: cc}, Defs: []mir.VReg{dst}})
	return nil
}

// iselDivide lowers §A's four divisions.
//
// The dividend is RAX, its high half is RDX, and the quotient and
// remainder come back in those same two registers.
func iselDivide(c *cursor, vr *vregs, in *ir.Inst, signed, wantRem bool) error {
	op := in.Op()
	a, ok := vr.lookup(in.Arg(0))
	if !ok {
		return fmt.Errorf("%s: operand defined outside the function", op)
	}
	b, ok := vr.lookup(in.Arg(1))
	if !ok {
		return fmt.Errorf("%s: operand defined outside the function", op)
	}
	result, err := vr.define(in.Result(0))
	if err != nil {
		return fmt.Errorf("%s: %w", op, err)
	}
	w := vr.widthOfVReg(result)

	lo := vr.physical(reg.RAX, w)
	hi := vr.physical(reg.RDX, w)

	// The divisor is copied rather than read where it lies. It may
	// already be in RAX or RDX — the third SysV argument arrives in RDX —
	// and those are the two registers this instruction overwrites. The
	// copy gives the allocator a vreg it is free to place elsewhere, and
	// the interference graph keeps it out of both, since it is an operand
	// of an instruction that defines them. When the divisor is somewhere
	// else already the copy costs nothing, because a copy's two ends are
	// allowed to share a register.
	div := vr.temp(w)
	emitCopy(c, div, b, w)

	emitCopy(c, lo, a, w)
	if signed {
		c.Emit(mir.Instr{Op: signExtendOp{w: w}, Defs: []mir.VReg{hi}, Uses: []mir.VReg{lo}})
	} else {
		c.Emit(mir.Instr{Op: zeroOp{}, Defs: []mir.VReg{hi}})
	}

	c.Emit(mir.Instr{
		Op:   divOp{signed: signed, w: w},
		Defs: []mir.VReg{lo, hi},
		Uses: []mir.VReg{lo, hi, div},
	})

	src := lo
	if wantRem {
		src = hi
	}
	emitCopy(c, result, src, w)
	return nil
}

// iselShift lowers §A5's five verbs.
func iselShift(c *cursor, vr *vregs, in *ir.Inst) error {
	op := in.Op()
	a, ok := vr.lookup(in.Arg(0))
	if !ok {
		return fmt.Errorf("%s: operand defined outside the function", op)
	}
	amt, ok := vr.lookup(in.Arg(1))
	if !ok {
		return fmt.Errorf("%s: shift amount defined outside the function", op)
	}
	result, err := vr.define(in.Result(0))
	if err != nil {
		return fmt.Errorf("%s: %w", op, err)
	}
	w := vr.widthOfVReg(result)

	// The count's width is the namespace's — §A5 shifts an i64 by an
	// i64 — but only CL is ever read, so the copy moves whichever half
	// the value occupies and the shift ignores the rest.
	cl := vr.physical(reg.RCX, w)
	emitCopy(c, cl, amt, w)

	c.Emit(mir.Instr{
		Op:   shiftOp{verb: op.Verb, w: w},
		Defs: []mir.VReg{result},
		Uses: []mir.VReg{a, cl},
	})
	return nil
}

// iselUnary lowers neg and not: one register in, the same register out.
func iselUnary(mf *mir.Func, c *cursor, vr *vregs, in *ir.Inst) error {
	op := in.Op()
	a, ok := vr.lookup(in.Arg(0))
	if !ok {
		return fmt.Errorf("%s: operand defined outside the function", op)
	}
	result, err := vr.define(in.Result(0))
	if err != nil {
		return fmt.Errorf("%s: %w", op, err)
	}
	w := vr.widthOfVReg(result)

	if w.isFloat() {
		if op.Verb != ir.VNeg {
			return fmt.Errorf("%s: not a float operation this package emits", op)
		}
		// A negation is the sign bit flipped, which is one mask and one
		// XOR — and not the integer NEG, which is a subtraction from
		// zero and would give the wrong answer for a zero and for a NaN.
		mask := emitSignMask(c, vr, w, false)
		emitCopy(c, result, a, w)
		c.Emit(mir.Instr{
			Op:   fLogicOp{op: fXor, w: w},
			Defs: []mir.VReg{result},
			Uses: []mir.VReg{result, mask},
		})
		return nil
	}

	var mop any = unOp{verb: op.Verb, w: w}
	if op.Type == ir.TypeI1 && op.Verb == ir.VNot {
		mop = i1NotOp{}
	}
	c.Emit(mir.Instr{
		Op:   mop,
		Defs: []mir.VReg{result},
		Uses: []mir.VReg{a},
	})
	return nil
}

// emitSignMask materializes the float sign-bit mask in a vector register.
func emitSignMask(c *cursor, vr *vregs, w width, inverted bool) mir.VReg {
	bits, iw := uint64(1)<<63, w64
	if w == wf32 {
		bits, iw = uint64(1)<<31, w32
	}
	if inverted {
		bits = ^bits
		if iw == w32 {
			bits &= 0xffffffff
		}
	}

	tmp := vr.temp(iw)
	mask := vr.temp(w)
	c.Emit(mir.Instr{Op: constOp{imm: int64(bits), w: iw}, Defs: []mir.VReg{tmp}})
	c.Emit(mir.Instr{Op: bitsToFloatOp{w: w}, Defs: []mir.VReg{mask}, Uses: []mir.VReg{tmp}})
	return mask
}

// iselFloatUnary lowers §A3's abs and sqrt.
func iselFloatUnary(c *cursor, vr *vregs, in *ir.Inst) error {
	op := in.Op()
	a, ok := vr.lookup(in.Arg(0))
	if !ok {
		return fmt.Errorf("%s: operand defined outside the function", op)
	}
	result, err := vr.define(in.Result(0))
	if err != nil {
		return fmt.Errorf("%s: %w", op, err)
	}
	w := vr.widthOfVReg(result)
	if !w.isFloat() {
		return fmt.Errorf("%s: not a float operation", op)
	}

	if op.Verb == ir.VSqrt {
		c.Emit(mir.Instr{Op: fSqrtOp{w: w}, Defs: []mir.VReg{result}, Uses: []mir.VReg{a}})
		return nil
	}

	// abs: the sign bit cleared, which is an AND with everything else.
	mask := emitSignMask(c, vr, w, true)
	emitCopy(c, result, a, w)
	c.Emit(mir.Instr{
		Op:   fLogicOp{op: fAnd, w: w},
		Defs: []mir.VReg{result},
		Uses: []mir.VReg{result, mask},
	})
	return nil
}

// iselCopySign lowers §A3's copysign.
func iselCopySign(c *cursor, vr *vregs, in *ir.Inst) error {
	op := in.Op()
	a, ok := vr.lookup(in.Arg(0))
	if !ok {
		return fmt.Errorf("%s: operand defined outside the function", op)
	}
	b, ok := vr.lookup(in.Arg(1))
	if !ok {
		return fmt.Errorf("%s: operand defined outside the function", op)
	}
	result, err := vr.define(in.Result(0))
	if err != nil {
		return fmt.Errorf("%s: %w", op, err)
	}
	w := vr.widthOfVReg(result)
	if !w.isFloat() {
		return fmt.Errorf("%s: not a float operation", op)
	}

	// The sign of b.
	sign := emitSignMask(c, vr, w, false)
	signOfB := vr.temp(w)
	emitCopy(c, signOfB, sign, w)
	c.Emit(mir.Instr{
		Op:   fLogicOp{op: fAnd, w: w},
		Defs: []mir.VReg{signOfB},
		Uses: []mir.VReg{signOfB, b},
	})

	// The magnitude of a: the same mask inverted, which is what the
	// destination-inverting AND does without a second constant.
	mag := emitSignMask(c, vr, w, false)
	c.Emit(mir.Instr{
		Op:   fLogicOp{op: fAndn, w: w},
		Defs: []mir.VReg{mag},
		Uses: []mir.VReg{mag, a},
	})

	emitCopy(c, result, mag, w)
	c.Emit(mir.Instr{
		Op:   fLogicOp{op: fOr, w: w},
		Defs: []mir.VReg{result},
		Uses: []mir.VReg{result, signOfB},
	})
	return nil
}

// iselLoad lowers a full-width load.
func iselLoad(mf *mir.Func, c *cursor, vr *vregs, in *ir.Inst) error {
	op := in.Op()
	addr, ok := vr.lookup(in.Arg(0))
	if !ok {
		return fmt.Errorf("%s: address defined outside the function", op)
	}
	dst, err := vr.define(in.Result(0))
	if err != nil {
		return fmt.Errorf("%s: %w", op, err)
	}
	w := vr.widthOfVReg(dst)
	c.Emit(mir.Instr{Op: loadOp{w: w, unaligned: statedUnaligned(in, w)},
		Defs: []mir.VReg{dst}, Uses: []mir.VReg{addr}})
	return nil
}

// iselStore is iselLoad's mirror, and the first instruction this package lowers that defines nothing.
func iselStore(c *cursor, vr *vregs, in *ir.Inst) error {
	op := in.Op()
	w, ok := widthOf(op.Type)
	if !ok {
		return fmt.Errorf("%s: only i32, i64, and ptr stores are supported", op)
	}
	val, ok := vr.lookup(in.Arg(0))
	if !ok {
		return fmt.Errorf("%s: value defined outside the function", op)
	}
	addr, ok := vr.lookup(in.Arg(1))
	if !ok {
		return fmt.Errorf("%s: address defined outside the function", op)
	}
	c.Emit(mir.Instr{Op: storeOp{w: w, unaligned: statedUnaligned(in, w)}, Uses: []mir.VReg{val, addr}})
	return nil
}

// iselExtLoad lowers one of §D2's six loads.
func iselExtLoad(mf *mir.Func, c *cursor, vr *vregs, in *ir.Inst, from access, signed bool) error {
	op := in.Op()
	addr, ok := vr.lookup(in.Arg(0))
	if !ok {
		return fmt.Errorf("%s: address defined outside the function", op)
	}
	dst, err := vr.define(in.Result(0))
	if err != nil {
		return fmt.Errorf("%s: %w", op, err)
	}
	w := vr.widthOfVReg(dst)
	if from == a32 && w != w64 {
		// §D2 puts sload32 and uload32 in the i64 namespace only, so this
		// is a hand-built instruction rather than anything the builder
		// can produce — but emit would have to pick an encoding for it,
		// and there is no four-byte extension into a four-byte register.
		return fmt.Errorf("%s: a 32-bit extending load needs a 64-bit result", op)
	}
	c.Emit(mir.Instr{
		Op:   extLoadOp{from: from, signed: signed, w: w},
		Defs: []mir.VReg{dst},
		Uses: []mir.VReg{addr},
	})
	return nil
}

// iselSubStore lowers one of §D2's three stores.
func iselSubStore(c *cursor, vr *vregs, in *ir.Inst, to access) error {
	op := in.Op()
	w, ok := widthOf(op.Type)
	if !ok {
		return fmt.Errorf("%s: only i32 and i64 sub-width stores are supported", op)
	}
	if to == a32 && w != w64 {
		// store32 lives in the i64 namespace, where it means something:
		// from an i32 it would be i32.store under another name.
		return fmt.Errorf("%s: a 32-bit truncating store needs a 64-bit value", op)
	}
	val, ok := vr.lookup(in.Arg(0))
	if !ok {
		return fmt.Errorf("%s: value defined outside the function", op)
	}
	addr, ok := vr.lookup(in.Arg(1))
	if !ok {
		return fmt.Errorf("%s: address defined outside the function", op)
	}
	c.Emit(mir.Instr{Op: subStoreOp{to: to}, Uses: []mir.VReg{val, addr}})
	return nil
}

// iselAlloc lowers ptr.alloc to the address of the slot planFrame gave it.
func iselAlloc(mf *mir.Func, c *cursor, vr *vregs, fr *frame, in *ir.Inst) error {
	off, ok := fr.slot[in]
	if !ok {
		// planFrame walked every block of this function and either gave
		// each ptr.alloc a slot or refused the function outright.
		return fmt.Errorf("%s: no frame slot was planned", in.Op())
	}
	dst, err := vr.define(in.Result(0))
	if err != nil {
		return fmt.Errorf("%s: %w", in.Op(), err)
	}
	c.Emit(mir.Instr{Op: leaOp{off: off}, Defs: []mir.VReg{dst}})

	if !in.Zeroed() {
		return nil
	}

	// §D3's zeroed alloc guarantees the storage reads as zero on entry
	// to the live range, which means emitting the stores that make it
	// so. That is a memset, and this is the milestone that has one.
	//
	// The size comes from the frame, not from the instruction: planFrame
	// has already resolved a stated size or a named type into the bytes
	// it reserved, and asking it again here would be asking the same
	// question twice and risking two answers.
	size, _, err := allocSizeAlign(in)
	if err != nil {
		return fmt.Errorf("%s: %w", in.Op(), err)
	}
	zero := vr.temp(w32)
	c.Emit(mir.Instr{Op: constOp{imm: 0, w: w32}, Defs: []mir.VReg{zero}})
	n := vr.temp(w64)
	c.Emit(mir.Instr{Op: constOp{imm: int64(size), w: w64}, Defs: []mir.VReg{n}})

	return emitLibcall(c, vr, memsetSym, []libArg{
		{v: dst, w: w64},
		{v: zero, w: w32},
		{v: n, w: w64},
	}, 0, false)
}

// iselAlloca lowers §D3's ptr.alloca: storage whose size is a value and
// whose home is therefore the stack itself rather than a slot planned
// for it.
//
// Everything ptr.alloc needed was known before the function ran, which
// is what let planFrame give it an offset from RBP. An alloca knows its
// size when it reaches the instruction, so the only place to put it is
// below RSP — and moving RSP is what makes the frame dynamic and what
// the rest of this has to be careful about.
func iselAlloca(c *cursor, vr *vregs, fr *frame, in *ir.Inst) error {
	op := in.Op()
	align, stated := in.Align()
	if stated && align > maxAlign {
		return fmt.Errorf("%s: wants %d-byte alignment; the stack guarantees %d", op, align, maxAlign)
	}

	n, ok := vr.lookup(in.Arg(0))
	if !ok {
		return fmt.Errorf("%s: size defined outside the function", op)
	}
	dst, err := vr.define(in.Result(0))
	if err != nil {
		return fmt.Errorf("%s: %w", op, err)
	}

	c.Emit(mir.Instr{
		Op:   allocaOp{outArgs: int32(fr.outgoing())},
		Defs: []mir.VReg{dst, vr.temp(w64)},
		Uses: []mir.VReg{n},
	})

	if !in.Zeroed() {
		return nil
	}
	// The same memset a zeroed ptr.alloc gets, over a size that is a
	// value rather than a constant. The requested count and not the
	// rounded one: what §D3 guarantees reads as zero is the allocation,
	// and the bytes the rounding took are padding that is not part of
	// it. That keeps the unrounded size live across the subtraction,
	// which is the price.
	zero := vr.temp(w32)
	c.Emit(mir.Instr{Op: constOp{imm: 0, w: w32}, Defs: []mir.VReg{zero}})
	return emitLibcall(c, vr, memsetSym, []libArg{
		{v: dst, w: w64},
		{v: zero, w: w32},
		{v: n, w: w64},
	}, 0, false)
}

// iselStackSave lowers §D3's ptr.stacksave: the opaque token that names
// where the stack was.
func iselStackSave(c *cursor, vr *vregs, in *ir.Inst) error {
	dst, err := vr.define(in.Result(0))
	if err != nil {
		return fmt.Errorf("%s: %w", in.Op(), err)
	}
	c.Emit(mir.Instr{Op: stackSaveOp{}, Defs: []mir.VReg{dst}})
	return nil
}

// iselStackRestore lowers §D3's ptr.stackrestore, which frees every
// allocation made since the token was taken by putting RSP back.
func iselStackRestore(c *cursor, vr *vregs, in *ir.Inst) error {
	tok, ok := vr.lookup(in.Arg(0))
	if !ok {
		return fmt.Errorf("%s: token defined outside the function", in.Op())
	}
	c.Emit(mir.Instr{Op: stackRestoreOp{}, Uses: []mir.VReg{tok}})
	return nil
}

// iselBlockAddr lowers §D3's ptr.blockaddr, which is what gives brind
// something to jump to: brind reads an address out of a value, and this
// is the only verb that makes one out of a block.
//
// §19.6 has already refused a blockaddr of a block no brind in the
// function targets, so the label is one control can actually reach.
func iselBlockAddr(c *cursor, vr *vregs, in *ir.Inst) error {
	op := in.Op()
	labels := in.Labels()
	if len(labels) != 1 {
		return fmt.Errorf("%s: %d blocks named, want exactly one", op, len(labels))
	}
	dst, err := vr.define(in.Result(0))
	if err != nil {
		return fmt.Errorf("%s: %w", op, err)
	}
	// The block's own function, not one threaded in: the builder panics
	// on a blockaddr naming a block from another function, so this is
	// the function isel is lowering.
	blk := labels[0]
	c.Emit(mir.Instr{Op: leaBlockOp{label: blockLabel(blk.Func(), blk)}, Defs: []mir.VReg{dst}})
	return nil
}

// iselFrameAddr lowers §D3's ptr.frameaddr, which is RBP — the frame
// pointer this package's prologue always establishes.
//
// It is a lea of RBP at offset zero rather than a move out of it,
// because RBP is not a register the allocator hands out and a pinned
// vreg would be claiming otherwise. leaOp already names an address as
// RBP plus a displacement, and zero is a displacement.
func iselFrameAddr(c *cursor, vr *vregs, in *ir.Inst) error {
	dst, err := vr.define(in.Result(0))
	if err != nil {
		return fmt.Errorf("%s: %w", in.Op(), err)
	}
	c.Emit(mir.Instr{Op: leaOp{off: 0}, Defs: []mir.VReg{dst}})
	return nil
}

// iselReturnAddr lowers §D3's ptr.returnaddr.
//
// The call that reached this function pushed the return address, and the
// prologue's push of RBP went on top of it, so it is the eightbyte just
// above the saved RBP. That is a load from RBP plus eight, which is the
// same shape argLoadOp already emits for an incoming stack parameter —
// those start at RBP plus sixteen, one slot further up, for exactly this
// reason.
func iselReturnAddr(c *cursor, vr *vregs, in *ir.Inst) error {
	dst, err := vr.define(in.Result(0))
	if err != nil {
		return fmt.Errorf("%s: %w", in.Op(), err)
	}
	c.Emit(mir.Instr{Op: argLoadOp{off: 8, w: w64}, Defs: []mir.VReg{dst}})
	return nil
}

// iselDiff lowers §D3's ptr.diff: the signed byte distance between two
// pointers.
//
// A 64-bit subtraction and nothing else. §D3 says the result is
// sign-extended from ptrbits, and ptrbits is 64 here — checkLayout is
// where that stopped being an assumption — so there is nothing left to
// extend.
func iselDiff(c *cursor, vr *vregs, in *ir.Inst) error {
	op := in.Op()
	a, ok := vr.lookup(in.Arg(0))
	if !ok {
		return fmt.Errorf("%s: operand 0 defined outside the function", op)
	}
	b, ok := vr.lookup(in.Arg(1))
	if !ok {
		return fmt.Errorf("%s: operand 1 defined outside the function", op)
	}
	dst, err := vr.define(in.Result(0))
	if err != nil {
		return fmt.Errorf("%s: %w", op, err)
	}
	c.Emit(mir.Instr{Op: aluOp{verb: ir.VSub, w: w64}, Defs: []mir.VReg{dst}, Uses: []mir.VReg{a, b}})
	return nil
}

// iselPtrIntCast lowers §C4: ptr.from_i64 and i64.from_ptr.
//
// A copy, in both directions. §C4 says from_i64 truncates and from_ptr
// zero-extends where ptrbits is less than 64; here ptrbits is 64, which
// checkLayout is what stops being an assumption, so both are the
// identity and the only thing that changes is which namespace the value
// is read as. It is emitted as a copy rather than as nothing because
// the two ends are different vregs; coalescing is what makes it free.
func iselPtrIntCast(c *cursor, vr *vregs, in *ir.Inst) error {
	op := in.Op()
	src, ok := vr.lookup(in.Arg(0))
	if !ok {
		return fmt.Errorf("%s: operand defined outside the function", op)
	}
	dst, err := vr.define(in.Result(0))
	if err != nil {
		return fmt.Errorf("%s: %w", op, err)
	}
	emitCopy(c, dst, src, w64)
	return nil
}

// iselGetAddr lowers ptr.getaddr: the address of a global or a function,
// into a register.
func iselGetAddr(mf *mir.Func, c *cursor, vr *vregs, in *ir.Inst) error {
	op := in.Op()
	sym := in.Symbol()
	if sym == nil {
		return fmt.Errorf("%s: no symbol named", op)
	}
	dst, err := vr.define(in.Result(0))
	if err != nil {
		return fmt.Errorf("%s: %w", op, err)
	}
	if isImportSym(sym) && c.fn.Module().Layout().ABI != abiMS {
		// A symbol this module only declares may be defined in a shared
		// library, whose address is not a link-time constant. A RIP-relative
		// LEA would bake one in; the address has to be read from the GOT,
		// which the loader fills. It is what the platform's compilers emit,
		// and where the definition turns out to be in this link the linker
		// relaxes the load back into an LEA.
		//
		// PE has no GOT and COFF has no relocation that would name one, so
		// the Microsoft ABI takes the LEA below. That is what MSVC emits for
		// an ordinary extern, and it is right for everything a static
		// library provides. A symbol that really does live in a DLL needs
		// __declspec(dllimport), which makes the reference an indirection
		// through __imp_ — a different mechanism at the source level, not a
		// decision this can make from the declaration alone.
		c.Emit(mir.Instr{Op: movSymGotOp{sym: sym.Name()}, Defs: []mir.VReg{dst}})
		return nil
	}
	c.Emit(mir.Instr{Op: leaSymOp{sym: sym.Name()}, Defs: []mir.VReg{dst}})
	return nil
}

// isImportSym reports whether a symbol is declared here and defined
// elsewhere.
func isImportSym(s ir.Symbol) bool {
	switch s.(type) {
	case *ir.GlobalImport, *ir.FuncImport:
		return true
	}
	return false
}

// PE's static thread-local model, which is the one Windows uses and the one
// this backend implements.
//
// A thread-local does not have an address; it has an offset into a block
// each thread owns a copy of. Finding the copy is four instructions and two
// symbols the CRT supplies:
//
//	mov rax, gs:[0x58]          the TEB's ThreadLocalStoragePointer
//	mov ecx, [rip + _tls_index] this module's slot in it
//	mov rax, [rax + rcx*8]      this thread's block for this module
//	lea rax, [rax + v@SECREL32] the variable within the block
//
// which is what clang emits, instruction for instruction. MSVC writes the
// last step as a mov of the offset into a register and an add, because its
// assembler will not put a relocation in a displacement field; ours will,
// for an offset kind, so the shorter form is available. Both are the same
// four values in the same order.
//
// The 0x58 is the TEB layout and not a number this could derive. It is fixed
// by the platform, and the same constant appears in every Windows compiler.
const tebTLSPointer = 0x58

// tlsIndexSymbol is the CRT's, one per image: which slot of the thread's
// array of blocks belongs to this module. The CRT defines it and the loader
// fills it, so this is a reference and never a definition.
const tlsIndexSymbol = "_tls_index"

func iselTLSAddr(c *cursor, vr *vregs, in *ir.Inst) error {
	op := in.Op()
	sym := in.Symbol()
	if sym == nil {
		return fmt.Errorf("%s: no symbol named", op)
	}
	if c.fn.Module().Layout().ABI != abiMS {
		// The ELF models and Mach-O's are a different question with
		// different relocations, and globals.Lower has already refused
		// the declaration on those targets — so reaching here means a
		// module declared a thread-local somewhere this cannot address.
		return fmt.Errorf("%s: no thread-local model for ABI %q",
			op, c.fn.Module().Layout().ABI)
	}
	dst, err := vr.define(in.Result(0))
	if err != nil {
		return fmt.Errorf("%s: %w", op, err)
	}
	c.Emit(mir.Instr{
		Op:   tlsAddrOp{sym: sym.Name()},
		Defs: []mir.VReg{dst, vr.temp(w64)},
	})
	return nil
}
