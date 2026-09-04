package i386

import (
	"fmt"

	"github.com/vertex-language/ir"
	"github.com/vertex-language/ir/lower/globals"
	"github.com/vertex-language/ir/lower/mir"
)

// pairwise is every §A verb whose 64-bit form is the same instruction applied
// to each half independently: the low halves do not tell the high halves
// anything.
var pairwise = map[ir.Verb]bool{
	ir.VAnd: true, ir.VOr: true, ir.VXor: true,
}

// carried is every §A verb whose 64-bit form is one instruction and then its
// carry-reading sibling.
var carried = map[ir.Verb]bool{ir.VAdd: true, ir.VSub: true}

// compareConds maps §B's verbs onto the condition each answers.
var compareConds = map[ir.Verb]condCode{
	ir.VEq:  condE,
	ir.VNe:  condNE,
	ir.VSLt: condL,
	ir.VSLe: condLE,
	ir.VULt: condB,
	ir.VULe: condBE,
}

// iselInst lowers one non-terminator instruction.
func iselInst(c *cursor, vr *vregs, fr *frame, in *ir.Inst, opts Options) error {
	op := in.Op()
	verb := op.Verb

	if handled, err := iselFloatInst(c, vr, fr, in, opts); handled {
		return err
	}

	switch {
	case pairwise[verb], carried[verb]:
		return iselBinary(c, vr, in)
	case verb == ir.VNot, verb == ir.VNeg:
		return iselUnary(c, vr, in)
	}
	if cond, ok := compareConds[verb]; ok {
		return iselCompare(c, vr, in, cond)
	}

	if shifts[verb] {
		return iselShift(c, vr, in)
	}

	if _, ok := libcalls[verb]; ok {
		return iselBulk(c, vr, fr, in)
	}
	if narrow, ok := atomicLoads[verb]; ok {
		return iselAtomicLoad(c, vr, in, narrow)
	}
	if narrow, ok := atomicStores[verb]; ok {
		return iselAtomicStore(c, vr, in, narrow)
	}
	if narrow, ok := atomicCases[verb]; ok {
		return iselAtomicCas(c, vr, in, narrow)
	}
	if rmw, ok := atomicRmwAlus[verb]; ok {
		return iselAtomicRmw(c, vr, in, rmw.a, rmw.alu)
	}
	if verb == ir.VFence {
		return iselFence(c, in)
	}

	switch verb {
	case ir.VSDiv:
		return iselDivide(c, vr, fr, in, true, false)
	case ir.VUDiv:
		return iselDivide(c, vr, fr, in, false, false)
	case ir.VSRem:
		return iselDivide(c, vr, fr, in, true, true)
	case ir.VURem:
		return iselDivide(c, vr, fr, in, false, true)
	case ir.VSMulHi:
		return iselMulHi(c, vr, in, true)
	case ir.VUMulHi:
		return iselMulHi(c, vr, in, false)
	case ir.VSAddO, ir.VUAddO, ir.VSSubO, ir.VSMulO, ir.VUMulO:
		return iselOverflow(c, vr, in, verb)
	case ir.VClz, ir.VCtz, ir.VPopcnt:
		return iselBitCount(c, vr, in)
	case ir.VBswap:
		return iselBswap(c, vr, in)
	case ir.VFromI64, ir.VFromPtr:
		return iselPtrInt(c, vr, in)
	case ir.VDiff:
		return iselDiff(c, vr, in)
	case ir.VMul:
		return iselMul(c, vr, in)
	case ir.VConst:
		return iselConst(c, vr, fr, in)
	case ir.VSelect:
		return iselSelect(c, vr, in)
	case ir.VLoad:
		return iselLoad(c, vr, in)
	case ir.VStore:
		return iselStore(c, vr, in)
	case ir.VAlloc:
		return iselAlloc(c, vr, fr, in)
	case ir.VGetAddr:
		return iselGetAddr(c, vr, in)
	case ir.VAlloca:
		return iselAlloca(c, vr, fr, in)
	case ir.VStackSave:
		return iselStackSave(c, vr, in)
	case ir.VStackRestore:
		return iselStackRestore(c, vr, in)
	case ir.VFrameAddr:
		return iselFrameAddr(c, vr, in, false)
	case ir.VReturnAddr:
		return iselFrameAddr(c, vr, in, true)
	case ir.VBlockAddr:
		return iselBlockAddr(c, vr, in)
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
	case ir.VCallInd:
		return iselCallInd(c, vr, fr, in)
	case ir.VAsm:
		return iselAsm(c.fn, c, vr, in, vr.nextAsmID(), nil)
	case ir.VCall:
		return iselCall(c, vr, fr, in)
	case ir.VWrapI64:
		return iselWrap(c, vr, in)
	case ir.VSExtI32:
		return iselExtendTo64(c, vr, in, true)
	case ir.VZExtI32:
		return iselExtendTo64(c, vr, in, false)
	case ir.VZExtI1:
		return iselZExtI1(c, vr, in)
	case ir.VSLoad8:
		return iselExtLoad(c, vr, in, a8, true)
	case ir.VSLoad16:
		return iselExtLoad(c, vr, in, a16, true)
	case ir.VSLoad32:
		return iselExtLoad(c, vr, in, a32, true)
	case ir.VULoad8:
		return iselExtLoad(c, vr, in, a8, false)
	case ir.VULoad16:
		return iselExtLoad(c, vr, in, a16, false)
	case ir.VULoad32:
		return iselExtLoad(c, vr, in, a32, false)
	case ir.VStore8:
		return iselSubStore(c, vr, in, a8)
	case ir.VStore16:
		return iselSubStore(c, vr, in, a16)
	case ir.VStore32:
		return iselSubStore(c, vr, in, a32)
	}
	return fmt.Errorf("unsupported instruction %s", op)
}

// operands is the values a two-operand instruction reads.
func operands(vr *vregs, in *ir.Inst, n int) ([]value, error) {
	out := make([]value, n)
	for i := 0; i < n; i++ {
		v, ok := vr.lookup(in.Arg(i))
		if !ok {
			return nil, fmt.Errorf("%s: operand %d defined outside the function", in.Op(), i)
		}
		out[i] = v
	}
	return out, nil
}

// two puts a's value into dst, so that a two-address instruction can write
// dst without destroying a.
//
// Every x86 arithmetic instruction reads and writes the same operand. The
// copy is marked so the allocator can coalesce it away when a is dead after
// this, which is the common case.
func two(c *cursor, dst, a mir.VReg) { emitCopy(c, dst, a) }

func iselBinary(c *cursor, vr *vregs, in *ir.Inst) error {
	verb := in.Op().Verb
	ops, err := operands(vr, in, 2)
	if err != nil {
		return err
	}
	dst, err := vr.define(in.Result(0))
	if err != nil {
		return fmt.Errorf("%s: %w", in.Op(), err)
	}

	two(c, dst.lo, ops[0].lo)
	c.Emit(mir.Instr{
		Op:   aluOp{verb: verb},
		Defs: []mir.VReg{dst.lo},
		Uses: []mir.VReg{dst.lo, ops[1].lo},
	})
	if !dst.w.pairs() {
		return nil
	}

	two(c, dst.hi, ops[0].hi)
	if pairwise[verb] {
		c.Emit(mir.Instr{
			Op:   aluOp{verb: verb},
			Defs: []mir.VReg{dst.hi},
			Uses: []mir.VReg{dst.hi, ops[1].hi},
		})
		return nil
	}
	// The carry the low half left, which is the whole reason a 64-bit add
	// is two instructions and not two independent ones. Nothing may come
	// between them that writes the flags, and nothing does: the allocator
	// inserts only moves, and an x86 move leaves the flags alone.
	c.Emit(mir.Instr{
		Op:   carryOp{sub: verb == ir.VSub},
		Defs: []mir.VReg{dst.hi},
		Uses: []mir.VReg{dst.hi, ops[1].hi},
	})
	return nil
}

// iselUnary lowers §A4's not and §A's neg.
//
// not is per-half and says nothing between them. neg is not: negating a
// 64-bit value is negating the low half, then subtracting the high half and
// whatever the low half borrowed from zero.
func iselUnary(c *cursor, vr *vregs, in *ir.Inst) error {
	verb := in.Op().Verb
	ops, err := operands(vr, in, 1)
	if err != nil {
		return err
	}
	dst, err := vr.define(in.Result(0))
	if err != nil {
		return fmt.Errorf("%s: %w", in.Op(), err)
	}

	if verb == ir.VNot || !dst.w.pairs() {
		two(c, dst.lo, ops[0].lo)
		c.Emit(mir.Instr{Op: unOp{verb: verb}, Defs: []mir.VReg{dst.lo}, Uses: []mir.VReg{dst.lo}})
		if dst.w.pairs() {
			two(c, dst.hi, ops[0].hi)
			c.Emit(mir.Instr{Op: unOp{verb: verb}, Defs: []mir.VReg{dst.hi}, Uses: []mir.VReg{dst.hi}})
		}
		return nil
	}

	// 0 − x at sixty-four bits: NEG the low half, which sets the borrow,
	// then subtract the high half and that borrow from zero.
	two(c, dst.lo, ops[0].lo)
	c.Emit(mir.Instr{Op: unOp{verb: ir.VNeg}, Defs: []mir.VReg{dst.lo}, Uses: []mir.VReg{dst.lo}})
	c.Emit(mir.Instr{Op: constOp{imm: 0}, Defs: []mir.VReg{dst.hi}})
	c.Emit(mir.Instr{
		Op:   carryOp{sub: true},
		Defs: []mir.VReg{dst.hi},
		Uses: []mir.VReg{dst.hi, ops[0].hi},
	})
	return nil
}

// iselMul lowers §A's multiply.
//
// At thirty-two bits it is IMUL, whose low half is the same bits signed or
// unsigned. At sixty-four it is three multiplies and two adds, which is the
// schoolbook expansion of (ah·2^32 + al)(bh·2^32 + bl) with the term that
// overflows the width dropped:
//
//	lo = al*bl                     (the full 64-bit product)
//	hi = al*bh + ah*bl + high(al*bl)
//
// MUL's 64-bit product lands in EDX:EAX, which is why the low half of this
// runs through those two registers rather than wherever the allocator would
// have put it.
func iselMul(c *cursor, vr *vregs, in *ir.Inst) error {
	ops, err := operands(vr, in, 2)
	if err != nil {
		return err
	}
	dst, err := vr.define(in.Result(0))
	if err != nil {
		return fmt.Errorf("%s: %w", in.Op(), err)
	}

	if !dst.w.pairs() {
		two(c, dst.lo, ops[0].lo)
		c.Emit(mir.Instr{
			Op:   aluOp{verb: ir.VMul},
			Defs: []mir.VReg{dst.lo},
			Uses: []mir.VReg{dst.lo, ops[1].lo},
		})
		return nil
	}
	return iselMul64(c, vr, dst, ops[0], ops[1])
}

// iselBulk lowers §E's verbs into the library calls they are.
func iselBulk(c *cursor, vr *vregs, fr *frame, in *ir.Inst) error {
	op := in.Op()
	if in.Volatile() {
		// Volatile is a promise about how the bytes are touched, and
		// memcpy makes no such promise: the call is the wrong lowering
		// rather than a slow one.
		return fmt.Errorf("%s: a volatile bulk operation cannot be a library call, and no open-coded form is emitted yet", op)
	}
	args := make([]value, len(in.Args()))
	for i, a := range in.Args() {
		v, ok := vr.lookup(a)
		if !ok {
			return fmt.Errorf("%s: operand %d defined outside the function", op, i)
		}
		args[i] = v
	}
	var result value
	want := len(in.Results()) == 1
	if want {
		r, err := vr.define(in.Result(0))
		if err != nil {
			return fmt.Errorf("%s: %w", op, err)
		}
		result = r
	}
	return emitLibcall(c, vr, fr, libcalls[op.Verb], args, result, want)
}

// iselCompare lowers §B.
//
// At thirty-two bits it is a compare and a SETcc. At sixty-four the two halves
// have to be reduced to one flag, and how depends on the verb: equality is
// the two differences OR'd together, and an ordering is a compare of the high
// halves with the low halves deciding only when the high halves match.
func iselCompare(c *cursor, vr *vregs, in *ir.Inst, cond condCode) error {
	ops, err := operands(vr, in, 2)
	if err != nil {
		return err
	}
	dst, err := vr.define(in.Result(0))
	if err != nil {
		return fmt.Errorf("%s: %w", in.Op(), err)
	}

	if !ops[0].w.pairs() {
		c.Emit(mir.Instr{Op: cmpOp{}, Uses: []mir.VReg{ops[0].lo, ops[1].lo}})
		c.Emit(mir.Instr{Op: setccOp{cond: cond}, Defs: []mir.VReg{dst.lo}})
		return nil
	}

	switch cond {
	case condE, condNE:
		// XOR each half, OR the results: zero exactly when both halves
		// matched. One flag out of two comparisons, with no branch.
		lo := vr.reg32()
		hi := vr.reg32()
		two(c, lo, ops[0].lo)
		c.Emit(mir.Instr{Op: aluOp{verb: ir.VXor}, Defs: []mir.VReg{lo}, Uses: []mir.VReg{lo, ops[1].lo}})
		two(c, hi, ops[0].hi)
		c.Emit(mir.Instr{Op: aluOp{verb: ir.VXor}, Defs: []mir.VReg{hi}, Uses: []mir.VReg{hi, ops[1].hi}})
		c.Emit(mir.Instr{Op: aluOp{verb: ir.VOr}, Defs: []mir.VReg{lo}, Uses: []mir.VReg{lo, hi}})
		c.Emit(mir.Instr{Op: setccOp{cond: cond}, Defs: []mir.VReg{dst.lo}})
		return nil
	}

	// An ordering. Subtracting the whole 64-bit value and reading the
	// flags of the high half's SBB answers it: the low halves contribute
	// exactly the borrow the high half needs, which is the same reason the
	// 64-bit subtract is written the way it is.
	//
	// But only the strict orderings, and this is the trap. After the SBB,
	// ZF says the *high* half came out zero and nothing about the low one
	// — so SETLE would call −1 ≤ −2 true, since their difference is 1 with
	// matching high halves. The non-strict comparisons are written as the
	// negation of the opposite strict one instead: a ≤ b is not (b < a),
	// which is the same subtract the other way round read with a condition
	// that does not consult ZF at all.
	a, b := ops[0], ops[1]
	switch cond {
	case condLE:
		a, b, cond = b, a, condGE
	case condBE:
		a, b, cond = b, a, condAE
	}

	lo := vr.reg32()
	hi := vr.reg32()
	two(c, lo, a.lo)
	c.Emit(mir.Instr{Op: aluOp{verb: ir.VSub}, Defs: []mir.VReg{lo}, Uses: []mir.VReg{lo, b.lo}})
	two(c, hi, a.hi)
	c.Emit(mir.Instr{Op: carryOp{sub: true}, Defs: []mir.VReg{hi}, Uses: []mir.VReg{hi, b.hi}})
	c.Emit(mir.Instr{Op: setccOp{cond: cond}, Defs: []mir.VReg{dst.lo}})
	return nil
}

func iselConst(c *cursor, vr *vregs, fr *frame, in *ir.Inst) error {
	lit, ok := in.Lit()
	if !ok {
		return fmt.Errorf("%s: only a plain literal is supported", in.Op())
	}
	dst, err := vr.define(in.Result(0))
	if err != nil {
		return fmt.Errorf("%s: %w", in.Op(), err)
	}
	if dst.w.isFloat() {
		if lit.Kind() != ir.ConstFloat {
			return fmt.Errorf("%s: only a plain float literal is supported", in.Op())
		}
		iselFloatConst(c, vr, fr, dst, lit.Float())
		return nil
	}
	// An integer literal, or one of §2's three symbolic constants, which
	// are integers this target has to be asked for.
	v, err := globals.ConstInt(layout{}, lit)
	if err != nil {
		return fmt.Errorf("%s: %w", in.Op(), err)
	}
	c.Emit(mir.Instr{Op: constOp{imm: int64(int32(v))}, Defs: []mir.VReg{dst.lo}})
	if dst.w.pairs() {
		c.Emit(mir.Instr{Op: constOp{imm: int64(int32(uint64(v) >> 32))}, Defs: []mir.VReg{dst.hi}})
	}
	return nil
}

// iselSelect lowers §F with CMOV, per half.
func iselSelect(c *cursor, vr *vregs, in *ir.Inst) error {
	ops, err := operands(vr, in, 3)
	if err != nil {
		return err
	}
	dst, err := vr.define(in.Result(0))
	if err != nil {
		return fmt.Errorf("%s: %w", in.Op(), err)
	}

	// The condition against itself, then the choice. The flags survive
	// both CMOVs, which is what lets one test serve two halves.
	if dst.w.isFloat() {
		// No conditional move in the vector file, so §F on floats is a
		// branch. CMOV is integer-only and there is no BLENDV before
		// SSE4.1.
		then := c.open("then")
		els := c.open("else")
		done := c.open("done")
		c.Emit(mir.Instr{Op: testOp{}, Uses: []mir.VReg{ops[0].lo}})
		c.branch(condNE, then, els)
		els.Emit(mir.Instr{Op: fmovOp{}, Defs: []mir.VReg{dst.lo}, Uses: []mir.VReg{ops[2].lo}})
		els.Emit(mir.Instr{Op: jmpOp{target: done.Label}})
		c.mf.Succ(els, done.Label)
		then.Emit(mir.Instr{Op: fmovOp{}, Defs: []mir.VReg{dst.lo}, Uses: []mir.VReg{ops[1].lo}})
		then.Emit(mir.Instr{Op: jmpOp{target: done.Label}})
		c.mf.Succ(then, done.Label)
		c.resume(done)
		return nil
	}
	c.Emit(mir.Instr{Op: testOp{}, Uses: []mir.VReg{ops[0].lo}})
	two(c, dst.lo, ops[2].lo)
	c.Emit(mir.Instr{
		Op:   cmovOp{cond: condNE},
		Defs: []mir.VReg{dst.lo},
		Uses: []mir.VReg{dst.lo, ops[1].lo},
	})
	if dst.w.pairs() {
		two(c, dst.hi, ops[2].hi)
		c.Emit(mir.Instr{
			Op:   cmovOp{cond: condNE},
			Defs: []mir.VReg{dst.hi},
			Uses: []mir.VReg{dst.hi, ops[1].hi},
		})
	}
	return nil
}

func iselLoad(c *cursor, vr *vregs, in *ir.Inst) error {
	ops, err := operands(vr, in, 1)
	if err != nil {
		return err
	}
	dst, err := vr.define(in.Result(0))
	if err != nil {
		return fmt.Errorf("%s: %w", in.Op(), err)
	}
	if dst.w.isFloat() {
		c.Emit(mir.Instr{Op: floadOp{w: dst.w}, Defs: []mir.VReg{dst.lo}, Uses: []mir.VReg{ops[0].lo}})
		return nil
	}
	c.Emit(mir.Instr{Op: loadOp{}, Defs: []mir.VReg{dst.lo}, Uses: []mir.VReg{ops[0].lo}})
	if dst.w.pairs() {
		// The high half is four bytes on, which is what makes a 64-bit
		// load two loads and not one wide one.
		c.Emit(mir.Instr{Op: loadOp{off: 4}, Defs: []mir.VReg{dst.hi}, Uses: []mir.VReg{ops[0].lo}})
	}
	return nil
}

func iselStore(c *cursor, vr *vregs, in *ir.Inst) error {
	ops, err := operands(vr, in, 2)
	if err != nil {
		return err
	}
	if ops[0].w.isFloat() {
		c.Emit(mir.Instr{Op: fstoreOp{w: ops[0].w}, Uses: []mir.VReg{ops[0].lo, ops[1].lo}})
		return nil
	}
	c.Emit(mir.Instr{Op: storeOp{}, Uses: []mir.VReg{ops[0].lo, ops[1].lo}})
	if ops[0].w.pairs() {
		c.Emit(mir.Instr{Op: storeOp{off: 4}, Uses: []mir.VReg{ops[0].hi, ops[1].lo}})
	}
	return nil
}

func iselExtLoad(c *cursor, vr *vregs, in *ir.Inst, from access, signed bool) error {
	ops, err := operands(vr, in, 1)
	if err != nil {
		return err
	}
	dst, err := vr.define(in.Result(0))
	if err != nil {
		return fmt.Errorf("%s: %w", in.Op(), err)
	}
	if from == a32 {
		c.Emit(mir.Instr{Op: loadOp{}, Defs: []mir.VReg{dst.lo}, Uses: []mir.VReg{ops[0].lo}})
	} else {
		c.Emit(mir.Instr{
			Op:   extLoadOp{from: from, signed: signed},
			Defs: []mir.VReg{dst.lo}, Uses: []mir.VReg{ops[0].lo},
		})
	}
	if dst.w.pairs() {
		return fillHigh(c, dst, signed)
	}
	return nil
}

func iselSubStore(c *cursor, vr *vregs, in *ir.Inst, to access) error {
	ops, err := operands(vr, in, 2)
	if err != nil {
		return err
	}
	c.Emit(mir.Instr{Op: subStoreOp{to: to}, Uses: []mir.VReg{ops[0].lo, ops[1].lo}})
	return nil
}

// iselWrap lowers §C's wrap_i64, which is the low half and nothing else: the
// high half simply stops being part of the value.
func iselWrap(c *cursor, vr *vregs, in *ir.Inst) error {
	ops, err := operands(vr, in, 1)
	if err != nil {
		return err
	}
	dst, err := vr.define(in.Result(0))
	if err != nil {
		return fmt.Errorf("%s: %w", in.Op(), err)
	}
	emitCopy(c, dst.lo, ops[0].lo)
	return nil
}

// iselExtendTo64 lowers §C's widening into a pair.
func iselExtendTo64(c *cursor, vr *vregs, in *ir.Inst, signed bool) error {
	ops, err := operands(vr, in, 1)
	if err != nil {
		return err
	}
	dst, err := vr.define(in.Result(0))
	if err != nil {
		return fmt.Errorf("%s: %w", in.Op(), err)
	}
	emitCopy(c, dst.lo, ops[0].lo)
	return fillHigh(c, dst, signed)
}

// fillHigh writes the high half of a widened value: zero, or every bit of the
// low half's sign.
func fillHigh(c *cursor, dst value, signed bool) error {
	if !signed {
		c.Emit(mir.Instr{Op: constOp{imm: 0}, Defs: []mir.VReg{dst.hi}})
		return nil
	}
	c.Emit(mir.Instr{Op: signFillOp{}, Defs: []mir.VReg{dst.hi}, Uses: []mir.VReg{dst.lo}})
	return nil
}

// iselZExtI1 lowers §C's zext_i1. An i1 is already a register holding 0 or 1,
// so the widening is a move and, at sixty-four bits, a zero.
func iselZExtI1(c *cursor, vr *vregs, in *ir.Inst) error {
	ops, err := operands(vr, in, 1)
	if err != nil {
		return err
	}
	dst, err := vr.define(in.Result(0))
	if err != nil {
		return fmt.Errorf("%s: %w", in.Op(), err)
	}
	emitCopy(c, dst.lo, ops[0].lo)
	if dst.w.pairs() {
		c.Emit(mir.Instr{Op: constOp{imm: 0}, Defs: []mir.VReg{dst.hi}})
	}
	return nil
}

func iselAlloc(c *cursor, vr *vregs, fr *frame, in *ir.Inst) error {
	off, ok := fr.slot[in]
	if !ok {
		return fmt.Errorf("%s: no frame slot was planned", in.Op())
	}
	dst, err := vr.define(in.Result(0))
	if err != nil {
		return fmt.Errorf("%s: %w", in.Op(), err)
	}
	c.Emit(mir.Instr{Op: frameOp{off: off}, Defs: []mir.VReg{dst.lo}})

	if !in.Zeroed() {
		return nil
	}
	// §D3's zeroed alloc guarantees the storage reads as zero on entry to
	// the live range, which means emitting what makes it so. That is a
	// memset, and §E is what has one.
	//
	// The size is asked of the instruction rather than of the frame:
	// planFrame rounded the slot up to the next allocation's alignment,
	// and the bytes that rounding took are padding rather than part of
	// what §D3 promised.
	size, _, err := allocSizeAlign(in)
	if err != nil {
		return fmt.Errorf("%s: %w", in.Op(), err)
	}
	return emitMemsetZero(c, vr, fr, dst, constant32(c, vr, int64(size)))
}

// constant32 materializes a literal into a fresh register.
func constant32(c *cursor, vr *vregs, imm int64) value {
	v := value{lo: vr.reg32(), w: w32}
	c.Emit(mir.Instr{Op: constOp{imm: imm}, Defs: []mir.VReg{v.lo}})
	return v
}

// emitMemsetZero writes n zero bytes at addr, which is what both of §D3's
// zeroed allocations end with. size_t is thirty-two bits here, so the count
// is one slot like everything else.
func emitMemsetZero(c *cursor, vr *vregs, fr *frame, addr, n value) error {
	zero := constant32(c, vr, 0)
	return emitLibcall(c, vr, fr, libcalls[ir.VMemSet],
		[]value{addr, zero, n}, value{}, false)
}

func iselGetAddr(c *cursor, vr *vregs, in *ir.Inst) error {
	sym := in.Symbol()
	if sym == nil {
		return fmt.Errorf("%s: no symbol named", in.Op())
	}
	dst, err := vr.define(in.Result(0))
	if err != nil {
		return fmt.Errorf("%s: %w", in.Op(), err)
	}
	c.Emit(mir.Instr{Op: symAddrOp{sym: sym.Name()}, Defs: []mir.VReg{dst.lo}})
	return nil
}
