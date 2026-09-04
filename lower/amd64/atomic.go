package amd64

import (
	"fmt"

	"github.com/vertex-language/amd64/reg"

	"github.com/vertex-language/ir"
	"github.com/vertex-language/ir/lower/mir"
)

// atomicRmws is §H's read-modify-write family: which instruction answers
// it, and the access width for the narrow forms.
var atomicRmws = map[ir.Verb]struct {
	kind rmwKind
	a    access
	alu  ir.Verb // the operation the loop applies, for rmwLoop
}{
	ir.VAtomicRmwAdd:  {kind: rmwAdd},
	ir.VAtomicRmwSub:  {kind: rmwSub},
	ir.VAtomicRmwXchg: {kind: rmwXchg},
	ir.VAtomicRmwAnd:  {kind: rmwLoop, alu: ir.VAnd},
	ir.VAtomicRmwOr:   {kind: rmwLoop, alu: ir.VOr},
	ir.VAtomicRmwXor:  {kind: rmwLoop, alu: ir.VXor},

	ir.VAtomicRmwAdd8:  {kind: rmwAdd, a: a8},
	ir.VAtomicRmwSub8:  {kind: rmwSub, a: a8},
	ir.VAtomicRmwXchg8: {kind: rmwXchg, a: a8},
	ir.VAtomicRmwAnd8:  {kind: rmwLoop, a: a8, alu: ir.VAnd},
	ir.VAtomicRmwOr8:   {kind: rmwLoop, a: a8, alu: ir.VOr},
	ir.VAtomicRmwXor8:  {kind: rmwLoop, a: a8, alu: ir.VXor},

	ir.VAtomicRmwAdd16:  {kind: rmwAdd, a: a16},
	ir.VAtomicRmwSub16:  {kind: rmwSub, a: a16},
	ir.VAtomicRmwXchg16: {kind: rmwXchg, a: a16},
	ir.VAtomicRmwAnd16:  {kind: rmwLoop, a: a16, alu: ir.VAnd},
	ir.VAtomicRmwOr16:   {kind: rmwLoop, a: a16, alu: ir.VOr},
	ir.VAtomicRmwXor16:  {kind: rmwLoop, a: a16, alu: ir.VXor},
}

// The other three §H families, by the access they make.
var (
	atomicLoads = map[ir.Verb]access{
		ir.VAtomicLoad:    0,
		ir.VAtomicULoad8:  a8,
		ir.VAtomicULoad16: a16,
	}
	atomicStores = map[ir.Verb]access{
		ir.VAtomicStore:   0,
		ir.VAtomicStore8:  a8,
		ir.VAtomicStore16: a16,
	}
	atomicCases = map[ir.Verb]access{
		ir.VAtomicCas:   0,
		ir.VAtomicCas8:  a8,
		ir.VAtomicCas16: a16,
	}
)

// atomicWidths is one §H instruction's access width and the register
// view its value lives in.
func atomicWidths(op ir.Op, narrow access) (access, width, error) {
	w, ok := widthOf(op.Type)
	if !ok || w.isFloat() {
		return 0, 0, fmt.Errorf("%s: only i32 and i64 atomics are emitted", op)
	}
	if narrow != 0 {
		return narrow, w, nil
	}
	if w == w32 {
		return a32, w32, nil
	}
	return a64, w64, nil
}

// iselAtomicLoad lowers §H's atomic loads, which are ordinary loads
// because x86-64's memory model is strongly ordered.
func iselAtomicLoad(c *cursor, vr *vregs, in *ir.Inst, narrow access) error {
	op := in.Op()
	a, w, err := atomicWidths(op, narrow)
	if err != nil {
		return err
	}
	addr, ok := vr.lookup(in.Arg(0))
	if !ok {
		return fmt.Errorf("%s: address defined outside the function", op)
	}
	dst, err := vr.define(in.Result(0))
	if err != nil {
		return fmt.Errorf("%s: %w", op, err)
	}

	if narrow != 0 {
		// §H's narrow loads are the unsigned ones only, which is what
		// makes this the zero-extending §D2 load and not a family of its
		// own.
		c.Emit(mir.Instr{
			Op:   extLoadOp{from: a, signed: false, w: w},
			Defs: []mir.VReg{dst}, Uses: []mir.VReg{addr},
		})
		return nil
	}
	c.Emit(mir.Instr{Op: loadOp{w: w}, Defs: []mir.VReg{dst}, Uses: []mir.VReg{addr}})
	return nil
}

// iselAtomicStore lowers §H's atomic stores, which are ordinary stores
// unless sequential consistency is requested.
func iselAtomicStore(c *cursor, vr *vregs, in *ir.Inst, narrow access) error {
	op := in.Op()
	a, w, err := atomicWidths(op, narrow)
	if err != nil {
		return err
	}
	val, ok := vr.lookup(in.Arg(0))
	if !ok {
		return fmt.Errorf("%s: value defined outside the function", op)
	}
	addr, ok := vr.lookup(in.Arg(1))
	if !ok {
		return fmt.Errorf("%s: address defined outside the function", op)
	}

	if seqCst(in) {
		// The exchange writes its register as well as memory, so the
		// value goes through a temporary rather than through the vreg
		// the caller still owns.
		dead := vr.temp(w)
		c.Emit(mir.Instr{
			Op:   atomicXchgOp{a: a, w: w},
			Defs: []mir.VReg{dead}, Uses: []mir.VReg{val, addr},
		})
		return nil
	}

	if narrow != 0 {
		c.Emit(mir.Instr{Op: subStoreOp{to: a}, Uses: []mir.VReg{val, addr}})
		return nil
	}
	c.Emit(mir.Instr{Op: storeOp{w: w}, Uses: []mir.VReg{val, addr}})
	return nil
}

// iselAtomicRmw lowers §H's read-modify-writes, which return the value
// that was there before.
func iselAtomicRmw(c *cursor, vr *vregs, in *ir.Inst, kind rmwKind, narrow access, alu ir.Verb) error {
	op := in.Op()
	a, w, err := atomicWidths(op, narrow)
	if err != nil {
		return err
	}
	val, ok := vr.lookup(in.Arg(0))
	if !ok {
		return fmt.Errorf("%s: value defined outside the function", op)
	}
	addr, ok := vr.lookup(in.Arg(1))
	if !ok {
		return fmt.Errorf("%s: address defined outside the function", op)
	}

	if kind == rmwLoop {
		return iselAtomicLoop(c, vr, in, a, w, alu, val, addr)
	}

	dst, err := vr.define(in.Result(0))
	if err != nil {
		return fmt.Errorf("%s: %w", op, err)
	}

	if kind == rmwXchg {
		c.Emit(mir.Instr{
			Op:   atomicXchgOp{a: a, w: w},
			Defs: []mir.VReg{dst}, Uses: []mir.VReg{val, addr},
		})
		return nil
	}

	operand := val
	if kind == rmwSub {
		// Negated in a register of its own, because the caller's value
		// is still theirs. At the narrow widths the negation runs at 32
		// bits and only the low byte or halfword reaches memory, which
		// is the same answer.
		neg := vr.temp(w)
		emitCopy(c, neg, val, w)
		c.Emit(mir.Instr{Op: unOp{verb: ir.VNeg, w: w}, Defs: []mir.VReg{neg}, Uses: []mir.VReg{neg}})
		operand = neg
	}

	c.Emit(mir.Instr{
		Op:   atomicRmwOp{a: a, w: w},
		Defs: []mir.VReg{dst}, Uses: []mir.VReg{operand, addr},
	})
	return nil
}

// iselAtomicLoop is the compare-and-swap loop that answers an atomic
// and, or or xor.
func iselAtomicLoop(c *cursor, vr *vregs, in *ir.Inst, a access, w width, alu ir.Verb, val, addr mir.VReg) error {
	op := in.Op()

	acc := vr.physical(reg.RAX, w)
	if a == a32 || a == a64 {
		c.Emit(mir.Instr{Op: loadOp{w: w}, Defs: []mir.VReg{acc}, Uses: []mir.VReg{addr}})
	} else {
		c.Emit(mir.Instr{
			Op:   extLoadOp{from: a, signed: false, w: w},
			Defs: []mir.VReg{acc}, Uses: []mir.VReg{addr},
		})
	}

	retry, cont := c.open("retry"), c.open("cont")
	c.to(retry)

	next := vr.temp(w)
	emitCopy(c, next, acc, w)
	c.Emit(mir.Instr{Op: aluOp{verb: alu, w: w}, Defs: []mir.VReg{next}, Uses: []mir.VReg{next, val}})
	c.Emit(mir.Instr{
		Op:   atomicCasOp{a: a, w: w},
		Defs: []mir.VReg{acc}, Uses: []mir.VReg{acc, next, addr},
	})
	c.branch(condNE, retry, cont)

	dst, err := vr.define(in.Result(0))
	if err != nil {
		return fmt.Errorf("%s: %w", op, err)
	}
	emitCopy(c, dst, acc, w)
	return nil
}

// iselAtomicCas lowers §H's compare-and-swap, which natively
// compares against the accumulator and leaves what it read there.
func iselAtomicCas(c *cursor, vr *vregs, in *ir.Inst, narrow access) error {
	op := in.Op()
	a, w, err := atomicWidths(op, narrow)
	if err != nil {
		return err
	}
	expect, ok := vr.lookup(in.Arg(0))
	if !ok {
		return fmt.Errorf("%s: expected value defined outside the function", op)
	}
	newVal, ok := vr.lookup(in.Arg(1))
	if !ok {
		return fmt.Errorf("%s: new value defined outside the function", op)
	}
	addr, ok := vr.lookup(in.Arg(2))
	if !ok {
		return fmt.Errorf("%s: address defined outside the function", op)
	}

	rax := vr.physical(reg.RAX, w)
	emitCopy(c, rax, expect, w)
	c.Emit(mir.Instr{
		Op:   atomicCasOp{a: a, w: w},
		Defs: []mir.VReg{rax}, Uses: []mir.VReg{rax, newVal, addr},
	})

	dst, err := vr.define(in.Result(0))
	if err != nil {
		return fmt.Errorf("%s: %w", op, err)
	}
	emitCopy(c, dst, rax, w)
	return nil
}

// iselFence lowers §I's fence, which only emits an MFENCE for
// sequentially consistent ordering.
func iselFence(c *cursor, in *ir.Inst) error {
	if in.SingleThread() {
		return nil
	}
	if seqCst(in) {
		c.Emit(mir.Instr{Op: mfenceOp{}})
	}
	return nil
}

// seqCst reports whether an instruction's first ordering is sequential consistency.
func seqCst(in *ir.Inst) bool {
	ords := in.Orderings()
	return len(ords) > 0 && ords[0] == ir.SeqCst
}
