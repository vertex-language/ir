package arm64

import (
	"fmt"

	"github.com/vertex-language/ir"
	"github.com/vertex-language/ir/lower/mir"
)

// §H, on the Armv8-A baseline.
//
// Every read-modify-write here is a retry loop around LDXR and STXR: the load
// takes an exclusive reservation on the address, the store succeeds only if
// nothing else touched it in between, and a non-zero status means it did and
// the loop runs again.
//
// FEAT_LSE, mandatory from Armv8.1-A, has one instruction for each of these
// loops — LDADD, SWP, CAS. They are not used. Options defaults to the Armv8-A
// baseline, which does not have them at all, and this package has one opt
// level; emitting the loop always is correct on every AArch64 processor, where
// emitting LSE would be correct on some. That is the obvious thing to add when
// there is a reason to have two lowerings of the same verb.
//
// The alignment §H requires is the hardware's business here rather than this
// package's: LDXR and LDAR fault on a misaligned address, so the trap the
// spec asks for is raised without an instruction to check for it.

// atomicRmwAlus is §H's read-modify-write family: the operation the loop
// applies and the width it accesses. VNone means the loop stores its operand
// and does not compute anything, which is xchg.
var atomicRmwAlus = map[ir.Verb]struct {
	alu ir.Verb
	a   access
}{
	ir.VAtomicRmwAdd:  {alu: ir.VAdd},
	ir.VAtomicRmwSub:  {alu: ir.VSub},
	ir.VAtomicRmwAnd:  {alu: ir.VAnd},
	ir.VAtomicRmwOr:   {alu: ir.VOr},
	ir.VAtomicRmwXor:  {alu: ir.VXor},
	ir.VAtomicRmwXchg: {},

	ir.VAtomicRmwAdd8:  {alu: ir.VAdd, a: a8},
	ir.VAtomicRmwSub8:  {alu: ir.VSub, a: a8},
	ir.VAtomicRmwAnd8:  {alu: ir.VAnd, a: a8},
	ir.VAtomicRmwOr8:   {alu: ir.VOr, a: a8},
	ir.VAtomicRmwXor8:  {alu: ir.VXor, a: a8},
	ir.VAtomicRmwXchg8: {a: a8},

	ir.VAtomicRmwAdd16:  {alu: ir.VAdd, a: a16},
	ir.VAtomicRmwSub16:  {alu: ir.VSub, a: a16},
	ir.VAtomicRmwAnd16:  {alu: ir.VAnd, a: a16},
	ir.VAtomicRmwOr16:   {alu: ir.VOr, a: a16},
	ir.VAtomicRmwXor16:  {alu: ir.VXor, a: a16},
	ir.VAtomicRmwXchg16: {a: a16},
}

// The other three families, by the access each makes.
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

// atomicWidths is one §H instruction's memory access width and the register
// view its value lives in.
func atomicWidths(op ir.Op, narrow access) (access, width, error) {
	w, ok := widthOf(op.Type)
	if !ok || w.isFloat() {
		return 0, 0, fmt.Errorf("%s: only i32, i64 and ptr atomics are emitted", op)
	}
	if narrow != 0 {
		return narrow, w, nil
	}
	if w == w32 {
		return a32, w32, nil
	}
	return a64, w64, nil
}

// acquires and releases are which half of an ordering one access owes.
//
// An acquire load is LDAR or LDAXR and a release store is STLR or STLXR;
// seq_cst is both, which is the standard Armv8 mapping and needs no barrier
// alongside it.
func acquires(o ir.Ordering) bool {
	return o == ir.Acquire || o == ir.AcqRel || o == ir.SeqCst
}

func releases(o ir.Ordering) bool {
	return o == ir.Release || o == ir.AcqRel || o == ir.SeqCst
}

// ordering is an atomic's first ordering.
func ordering(in *ir.Inst) (ir.Ordering, error) {
	ords := in.Orderings()
	if len(ords) == 0 {
		return 0, fmt.Errorf("%s: no ordering", in.Op())
	}
	return ords[0], nil
}

// iselAtomicLoad lowers §H's loads: a plain load where the ordering does not
// ask for anything, and LDAR where it does.
func iselAtomicLoad(c *cursor, vr *vregs, in *ir.Inst, narrow access) error {
	op := in.Op()
	a, w, err := atomicWidths(op, narrow)
	if err != nil {
		return err
	}
	ord, err := ordering(in)
	if err != nil {
		return err
	}
	if releases(ord) && !acquires(ord) {
		return fmt.Errorf("%s: %s is not an ordering a load takes", op, ord)
	}
	addr, ok := vr.lookup(in.Arg(0))
	if !ok {
		return fmt.Errorf("%s: address defined outside the function", op)
	}
	dst, err := vr.define(in.Result(0))
	if err != nil {
		return fmt.Errorf("%s: %w", op, err)
	}
	c.Emit(mir.Instr{
		Op:   atomicLoadOp{a: a, w: w, ordered: acquires(ord)},
		Defs: []mir.VReg{dst}, Uses: []mir.VReg{addr},
	})
	return nil
}

// iselAtomicStore lowers §H's stores: a plain store, or STLR.
func iselAtomicStore(c *cursor, vr *vregs, in *ir.Inst, narrow access) error {
	op := in.Op()
	a, _, err := atomicWidths(op, narrow)
	if err != nil {
		return err
	}
	ord, err := ordering(in)
	if err != nil {
		return err
	}
	if acquires(ord) && !releases(ord) {
		return fmt.Errorf("%s: %s is not an ordering a store takes", op, ord)
	}
	val, ok := vr.lookup(in.Arg(0))
	if !ok {
		return fmt.Errorf("%s: value defined outside the function", op)
	}
	addr, ok := vr.lookup(in.Arg(1))
	if !ok {
		return fmt.Errorf("%s: address defined outside the function", op)
	}
	c.Emit(mir.Instr{
		Op:   atomicStoreOp{a: a, ordered: releases(ord)},
		Uses: []mir.VReg{val, addr},
	})
	return nil
}

// iselAtomicRmw lowers §H's read-modify-writes as the exclusive loop.
//
//	retry: LDXR  old, [addr]
//	       <alu> new, old, val
//	       STXR  status, new, [addr]
//	       CBNZ  status, retry
//
// The result is old, which is the value the loop read on the iteration whose
// store succeeded.
func iselAtomicRmw(c *cursor, vr *vregs, in *ir.Inst, narrow access, alu ir.Verb) error {
	op := in.Op()
	a, w, err := atomicWidths(op, narrow)
	if err != nil {
		return err
	}
	ord, err := ordering(in)
	if err != nil {
		return err
	}
	if ord == ir.Unordered {
		return fmt.Errorf("%s: a read-modify-write is not unordered", op)
	}
	val, ok := vr.lookup(in.Arg(0))
	if !ok {
		return fmt.Errorf("%s: value defined outside the function", op)
	}
	addr, ok := vr.lookup(in.Arg(1))
	if !ok {
		return fmt.Errorf("%s: address defined outside the function", op)
	}
	dst, err := vr.define(in.Result(0))
	if err != nil {
		return fmt.Errorf("%s: %w", op, err)
	}

	retry, done := c.open("retry"), c.open("done")
	c.to(retry)

	c.Emit(mir.Instr{
		Op:   ldxrOp{a: a, w: w, acquire: acquires(ord)},
		Defs: []mir.VReg{dst}, Uses: []mir.VReg{addr},
	})

	// xchg computes nothing: the value the caller supplied is the value
	// stored, and the loop exists only to make the exchange atomic.
	next := val
	if alu != "" {
		next = vr.temp(w)
		c.Emit(mir.Instr{
			Op:   aluOp{verb: alu, w: w},
			Defs: []mir.VReg{next}, Uses: []mir.VReg{dst, val},
		})
	}

	status := vr.temp(w32)
	c.Emit(mir.Instr{
		Op:   stxrOp{a: a, release: releases(ord)},
		Defs: []mir.VReg{status}, Uses: []mir.VReg{next, addr},
	})
	c.branchNonZero(status, retry, done)
	return nil
}

// iselAtomicCas lowers §H's compare-and-swap.
//
//	retry: LDXR  old, [addr]
//	       CMP   old, expect
//	       B.NE  fail
//	       STXR  status, new, [addr]
//	       CBNZ  status, retry
//	       B     done
//	fail:  CLREX
//	done:
//
// The result is old either way, which is what §H asks for: the value read,
// with success being eq against expect.
//
// CLREX on the failure path is what the reservation is owed. The loop took one
// with LDXR and then did not store, and nothing else would consume it before
// the next exclusive access — which might be in another function entirely.
func iselAtomicCas(c *cursor, vr *vregs, in *ir.Inst, narrow access) error {
	op := in.Op()
	a, w, err := atomicWidths(op, narrow)
	if err != nil {
		return err
	}
	ords := in.Orderings()
	if len(ords) != 2 {
		return fmt.Errorf("%s: %d orderings, want a success and a failure", op, len(ords))
	}
	success, failure := ords[0], ords[1]
	if success == ir.Unordered || failure == ir.Unordered {
		return fmt.Errorf("%s: a compare-and-swap is not unordered", op)
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
	dst, err := vr.define(in.Result(0))
	if err != nil {
		return fmt.Errorf("%s: %w", op, err)
	}

	// A narrow compare-and-swap compares only the bytes it accesses. The
	// loaded value arrives zero-extended, so the expected one is cut to
	// the same width rather than the comparison being made narrow.
	want := expect
	if a == a8 || a == a16 {
		want = vr.temp(w)
		c.Emit(mir.Instr{
			Op:   extOp{from: a, signed: false},
			Defs: []mir.VReg{want}, Uses: []mir.VReg{expect},
		})
	}

	// The load side acquires if either ordering asks it to: a failing
	// compare-and-swap still performs the load, and its ordering is the
	// failure one.
	acquire := acquires(success) || acquires(failure)

	retry, fail, done := c.open("retry"), c.open("fail"), c.open("done")
	c.to(retry)

	c.Emit(mir.Instr{
		Op:   ldxrOp{a: a, w: w, acquire: acquire},
		Defs: []mir.VReg{dst}, Uses: []mir.VReg{addr},
	})
	c.Emit(mir.Instr{Op: cmpOp{w: w}, Uses: []mir.VReg{dst, want}})
	store := c.open("store")
	c.branch(condNE, fail, store)

	status := vr.temp(w32)
	c.Emit(mir.Instr{
		Op:   stxrOp{a: a, release: releases(success)},
		Defs: []mir.VReg{status}, Uses: []mir.VReg{newVal, addr},
	})
	c.branchNonZero(status, retry, done)

	fail.Emit(mir.Instr{Op: clrexOp{}})
	fail.Emit(mir.Instr{Op: bOp{target: done.Label}})
	c.mf.Succ(fail, done.Label)

	c.resume(done)
	return nil
}

// iselFence lowers §H's fence.
//
// A single-thread fence emits nothing: it orders this thread against its own
// interrupted execution, which no machine barrier is needed for and which the
// IR carries so that the optimizer can see it. Everything else is DMB —
// ISHLD for an acquire, which orders loads against what follows and is what
// acquire means, and the full ISH otherwise.
func iselFence(c *cursor, in *ir.Inst) error {
	if in.SingleThread() {
		return nil
	}
	ord, err := ordering(in)
	if err != nil {
		return err
	}
	b := barrierISH
	if ord == ir.Acquire {
		b = barrierISHLD
	}
	c.Emit(mir.Instr{Op: dmbOp{barrier: b}})
	return nil
}
