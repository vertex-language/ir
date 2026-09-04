package i386

import (
	"fmt"

	"github.com/vertex-language/i386/reg"

	"github.com/vertex-language/ir"
	"github.com/vertex-language/ir/lower/mir"
)

// shifts is §A5's five verbs.
var shifts = map[ir.Verb]bool{
	ir.VShl: true, ir.VSShr: true, ir.VUShr: true,
	ir.VRotL: true, ir.VRotR: true,
}

// counted puts the shift amount in ECX, which is the only register the
// variable-count forms read.
func counted(c *cursor, vr *vregs, n mir.VReg) mir.VReg {
	cl := vr.physical(reg.ECX)
	emitCopy(c, cl, n)
	return cl
}

// iselShift lowers §A5.
func iselShift(c *cursor, vr *vregs, in *ir.Inst) error {
	verb := in.Op().Verb
	ops, err := operands(vr, in, 2)
	if err != nil {
		return err
	}
	dst, err := vr.define(in.Result(0))
	if err != nil {
		return fmt.Errorf("%s: %w", in.Op(), err)
	}

	if !dst.w.pairs() {
		cl := counted(c, vr, ops[1].lo)
		two(c, dst.lo, ops[0].lo)
		c.Emit(mir.Instr{
			Op:   shiftOp{verb: verb},
			Defs: []mir.VReg{dst.lo},
			Uses: []mir.VReg{dst.lo, cl},
		})
		return nil
	}

	switch verb {
	case ir.VRotL, ir.VRotR:
		return iselRotate64(c, vr, dst, ops[0], ops[1], verb == ir.VRotR)
	}
	return iselShift64(c, vr, dst, ops[0], ops[1], verb)
}

// iselShift64 lowers a 64-bit shift.
//
//	shl:  SHLD hi, lo, cl   then SHL lo, cl
//	ushr: SHRD lo, hi, cl   then SHR hi, cl
//	sshr: SHRD lo, hi, cl   then SAR hi, cl
//
// which is right for a count below thirty-two and wrong above it, because
// x86 masks the count to five bits and the halves have to cross. The fixup is
// two CMOVs on bit five of the count: at thirty-two or more the shifted-by-
// (n−32) half becomes the whole answer and the other half is filled — with
// zero, or with the sign for an arithmetic shift.
//
// Branchless on purpose. The two CMOVs read the flags one TEST left, and
// nothing between them may write flags — nothing does, because the only
// instructions the allocator inserts are moves.
func iselShift64(c *cursor, vr *vregs, dst, a, n value, verb ir.Verb) error {
	cl := counted(c, vr, n.lo)
	two(c, dst.lo, a.lo)
	two(c, dst.hi, a.hi)

	// The sign fill an arithmetic shift needs, taken before the high half
	// is shifted over.
	var sign mir.VReg
	if verb == ir.VSShr {
		sign = vr.reg32()
		emitCopy(c, sign, dst.hi)
		c.Emit(mir.Instr{
			Op:   shiftImmOp{verb: ir.VSShr, n: 31},
			Defs: []mir.VReg{sign}, Uses: []mir.VReg{sign},
		})
	}

	if verb == ir.VShl {
		c.Emit(mir.Instr{
			Op:   shiftDblOp{},
			Defs: []mir.VReg{dst.hi},
			Uses: []mir.VReg{dst.hi, dst.lo, cl},
		})
		c.Emit(mir.Instr{
			Op:   shiftOp{verb: ir.VShl},
			Defs: []mir.VReg{dst.lo}, Uses: []mir.VReg{dst.lo, cl},
		})
	} else {
		c.Emit(mir.Instr{
			Op:   shiftDblOp{right: true},
			Defs: []mir.VReg{dst.lo},
			Uses: []mir.VReg{dst.lo, dst.hi, cl},
		})
		c.Emit(mir.Instr{
			Op:   shiftOp{verb: verb},
			Defs: []mir.VReg{dst.hi}, Uses: []mir.VReg{dst.hi, cl},
		})
	}

	// The filler, materialized before the TEST so that materializing it
	// cannot disturb the flags the CMOVs read.
	fill := sign
	if verb != ir.VSShr {
		fill = vr.reg32()
		c.Emit(mir.Instr{Op: constOp{imm: 0}, Defs: []mir.VReg{fill}})
	}

	c.Emit(mir.Instr{Op: testImmOp{imm: 32}, Uses: []mir.VReg{cl}})
	if verb == ir.VShl {
		c.Emit(mir.Instr{Op: cmovOp{cond: condNE}, Defs: []mir.VReg{dst.hi}, Uses: []mir.VReg{dst.hi, dst.lo}})
		c.Emit(mir.Instr{Op: cmovOp{cond: condNE}, Defs: []mir.VReg{dst.lo}, Uses: []mir.VReg{dst.lo, fill}})
		return nil
	}
	c.Emit(mir.Instr{Op: cmovOp{cond: condNE}, Defs: []mir.VReg{dst.lo}, Uses: []mir.VReg{dst.lo, dst.hi}})
	c.Emit(mir.Instr{Op: cmovOp{cond: condNE}, Defs: []mir.VReg{dst.hi}, Uses: []mir.VReg{dst.hi, fill}})
	return nil
}

// iselRotate64 lowers a 64-bit rotate.
//
// A rotate is what SHLD already is, applied both ways round: the bits leaving
// the low half's top enter the high half's bottom, and the bits leaving the
// high half's top wrap into the low half's bottom. So the whole rotate is two
// SHLDs, each taking the other half as its source — once the halves have been
// swapped for a count of thirty-two or more, which is what that count means.
func iselRotate64(c *cursor, vr *vregs, dst, a, n value, right bool) error {
	// Rotating right by n is rotating left by 64−n. The negation is taken
	// modulo 64 by the same masking every shift here relies on.
	amount := n.lo
	if right {
		neg := vr.reg32()
		emitCopy(c, neg, n.lo)
		c.Emit(mir.Instr{Op: unOp{verb: ir.VNeg}, Defs: []mir.VReg{neg}, Uses: []mir.VReg{neg}})
		amount = neg
	}
	cl := counted(c, vr, amount)

	lo := vr.reg32()
	hi := vr.reg32()
	emitCopy(c, lo, a.lo)
	emitCopy(c, hi, a.hi)

	// The half-swap for a count of thirty-two or more, branchless.
	spare := vr.reg32()
	emitCopy(c, spare, lo)
	c.Emit(mir.Instr{Op: testImmOp{imm: 32}, Uses: []mir.VReg{cl}})
	c.Emit(mir.Instr{Op: cmovOp{cond: condNE}, Defs: []mir.VReg{lo}, Uses: []mir.VReg{lo, hi}})
	c.Emit(mir.Instr{Op: cmovOp{cond: condNE}, Defs: []mir.VReg{hi}, Uses: []mir.VReg{hi, spare}})

	// Both halves are needed twice, so the low one is kept aside before
	// the first SHLD overwrites it.
	keep := vr.reg32()
	emitCopy(c, keep, lo)
	emitCopy(c, dst.lo, lo)
	c.Emit(mir.Instr{Op: shiftDblOp{}, Defs: []mir.VReg{dst.lo}, Uses: []mir.VReg{dst.lo, hi, cl}})
	emitCopy(c, dst.hi, hi)
	c.Emit(mir.Instr{Op: shiftDblOp{}, Defs: []mir.VReg{dst.hi}, Uses: []mir.VReg{dst.hi, keep, cl}})
	return nil
}

// iselBitCount lowers §A6's clz, ctz and popcnt.
func iselBitCount(c *cursor, vr *vregs, in *ir.Inst) error {
	verb := in.Op().Verb
	ops, err := operands(vr, in, 1)
	if err != nil {
		return err
	}
	dst, err := vr.define(in.Result(0))
	if err != nil {
		return fmt.Errorf("%s: %w", in.Op(), err)
	}

	if verb == ir.VPopcnt {
		lo := popcnt32(c, vr, ops[0].lo)
		if !dst.w.pairs() {
			emitCopy(c, dst.lo, lo)
			return nil
		}
		hi := popcnt32(c, vr, ops[0].hi)
		emitCopy(c, dst.lo, lo)
		c.Emit(mir.Instr{Op: aluOp{verb: ir.VAdd}, Defs: []mir.VReg{dst.lo}, Uses: []mir.VReg{dst.lo, hi}})
		c.Emit(mir.Instr{Op: constOp{imm: 0}, Defs: []mir.VReg{dst.hi}})
		return nil
	}

	if !dst.w.pairs() {
		r := scan32(c, vr, ops[0].lo, verb == ir.VClz)
		emitCopy(c, dst.lo, r)
		return nil
	}

	// At sixty-four the answer comes from whichever half has a set bit
	// nearest the end being counted from, and the other half's count plus
	// thirty-two when it does not.
	near, far := ops[0].hi, ops[0].lo // clz counts down from the top
	if verb == ir.VCtz {
		near, far = ops[0].lo, ops[0].hi
	}
	nearCount := scan32(c, vr, near, verb == ir.VClz)
	farCount := scan32(c, vr, far, verb == ir.VClz)
	c.Emit(mir.Instr{Op: aluOp{verb: ir.VAdd}, Defs: []mir.VReg{farCount}, Uses: []mir.VReg{farCount, thirtyTwo(c, vr)}})

	emitCopy(c, dst.lo, nearCount)
	c.Emit(mir.Instr{Op: testOp{}, Uses: []mir.VReg{near}})
	c.Emit(mir.Instr{Op: cmovOp{cond: condE}, Defs: []mir.VReg{dst.lo}, Uses: []mir.VReg{dst.lo, farCount}})
	c.Emit(mir.Instr{Op: constOp{imm: 0}, Defs: []mir.VReg{dst.hi}})
	return nil
}

func thirtyTwo(c *cursor, vr *vregs) mir.VReg {
	r := vr.reg32()
	c.Emit(mir.Instr{Op: constOp{imm: 32}, Defs: []mir.VReg{r}})
	return r
}

// scan32 counts one register's leading or trailing zeroes.
//
// BSR and BSF give the index of the highest or lowest set bit and leave the
// destination alone when there is none, which is why §A6's specified answer
// for zero — the width, not something target-defined — needs the CMOV: the
// register is undefined in exactly the case the spec pins down.
func scan32(c *cursor, vr *vregs, x mir.VReg, leading bool) mir.VReg {
	idx := vr.reg32()
	if !leading {
		miss := vr.reg32()
		c.Emit(mir.Instr{Op: constOp{imm: 32}, Defs: []mir.VReg{miss}})
		c.Emit(mir.Instr{Op: bitScanOp{}, Defs: []mir.VReg{idx}, Uses: []mir.VReg{x}})
		c.Emit(mir.Instr{Op: cmovOp{cond: condE}, Defs: []mir.VReg{idx}, Uses: []mir.VReg{idx, miss}})
		return idx
	}
	// Leading zeroes are 31 minus the highest set bit's index, so the
	// miss value is −1: 31 − (−1) is the 32 the spec asks for.
	miss := vr.reg32()
	c.Emit(mir.Instr{Op: constOp{imm: -1}, Defs: []mir.VReg{miss}})
	c.Emit(mir.Instr{Op: bitScanOp{reverse: true}, Defs: []mir.VReg{idx}, Uses: []mir.VReg{x}})
	c.Emit(mir.Instr{Op: cmovOp{cond: condE}, Defs: []mir.VReg{idx}, Uses: []mir.VReg{idx, miss}})

	out := vr.reg32()
	c.Emit(mir.Instr{Op: constOp{imm: 31}, Defs: []mir.VReg{out}})
	c.Emit(mir.Instr{Op: aluOp{verb: ir.VSub}, Defs: []mir.VReg{out}, Uses: []mir.VReg{out, idx}})
	return out
}

// popcnt32 is the SWAR sequence: pairs, then nibbles, then bytes, then one
// multiply that sums the bytes into the top one.
//
// The 386 has no POPCNT — it arrived with SSE4.2 — and the assembler declares
// no SSE at all, so there is nothing to gate this behind.
func popcnt32(c *cursor, vr *vregs, x mir.VReg) mir.VReg {
	konst := func(v int64) mir.VReg {
		r := vr.reg32()
		c.Emit(mir.Instr{Op: constOp{imm: v}, Defs: []mir.VReg{r}})
		return r
	}
	alu := func(verb ir.Verb, a, b mir.VReg) mir.VReg {
		r := vr.reg32()
		emitCopy(c, r, a)
		c.Emit(mir.Instr{Op: aluOp{verb: verb}, Defs: []mir.VReg{r}, Uses: []mir.VReg{r, b}})
		return r
	}
	shr := func(a mir.VReg, n int64) mir.VReg {
		r := vr.reg32()
		emitCopy(c, r, a)
		c.Emit(mir.Instr{Op: shiftImmOp{verb: ir.VUShr, n: n}, Defs: []mir.VReg{r}, Uses: []mir.VReg{r}})
		return r
	}

	v := alu(ir.VSub, x, alu(ir.VAnd, shr(x, 1), konst(0x55555555)))
	v = alu(ir.VAdd, alu(ir.VAnd, v, konst(0x33333333)), alu(ir.VAnd, shr(v, 2), konst(0x33333333)))
	v = alu(ir.VAnd, alu(ir.VAdd, v, shr(v, 4)), konst(0x0f0f0f0f))
	return shr(alu(ir.VMul, v, konst(0x01010101)), 24)
}

// iselBswap lowers §A6's byte reversal. At sixty-four bits it is the two
// halves reversed and then exchanged, since reversing the whole value moves
// the high bytes to the low end.
func iselBswap(c *cursor, vr *vregs, in *ir.Inst) error {
	ops, err := operands(vr, in, 1)
	if err != nil {
		return err
	}
	dst, err := vr.define(in.Result(0))
	if err != nil {
		return fmt.Errorf("%s: %w", in.Op(), err)
	}
	if !dst.w.pairs() {
		two(c, dst.lo, ops[0].lo)
		c.Emit(mir.Instr{Op: bswapOp{}, Defs: []mir.VReg{dst.lo}, Uses: []mir.VReg{dst.lo}})
		return nil
	}
	two(c, dst.lo, ops[0].hi)
	c.Emit(mir.Instr{Op: bswapOp{}, Defs: []mir.VReg{dst.lo}, Uses: []mir.VReg{dst.lo}})
	two(c, dst.hi, ops[0].lo)
	c.Emit(mir.Instr{Op: bswapOp{}, Defs: []mir.VReg{dst.hi}, Uses: []mir.VReg{dst.hi}})
	return nil
}

// iselPtrInt lowers §C4, which on a target whose pointers are already the
// register width is a move — and, from an i64, the low half alone.
func iselPtrInt(c *cursor, vr *vregs, in *ir.Inst) error {
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
		// i64.from_ptr zero-extends, which §C4 says because ptrbits is
		// less than 64 here. This is the one target where that clause
		// does something.
		c.Emit(mir.Instr{Op: constOp{imm: 0}, Defs: []mir.VReg{dst.hi}})
	}
	return nil
}

// iselDiff lowers §D3's ptr.diff: a subtraction, sign-extended from ptrbits
// — which at thirty-two bits into an i64 is a real sign extension, not the
// no-op it is on the other two targets.
func iselDiff(c *cursor, vr *vregs, in *ir.Inst) error {
	ops, err := operands(vr, in, 2)
	if err != nil {
		return err
	}
	dst, err := vr.define(in.Result(0))
	if err != nil {
		return fmt.Errorf("%s: %w", in.Op(), err)
	}
	two(c, dst.lo, ops[0].lo)
	c.Emit(mir.Instr{Op: aluOp{verb: ir.VSub}, Defs: []mir.VReg{dst.lo}, Uses: []mir.VReg{dst.lo, ops[1].lo}})
	if dst.w.pairs() {
		return fillHigh(c, dst, true)
	}
	return nil
}
