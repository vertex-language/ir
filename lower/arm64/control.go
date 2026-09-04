package arm64

import (
	"fmt"

	"github.com/vertex-language/ir"
	"github.com/vertex-language/ir/lower/mir"
)

// A cursor is the block isel is filling, and the ability to start another.
type cursor struct {
	fn   *ir.Func
	mf   *mir.Func
	blk  *mir.Block
	base string
	n    int
}

func newCursor(fn *ir.Func, mf *mir.Func, blk *mir.Block) *cursor {
	return &cursor{fn: fn, mf: mf, blk: blk, base: blk.Label}
}

func (c *cursor) Emit(in mir.Instr) { c.blk.Emit(in) }

// open starts a block under the one isel began in, and does not move the cursor.
func (c *cursor) open(kind string) *mir.Block {
	c.n++
	return c.mf.NewBlock(fmt.Sprintf("%s.%s%d", c.base, kind, c.n))
}

// to ends the block being filled with a branch to b, and makes b the block
// being filled.
func (c *cursor) to(b *mir.Block) {
	c.Emit(mir.Instr{Op: bOp{target: b.Label}})
	c.mf.Succ(c.blk, b.Label)
	c.blk = b
}

// resume makes b the block being filled, without a branch into it.
func (c *cursor) resume(b *mir.Block) { c.blk = b }

// branch ends the block being filled with a two-way branch on the flags and
// makes els the block being filled — so the code that follows is the
// condition's false arm, which is the arm a guard falls through into.
func (c *cursor) branch(cond condCode, then, els *mir.Block) {
	c.Emit(mir.Instr{Op: bcondOp{cond: cond, then: then.Label, els: els.Label}})
	c.mf.Succ(c.blk, then.Label)
	c.mf.Succ(c.blk, els.Label)
	c.blk = els
}

// branchNonZero ends the block being filled with CBNZ and makes els the block
// being filled — the shape a retry loop wants, where the fallthrough is the
// way out.
func (c *cursor) branchNonZero(v mir.VReg, then, els *mir.Block) {
	c.Emit(mir.Instr{Op: cbnzOp{then: then.Label, els: els.Label}, Uses: []mir.VReg{v}})
	c.mf.Succ(c.blk, then.Label)
	c.mf.Succ(c.blk, els.Label)
	c.blk = els
}

// blockLabel is the section label one ir block gets.
func blockLabel(fn *ir.Func, blk *ir.Block) string {
	if blk.IsEntry() {
		return fn.Name()
	}
	return fn.Name() + "." + blk.Label()
}

// emitCopy is a register-to-register move, marked as a copy so the allocator
// can coalesce it away.
func emitCopy(e emitter, dst, src mir.VReg, w width) {
	e.Emit(mir.Instr{
		Op:   movOp{w: w},
		Defs: []mir.VReg{dst},
		Uses: []mir.VReg{src},
		Copy: true,
	})
}

// iselBlock lowers one ir block's instructions and its terminator.
func iselBlock(fn *ir.Func, mf *mir.Func, vr *vregs, fr *frame, blk *ir.Block, mb *mir.Block, opts Options) error {
	c := newCursor(fn, mf, mb)
	term := blk.Term()

	for _, in := range blk.Insts() {
		if in == term {
			continue
		}
		if err := iselInst(c, vr, fr, in, opts); err != nil {
			return err
		}
	}

	if term == nil {
		return fmt.Errorf("block @%s has no terminator", blk.Label())
	}
	switch term.Op().Verb {
	case ir.VBr:
		return iselBr(fn, mf, c, vr, term)
	case ir.VBrIf:
		return iselBrIf(fn, mf, c, vr, term)
	case ir.VBrTable:
		return iselBrTable(fn, mf, c, vr, term)
	case ir.VBrInd:
		return iselBrInd(fn, mf, c, vr, term)
	case ir.VTrap:
		c.Emit(mir.Instr{Op: trapOp{}})
		return nil
	case ir.VReturn:
		return iselReturn(fn, c, vr, term)
	case ir.VAsmGoto:
		return iselAsmGoto(fn, mf, c, vr, term)
	}
	return fmt.Errorf("%s is not a terminator this package lowers", term.Op())
}

// iselReturn lowers a return of the values AAPCS64 brings back in registers.
//
// The same eight registers arguments arrive in, and the first two of each file
// is where this package stops — for the reason the other architecture stops at
// two, which is that more than that is a hidden pointer and a memory result.
func iselReturn(fn *ir.Func, c *cursor, vr *vregs, term *ir.Inst) error {
	args := term.Args()
	if len(args) > 2 {
		return fmt.Errorf("return: %d values; more than two comes back through memory, which is sret and is not written yet", len(args))
	}

	// A result §5.5 brings back in registers, out of the storage the body
	// wrote it into. The front end declared an sret parameter and returns
	// nothing, so there is no operand here to read: the slot that parameter
	// names is the value, and this is where it becomes a register again.
	if agg, inRegs, err := sretInRegs(sretParamType(fn)); err != nil {
		return err
	} else if inRegs {
		return returnAggregate(fn, c, vr, agg)
	}

	uses := make([]mir.VReg, 0, len(args))
	var ints, floats int
	for i, a := range args {
		v, ok := vr.lookup(a)
		if !ok {
			return fmt.Errorf("return: operand %d defined outside the function", i)
		}
		w := vr.widthOfVReg(v)
		var dst mir.VReg
		if w.isFloat() {
			dst = vr.physicalVec(aapcsFloatArgs[floats], w)
			floats++
		} else {
			dst = vr.physical(aapcsIntArgs[ints], w)
			ints++
		}
		emitCopy(c, dst, v, w)
		uses = append(uses, dst)
	}

	c.Emit(mir.Instr{Op: retOp{}, Uses: uses})
	return nil
}

// returnAggregate brings a byval result back in registers. The sret
// parameter's vreg holds the address of the storage the body wrote through —
// a slot of this function's own, since a caller expecting registers supplied
// no address — and each register is one load out of it, at that register's
// own offset.
func returnAggregate(fn *ir.Func, c *cursor, vr *vregs, agg aggregate) error {
	ps := fn.Params()
	if len(ps) == 0 {
		return fmt.Errorf("return: an sret signature with no parameters")
	}
	base, ok := vr.lookup(ps[0])
	if !ok {
		return fmt.Errorf("return: the sret parameter is defined outside the function")
	}

	uses := make([]mir.VReg, 0, agg.n)
	for k := 0; k < agg.n; k++ {
		var dst mir.VReg
		if agg.kind == aggHFA {
			dst = vr.physicalVec(aapcsFloatArgs[k], agg.w)
		} else {
			dst = vr.physical(aapcsIntArgs[k], agg.w)
		}
		c.Emit(mir.Instr{
			Op:   loadAtOp{off: int64(uint64(k) * agg.step), w: agg.w},
			Defs: []mir.VReg{dst},
			Uses: []mir.VReg{base},
		})
		uses = append(uses, dst)
	}
	c.Emit(mir.Instr{Op: retOp{}, Uses: uses})
	return nil
}

// iselBr lowers an unconditional branch, with the edge's block arguments
// assigned before it.
func iselBr(fn *ir.Func, mf *mir.Func, c *cursor, vr *vregs, term *ir.Inst) error {
	targets := term.Targets()
	if len(targets) != 1 {
		return fmt.Errorf("br: %d targets, want one", len(targets))
	}
	target := targets[0]
	moves, err := edgeCopies(vr, target)
	if err != nil {
		return err
	}
	emitParallelCopy(c, vr, moves)

	label := blockLabel(fn, target.Block())
	c.Emit(mir.Instr{Op: bOp{target: label}})
	mf.Succ(c.blk, label)
	return nil
}

// iselBrIf lowers a two-way branch.
//
// The condition is tested against zero rather than fused into a compare. A
// fused compare-and-branch is what CBZ and TBZ are and is a peephole this
// package does not have a pass for; the flags are set by a CMP that isel
// emitted for the comparison, and reading them back through a value is the
// shape that always works.
func iselBrIf(fn *ir.Func, mf *mir.Func, c *cursor, vr *vregs, term *ir.Inst) error {
	cond, ok := vr.lookup(term.Arg(0))
	if !ok {
		return fmt.Errorf("brif: condition defined outside the function")
	}

	targets := term.Targets()
	if len(targets) != 2 {
		return fmt.Errorf("brif: %d targets, want two", len(targets))
	}
	then, els := targets[0], targets[1]
	thenLabel, err := edgeTarget(fn, mf, c, vr, then, "then")
	if err != nil {
		return err
	}
	elsLabel, err := edgeTarget(fn, mf, c, vr, els, "else")
	if err != nil {
		return err
	}

	c.Emit(mir.Instr{Op: cmpImmOp{imm: 0, w: w32}, Uses: []mir.VReg{cond}})
	c.Emit(mir.Instr{Op: bcondOp{cond: condNE, then: thenLabel, els: elsLabel}})
	mf.Succ(c.blk, thenLabel)
	mf.Succ(c.blk, elsLabel)
	return nil
}

// edgeTarget is the label a branch should name for one target: the block
// itself when the edge carries no arguments, and a block made to assign them
// when it does.
func edgeTarget(fn *ir.Func, mf *mir.Func, c *cursor, vr *vregs, t ir.BlockTarget, kind string) (string, error) {
	moves, err := edgeCopies(vr, t)
	if err != nil {
		return "", err
	}
	dest := blockLabel(fn, t.Block())
	if len(moves) == 0 {
		return dest, nil
	}
	edge := c.open(kind)
	ec := newCursor(fn, mf, edge)
	emitParallelCopy(ec, vr, moves)
	ec.Emit(mir.Instr{Op: bOp{target: dest}})
	mf.Succ(edge, dest)
	return edge.Label, nil
}

// copyPair is one assignment inside a branch edge's parallel copy.
type copyPair struct {
	dst, src mir.VReg
	w        width
}

// edgeCopies pairs every argument a branch supplies with the block parameter
// it lands in.
func edgeCopies(vr *vregs, target ir.BlockTarget) ([]copyPair, error) {
	return edgeCopiesTrailing(vr, target, 0)
}

// edgeCopiesTrailing is edgeCopies for an edge whose target takes trailing
// parameters from somewhere other than the branch: an asm goto's outputs are
// written by the assembled text rather than copied in, so extra of them are
// already in place and the copies cover the leading arguments alone.
func edgeCopiesTrailing(vr *vregs, target ir.BlockTarget, extra int) ([]copyPair, error) {
	blk := target.Block()
	args := target.Args()
	params := blk.Params()
	if len(args)+extra != len(params) {
		return nil, fmt.Errorf("@%s takes %d parameters, the edge supplies %d", blk.Label(), len(params), len(args)+extra)
	}

	var out []copyPair
	for i, a := range args {
		src, ok := vr.lookup(a)
		if !ok {
			return nil, fmt.Errorf("@%s: edge argument %d defined outside the function", blk.Label(), i)
		}
		dst, ok := vr.lookup(params[i])
		if !ok {
			return nil, fmt.Errorf("@%s: parameter %d has no vreg", blk.Label(), i)
		}
		if src == dst {
			continue
		}
		out = append(out, copyPair{dst: dst, src: src, w: vr.widthOfVReg(dst)})
	}
	return out, nil
}

// emitParallelCopy issues a set of simultaneous assignments as a sequence.
//
// A move whose destination is nobody else's source is free to go first. When
// none is — which is a cycle — one is broken by copying its source aside, and
// the rest then unblock in order. A scratch register rather than the swap
// instruction the other architecture uses: A64 has no XCHG.
func emitParallelCopy(c *cursor, vr *vregs, moves []copyPair) {
	pending := append([]copyPair(nil), moves...)
	for len(pending) > 0 {
		i := readyMove(pending)
		if i < 0 {
			// Every remaining move is blocked, so they form a cycle.
			// Copying one source into a temporary makes that move's
			// destination free and the cycle a chain.
			tmp := vr.temp(pending[0].w)
			emitCopy(c, tmp, pending[0].src, pending[0].w)
			pending[0].src = tmp
			continue
		}
		m := pending[i]
		emitCopy(c, m.dst, m.src, m.w)
		pending = append(pending[:i], pending[i+1:]...)
	}
}

// readyMove is the index of a move no other pending move still reads from.
func readyMove(pending []copyPair) int {
	for i, m := range pending {
		blocked := false
		for j, other := range pending {
			if i != j && other.src == m.dst {
				blocked = true
				break
			}
		}
		if !blocked {
			return i
		}
	}
	return -1
}
