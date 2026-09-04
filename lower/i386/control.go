package i386

import (
	"fmt"

	"github.com/vertex-language/i386/reg"

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

// to ends the block being filled with a jump to b, and makes b the block
// being filled.
func (c *cursor) to(b *mir.Block) {
	c.Emit(mir.Instr{Op: jmpOp{target: b.Label}})
	c.mf.Succ(c.blk, b.Label)
	c.blk = b
}

// branch ends the block being filled with a two-way branch on the flags and
// makes els the block being filled — so what follows is the condition's false
// arm, which is the arm a retry loop falls out into.
func (c *cursor) branch(cond condCode, then, els *mir.Block) {
	c.Emit(mir.Instr{Op: jccOp{cond: cond, then: then.Label, els: els.Label}})
	c.mf.Succ(c.blk, then.Label)
	c.mf.Succ(c.blk, els.Label)
	c.blk = els
}

// resume makes b the block being filled, without a branch into it.
func (c *cursor) resume(b *mir.Block) { c.blk = b }

// open starts a block under the one isel began in, and does not move the cursor.
func (c *cursor) open(kind string) *mir.Block {
	c.n++
	return c.mf.NewBlock(fmt.Sprintf("%s.%s%d", c.base, kind, c.n))
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
func emitCopy(e emitter, dst, src mir.VReg) {
	e.Emit(mir.Instr{
		Op:   movOp{},
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
	case ir.VAsmGoto:
		return iselAsmGoto(fn, mf, c, vr, term)
	case ir.VTrap:
		c.Emit(mir.Instr{Op: trapOp{}})
		return nil
	case ir.VReturn:
		return iselReturn(c, vr, fr, term)
	}
	return fmt.Errorf("%s is not a terminator this package lowers", term.Op())
}

// iselReturn lowers a return of the value the psABI brings back in registers.
//
// EAX, or EDX:EAX for an i64 — which is the only place the two halves of a
// value are named by different registers rather than by the allocator.
func iselReturn(c *cursor, vr *vregs, fr *frame, term *ir.Inst) error {
	args := term.Args()
	if len(args) > 1 {
		return fmt.Errorf("return: %d values; the psABI returns one, and more comes back through memory, which is sret and is not written yet", len(args))
	}

	var uses []mir.VReg
	for i, a := range args {
		v, ok := vr.lookup(a)
		if !ok {
			return fmt.Errorf("return: operand %d defined outside the function", i)
		}
		if v.w.isFloat() {
			// The psABI returns a float in ST(0), which is the one
			// place x87 survives in a package that computes in XMM
			// registers: the value goes out to memory and comes
			// back onto the x87 stack, and the FSTP the caller
			// does is what takes it off again.
			c.Emit(mir.Instr{
				Op:   fstReturnOp{off: fr.scratch(), w: v.w},
				Uses: []mir.VReg{v.lo},
			})
			continue
		}
		eax := vr.physical(reg.EAX)
		emitCopy(c, eax, v.lo)
		uses = append(uses, eax)
		if v.w.pairs() {
			edx := vr.physical(reg.EDX)
			emitCopy(c, edx, v.hi)
			uses = append(uses, edx)
		}
	}

	c.Emit(mir.Instr{Op: retOp{}, Uses: uses})
	return nil
}

// iselCallInd lowers §G's indirect call, whose convention comes from the
// named func type rather than from a declaration.
func iselCallInd(c *cursor, vr *vregs, fr *frame, in *ir.Inst) error {
	addr, ok := vr.lookup(in.Arg(0))
	if !ok {
		return fmt.Errorf("callind: callee defined outside the function")
	}
	var sig *ir.Sig
	if t := in.NamedType(); t != nil {
		sig = t.Sig()
	}
	// The address into a vreg of this call's own: every caller-saved
	// register is a destination of the call, so the allocator can only
	// place it out of the way if it is a value the call uses rather than
	// one of the pinned vregs it also defines.
	target := vr.reg32()
	emitCopy(c, target, addr.lo)
	return iselCallSeq(c, vr, fr, "callind", sig, in.Args()[1:], in.Results(), target, true)
}

// iselCall lowers §G's direct call.
//
// Every argument goes to the outgoing area, in order. There is no register
// argument to place and no file to run out of, so the whole of the calling
// convention here is a run of stores.
func iselCall(c *cursor, vr *vregs, fr *frame, in *ir.Inst) error {
	sym := in.Symbol()
	if sym == nil {
		return fmt.Errorf("call: no callee named")
	}
	var sig *ir.Sig
	if callee := in.Callee(); callee != nil {
		sig = callee.Signature()
	}
	return iselCallSeq(c, vr, fr, "call @"+sym.Name(), sig, in.Args(), in.Results(),
		0, false, sym.Name())
}

// iselCallSeq is the body both call forms share.
//
// Every argument goes to the outgoing area, in order. There is no register
// argument to place and no file to run out of, so the whole of the calling
// convention here is a run of stores — which is why this is short where the
// other two backends' is not.
func iselCallSeq(c *cursor, vr *vregs, fr *frame, what string, sig *ir.Sig,
	args []*ir.Def, results []*ir.Def, target mir.VReg, indirect bool, sym ...string) error {

	// A variadic call needs nothing extra: the psABI passes a variadic
	// argument exactly where a named one would go, which is the stack.
	// That is the whole reason §I is short on this target.
	_ = sig
	if len(results) > 1 {
		return fmt.Errorf("%s: %d results; the psABI returns one", what, len(results))
	}

	places, err := classifySysV(typesOf(args))
	if err != nil {
		return fmt.Errorf("%s: %w", what, err)
	}
	for i, pl := range places {
		src, ok := vr.lookup(args[i])
		if !ok {
			return fmt.Errorf("%s: argument %d defined outside the function", what, i)
		}
		if pl.w.isFloat() {
			c.Emit(mir.Instr{
				Op:   fargStoreOp{off: pl.off, w: pl.w},
				Uses: []mir.VReg{src.lo},
			})
			continue
		}
		c.Emit(mir.Instr{Op: argStoreOp{off: pl.off}, Uses: []mir.VReg{src.lo}})
		if pl.w.pairs() {
			c.Emit(mir.Instr{Op: argStoreOp{off: pl.off + 4}, Uses: []mir.VReg{src.hi}})
		}
	}

	// Every caller-saved register is a destination whether or not the call
	// names one: that is the list of places a value live across it cannot
	// be.
	eax := vr.physical(reg.EAX)
	edx := vr.physical(reg.EDX)
	ecx := vr.physical(reg.ECX)
	call := mir.Instr{Defs: []mir.VReg{eax, edx, ecx}}
	if indirect {
		call.Op = callIndOp{}
		call.Uses = []mir.VReg{target}
	} else {
		call.Op = callOp{sym: sym[0]}
	}
	c.Emit(call)

	for _, d := range results {
		result, err := vr.define(d)
		if err != nil {
			return fmt.Errorf("%s: %w", what, err)
		}
		if result.w.isFloat() {
			// Off the x87 stack, through memory, into a vector
			// register — the return convention in reverse.
			c.Emit(mir.Instr{
				Op:   fstpResultOp{off: fr.scratch(), w: result.w},
				Defs: []mir.VReg{result.lo},
			})
			continue
		}
		emitCopy(c, result.lo, eax)
		if result.w.pairs() {
			emitCopy(c, result.hi, edx)
		}
	}
	return nil
}

// iselBr lowers an unconditional branch, with the edge's block arguments
// assigned before it.
func iselBr(fn *ir.Func, mf *mir.Func, c *cursor, vr *vregs, term *ir.Inst) error {
	targets := term.Targets()
	if len(targets) != 1 {
		return fmt.Errorf("br: %d targets, want one", len(targets))
	}
	moves, err := edgeCopies(vr, targets[0])
	if err != nil {
		return err
	}
	emitParallelCopy(c, vr, moves)

	label := blockLabel(fn, targets[0].Block())
	c.Emit(mir.Instr{Op: jmpOp{target: label}})
	mf.Succ(c.blk, label)
	return nil
}

// iselBrIf lowers a two-way branch.
//
// The condition is tested against itself rather than fused into a compare.
// Fusing is a peephole and this package has nowhere to put one; the flags
// were set by a compare isel already emitted, and reading them back through
// a value is the shape that always works.
func iselBrIf(fn *ir.Func, mf *mir.Func, c *cursor, vr *vregs, term *ir.Inst) error {
	cond, ok := vr.lookup(term.Arg(0))
	if !ok {
		return fmt.Errorf("brif: condition defined outside the function")
	}

	targets := term.Targets()
	if len(targets) != 2 {
		return fmt.Errorf("brif: %d targets, want two", len(targets))
	}
	thenLabel, err := edgeTarget(fn, mf, c, vr, targets[0], "then")
	if err != nil {
		return err
	}
	elsLabel, err := edgeTarget(fn, mf, c, vr, targets[1], "else")
	if err != nil {
		return err
	}

	c.Emit(mir.Instr{Op: testOp{}, Uses: []mir.VReg{cond.lo}})
	c.Emit(mir.Instr{Op: jccOp{cond: condNE, then: thenLabel, els: elsLabel}})
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
	ec := newCursor(c.fn, mf, edge)
	emitParallelCopy(ec, vr, moves)
	ec.Emit(mir.Instr{Op: jmpOp{target: dest}})
	mf.Succ(edge, dest)
	return edge.Label, nil
}

// copyPair is one assignment inside a branch edge's parallel copy.
type copyPair struct{ dst, src mir.VReg }

// edgeCopies pairs every argument a branch supplies with the block parameter
// it lands in, one pair per register — so a 64-bit argument is two of them,
// and the two halves move independently.
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
			return nil, fmt.Errorf("@%s: parameter %d has no register", blk.Label(), i)
		}
		if src.lo != dst.lo {
			out = append(out, copyPair{dst: dst.lo, src: src.lo})
		}
		if dst.w.pairs() && src.hi != dst.hi {
			out = append(out, copyPair{dst: dst.hi, src: src.hi})
		}
	}
	return out, nil
}

// emitParallelCopy issues a set of simultaneous assignments as a sequence.
//
// A move whose destination is nobody else's source is free to go first. When
// none is — which is a cycle — one source is copied aside and the rest
// unblock in order. A scratch register rather than XCHG, which is a slow
// instruction on every x86 made since the 486 and would need the two
// registers named as both sources and destinations.
func emitParallelCopy(c *cursor, vr *vregs, moves []copyPair) {
	pending := append([]copyPair(nil), moves...)
	for len(pending) > 0 {
		i := readyMove(pending)
		if i < 0 {
			tmp := vr.reg32()
			emitCopy(c, tmp, pending[0].src)
			pending[0].src = tmp
			continue
		}
		m := pending[i]
		emitCopy(c, m.dst, m.src)
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

// iselBrInd lowers §G2's computed goto: an indirect jump.
//
// The named labels are declared as successors and nothing else is done with
// them. They are what liveness needs — a value the target block reads has to
// still be live at a branch that can reach it — and the address itself came
// from a blockaddr.
func iselBrInd(fn *ir.Func, mf *mir.Func, c *cursor, vr *vregs, term *ir.Inst) error {
	ptr, ok := vr.lookup(term.Arg(0))
	if !ok {
		return fmt.Errorf("brind: pointer defined outside the function")
	}
	for _, b := range term.Labels() {
		mf.Succ(c.blk, blockLabel(fn, b))
	}
	target := vr.reg32()
	emitCopy(c, target, ptr.lo)
	c.Emit(mir.Instr{Op: brIndOp{}, Uses: []mir.VReg{target}})
	return nil
}

// iselBrTable lowers §G2's br_table.
//
// One unsigned compare covers both ends of the range: a negative selector
// read as unsigned is a very large one, so the branch that catches an index
// past the last entry catches a negative one too.
//
// The table holds addresses rather than distances, and the jump reads it in
// one instruction — an indexed memory operand is a jump target here. Both are
// things the 64-bit backends cannot have: an absolute pointer into text is
// what a position-independent image may not hold, and this object is not one.
func iselBrTable(fn *ir.Func, mf *mir.Func, c *cursor, vr *vregs, term *ir.Inst) error {
	selector, ok := vr.lookup(term.Arg(0))
	if !ok {
		return fmt.Errorf("br_table: selector defined outside the function")
	}
	targets := term.Targets()
	if len(targets) == 0 {
		return fmt.Errorf("br_table: no targets")
	}

	labels := make([]string, len(targets)-1)
	for i := range labels {
		l, err := edgeTarget(fn, mf, c, vr, targets[i], fmt.Sprintf("case%d", i))
		if err != nil {
			return err
		}
		labels[i] = l
		mf.Succ(c.blk, l)
	}
	dflt, err := edgeTarget(fn, mf, c, vr, targets[len(targets)-1], "default")
	if err != nil {
		return err
	}
	mf.Succ(c.blk, dflt)

	// The selector into a vreg of this branch's own, named as a use of a
	// terminator that also defines a scratch: what is free at a branch is
	// what allocation decides.
	sel := vr.reg32()
	emitCopy(c, sel, selector.lo)
	c.Emit(mir.Instr{
		Op: brTableOp{
			id:      fmt.Sprintf("%s.table", c.blk.Label),
			targets: labels,
			dflt:    dflt,
		},
		Defs: []mir.VReg{vr.reg32()},
		Uses: []mir.VReg{sel},
	})
	return nil
}

// labeledBlocks is every block label that something other than a branch
// names — a jump table entry or a blockaddr — and therefore has to be a
// symbol rather than a bare label.
func labeledBlocks(mf *mir.Func) map[string]bool {
	out := map[string]bool{}
	for _, mb := range mf.Blocks {
		for _, in := range mb.Instrs {
			switch op := in.Op.(type) {
			case brTableOp:
				for _, t := range op.targets {
					out[t] = true
				}
				out[op.dflt] = true
			case blockAddrOp:
				out[op.label] = true
			case asmOp:
				// An asm goto's labels are branched to by text this
				// package assembled rather than emitted, so the
				// reference arrives as a symbol and needs one.
				for _, l := range op.emitted {
					out[l] = true
				}
			}
		}
	}
	return out
}
