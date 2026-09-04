package amd64

import (
	"fmt"

	"github.com/vertex-language/ir"
	"github.com/vertex-language/ir/lower/mir"
)

// A cursor is the block isel is filling, and the ability to start
// another.
type cursor struct {
	fn   *ir.Func
	mf   *mir.Func
	blk  *mir.Block
	base string // the ir block's label, which the new ones extend
	n    int

	// prefix is Options.LibcallPrefix. It rides here because a library
	// call is emitted from wherever isel happens to be, and the cursor is
	// the one thing every one of those places already holds.
	prefix string
}

func newCursor(fn *ir.Func, mf *mir.Func, blk *mir.Block, prefix string) *cursor {
	return &cursor{fn: fn, mf: mf, blk: blk, base: blk.Label, prefix: prefix}
}

func (c *cursor) Emit(in mir.Instr) { c.blk.Emit(in) }

// open starts a block under the one isel began in, and does not move the cursor.
func (c *cursor) open(kind string) *mir.Block {
	c.n++
	return c.mf.NewBlock(edgeLabel(c.mf, fmt.Sprintf("%s.%s%d", c.base, kind, c.n)))
}

// to ends the block being filled with a branch to b, and makes b the block being filled.
func (c *cursor) to(b *mir.Block) {
	c.Emit(mir.Instr{Op: jmpOp{target: b.Label}})
	c.mf.Succ(c.blk, b.Label)
	c.blk = b
}

// resume makes b the block being filled, without a branch.
func (c *cursor) resume(b *mir.Block) { c.blk = b }

// branch ends the block being filled with a two-way branch on the flags and makes els the block being filled.
func (c *cursor) branch(cond condCode, then, els *mir.Block) {
	c.Emit(mir.Instr{Op: brccOp{cond: cond, then: then.Label, els: els.Label}})
	c.mf.Succ(c.blk, then.Label)
	c.mf.Succ(c.blk, els.Label)
	c.blk = els
}

func blockLabel(fn *ir.Func, blk *ir.Block) string {
	if blk.IsEntry() {
		return fn.Name()
	}
	return fn.Name() + "." + blk.Label()
}

// iselBlock isels one ir.Block's body and terminator.
// vr is shared across every block of the function.
func iselBlock(fn *ir.Func, mf *mir.Func, c *cursor, vr *vregs, fr *frame, blk *ir.Block, uses useIndex) error {
	term := blk.Term()

	// A brif fuses its condition's defining compare directly into the
	// branch, and a select fuses its own — see the package doc — so those
	// instructions are skipped in the ordinary walk below rather than
	// isel'd twice.
	//
	// Skipped only when fusing is every reader's answer. A compare whose
	// result is also read as a value has to exist as one, so it is
	// isel'd here into a setcc as well; the branch still compares for
	// itself, which costs a second cmp and is the price of having no
	// pass that could notice the first one is still good.
	var fused *ir.Inst
	var cmp compare
	if term != nil && term.Op().Verb == ir.VBrIf {
		fused, cmp, _ = fusedCompare(term)
	}
	skip := map[*ir.Inst]bool{}
	if fused != nil && fusesEveryUse(uses, fused) {
		skip[fused] = true
	}
	for _, in := range blk.Insts() {
		if in.Op().Verb != ir.VSelect {
			continue
		}
		if c := in.Arg(0).Inst(); c != nil && fusesEveryUse(uses, c) {
			skip[c] = true
		}
	}

	for _, in := range blk.Insts() {
		if skip[in] {
			continue
		}
		if err := iselInst(mf, c, vr, fr, in); err != nil {
			return err
		}
	}

	if fused != nil {
		return iselBrIf(fn, mf, c, vr, blk, fused, cmp, term)
	}
	if term != nil && term.Op().Verb == ir.VBrIf {
		return iselBrIfValue(fn, mf, c, vr, blk, term)
	}
	if term != nil && term.Op().Verb == ir.VBr {
		return iselBr(fn, mf, c, vr, term)
	}
	if term != nil && term.Op().Verb == ir.VBrInd {
		return iselBrInd(fn, mf, c, vr, term)
	}
	if term != nil && term.Op().Verb == ir.VBrTable {
		return iselBrTable(fn, mf, c, vr, term)
	}
	if term != nil && term.Op().Verb == ir.VAsmGoto {
		return iselAsmGoto(fn, mf, c, vr, term)
	}
	if term != nil && term.Op().Verb == ir.VTrap {
		// ud2, which raises #UD. It is a terminator with no successors
		// and no frame to tear down: control does not leave this
		// instruction, so there is nothing after it to restore RSP for.
		c.Emit(mir.Instr{Op: trapOp{}})
		return nil
	}
	return iselReturn(fn, c, vr, term)
}

// useIndex is every slot in a function that reads a definition, by
// definition. It is ir.Func.Uses' answer for every value at once.
//
// The index exists because the question is asked per block and the walk
// behind Uses is over the whole function, which made isel quadratic in
// function size: a switch with sixteen hundred arms took eleven times as
// long to select as one with four hundred, and SQLite's amalgamation, whose
// interpreter loop is one enormous function, did not finish. Building it
// once is one walk instead of one per compare.
type useIndex map[*ir.Def][]ir.Use

func indexUses(fn *ir.Func) useIndex {
	idx := useIndex{}
	fn.WalkUses(func(u ir.Use) bool {
		if d := u.Def(); d != nil {
			idx[d] = append(idx[d], u)
		}
		return true
	})
	return idx
}

// fusesEveryUse reports whether every reader of cmp's result is one that
// fuses the compare into itself — a brif's condition or a select's.
func fusesEveryUse(uses useIndex, cmp *ir.Inst) bool {
	// Fusable at all, first. §B's float eq and ne are two conditions and
	// no branch takes two, so they are values whatever reads them — and
	// a reader that cannot fuse one still has to find it in a register.
	if _, ok := condFor(cmp.Op()); !ok {
		return false
	}
	res := cmp.Result(0)
	if res == nil {
		return false
	}
	refs := uses[res]
	if len(refs) == 0 {
		return false
	}
	for _, u := range refs {
		// The condition is operand zero of both forms, and never a
		// branch argument: a target's argument list is values the
		// successor takes as parameters, which is a use no fusing can
		// reach into.
		if u.Target >= 0 || u.Index != 0 {
			return false
		}
		switch u.Inst.Op().Verb {
		case ir.VBrIf, ir.VSelect:
		default:
			return false
		}
	}
	return true
}

// fusedCompare resolves a condition back to the single compare
// instruction it can fuse into, along with the condition code and width.
func fusedCompare(term *ir.Inst) (*ir.Inst, compare, bool) {
	in := term.Arg(0).Inst()
	if in == nil {
		return nil, compare{}, false
	}
	c, ok := condFor(in.Op())
	if !ok {
		return nil, compare{}, false
	}
	return in, c, true
}

// compareOperands is a compare's two operands in the order the instruction has to take them.
func compareOperands(vr *vregs, cmp *ir.Inst, c compare) (a, b mir.VReg, err error) {
	first, second := cmp.Arg(0), cmp.Arg(1)
	if c.swap {
		first, second = second, first
	}
	a, ok := vr.lookup(first)
	if !ok {
		return 0, 0, fmt.Errorf("%s: operand defined outside the function", cmp.Op())
	}
	b, ok = vr.lookup(second)
	if !ok {
		return 0, 0, fmt.Errorf("%s: operand defined outside the function", cmp.Op())
	}
	return a, b, nil
}

// brifEdges is the two labels a conditional branch out of blk names, one
// per arm, each either the target block itself or an edge block that
// carries its arguments.
func brifEdges(fn *ir.Func, mf *mir.Func, vr *vregs, blk *ir.Block, term *ir.Inst) (then, els string, err error) {
	targets := term.Targets()
	then, err = branchEdge(fn, mf, vr, blk, "then", targets[0])
	if err != nil {
		return "", "", err
	}
	els, err = branchEdge(fn, mf, vr, blk, "else", targets[1])
	if err != nil {
		return "", "", err
	}
	return then, els, nil
}

func iselBrIf(fn *ir.Func, mf *mir.Func, c *cursor, vr *vregs, blk *ir.Block, cmp *ir.Inst, cc compare, term *ir.Inst) error {
	a, b, err := compareOperands(vr, cmp, cc)
	if err != nil {
		return err
	}

	then, els, err := brifEdges(fn, mf, vr, blk, term)
	if err != nil {
		return err
	}

	c.Emit(mir.Instr{Op: cmpOp{w: cc.w}, Uses: []mir.VReg{a, b}})
	c.Emit(mir.Instr{Op: brccOp{cond: cc.cond, then: then, els: els}})
	// Both arms, including an edge block standing in for one of them.
	// Liveness reads these and nothing else: a missing successor makes a
	// value look dead early and two of them share one register.
	mf.Succ(c.blk, then)
	mf.Succ(c.blk, els)
	return nil
}

// iselBrIfValue lowers a brif whose condition is an i1 in a register
// rather than a compare to fuse.
func iselBrIfValue(fn *ir.Func, mf *mir.Func, c *cursor, vr *vregs, blk *ir.Block, term *ir.Inst) error {
	cond := term.Arg(0)
	v, ok := vr.lookup(cond)
	if !ok {
		return fmt.Errorf("brif: condition %s is not a value this package produced", cond)
	}

	then, els, err := brifEdges(fn, mf, vr, blk, term)
	if err != nil {
		return err
	}

	c.Emit(mir.Instr{Op: testOp{}, Uses: []mir.VReg{v}})
	c.Emit(mir.Instr{Op: brccOp{cond: condNE, then: then, els: els}})
	mf.Succ(c.blk, then)
	mf.Succ(c.blk, els)
	return nil
}

// branchEdge returns the label a conditional branch out of blk should name
// for one of its two targets: the target block itself when the edge is bare.
func branchEdge(fn *ir.Func, mf *mir.Func, vr *vregs, blk *ir.Block, role string, target ir.BlockTarget) (string, error) {
	dst := blockLabel(fn, target.Block())
	if len(target.Args()) == 0 {
		return dst, nil
	}

	moves, err := edgeCopies(vr, target)
	if err != nil {
		return "", err
	}

	edge := mf.NewBlock(edgeLabel(mf, blockLabel(fn, blk)+"."+role))
	emitParallelCopy(edge, moves)
	edge.Emit(mir.Instr{Op: jmpOp{target: dst}})
	mf.Succ(edge, dst)
	return edge.Label, nil
}

// edgeLabel returns want, or want with a suffix, whichever is not already
// some block's label.
func edgeLabel(mf *mir.Func, want string) string {
	taken := make(map[string]bool, len(mf.Blocks))
	for _, b := range mf.Blocks {
		taken[b.Label] = true
	}
	if !taken[want] {
		return want
	}
	for n := 2; ; n++ {
		if cand := fmt.Sprintf("%s.%d", want, n); !taken[cand] {
			return cand
		}
	}
}

// iselBr lowers an unconditional branch: the edge's argument moves, then the jump.
func iselBr(fn *ir.Func, mf *mir.Func, c *cursor, vr *vregs, term *ir.Inst) error {
	target := term.Targets()[0]
	moves, err := edgeCopies(vr, target)
	if err != nil {
		return err
	}
	emitParallelCopy(c, moves)
	dst := blockLabel(fn, target.Block())
	c.Emit(mir.Instr{Op: jmpOp{target: dst}})
	mf.Succ(c.blk, dst)
	return nil
}

// emitCopy moves src into dst. Every move this package builds goes
// through here, so that Copy is set on all of them and none by accident.
func emitCopy(e emitter, dst, src mir.VReg, w width) {
	e.Emit(mir.Instr{
		Op:   movOp{w: w},
		Defs: []mir.VReg{dst},
		Uses: []mir.VReg{src},
		Copy: true,
	})
}

// copyPair is one assignment inside a branch edge's parallel copy: dst
// takes the value src held before any of the edge's moves ran, at w.
//
// Both ends are always the same width: an argument's type is the
// parameter's, which is §19.3, so w is a property of the pair and not of
// either end. That is what lets emitParallelCopy rewrite a cycle's
// sources without ever having to re-derive one.
type copyPair struct {
	dst, src mir.VReg
	w        width
}

// edgeCopies pairs every argument a branch supplies with the block parameter it lands in.
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
		return nil, fmt.Errorf("branch to @%s: %d arguments for %d parameters",
			blk.Label(), len(args)+extra, len(params))
	}

	moves := make([]copyPair, len(args))
	for i, arg := range args {
		src, ok := vr.lookup(arg)
		if !ok {
			return nil, fmt.Errorf("branch to @%s: argument %d defined outside the function", blk.Label(), i)
		}
		dst, ok := vr.lookup(params[i])
		if !ok {
			return nil, fmt.Errorf("branch to @%s: parameter %q was not classified", blk.Label(), params[i].Name())
		}
		moves[i] = copyPair{dst: dst, src: src, w: vr.widthOfVReg(dst)}
	}
	return moves, nil
}

// emitParallelCopy sequentializes one edge's moves.
// A branch's arguments are a simultaneous assignment: every parameter
// takes the value its argument had before the branch.
func emitParallelCopy(e emitter, moves []copyPair) {
	pending := live(moves)

	for len(pending) > 0 {
		if i := readyMove(pending); i >= 0 {
			mv := pending[i]
			emitCopy(e, mv.dst, mv.src, mv.w)
			pending = append(pending[:i], pending[i+1:]...)
			continue
		}

		mv := pending[0]
		e.Emit(mir.Instr{
			Op:   swapOp{w: mv.w},
			Defs: []mir.VReg{mv.dst, mv.src},
			Uses: []mir.VReg{mv.dst, mv.src},
		})
		pending = pending[1:]
		for i := range pending {
			if pending[i].src == mv.dst {
				pending[i].src = mv.src
			}
		}
		pending = live(pending)
	}
}

// live drops the moves that have nothing to do.
func live(moves []copyPair) []copyPair {
	out := make([]copyPair, 0, len(moves))
	for _, mv := range moves {
		if mv.dst != mv.src {
			out = append(out, mv)
		}
	}
	return out
}

// readyMove returns the index of a move whose destination no other pending move still has to read.
func readyMove(pending []copyPair) int {
	for i, mv := range pending {
		blocked := false
		for j, other := range pending {
			if i != j && other.src == mv.dst {
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

// iselReturn lowers the block's one other allowed terminator: a return
// of the values the ABI brings back in registers.
func iselReturn(fn *ir.Func, c *cursor, vr *vregs, term *ir.Inst) error {
	if term == nil || term.Op().Verb != ir.VReturn {
		return fmt.Errorf("only a bare return, a trap, a fused brif, or a br terminator is supported")
	}
	args := term.Args()
	abi := fn.Module().Layout().ABI
	places, err := classifyRet(abi, typesOf(args))
	if err != nil {
		return fmt.Errorf("return: %w", err)
	}

	// A call's arguments read backwards: each value copied into a vreg
	// pinned to the register the ABI names, all of them used by the
	// return so the copies cannot be sunk past it or dropped. Two pinned
	// vregs are two registers the allocator keeps apart, so returning a
	// permutation becomes its swap rather than a lost value. Here rather
	// than in emit because returnOp could only carry one width and one
	// register file, and these copies carry their own.
	uses := make([]mir.VReg, 0, len(places))
	for i, pl := range places {
		v, ok := vr.lookup(args[i])
		if !ok {
			return fmt.Errorf("return: operand %d defined outside the function", i)
		}
		slot := pl.regs[0]
		var dst mir.VReg
		if slot.kind == placeFloat {
			dst = vr.physicalXmm(floatRetReg(abi, slot.i), slot.w)
		} else {
			dst = vr.physical(intRetReg(abi, slot.i), slot.w)
		}
		emitCopy(c, dst, v, slot.w)
		uses = append(uses, dst)
	}

	// §3.2.3's other half of sret, in the two shapes it comes in.
	//
	// Only when the signature declares no result of its own. A function
	// that returns something as well has already claimed RAX above, and
	// what it put there is what it said it returns.
	if len(places) == 0 {
		if p, ok := sretParam(fn); ok {
			v, found := vr.lookup(p)
			if !found {
				return fmt.Errorf("return: the sret parameter is defined outside the function")
			}
			agg, inRegs, err := sretRegs(abi, sretParamType(fn))
			if err != nil {
				return err
			}
			if inRegs {
				// The result is small enough to come back in registers, so
				// the parameter names a slot of this function's own and the
				// body has written the value into it. Each eightbyte is one
				// load out of that slot, into the register its class named.
				for k, slot := range sretSlots(abi, agg) {
					var dst mir.VReg
					if slot.kind == placeFloat {
						dst = vr.physicalXmm(floatRetReg(abi, slot.i), slot.w)
					} else {
						dst = vr.physical(intRetReg(abi, slot.i), slot.w)
					}
					c.Emit(mir.Instr{
						Op:   loadAtOp{off: int32(k * 8), w: slot.w},
						Defs: []mir.VReg{dst},
						Uses: []mir.VReg{v},
					})
					uses = append(uses, dst)
				}
			} else {
				// MEMORY: the result was written through a hidden pointer
				// the caller passed in RDI, and that pointer comes back in
				// RAX. The IR models it as an ordinary first parameter, so
				// passing it already happens — this is the part that does
				// not.
				dst := vr.physical(intRetReg(abi, 0), w64)
				emitCopy(c, dst, v, w64)
				uses = append(uses, dst)
			}
		}
	}

	c.Emit(mir.Instr{Op: returnOp{}, Uses: uses})
	return nil
}

// sretParam is fn's sret parameter, if it has one.
//
// §19.13 admits sret on at most one parameter and that parameter is the
// first, so this looks at the first and no further.
func sretParam(fn *ir.Func) (*ir.Def, bool) {
	sig := fn.Signature()
	if sig == nil || len(sig.Params()) == 0 || len(fn.Params()) == 0 {
		return nil, false
	}
	for _, a := range sig.Params()[0].Attrs {
		if a.IsSRet() {
			return fn.Params()[0], true
		}
	}
	return nil, false
}

func iselBrInd(fn *ir.Func, mf *mir.Func, c *cursor, vr *vregs, term *ir.Inst) error {
	ptr, ok := vr.lookup(term.Arg(0))
	if !ok {
		return fmt.Errorf("brind: pointer defined outside the function")
	}

	for _, b := range term.Labels() {
		mf.Succ(c.blk, blockLabel(fn, b))
	}

	target := vr.temp(w64)
	emitCopy(c, target, ptr, w64)
	c.Emit(mir.Instr{Op: jmpIndOp{}, Uses: []mir.VReg{target}})
	return nil
}

func iselBrTable(fn *ir.Func, mf *mir.Func, c *cursor, vr *vregs, term *ir.Inst) error {
	selector, ok := vr.lookup(term.Arg(0))
	if !ok {
		return fmt.Errorf("br_table: selector defined outside the function")
	}

	targets := term.Targets()
	if len(targets) == 0 {
		return fmt.Errorf("br_table: no targets")
	}

	targetLabels := make([]string, len(targets)-1)
	for i := 0; i < len(targets)-1; i++ {
		moves, err := edgeCopies(vr, targets[i])
		if err != nil {
			return err
		}
		if len(moves) > 0 {
			edge := c.open(fmt.Sprintf("table_edge_%d", i))
			c_edge := newCursor(c.fn, mf, edge, c.prefix)
			emitParallelCopy(c_edge, moves)
			c_edge.to(mf.Block(blockLabel(fn, targets[i].Block())))
			targetLabels[i] = edge.Label
			mf.Succ(c.blk, edge.Label)
		} else {
			lbl := blockLabel(fn, targets[i].Block())
			targetLabels[i] = lbl
			mf.Succ(c.blk, lbl)
		}
	}

	defTarget := targets[len(targets)-1]
	defMoves, err := edgeCopies(vr, defTarget)
	if err != nil {
		return err
	}
	var defLabel string
	if len(defMoves) > 0 {
		edge := c.open("table_edge_def")
		c_edge := newCursor(c.fn, mf, edge, c.prefix)
		emitParallelCopy(c_edge, defMoves)
		c_edge.to(mf.Block(blockLabel(fn, defTarget.Block())))
		defLabel = edge.Label
		mf.Succ(c.blk, edge.Label)
	} else {
		defLabel = blockLabel(fn, defTarget.Block())
		mf.Succ(c.blk, defLabel)
	}

	id := fmt.Sprintf("%s.table", c.blk.Label)
	w := vr.widthOfVReg(selector)
	selReg := vr.temp(w)
	emitCopy(c, selReg, selector, w)

	// The scratch register the lookup needs, named as a destination so
	// the allocator knows it is written. A destination interferes with
	// everything live after its instruction — after a terminator, with
	// everything live into the successors — which is what keeps a value
	// they still want out of the register this sequence destroys, and
	// keeps it off the selector the jump reads afterwards.
	base := vr.temp(w64)

	c.Emit(mir.Instr{
		Op: brTableOp{
			id:            id,
			targets:       targetLabels,
			defaultTarget: defLabel,
		},
		Defs: []mir.VReg{base},
		Uses: []mir.VReg{selReg},
	})
	return nil
}
