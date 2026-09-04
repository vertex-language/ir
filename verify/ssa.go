package verify

import "github.com/vertex-language/ir"

// §19.1 (single assignment and dominance), §19.2 (terminators and
// reachability), and §19.17 (the entry block is nobody's target).
//
// §19.3 — a branch's arguments against its target's parameters — is not
// here. It is a deferred check on the builder, run by ir.Module.Err,
// because a forward branch may name a block whose parameter list is not
// declared yet, and Module has already returned it by the time any of
// this runs.

// terminators is §19.2's first half: every block ends in exactly one
// terminator. It reports whether the function's CFG is worth reading
// further — see checker.function for why a missing terminator stops the
// function rather than degrading the rules after it.
//
// "Exactly one" is only ever checked in the one direction. Two
// terminators in a block is not a shape the builder can produce: it
// refuses any instruction emitted into an already-terminated block with
// ir.ErrTerminated, which is sticky, which Module returned before this
// ran.
func (c *checker) terminators(blocks []*ir.Block) bool {
	ok := true
	for _, b := range blocks {
		if b.Term() == nil {
			ok = false
			c.fail(b, -1, ir.Op{}, ErrTerminator,
				"the block ends after %d instructions with no br, return, trap, or other terminator", len(b.Insts()))
			if c.full() {
				return false
			}
		}
	}
	return ok
}

// reachability is §19.2's second half and §19.17: every block but the
// entry block is reached by some path from it, and the entry block is
// reached by no edge at all.
//
// The two are one walk because they are the same question asked at both
// ends of the graph. A block with no predecessors is dead code the
// builder was still willing to emit — no rule of §19 makes it
// unconstructible, only unsound. The entry block with a predecessor is
// the mirror image: its inputs are the signature's parameter registers,
// so an edge into it would have to supply parameters that no branch
// argument list can name.
func (c *checker) reachability(f *ir.Func, blocks []*ir.Block) {
	entry := entryBlock(blocks)
	reached := make(map[*ir.Block]bool, len(blocks))
	for _, b := range f.RPO() {
		reached[b] = true
	}

	for _, b := range blocks {
		if b == entry || reached[b] {
			continue
		}
		c.fail(b, -1, ir.Op{}, ErrUnreachable,
			"no path from @%s reaches this block", entry.Label())
		if c.full() {
			return
		}
	}

	// entry is never nil here: checker.function returned already if the
	// function had no blocks, and entryBlock falls back to the first one.
	for _, p := range entry.Preds() {
		c.fail(p, instIndex(p, p.Term()), termOp(p), ErrEntryTarget,
			"@%s branches to the entry block @%s", p.Label(), entry.Label())
		if c.full() {
			return
		}
	}

	// ptr.blockaddr does not make an edge, so Preds cannot see it, but
	// §19.17 names it alongside the branches for the same reason: the
	// address is only ever used by a brind, which would enter the entry
	// block at run time exactly as a br would.
	for _, b := range blocks {
		for i, in := range b.All() {
			if in.Op() != (ir.Op{Type: ir.TypePtr, Verb: ir.VBlockAddr}) {
				continue
			}
			for _, l := range in.Labels() {
				if l != entry {
					continue
				}
				c.fail(b, i, in.Op(), ErrEntryTarget,
					"the address of the entry block @%s is taken", entry.Label())
				if c.full() {
					return
				}
			}
		}
	}
}

// dominance is §19.1's surviving half: every use of a definition is
// dominated by it.
//
// The other half — that every register is assigned exactly once — is
// structural rather than checkable. An ir.Def is created by the one thing
// that defines it, a signature parameter, a block parameter, or one
// result of one instruction, and there is no operation that assigns to an
// existing one. A verifier cannot find a second assignment because the
// surface has no way to write one.
func (c *checker) dominance(f *ir.Func, dt *domTree, pos map[*ir.Inst]int) {
	f.WalkUses(func(u ir.Use) bool {
		d := u.Def()
		if d == nil {
			// A zero Value is ir.ErrPoison, sticky, and already
			// returned; a slot that lost its definition to a rewrite is
			// that pass's bug and has no §19 rule to break.
			return true
		}
		if c.dominatesUse(dt, pos, d, u.Inst) {
			return true
		}
		in := u.Inst
		c.fail(in.Block(), pos[in], in.Op(), ErrDominance,
			"%%%s is defined in %s", d, defSite(d))
		return !c.full()
	})
}

// dominatesUse reports whether d's definition dominates the instruction
// using it.
//
// Uses in or of an unreachable block are nobody's business here: §19.2
// already reported the block, and reporting every value that crosses it
// as undominated too would bury the one fault that matters under its
// consequences.
func (c *checker) dominatesUse(dt *domTree, pos map[*ir.Inst]int, d *ir.Def, use *ir.Inst) bool {
	useBlk := use.Block()
	if useBlk == nil || !dt.reachable(useBlk) {
		return true
	}

	defBlk := d.Block()
	if defBlk == nil {
		// A signature parameter. Its home is the function, and the entry
		// block dominates every reachable block by definition.
		return true
	}
	if !dt.reachable(defBlk) {
		return true
	}

	if defBlk != useBlk {
		return dt.dominates(defBlk, useBlk)
	}

	// Within one block, dominance is position: a block parameter precedes
	// every instruction, and an instruction result precedes only what
	// comes after it. The builder cannot emit a use ahead of its
	// definition — an operand is a value the caller already holds — but a
	// pass that moves an instruction can, which is the case this catches.
	if d.IsParam() {
		return true
	}
	def := d.Inst()
	if def == nil {
		return true
	}
	return pos[def] < pos[use]
}

// entryBlock is the block a function begins in. It is blocks[0] in every
// module the builder produces; the flag is what makes that a fact about
// the block rather than about the slice a pass may have reordered.
func entryBlock(blocks []*ir.Block) *ir.Block {
	for _, b := range blocks {
		if b.IsEntry() {
			return b
		}
	}
	if len(blocks) > 0 {
		return blocks[0]
	}
	return nil
}

// instPositions numbers every instruction by its index within its own
// block, terminator last. One map answers both questions dominance asks:
// which of two instructions in a block comes first, and where to point a
// fault.
func instPositions(f *ir.Func) map[*ir.Inst]int {
	pos := map[*ir.Inst]int{}
	for _, b := range f.Blocks() {
		for i, in := range b.All() {
			pos[in] = i
		}
	}
	return pos
}

// instIndex is one instruction's position in a block, or -1 if it is not
// in it.
func instIndex(b *ir.Block, in *ir.Inst) int {
	for i, x := range b.All() {
		if x == in {
			return i
		}
	}
	return -1
}

// termOp is a block's terminator mnemonic, or the zero Op if it has none.
func termOp(b *ir.Block) ir.Op {
	if t := b.Term(); t != nil {
		return t.Op()
	}
	return ir.Op{}
}

// defSite says where a definition lives, for a fault that has to explain
// why a use cannot see it.
func defSite(d *ir.Def) string {
	switch {
	case d.Block() == nil:
		return "the signature of @" + d.Func().Name()
	case d.IsParam():
		return "the parameters of @" + d.Block().Label()
	default:
		return "@" + d.Block().Label()
	}
}
