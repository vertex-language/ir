package ir

// CFG shape is traversal, so it lives here rather than in an analysis package:
// predecessors are computed, not stored, which is what block parameters buy in
// place of predecessor-indexed operand lists.

// Walk visits every module item in declaration order. It stops early if fn
// returns false.
func (m *Module) Walk(fn func(Item) bool) {
	for _, it := range m.items {
		if !fn(it) {
			return
		}
	}
}

// WalkFuncs visits every function definition in declaration order.
func (m *Module) WalkFuncs(fn func(*Func) bool) {
	for _, it := range m.items {
		if f, ok := it.(*Func); ok {
			if !fn(f) {
				return
			}
		}
	}
}

// WalkInsts visits every instruction of every block in declaration order,
// terminators included and last within their block.
func (f *Func) WalkInsts(fn func(*Inst) bool) {
	for _, b := range f.blocks {
		for _, in := range b.insts {
			if !fn(in) {
				return
			}
		}
		if b.term != nil && !fn(b.term) {
			return
		}
	}
}

// Insts returns every instruction of the block in order, the terminator last.
func (b *Block) All() []*Inst {
	out := make([]*Inst, 0, len(b.insts)+1)
	out = append(out, b.insts...)
	if b.term != nil {
		out = append(out, b.term)
	}
	return out
}

// WalkDefs visits every definition in the function: signature parameters, block
// parameters, then instruction results, in block order.
func (f *Func) WalkDefs(fn func(*Def) bool) {
	for _, d := range f.params {
		if !fn(d) {
			return
		}
	}
	for _, b := range f.blocks {
		for _, d := range b.params {
			if !fn(d) {
				return
			}
		}
		for _, in := range b.All() {
			for _, d := range in.results {
				if !fn(d) {
					return
				}
			}
		}
	}
}

// Succs returns the blocks control may reach from this one: branch targets, an
// invoke's normal and unwind edges, a brind's or asm goto's label list.
// Duplicates are preserved, since a brif to the same block twice is two edges.
func (b *Block) Succs() []*Block {
	if b.term == nil {
		return nil
	}
	return b.term.Succs()
}

// Succs returns the blocks a terminator may transfer control to.
func (in *Inst) Succs() []*Block {
	if !in.op.IsTerminator() || in.im == nil {
		return nil
	}
	var out []*Block
	for _, t := range in.im.targets {
		if t.blk != nil {
			out = append(out, t.blk)
		}
	}
	out = append(out, in.im.labels...)
	if in.im.unwind != nil {
		out = append(out, in.im.unwind)
	}
	return out
}

// Preds computes the function's predecessor map. It is computed rather than
// stored, so it is a snapshot: a rewrite invalidates it.
func (f *Func) Preds() map[*Block][]*Block {
	preds := make(map[*Block][]*Block, len(f.blocks))
	for _, b := range f.blocks {
		for _, s := range b.Succs() {
			preds[s] = append(preds[s], b)
		}
	}
	return preds
}

// Preds computes this block's predecessors. Use Func.Preds when asking about
// more than one block.
func (b *Block) Preds() []*Block {
	var out []*Block
	for _, p := range b.fn.blocks {
		for _, s := range p.Succs() {
			if s == b {
				out = append(out, p)
				break
			}
		}
	}
	return out
}

// RPO returns the reachable blocks in reverse postorder from the entry block.
// Unreachable blocks are omitted; only the entry block may be
// unreachable-from-nothing, and finding others is §19.2's business.
func (f *Func) RPO() []*Block {
	if len(f.blocks) == 0 {
		return nil
	}
	seen := make(map[*Block]bool, len(f.blocks))
	var post []*Block
	var visit func(*Block)
	visit = func(b *Block) {
		if b == nil || seen[b] {
			return
		}
		seen[b] = true
		for _, s := range b.Succs() {
			visit(s)
		}
		post = append(post, b)
	}
	visit(f.blocks[0])
	for i, j := 0, len(post)-1; i < j; i, j = i+1, j-1 {
		post[i], post[j] = post[j], post[i]
	}
	return post
}

// A Use is one operand slot referencing a definition. Target is -1 for a plain
// register operand, and otherwise indexes the terminator's target list, with
// Index naming the argument within that target's argument list.
type Use struct {
	Inst   *Inst
	Target int
	Index  int
}

// Def returns the definition the slot currently holds.
func (u Use) Def() *Def {
	if u.Inst == nil {
		return nil
	}
	if u.Target < 0 {
		return u.Inst.Arg(u.Index)
	}
	ts := u.Inst.Targets()
	if u.Target >= len(ts) || u.Index >= len(ts[u.Target].args) {
		return nil
	}
	return ts[u.Target].args[u.Index]
}

// Set rewrites the slot.
func (u Use) Set(d *Def) {
	if u.Inst == nil {
		return
	}
	if u.Target < 0 {
		if u.Index < len(u.Inst.args) {
			u.Inst.args[u.Index] = d
		}
		return
	}
	if u.Inst.im == nil || u.Target >= len(u.Inst.im.targets) {
		return
	}
	t := u.Inst.im.targets[u.Target]
	if u.Index < len(t.args) {
		t.args[u.Index] = d
	}
}

// WalkUses visits every operand slot of the function, branch argument lists
// included. Branch arguments are operands like any other; a pass that forgets
// them is a pass that misses the edges where block parameters are supplied.
func (f *Func) WalkUses(fn func(Use) bool) {
	for _, b := range f.blocks {
		for _, in := range b.All() {
			for i := range in.args {
				if !fn(Use{Inst: in, Target: -1, Index: i}) {
					return
				}
			}
			for ti, t := range in.Targets() {
				for ai := range t.args {
					if !fn(Use{Inst: in, Target: ti, Index: ai}) {
						return
					}
				}
			}
		}
	}
}

// Uses returns every slot referencing d.
func (f *Func) Uses(d *Def) []Use {
	var out []Use
	f.WalkUses(func(u Use) bool {
		if u.Def() == d {
			out = append(out, u)
		}
		return true
	})
	return out
}

// ReplaceUses rewrites every reference to old so that it names new, and reports
// how many slots changed. It does not touch the definition of old, which stays
// where it is until a caller removes it.
func (f *Func) ReplaceUses(old, new *Def) int {
	n := 0
	f.WalkUses(func(u Use) bool {
		if u.Def() == old {
			u.Set(new)
			n++
		}
		return true
	})
	return n
}

// SetArg rewrites one register operand.
func (in *Inst) SetArg(i int, d *Def) {
	if in != nil && i >= 0 && i < len(in.args) {
		in.args[i] = d
	}
}

// Remove deletes a non-terminator instruction from its block and reports whether
// it was found. Uses of its results are the caller's to fix first.
func (b *Block) Remove(in *Inst) bool {
	for i, x := range b.insts {
		if x == in {
			b.insts = append(b.insts[:i], b.insts[i+1:]...)
			in.blk = nil
			return true
		}
	}
	return false
}

// RemoveBlock deletes b from f and reports whether it was found.
//
// The entry block is not removable: a function without one is not a function.
// Nothing else is checked, because the case this exists for is a set of
// blocks that reach each other and nothing else reaches — a loop after a
// return — where checking each one for predecessors would refuse the whole
// set. What still branches to b, and what still uses the values b defined,
// are the caller's to have settled first; verify.Module is what says whether
// they did.
func (f *Func) RemoveBlock(b *Block) bool {
	if b == nil || b == f.entry {
		return false
	}
	for i, x := range f.blocks {
		if x != b {
			continue
		}
		f.blocks = append(f.blocks[:i], f.blocks[i+1:]...)
		if f.labels[b.label] == b {
			delete(f.labels, b.label)
		}
		// Index is a position in this slice, so every block after the
		// hole moved. Nothing here caches it, and a stale one would be
		// a silent aliasing of two blocks. b keeps its function: a
		// removed block that something still holds is an orphan, not a
		// crash waiting for the next Emit.
		for j := i; j < len(f.blocks); j++ {
			f.blocks[j].index = j
		}
		return true
	}
	return false
}

// InsertBefore moves in ahead of mark in mark's block. Both must be
// non-terminators of the same function.
func (b *Block) InsertBefore(mark, in *Inst) bool {
	for i, x := range b.insts {
		if x == mark {
			b.insts = append(b.insts, nil)
			copy(b.insts[i+1:], b.insts[i:])
			b.insts[i] = in
			in.blk = b
			return true
		}
	}
	return false
}

// SetTerm replaces the block's terminator. A pass that reshapes control flow
// rewrites this and lets Preds and RPO recompute.
func (b *Block) SetTerm(in *Inst) {
	if in != nil && !in.op.IsTerminator() {
		return
	}
	b.term = in
	if in != nil {
		in.blk = b
	}
}
