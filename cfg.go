package ir

// MergeBlocks straightens CFG control flow by iteratively applying two rules:
//   - A block ending in an unconditional branch to a single-predecessor successor
//     absorbs that successor's instructions and terminator.
//   - An empty trampoline block that only forwards a branch is bypassed by predecessors.
// Entry blocks and landing pads are preserved.

// MergeBlocks straightens f's control flow and reports how many blocks it
// removed.
func (f *Func) MergeBlocks() int {
	if f == nil || len(f.blocks) == 0 || f.m == nil || f.m.err != nil {
		return 0
	}
	removed := 0
	for {
		n := f.mergeOnce()
		if n == 0 {
			return removed
		}
		removed += n
	}
}

func (f *Func) mergeOnce() int {
	preds := f.Preds()
	for _, b := range f.blocks {
		t := b.term
		if t == nil || t.op.Verb != VBr || t.im == nil || len(t.im.targets) != 1 {
			continue
		}
		s := t.im.targets[0].blk
		if s == nil || s == b || s.isEntry || s.isPad || len(preds[s]) != 1 {
			continue
		}
		args := t.im.targets[0].args
		if len(args) != len(s.params) {
			continue
		}
		repl := make(map[*Def]*Def, len(s.params))
		for i, p := range s.params {
			repl[p] = args[i]
		}
		for _, in := range s.insts {
			in.move(b)
		}
		b.insts = append(b.insts, s.insts...)
		b.term = s.term
		if b.term != nil {
			b.term.move(b)
		}
		s.insts = nil
		s.term = nil
		f.RemoveBlock(s)
		f.replaceDefs(repl)
		return 1
	}
	for _, e := range f.blocks {
		if e.isEntry || e.isPad || len(e.insts) != 0 || e.term == nil || e.term.op.Verb != VBr ||
			e.term.im == nil || len(e.term.im.targets) != 1 {
			continue
		}
		to := e.term.im.targets[0]
		if to.blk == e || len(preds[e]) == 0 || !f.paramsOnlyForwarded(e) {
			continue
		}
		redirected := false
		for _, p := range preds[e] {
			pt := p.term
			if pt == nil || pt.im == nil {
				continue
			}
			switch pt.op.Verb {
			case VBr, VBrIf:
			default:
				// A table, an invoke or an asm goto: their edges are
				// left as they are.
				continue
			}
			for i, edge := range pt.im.targets {
				if edge.blk != e || len(edge.args) != len(e.params) {
					continue
				}
				pt.im.targets[i] = forwardTarget(to, e.params, edge.args)
				redirected = true
			}
		}
		if !redirected {
			continue
		}
		if len(f.Preds()[e]) == 0 {
			f.RemoveBlock(e)
		}
		f.collapseBrIfs()
		return 1
	}
	return 0
}

// paramsOnlyForwarded reports whether e's parameters are read nowhere but
// as arguments of e's own terminator.
func (f *Func) paramsOnlyForwarded(e *Block) bool {
	if len(e.params) == 0 {
		return true
	}
	own := map[*Def]bool{}
	for _, p := range e.params {
		own[p] = true
	}
	ok := true
	f.WalkUses(func(u Use) bool {
		if d := u.Def(); d != nil && own[d] && u.Inst != e.term {
			ok = false
			return false
		}
		return true
	})
	return ok
}

// forwardTarget is the edge to `to` that passes what an edge into a
// skipped block with parameters params, given args, would have passed on.
func forwardTarget(to BlockTarget, params, args []*Def) BlockTarget {
	out := BlockTarget{blk: to.blk}
	if len(to.args) == 0 {
		out.bare = to.bare
		return out
	}
	out.args = make([]*Def, len(to.args))
	for i, a := range to.args {
		out.args[i] = a
		for j, p := range params {
			if a == p {
				out.args[i] = args[j]
				break
			}
		}
	}
	return out
}

// collapseBrIfs turns a brif whose two edges are the same edge into a br.
func (f *Func) collapseBrIfs() {
	for _, b := range f.blocks {
		t := b.term
		if t == nil || t.op.Verb != VBrIf || t.im == nil || len(t.im.targets) != 2 {
			continue
		}
		a, c := t.im.targets[0], t.im.targets[1]
		if a.blk != c.blk || len(a.args) != len(c.args) {
			continue
		}
		same := true
		for i := range a.args {
			if a.args[i] != c.args[i] {
				same = false
				break
			}
		}
		if same {
			b.term = &Inst{op: Op{TypeNone, VBr}, blk: b, im: &imm{targets: []BlockTarget{copyTarget(a)}}}
		}
	}
}

// replaceDefs makes every operand that reads a key of repl read its value.
func (f *Func) replaceDefs(repl map[*Def]*Def) {
	if len(repl) == 0 {
		return
	}
	resolve := func(d *Def) *Def {
		for n := 0; n <= len(repl); n++ {
			r, ok := repl[d]
			if !ok {
				return d
			}
			d = r
		}
		return d
	}
	f.WalkUses(func(u Use) bool {
		if d := u.Def(); d != nil {
			if _, ok := repl[d]; ok {
				u.Set(resolve(d))
			}
		}
		return true
	})
}
