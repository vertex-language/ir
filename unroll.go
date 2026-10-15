package ir

import "fmt"

// Unrolling a loop that runs a known, small number of times.
//
// A kernel's innermost loops are often of this kind -- four rows of a
// block, eight quants of a lane -- and a GPU wants them straight: the
// loads of every iteration issued together, rather than one iteration's
// waiting on the last one's branch. A CPU compiler unrolls them itself;
// Apple's, reading AIR, does not, and a q6_K matrix-vector kernel ran 20%
// faster with its loop of four written out.
//
// UnrollConstantLoops takes the loop of this shape, the one an inliner and
// MergeBlocks leave for `var i = c0; while i < n { ...; i = i &+ s }`:
//
//	@pre:              ... br @h(c0, a...)
//	@h(%i, %a...):     [pure instructions]; %t = cmp %i, n; brif %t, @body, @exit(...)
//	@body:             ...; br @h(%i + s, b...)
//
// where c0, n and s are constants, @body has no other predecessor and @h
// no others than @pre and @body. Its iterations are counted by running
// the counter, and when there are at most maxTrips of them and the copies
// stay under maxInsts instructions, @h and @body are replaced by one block
// of them in order, which then branches to @exit as the last test would.

// The bounds on what is unrolled: iterations, and the instructions of
// the copies -- this IR's, in which every constant is one.
const (
	maxTrips = 16
	maxInsts = 2400
)

// UnrollConstantLoops unrolls f's small constant loops and reports how many
// it unrolled.
func (f *Func) UnrollConstantLoops() int {
	if f == nil || len(f.blocks) == 0 || f.m == nil || f.m.err != nil {
		return 0
	}
	n := 0
	for f.unrollOne() {
		n++
		if f.m.err != nil {
			return n
		}
	}
	return n
}

func (f *Func) unrollOne() bool {
	preds := f.Preds()
	for _, h := range f.blocks {
		if f.tryUnroll(h, preds) {
			return true
		}
	}
	return false
}

func (f *Func) tryUnroll(h *Block, preds map[*Block][]*Block) bool {
	if h.isEntry || h.isPad || h.term == nil || h.term.op.Verb != VBrIf || h.term.im == nil || len(h.term.im.targets) != 2 {
		return false
	}
	body, exit := h.term.im.targets[0], h.term.im.targets[1]
	b := body.blk
	if b == nil || b == h || len(body.args) != 0 || len(b.params) != 0 || exit.blk == h || exit.blk == b {
		return false
	}
	if len(preds[b]) != 1 || len(preds[h]) != 2 {
		return false
	}
	var pre *Block
	for _, p := range preds[h] {
		if p != b {
			pre = p
		}
	}
	if pre == nil || pre == h || b.term == nil || b.term.op.Verb != VBr || b.term.im == nil || len(b.term.im.targets) != 1 ||
		b.term.im.targets[0].blk != h || pre.term == nil || pre.term.op.Verb != VBr || pre.term.im == nil || len(pre.term.im.targets) != 1 {
		return false
	}
	back := b.term.im.targets[0].args
	init := pre.term.im.targets[0].args
	if len(back) != len(h.params) || len(init) != len(h.params) {
		return false
	}
	// The header computes the test and, before it, only pure values.
	for _, in := range h.insts {
		if !pureVerb(in.op.Verb) || traps(in.op.Verb) || len(in.results) != 1 {
			return false
		}
	}
	// The test: a compare of a header parameter with a constant.
	t := h.term.args[0]
	if t == nil || t.inst == nil || t.inst.blk != h || len(t.inst.args) != 2 {
		return false
	}
	cmp := t.inst
	iv := -1
	for i, p := range h.params {
		if cmp.args[0] == p {
			iv = i
		}
	}
	if iv < 0 {
		return false
	}
	limit, ok := constOf(cmp.args[1])
	if !ok {
		return false
	}
	start, ok := constOf(init[iv])
	if !ok {
		return false
	}
	next := back[iv]
	if next == nil || next.inst == nil || next.inst.op.Verb != VAdd || len(next.inst.args) != 2 || next.inst.args[0] != h.params[iv] {
		return false
	}
	step, ok := constOf(next.inst.args[1])
	if !ok || step <= 0 {
		return false
	}
	holds := func(i int64) (bool, bool) {
		switch cmp.op.Verb {
		case VSLt:
			return i < limit, true
		case VSLe:
			return i <= limit, true
		case VULt:
			return i >= 0 && limit >= 0 && i < limit, true
		case VULe:
			return i >= 0 && limit >= 0 && i <= limit, true
		case VNe:
			return i != limit, true
		}
		return false, false
	}
	trips := 0
	for i := start; ; i += step {
		yes, known := holds(i)
		if !known {
			return false
		}
		if !yes {
			break
		}
		trips++
		if trips > maxTrips {
			return false
		}
	}
	if trips*(len(b.insts)+len(h.insts)) > maxInsts {
		return false
	}
	// Nothing past the loop reads what the body made: only the header's
	// values flow out, and those are what the last test saw.
	inLoop := map[*Def]bool{}
	for _, in := range b.insts {
		for _, r := range in.results {
			inLoop[r] = true
		}
	}
	escapes := false
	f.WalkUses(func(u Use) bool {
		if u.Inst.blk != b && u.Inst.blk != h && inLoop[u.Def()] {
			escapes = true
			return false
		}
		return true
	})
	if escapes {
		return false
	}

	u := f.newBlock(fmt.Sprintf("%s_unrolled_%d", h.label, f.nextID))
	if f.m.err != nil {
		return false
	}
	cur := map[*Def]*Def{}
	for i, p := range h.params {
		cur[p] = init[i]
	}
	lookup := func(d *Def) *Def {
		if r, ok := cur[d]; ok {
			return r
		}
		return d
	}
	copyHeader := func() bool {
		for _, in := range h.insts {
			c := u.Clone(in, lookup, nil)
			if c == nil {
				return false
			}
			cur[in.results[0]] = c.results[0]
		}
		return true
	}
	for n := 0; n < trips; n++ {
		if !copyHeader() {
			return false
		}
		for _, in := range b.insts {
			c := u.Clone(in, lookup, nil)
			if c == nil {
				return false
			}
			for i, r := range in.results {
				cur[r] = c.results[i]
			}
		}
		nextVals := make([]*Def, len(back))
		for i, a := range back {
			nextVals[i] = lookup(a)
		}
		for i, p := range h.params {
			cur[p] = nextVals[i]
		}
	}
	// The values after the loop: the header's parameters as the last test
	// saw them, and its instructions computed once more on them.
	if !copyHeader() || f.m.err != nil {
		return false
	}
	out := BlockTarget{blk: exit.blk, bare: exit.bare}
	if len(exit.args) > 0 {
		out.args = make([]*Def, len(exit.args))
		for i, a := range exit.args {
			out.args[i] = lookup(a)
		}
	}
	u.term = &Inst{op: Op{TypeNone, VBr}, blk: u, im: &imm{targets: []BlockTarget{out}}}
	pre.term.im.targets[0] = BlockTarget{blk: u, bare: true}
	// What follows the loop read the header's values: it reads the last.
	final := map[*Def]*Def{}
	for _, p := range h.params {
		final[p] = cur[p]
	}
	for _, in := range h.insts {
		final[in.results[0]] = cur[in.results[0]]
	}
	h.insts, h.term, b.insts, b.term = nil, nil, nil, nil
	f.RemoveBlock(h)
	f.RemoveBlock(b)
	f.replaceDefs(final)
	return true
}
