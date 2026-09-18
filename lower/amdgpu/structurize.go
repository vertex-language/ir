package amdgpu

// Divergent control flow: the whole wave takes one path, so a branch
// whose condition differs across lanes cannot be a branch. What it can
// be is a mask. Every block gets a predicate — a lane mask saying which
// lanes want to execute it next — and the blocks are laid out in one
// linear order, each behind a flow block that narrows exec to the lanes
// whose predicate is set and skips the block when none is. A block's
// terminator no longer branches: it sets its successors' predicates for
// the lanes that reached it, and falls through to the next flow. A loop
// is the same, with its header's flow entered again from the end of the
// body and left when no lane's predicate remains.
//
// This is LLVM's StructurizeCFG followed by SILowerControlFlow, done on
// MIR rather than on the IR, which is what makes it small: block
// parameters are already copies in edge blocks, so the only values the
// pass invents are the predicates. What it needs from the CFG is
// reducibility — every loop has one header, every entry to it goes
// through that header — and it refuses the rest by name.
//
// # The invariants the lowering rests on
//
// A VGPR keeps its value in a lane that is not active: a value defined
// in a block executes only in that block's lanes, and every later use is
// by one of those lanes. So no merge is ever needed for a vector value;
// a def in a skipped block simply does not happen. The allocator does
// not know that, and would carry a value live-in from the function's
// entry along the skip path, so each guard defines the block's outgoing
// values with an undefOp that emits nothing.
//
// An SGPR pair does not: a scalar write is all sixty-four bits at once.
// So every mask that must survive a region another lane executes — a
// predicate, an i1 block parameter — is written read-modify-write under
// exec, which is also what keeps it live through the region in the
// allocator's eyes. A mask defined and consumed on one path is written
// plainly, since the bits of other lanes are never read.

import (
	"fmt"

	"github.com/vertex-language/ir/lower/mir"
)

// The structurizer's MIR ops, emitted by emit.go.
type (
	// execSaveOp is s_mov_b64 Defs[0], exec.
	execSaveOp struct{}
	// execRestoreOp is s_mov_b64 exec, Uses[0].
	execRestoreOp struct{}
	// execAndOp narrows exec to the lanes whose bit in Uses[0] is set.
	execAndOp struct{}
	// execzOp is s_cbranch_execz to target.
	execzOp struct{ target string }
	// maskInitOp is s_mov_b64 Defs[0], 0.
	maskInitOp struct{}
	// pendClearOp clears the active lanes' bits in the mask Defs[0] =
	// Uses[0]: the lanes entering a block have consumed its predicate.
	pendClearOp struct{}
	// pendSetOp sets the active lanes' bits in Defs[0] = Uses[0] to the
	// mask Uses[1], or its complement when neg.
	pendSetOp struct{ neg bool }
	// pendSetAllOp sets the active lanes' bits in Defs[0] = Uses[0].
	pendSetAllOp struct{}
	// trapAnyOp traps when any lane is active.
	trapAnyOp struct{}
	// undefOp defines Defs[0] and emits nothing: the point at which the
	// allocator should start considering the value live.
	undefOp struct{}
)

// A loop is a natural loop: its header, its blocks, and its place in
// the nest.
type loop struct {
	header *mir.Block
	blocks map[*mir.Block]bool
	parent *loop
}

// An item is one node of the linear order: a block, or a loop whose
// body is its own order.
type item struct {
	blk  *mir.Block
	loop *loop
	body []item // for a loop: the header first
}

type structurizer struct {
	x    *fnState
	mf   *mir.Func
	name string

	rpo    []*mir.Block
	rpoIdx map[*mir.Block]int
	preds  map[*mir.Block][]*mir.Block
	idom   map[*mir.Block]*mir.Block
	loops  []*loop
	loopOf map[*mir.Block]*loop

	pred    map[*mir.Block]mir.VReg // each block's predicate
	inits   map[int][]mir.VReg      // predicates initialised at a top-level position
	undefs  map[*mir.Block][]mir.VReg
	out     []*mir.Block
	nflow   int
	restore mir.VReg // the exec the next flow restores, or -1
	skips   []skip   // execz branches to patch once their target exists
}

// A skip is a flow whose execz goes to the block laid out at position at.
type skip struct {
	f  *mir.Block
	at int
}

// structurize rewrites mf into predicated linear form.
func (x *fnState) structurize() error {
	s := &structurizer{x: x, mf: x.mf, name: x.fn.Name(), restore: -1}
	if err := s.analyze(); err != nil {
		return err
	}
	seq := s.order(nil, s.rpo[0])
	s.plan(seq)
	s.crossings()
	s.gen(seq)
	// The exit: every lane retires here.
	exit := s.flow("exit")
	s.emitRestore(exit)
	exit.Emit(mir.Instr{Op: endpgmOp{}})
	s.out = append(s.out, exit)
	// The fall-throughs and the skips, now that every block has a place.
	for i, b := range s.out {
		if n := len(b.Instrs); n > 0 && i+1 < len(s.out) {
			if br, ok := b.Instrs[n-1].Op.(branchOp); ok && br.target == "" {
				b.Instrs[n-1].Op = branchOp{target: s.out[i+1].Label}
				b.Succs = append(b.Succs, s.out[i+1])
			}
		}
	}
	for _, sk := range s.skips {
		t := s.out[sk.at]
		for i := range sk.f.Instrs {
			if _, ok := sk.f.Instrs[i].Op.(execzOp); ok {
				sk.f.Instrs[i].Op = execzOp{target: t.Label}
			}
		}
		sk.f.Succs = append(sk.f.Succs, t)
	}
	s.mf.Blocks = s.out
	return nil
}

// —— analysis ——

func (s *structurizer) analyze() error {
	entry := s.mf.Blocks[0]
	// Reverse postorder over the reachable blocks.
	seen := map[*mir.Block]bool{}
	var post []*mir.Block
	var dfs func(b *mir.Block)
	dfs = func(b *mir.Block) {
		seen[b] = true
		for _, t := range b.Succs {
			if !seen[t] {
				dfs(t)
			}
		}
		post = append(post, b)
	}
	dfs(entry)
	s.rpoIdx = map[*mir.Block]int{}
	for i := len(post) - 1; i >= 0; i-- {
		s.rpoIdx[post[i]] = len(s.rpo)
		s.rpo = append(s.rpo, post[i])
	}
	s.preds = map[*mir.Block][]*mir.Block{}
	for _, b := range s.rpo {
		for _, t := range b.Succs {
			s.preds[t] = append(s.preds[t], b)
		}
	}

	// Dominators, the iterative way.
	s.idom = map[*mir.Block]*mir.Block{entry: entry}
	intersect := func(a, b *mir.Block) *mir.Block {
		for a != b {
			for s.rpoIdx[a] > s.rpoIdx[b] {
				a = s.idom[a]
			}
			for s.rpoIdx[b] > s.rpoIdx[a] {
				b = s.idom[b]
			}
		}
		return a
	}
	for changed := true; changed; {
		changed = false
		for _, b := range s.rpo[1:] {
			var d *mir.Block
			for _, p := range s.preds[b] {
				if s.idom[p] == nil {
					continue
				}
				if d == nil {
					d = p
				} else {
					d = intersect(d, p)
				}
			}
			if d != nil && s.idom[b] != d {
				s.idom[b] = d
				changed = true
			}
		}
	}
	dominates := func(a, b *mir.Block) bool {
		for {
			if a == b {
				return true
			}
			if b == entry {
				return false
			}
			b = s.idom[b]
		}
	}

	// Back edges, and the natural loop of each; an edge to an earlier
	// block that is not a back edge is what an irreducible CFG has.
	byHeader := map[*mir.Block]*loop{}
	for _, b := range s.rpo {
		for _, t := range b.Succs {
			if s.rpoIdx[t] > s.rpoIdx[b] {
				continue
			}
			if !dominates(t, b) {
				return fmt.Errorf("the control flow is irreducible at %s -> %s: a loop with two entries, which the execution mask cannot follow", b.Label, t.Label)
			}
			l := byHeader[t]
			if l == nil {
				l = &loop{header: t, blocks: map[*mir.Block]bool{t: true}}
				byHeader[t] = l
				s.loops = append(s.loops, l)
			}
			// Everything that reaches the latch without passing the header.
			stack := []*mir.Block{b}
			for len(stack) > 0 {
				n := stack[len(stack)-1]
				stack = stack[:len(stack)-1]
				if l.blocks[n] {
					continue
				}
				l.blocks[n] = true
				stack = append(stack, s.preds[n]...)
			}
		}
	}
	// Nesting: the innermost loop holding each block, and each loop's parent.
	s.loopOf = map[*mir.Block]*loop{}
	for _, b := range s.rpo {
		var inner *loop
		for _, l := range s.loops {
			if l.blocks[b] && (inner == nil || len(l.blocks) < len(inner.blocks)) {
				inner = l
			}
		}
		s.loopOf[b] = inner
	}
	for _, l := range s.loops {
		for _, m := range s.loops {
			if m != l && m.blocks[l.header] && (l.parent == nil || len(m.blocks) < len(l.parent.blocks)) {
				l.parent = m
			}
		}
	}
	return nil
}

// A key is one node of a region's collapsed graph: a block of the
// region, or an inner loop.
type key struct {
	blk  *mir.Block
	loop *loop
}

// order is the linear order of a region — the function, or a loop's
// body — as a reverse postorder over its blocks with each inner loop
// collapsed to one node.
func (s *structurizer) order(in *loop, entry *mir.Block) []item {
	// node is the item a successor edge lands on within this region:
	// the block, the inner loop holding it, or nothing for an edge out.
	node := func(b *mir.Block) (key, bool) {
		l := s.loopOf[b]
		if l == in {
			return key{blk: b}, true
		}
		for l != nil && l.parent != in {
			l = l.parent
		}
		if l == nil || (in != nil && !in.blocks[b]) {
			return key{}, false
		}
		return key{loop: l}, true
	}
	succs := func(k key) []key {
		var out []key
		add := func(b *mir.Block) {
			for _, t := range b.Succs {
				if in != nil && t == in.header {
					continue // the region's own back edge
				}
				if n, ok := node(t); ok && n != k {
					out = append(out, n)
				}
			}
		}
		if k.blk != nil {
			add(k.blk)
		} else {
			for b := range k.loop.blocks {
				add(b)
			}
		}
		return out
	}
	start, _ := node(entry)
	seen := map[key]bool{}
	var post []key
	var dfs func(k key)
	dfs = func(k key) {
		seen[k] = true
		// In reverse postorder of the original graph, for a stable order.
		ss := succs(k)
		at := func(k key) int {
			if k.blk != nil {
				return s.rpoIdx[k.blk]
			}
			return s.rpoIdx[k.loop.header]
		}
		for i := 1; i < len(ss); i++ {
			for j := i; j > 0 && at(ss[j]) < at(ss[j-1]); j-- {
				ss[j], ss[j-1] = ss[j-1], ss[j]
			}
		}
		for _, t := range ss {
			if !seen[t] {
				dfs(t)
			}
		}
		post = append(post, k)
	}
	dfs(start)
	var items []item
	for i := len(post) - 1; i >= 0; i-- {
		k := post[i]
		if k.blk != nil {
			items = append(items, item{blk: k.blk})
		} else {
			items = append(items, item{loop: k.loop, body: s.order(k.loop, k.loop.header)})
		}
	}
	return items
}

// plan gives every block a predicate and decides where each is
// initialised: at the top-level position of its earliest setter, so
// that it is live from there to the block's flow and no longer.
func (s *structurizer) plan(seq []item) {
	s.pred = map[*mir.Block]mir.VReg{}
	for _, b := range s.rpo {
		if len(s.preds[b]) > 0 {
			s.pred[b] = s.x.vr.temp(s64)
		}
	}
	top := map[*mir.Block]int{}
	var walk func(items []item, at int)
	walk = func(items []item, at int) {
		for i, it := range items {
			pos := at
			if at < 0 {
				pos = i
			}
			if it.blk != nil {
				top[it.blk] = pos
			} else {
				walk(it.body, pos)
			}
		}
	}
	walk(seq, -1)
	s.inits = map[int][]mir.VReg{}
	for _, b := range s.rpo {
		p, ok := s.pred[b]
		if !ok {
			continue
		}
		first := -1
		for _, u := range s.preds[b] {
			if first < 0 || top[u] < first {
				first = top[u]
			}
		}
		s.inits[first] = append(s.inits[first], p)
	}
}

// crossings finds the values a block defines that another block uses.
func (s *structurizer) crossings() {
	defIn := map[mir.VReg]map[*mir.Block]bool{}
	useIn := map[mir.VReg]map[*mir.Block]bool{}
	for _, b := range s.rpo {
		for _, in := range b.Instrs {
			for _, v := range in.Defs {
				if defIn[v] == nil {
					defIn[v] = map[*mir.Block]bool{}
				}
				defIn[v][b] = true
			}
			for _, v := range in.Uses {
				if useIn[v] == nil {
					useIn[v] = map[*mir.Block]bool{}
				}
				useIn[v][b] = true
			}
		}
	}
	s.undefs = map[*mir.Block][]mir.VReg{}
	for v, defs := range defIn {
		crosses := false
		for u := range useIn[v] {
			if !defs[u] {
				crosses = true
				break
			}
		}
		if !crosses && len(defs) == 1 {
			continue
		}
		for b := range defs {
			s.undefs[b] = append(s.undefs[b], v)
		}
	}
}

// —— generation ——

func (s *structurizer) flow(kind string) *mir.Block {
	s.nflow++
	return s.mf.NewBlock(fmt.Sprintf("%s.%s%d", s.name, kind, s.nflow))
}

func (s *structurizer) emitRestore(b *mir.Block) {
	if s.restore >= 0 {
		b.Emit(mir.Instr{Op: execRestoreOp{}, Uses: rs(s.restore)})
		s.restore = -1
	}
}

func (s *structurizer) emitInits(b *mir.Block, pos int) {
	for _, p := range s.inits[pos] {
		b.Emit(mir.Instr{Op: maskInitOp{}, Defs: rs(p)})
	}
	delete(s.inits, pos)
}

func (s *structurizer) emitUndefs(f, b *mir.Block) {
	for _, v := range s.undefs[b] {
		f.Emit(mir.Instr{Op: undefOp{}, Defs: rs(v)})
	}
}

// gen lays the sequence out. Inits go at top-level positions only: a
// flow inside a loop is skipped with the loop, and a predicate whose
// init was skipped is garbage.
func (s *structurizer) gen(seq []item) {
	s.genItems(seq, true)
}

func (s *structurizer) genItems(items []item, topLevel bool) {
	entry := s.rpo[0]
	for i, it := range items {
		pos := -1
		if topLevel {
			pos = i
		}
		if it.blk != nil {
			b := it.blk
			f := s.flow("flow")
			s.emitRestore(f)
			if topLevel {
				s.emitInits(f, pos)
			}
			if b == entry {
				// Unguarded: every lane executes the entry.
				f.Emit(mir.Instr{Op: branchOp{target: b.Label}})
				f.Succs = []*mir.Block{b}
				s.out = append(s.out, f, b)
				s.rewrite(b)
				continue
			}
			p := s.pred[b]
			save := s.x.vr.temp(s64)
			s.emitUndefs(f, b)
			f.Emit(mir.Instr{Op: execSaveOp{}, Defs: rs(save)})
			f.Emit(mir.Instr{Op: execAndOp{}, Uses: rs(p)})
			f.Emit(mir.Instr{Op: pendClearOp{}, Defs: rs(p), Uses: rs(p)})
			f.Emit(mir.Instr{Op: execzOp{}}) // target patched at the end
			f.Succs = []*mir.Block{b}
			s.out = append(s.out, f, b)
			s.rewrite(b)
			s.patchSkip(f, len(s.out)) // the block after b
			s.restore = save
			continue
		}
		// A loop: its pre-flow saves exec once, its header's flow narrows
		// exec each time round and leaves when nothing is left.
		l := it.loop
		h := l.header
		pre := s.flow("pre")
		s.emitRestore(pre)
		if topLevel {
			s.emitInits(pre, pos)
		}
		saveL := s.x.vr.temp(s64)
		pre.Emit(mir.Instr{Op: execSaveOp{}, Defs: rs(saveL)})
		p := s.pred[h]
		if h == entry {
			pre.Emit(mir.Instr{Op: pendSetAllOp{}, Defs: rs(p), Uses: rs(p)})
		}
		head := s.flow("loop")
		pre.Emit(mir.Instr{Op: branchOp{target: head.Label}})
		pre.Succs = []*mir.Block{head}
		s.emitUndefs(head, h)
		head.Emit(mir.Instr{Op: execAndOp{}, Uses: rs(p)})
		head.Emit(mir.Instr{Op: pendClearOp{}, Defs: rs(p), Uses: rs(p)})
		head.Emit(mir.Instr{Op: execzOp{}})
		head.Succs = []*mir.Block{h}
		s.out = append(s.out, pre, head, h)
		s.rewrite(h)
		s.restore = -1
		s.genItems(it.body[1:], false)
		back := s.flow("back")
		s.emitRestore(back)
		back.Emit(mir.Instr{Op: branchOp{target: head.Label}})
		back.Succs = []*mir.Block{head}
		s.out = append(s.out, back)
		s.patchSkip(head, len(s.out))
		s.restore = saveL
	}
}

// patchSkip points a flow's execz at the block that will be laid out at
// position at, once it exists.
func (s *structurizer) patchSkip(f *mir.Block, at int) {
	s.skips = append(s.skips, skip{f: f, at: at})
}

// rewrite replaces a block's terminator with predicate sets and a fall
// through to whatever follows it.
func (s *structurizer) rewrite(b *mir.Block) {
	n := len(b.Instrs)
	if n == 0 {
		return
	}
	term := b.Instrs[n-1]
	body := b.Instrs[:n-1]
	var out []mir.Instr
	switch op := term.Op.(type) {
	case branchOp:
		t := s.mf.Block(op.target)
		p := s.pred[t]
		out = append(out, mir.Instr{Op: pendSetAllOp{}, Defs: rs(p), Uses: rs(p)})
	case cbranchOp:
		m := term.Uses[0]
		pt, pe := s.pred[s.mf.Block(op.then)], s.pred[s.mf.Block(op.els)]
		out = append(out,
			mir.Instr{Op: pendSetOp{}, Defs: rs(pt), Uses: rs(pt, m)},
			mir.Instr{Op: pendSetOp{neg: true}, Defs: rs(pe), Uses: rs(pe, m)})
	case endpgmOp:
	case trapOp:
		out = append(out, mir.Instr{Op: trapAnyOp{}})
	default:
		// Not a terminator: a block isel left unterminated cannot happen,
		// but keep the instruction rather than lose it.
		body = b.Instrs
	}
	b.Instrs = append(append(body[:len(body):len(body)], out...), mir.Instr{Op: branchOp{target: ""}})
	b.Succs = nil
}
