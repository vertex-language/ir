package ir

// Frame-slot promotion: mem2reg.
//
// A frontend lowers a local variable the simple way -- an alloc in the
// entry block, a store for every assignment, a load for every read -- and
// that is the shape a C++ frontend has to start from, since it cannot know
// until the function is finished whether some later `&x` takes the
// variable's address. Lowered as it stands each read and write is a memory
// access, and a loop counter costs a load and a store per iteration.
//
// When nothing does take the address -- every use of the alloc is a plain
// load from it or a plain store to it, all of the one register type -- the
// slot is only a variable, and this pass makes it one: each load becomes the
// value last stored on the way to it, and a block where differing values
// meet takes the value as a block parameter. This is the textbook
// construction (Cytron et al.): parameters at the iterated dominance
// frontier of the stores, pruned to the blocks the slot is live into, then
// a walk of the dominator tree carrying the current value down it.
//
// What is left alone: a slot any other instruction touches (a call, an
// address computation, a sub-width access, an asm operand); a volatile
// access; an allocation sized by a named type; and a slot that would need a
// parameter on a block that cannot take one -- the entry, a pad, or a block
// some predecessor reaches by an edge with no argument list (brind, asm
// goto's label list, an unwind edge). A function that calls something
// that returns twice is left alone entirely, as LLVM leaves it.
//
// A load with no store on some path to it reads what an uninitialized
// local holds, which is undefined; here it reads zero, and a zeroed alloc
// starts from zero anyway.

// PromoteSlots rewrites every promotable alloc in f to SSA values, and
// reports how many it promoted.
func (f *Func) PromoteSlots() int {
	if f == nil || f.entry == nil || len(f.blocks) == 0 || f.m == nil || f.m.err != nil {
		return 0
	}
	var slots []*Inst
	for _, in := range f.entry.insts {
		if in.op.Verb == VAlloc && in.im != nil && in.im.typ == nil {
			slots = append(slots, in)
		}
	}
	if len(slots) == 0 || f.callsReturnsTwice() {
		return 0
	}
	uses := f.slotUses()
	var g *promoteCFG
	p := &promotion{f: f, repl: map[*Def]*Def{}, zeros: map[RegType]*Def{}}
	n := 0
	for _, alloc := range slots {
		acc, ok := slotAccessesOf(alloc, uses[alloc.results[0]])
		if !ok || !f.m.layout.Admits(acc.typ) {
			continue
		}
		if g == nil {
			g = newPromoteCFG(f)
		}
		if p.promote(g, alloc, acc) {
			n++
		}
	}
	if n > 0 {
		p.finish()
	}
	return n
}

// callsReturnsTwice reports whether f calls a function that returns twice.
// setjmp's second return resumes with the registers it saved, not the ones
// the code between left behind, so a local that was a slot has to stay one.
func (f *Func) callsReturnsTwice() bool {
	found := false
	f.WalkInsts(func(in *Inst) bool {
		if in.im == nil || in.im.callee == nil {
			return true
		}
		switch c := in.im.callee.(type) {
		case *Func:
			found = c.returnsTwice
		case *FuncImport:
			found = c.returnsTwice
		}
		return !found
	})
	return found
}

// slotUses maps each alloc's result to the instructions that use it, in
// one walk of the function.
func (f *Func) slotUses() map[*Def][]Use {
	allocs := map[*Def]bool{}
	for _, in := range f.entry.insts {
		if in.op.Verb == VAlloc {
			allocs[in.results[0]] = true
		}
	}
	out := map[*Def][]Use{}
	f.WalkUses(func(u Use) bool {
		if d := u.Def(); allocs[d] {
			out[d] = append(out[d], u)
		}
		return true
	})
	return out
}

type slotAccess struct {
	typ           RegType
	width         uint64 // bytes of a sub-width slot (store8 and uload8, say); 0 for full width
	loads, stores []*Inst
	of            map[*Inst]bool
}

// accessWidth is the bytes a memory verb moves, and whether a load sign-
// extends them; 0 for a full-width load or store.
func accessWidth(v Verb) (width uint64, signed, load, ok bool) {
	switch v {
	case VLoad:
		return 0, false, true, true
	case VStore:
		return 0, false, false, true
	case VULoad8:
		return 1, false, true, true
	case VULoad16:
		return 2, false, true, true
	case VULoad32:
		return 4, false, true, true
	case VSLoad8:
		return 1, true, true, true
	case VSLoad16:
		return 2, true, true, true
	case VSLoad32:
		return 4, true, true, true
	case VStore8:
		return 1, false, false, true
	case VStore16:
		return 2, false, false, true
	case VStore32:
		return 4, false, false, true
	}
	return 0, false, false, false
}

func slotAccessesOf(alloc *Inst, uses []Use) (slotAccess, bool) {
	acc := slotAccess{of: map[*Inst]bool{}}
	addr := alloc.results[0]
	typed := false
	for _, u := range uses {
		in := u.Inst
		if u.Target >= 0 || in.im != nil && in.im.volatile {
			return acc, false
		}
		width, _, load, ok := accessWidth(in.op.Verb)
		if !ok {
			return acc, false
		}
		switch {
		case load && u.Index == 0:
			acc.loads = append(acc.loads, in)
		case !load && u.Index == 1 && in.args[0] != addr:
			acc.stores = append(acc.stores, in)
		default:
			return acc, false
		}
		// One register type and one width for every access: a slot
		// written as a byte and read as a word is memory, not a value.
		t := in.op.Type
		if typed && (t != acc.typ || width != acc.width) {
			return acc, false
		}
		acc.typ, acc.width, typed = t, width, true
		acc.of[in] = true
	}
	return acc, typed
}

// promoteCFG is the function's reachable CFG with its dominator tree and
// dominance frontiers.
type promoteCFG struct {
	preds map[*Block][]*Block
	rpo   map[*Block]int
	idom  map[*Block]*Block
	kids  map[*Block][]*Block
	df    map[*Block][]*Block
}

func newPromoteCFG(f *Func) *promoteCFG {
	order := f.RPO()
	g := &promoteCFG{
		preds: f.Preds(),
		rpo:   make(map[*Block]int, len(order)),
		idom:  make(map[*Block]*Block, len(order)),
		kids:  map[*Block][]*Block{},
		df:    map[*Block][]*Block{},
	}
	for i, b := range order {
		g.rpo[b] = i
	}
	entry := order[0]
	g.idom[entry] = entry
	intersect := func(a, b *Block) *Block {
		for a != b {
			for g.rpo[a] > g.rpo[b] {
				a = g.idom[a]
			}
			for g.rpo[b] > g.rpo[a] {
				b = g.idom[b]
			}
		}
		return a
	}
	for changed := true; changed; {
		changed = false
		for _, b := range order[1:] {
			var nd *Block
			for _, p := range g.preds[b] {
				if _, ok := g.idom[p]; !ok {
					continue
				}
				if nd == nil {
					nd = p
				} else {
					nd = intersect(nd, p)
				}
			}
			if nd != nil && g.idom[b] != nd {
				g.idom[b] = nd
				changed = true
			}
		}
	}
	for _, b := range order[1:] {
		g.kids[g.idom[b]] = append(g.kids[g.idom[b]], b)
	}
	for _, b := range order {
		var reach []*Block
		for _, p := range g.preds[b] {
			if g.reachable(p) {
				reach = appendBlockOnce(reach, p)
			}
		}
		if len(reach) < 2 {
			continue
		}
		for _, p := range reach {
			for r := p; r != g.idom[b]; r = g.idom[r] {
				g.df[r] = appendBlockOnce(g.df[r], b)
				if r == entry {
					break
				}
			}
		}
	}
	return g
}

func (g *promoteCFG) reachable(b *Block) bool {
	_, ok := g.rpo[b]
	return ok
}

func appendBlockOnce(bs []*Block, b *Block) []*Block {
	for _, x := range bs {
		if x == b {
			return bs
		}
	}
	return append(bs, b)
}

// promotion is the state shared by the slots of one function: what each
// removed load is replaced by, and the zero each type's undefined reads
// take.
type promotion struct {
	f       *Func
	repl    map[*Def]*Def
	zeros   map[RegType]*Def
	dead    []*Inst
	inserts []insertion
}

// zero is a constant zero of type t at the top of the entry block.
func (p *promotion) zero(t RegType) *Def {
	if d, ok := p.zeros[t]; ok {
		return d
	}
	lit := Int(0)
	if t == TypeF32 || t == TypeF64 {
		lit = Float(0)
	}
	in := &Inst{op: Op{t, VConst}, blk: p.f.entry, im: &imm{lit: lit, hasLit: true}}
	d := p.f.newDef(t, "", in, 0)
	in.results = []*Def{d}
	p.f.entry.insts = append([]*Inst{in}, p.f.entry.insts...)
	p.zeros[t] = d
	return d
}

// narrow is what a sub-width load of the slot reads, given the value last
// stored whole: its low bytes, zero- or sign-extended as the load would
// have. The instructions that do it are placed before the load once the
// walk is done (see finish); a full-width load reads the value itself.
func (p *promotion) narrow(b *Block, load *Inst, v *Def, acc slotAccess) *Def {
	if acc.width == 0 {
		return v
	}
	_, signed, _, _ := accessWidth(load.op.Verb)
	t := acc.typ
	bits := int64(32)
	if t == TypeI64 {
		bits = 64
	}
	keep := int64(acc.width) * 8
	mk := func(op Op, args []*Def, im *imm) *Def {
		in := &Inst{op: op, blk: b, args: args, im: im}
		d := p.f.newDef(t, "", in, 0)
		in.results = []*Def{d}
		p.inserts = append(p.inserts, insertion{before: load, in: in})
		return d
	}
	konst := func(c int64) *Def {
		return mk(Op{t, VConst}, nil, &imm{lit: Int(c), hasLit: true})
	}
	if !signed {
		mask := int64(1)<<uint(keep) - 1
		return mk(Op{t, VAnd}, []*Def{v, konst(mask)}, nil)
	}
	sh := konst(bits - keep)
	up := mk(Op{t, VShl}, []*Def{v, sh}, nil)
	return mk(Op{t, VSShr}, []*Def{up, sh}, nil)
}

// isSlotStore reports whether in, an access of a promotable slot, writes
// it -- whole or a low part.
func isSlotStore(in *Inst) bool {
	_, _, load, ok := accessWidth(in.op.Verb)
	return ok && !load
}

// An insertion is an instruction to place before another once the walk
// that made it is done.
type insertion struct {
	before, in *Inst
}

// promote plans the promotion of one slot and carries it out, or reports
// false and changes nothing.
func (p *promotion) promote(g *promoteCFG, alloc *Inst, acc slotAccess) bool {
	for in := range acc.of {
		if !g.reachable(in.blk) {
			return false
		}
	}
	// Liveness: live into a block that loads before it stores, and into
	// every block that passes the slot untouched to a live-in successor.
	upLoad := map[*Block]bool{}
	stores := map[*Block]bool{}
	for _, b := range p.f.blocks {
		first := true
		for _, in := range b.insts {
			if !acc.of[in] {
				continue
			}
			if isSlotStore(in) {
				stores[b] = true
			} else if first {
				upLoad[b] = true
			}
			first = false
		}
	}
	liveIn := map[*Block]bool{}
	var work []*Block
	for b := range upLoad {
		liveIn[b] = true
		work = append(work, b)
	}
	for len(work) > 0 {
		b := work[len(work)-1]
		work = work[:len(work)-1]
		for _, q := range g.preds[b] {
			// A block that stores kills the slot -- unless it loads
			// first, and then it is live-in already.
			if !g.reachable(q) || liveIn[q] || stores[q] {
				continue
			}
			liveIn[q] = true
			work = append(work, q)
		}
	}
	// The iterated dominance frontier of the stores. The entry block
	// counts as storing the initial value.
	isPhi := map[*Block]bool{}
	var phis []*Block
	var defs []*Block
	queued := map[*Block]bool{p.f.entry: true}
	defs = append(defs, p.f.entry)
	for b := range stores {
		if !queued[b] {
			queued[b] = true
			defs = append(defs, b)
		}
	}
	for len(defs) > 0 {
		b := defs[len(defs)-1]
		defs = defs[:len(defs)-1]
		for _, d := range g.df[b] {
			if isPhi[d] || !liveIn[d] {
				continue
			}
			isPhi[d] = true
			phis = append(phis, d)
			if !queued[d] {
				queued[d] = true
				defs = append(defs, d)
			}
		}
	}
	for _, d := range phis {
		if d.isEntry || d.isPad {
			return false
		}
		for _, q := range g.preds[d] {
			if !g.reachable(q) || !passesArgs(q.term, d) {
				return false
			}
		}
	}

	// Carry it out: parameters first, so the walk can name them.
	params := map[*Block]*Def{}
	for _, d := range phis {
		prm := p.f.newDef(acc.typ, "", nil, len(d.params))
		prm.isParam = true
		prm.blk = d
		d.params = append(d.params, prm)
		params[d] = prm
	}
	var walk func(b *Block, cur *Def)
	walk = func(b *Block, cur *Def) {
		if prm, ok := params[b]; ok {
			cur = prm
		}
		for _, in := range b.insts {
			if !acc.of[in] {
				continue
			}
			if isSlotStore(in) {
				cur = in.args[0]
				continue
			}
			if cur == nil {
				cur = p.zero(acc.typ)
			}
			p.repl[in.results[0]] = p.narrow(b, in, cur, acc)
		}
		if b.term != nil && b.term.im != nil {
			for i := range b.term.im.targets {
				t := &b.term.im.targets[i]
				if _, ok := params[t.blk]; !ok {
					continue
				}
				v := cur
				if v == nil {
					v = p.zero(acc.typ)
				}
				t.args = append(t.args, v)
				t.bare = false
			}
		}
		for _, k := range g.kids[b] {
			walk(k, cur)
		}
	}
	var init *Def
	if alloc.im.zeroed {
		init = p.zero(acc.typ)
	}
	walk(p.f.entry, init)
	p.dead = append(p.dead, acc.loads...)
	p.dead = append(p.dead, acc.stores...)
	p.dead = append(p.dead, alloc)
	return true
}

// passesArgs reports whether every edge term has to d carries an argument
// list.
func passesArgs(term *Inst, d *Block) bool {
	if term == nil || term.im == nil {
		return false
	}
	if term.im.unwind == d {
		return false
	}
	for _, l := range term.im.labels {
		if l == d {
			return false
		}
	}
	for _, t := range term.im.targets {
		if t.blk == d {
			return true
		}
	}
	return false
}

// finish rewrites every use of a removed load to what it read -- through
// chains, since a store may have stored another promoted slot's load --
// and deletes the slots and their accesses.
func (p *promotion) finish() {
	for _, x := range p.inserts {
		b := x.before.blk
		for i, in := range b.insts {
			if in == x.before {
				b.insts = append(b.insts[:i], append([]*Inst{x.in}, b.insts[i:]...)...)
				break
			}
		}
	}
	resolve := func(d *Def) *Def {
		for n := 0; n <= len(p.repl); n++ {
			r, ok := p.repl[d]
			if !ok {
				return d
			}
			d = r
		}
		return d
	}
	p.f.WalkUses(func(u Use) bool {
		if d := u.Def(); d != nil {
			if _, ok := p.repl[d]; ok {
				u.Set(resolve(d))
			}
		}
		return true
	})
	for _, in := range p.dead {
		if in.blk != nil {
			in.blk.Remove(in)
		}
	}
}
