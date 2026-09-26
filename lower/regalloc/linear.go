package regalloc

// Linear scan over live intervals: the allocator a debug build wants.
//
// The graph colourer in regalloc.go builds an interference graph whose
// size is the number of simultaneously live pairs, and rebuilds it and the
// liveness under it every spill round. For the functions a front end
// produces without optimising -- long straight runs, a call every few
// instructions, each call writing thirty-nine clobbered registers -- that
// graph is most of the compile. clang -O0 (RegAllocFast), Go and Cranelift
// all allocate in time close to linear in the instruction count instead,
// and so does this.
//
// The shape is Poletto and Sarkar's, with Wimmer and Franz's intervals:
//
//   - Instructions are numbered in block order. Instruction k reads at
//     position 2k and writes at 2k+1.
//   - Liveness is solved once per round, as bitsets over the dense vreg
//     numbering rather than maps.
//   - Each vreg's lifetime is a list of ranges -- an interval with holes,
//     so a value live around a loop but not in its exit path does not hold
//     its register across the exit.
//   - A pinned vreg's ranges become part of its physical register's
//     fixed interval, which blocks the register for exactly those ranges.
//   - Intervals are visited by start; each takes a register free for the
//     whole of its lifetime -- its copy partner's if it can, otherwise the
//     first in the pool's order -- or, when there is none, it or a
//     longer-lived value holding one is spilled.
//
// What is shared with the graph colourer is the meaning of interference,
// so both give correct answers for the same MIR:
//
//   - A destination conflicts with everything live after its instruction:
//     a write lands at 2k+1, where those values still are.
//   - A destination conflicts with its own instruction's operands, except
//     for a copy: an operand of anything else stays live through 2k+1,
//     an operand of a copy ends at 2k and may share its destination's
//     register, which makes the copy a no-op.
//   - Two destinations of one instruction conflict: both begin at 2k+1.
//
// Nothing is split. A value that finds no register is spilled everywhere,
// by the same rewrite the graph colourer uses (spill.go), and the function
// is allocated again. The rewrite's reloads live for one instruction, so a
// second round almost always finishes it.

import (
	"fmt"
	"math"
	"math/bits"
	"os"
	"slices"
	"sync"

	"github.com/vertex-language/ir/lower/mir"
)

// useGraph selects the graph colourer, for comparing the two.
var useGraph = os.Getenv("IR_REGALLOC") == "graph"

// UseGraph selects the graph colourer (true) or linear scan (false) for
// every allocation after it, and returns the previous choice. It is for
// tests that pin the exact registers the graph colourer chooses: golden
// encodings written against it. It is not safe to call while another
// goroutine is allocating.
func UseGraph(on bool) (was bool) {
	was, useGraph = useGraph, on
	return was
}

type rng struct{ from, to int32 } // [from, to)

type ivl struct {
	v      mir.VReg
	ranges []rng // ascending once built
	class  Class
	reg    PhysReg
	has    bool // reg is set
	fixed  bool
	cur    int // first range that may still matter
}

func (a *ivl) start() int32 { return a.ranges[0].from }
func (a *ivl) end() int32   { return a.ranges[len(a.ranges)-1].to }

// advance moves the cursor past ranges that end at or before pos. Visits
// happen in increasing pos, so the cursor only moves forward.
func (a *ivl) advance(pos int32) {
	for a.cur < len(a.ranges) && a.ranges[a.cur].to <= pos {
		a.cur++
	}
}

func (a *ivl) covers(pos int32) bool {
	a.advance(pos)
	return a.cur < len(a.ranges) && a.ranges[a.cur].from <= pos
}

// intersect is the first position both cover, or MaxInt32.
func intersect(a, b *ivl) int32 {
	i, j := a.cur, b.cur
	for i < len(a.ranges) && j < len(b.ranges) {
		ra, rb := a.ranges[i], b.ranges[j]
		switch {
		case ra.to <= rb.from:
			i++
		case rb.to <= ra.from:
			j++
		default:
			return max(ra.from, rb.from)
		}
	}
	return math.MaxInt32
}

// addRange adds [from, to) to an interval being built backwards: every
// range added is at or before the ones already there.
func (a *ivl) addRange(from, to int32) {
	if from >= to {
		return
	}
	if n := len(a.ranges); n > 0 {
		last := &a.ranges[n-1]
		if to >= last.from {
			last.from = min(last.from, from)
			last.to = max(last.to, to)
			return
		}
	}
	a.ranges = append(a.ranges, rng{from, to})
}

// define shortens the range a later use opened at its block's start so
// that it begins at the write, or records a write nothing reads.
func (a *ivl) define(pos int32) {
	if n := len(a.ranges); n > 0 {
		last := &a.ranges[n-1]
		if last.from <= pos && pos < last.to {
			last.from = pos
			return
		}
	}
	a.ranges = append(a.ranges, rng{pos, pos + 1})
}

// bitset is a set of vregs.
type bitset []uint64

func newBitset(n int) bitset         { return make(bitset, (n+63)/64) }
func (s bitset) add(v mir.VReg)      { s[v>>6] |= 1 << (uint(v) & 63) }
func (s bitset) del(v mir.VReg)      { s[v>>6] &^= 1 << (uint(v) & 63) }
func (s bitset) has(v mir.VReg) bool { return s[v>>6]&(1<<(uint(v)&63)) != 0 }
func (s bitset) each(fn func(mir.VReg)) {
	for i, w := range s {
		for w != 0 {
			b := bits.TrailingZeros64(w)
			fn(mir.VReg(i*64 + b))
			w &= w - 1
		}
	}
}

// liveOut solves liveness for f and returns each block's live-out set, in
// f.Blocks order.
func liveOut(f *mir.Func, n int) []bitset {
	nb := len(f.Blocks)
	index := make(map[*mir.Block]int, nb)
	for i, b := range f.Blocks {
		index[b] = i
	}
	// Per block: the vregs read before any write (gen) and the vregs
	// written (kill), as lists -- they are short, and dense sets for
	// every block would dominate the memory.
	gen := make([][]mir.VReg, nb)
	kill := make([][]mir.VReg, nb)
	seen := newBitset(n)
	wrote := newBitset(n)
	for i, b := range f.Blocks {
		for _, in := range b.Instrs {
			for _, v := range in.Uses {
				if !wrote.has(v) && !seen.has(v) {
					seen.add(v)
					gen[i] = append(gen[i], v)
				}
			}
			for _, v := range in.Defs {
				if !wrote.has(v) {
					wrote.add(v)
					kill[i] = append(kill[i], v)
				}
			}
		}
		for _, v := range gen[i] {
			seen.del(v)
		}
		for _, v := range kill[i] {
			wrote.del(v)
		}
	}
	succ := make([][]int, nb)
	for i, b := range f.Blocks {
		for _, s := range b.Succs {
			if j, ok := index[s]; ok {
				succ[i] = append(succ[i], j)
			}
		}
	}
	in := make([]bitset, nb)
	out := make([]bitset, nb)
	for i := range f.Blocks {
		in[i] = newBitset(n)
		out[i] = newBitset(n)
	}
	tmp := newBitset(n)
	for changed := true; changed; {
		changed = false
		for i := nb - 1; i >= 0; i-- {
			o := out[i]
			for _, s := range succ[i] {
				for w, x := range in[s] {
					o[w] |= x
				}
			}
			copy(tmp, o)
			for _, v := range kill[i] {
				tmp.del(v)
			}
			for _, v := range gen[i] {
				tmp.add(v)
			}
			if !slices.Equal(tmp, in[i]) {
				copy(in[i], tmp)
				changed = true
			}
		}
	}
	return out
}

// scratch is an allocation's working memory, kept from round to round
// and, through scratchPool, from function to function. A debug build
// allocates thousands of functions, each more than once where it spills,
// and making and zeroing tables the size of each one's vreg numbering
// every round was most of the allocator's time. Only what a round touched
// is reset for the next.
type scratch struct {
	all     []*ivl
	slab    []ivl        // the intervals all points into, by vreg
	touched []mir.VReg   // the vregs all holds
	partner [][]mir.VReg // copy partners, by vreg
	paired  []mir.VReg   // the vregs partner holds
	desired []PhysReg
	// mark is visited-by-generation, and never needs clearing: a new
	// search takes a new generation.
	mark []int32
	gen  int32

	// fixed is each register's fixed interval, by class*64+register;
	// used lists the ones this round has, in order of first use.
	fixed [mir.MaxClobberClasses * 64]ivl
	used  []int

	unhandled, active, inactive []*ivl
}

var scratchPool = sync.Pool{New: func() any { return new(scratch) }}

// grow makes the per-vreg tables hold n vregs. The functions allocated
// get fresh vregs as they spill, so n only grows within one.
func (sc *scratch) grow(n int) {
	if cap(sc.all) < n {
		m := n + n/4
		sc.all = make([]*ivl, m)
		sc.slab = make([]ivl, m)
		sc.partner = make([][]mir.VReg, m)
		sc.desired = make([]PhysReg, m)
		mark := make([]int32, m)
		copy(mark, sc.mark)
		sc.mark = mark
	}
	sc.all, sc.slab = sc.all[:n], sc.slab[:n]
	sc.partner, sc.desired, sc.mark = sc.partner[:n], sc.desired[:n], sc.mark[:n]
}

// reset clears what the last round left, keeping the memory.
func (sc *scratch) reset() {
	for _, v := range sc.touched {
		a := &sc.slab[v]
		*a = ivl{ranges: a.ranges[:0]}
		sc.all[v] = nil
	}
	sc.touched = sc.touched[:0]
	for _, v := range sc.paired {
		sc.partner[v] = sc.partner[v][:0]
	}
	sc.paired = sc.paired[:0]
	for _, i := range sc.used {
		fx := &sc.fixed[i]
		*fx = ivl{ranges: fx.ranges[:0]}
	}
	sc.used = sc.used[:0]
}

// intervals builds every vreg's lifetime, indexed by vreg; nil for one the
// function never names.
func intervals(f *mir.Func, n int, sc *scratch) []*ivl {
	sc.reset()
	sc.grow(n)
	out := liveOut(f, n)
	all, slab := sc.all, sc.slab
	get := func(v mir.VReg) *ivl {
		a := all[v]
		if a == nil {
			a = &slab[v]
			a.v = v
			all[v] = a
			sc.touched = append(sc.touched, v)
		}
		return a
	}
	// Block starts, in instruction numbers.
	starts := make([]int32, len(f.Blocks)+1)
	for i, b := range f.Blocks {
		starts[i+1] = starts[i] + int32(len(b.Instrs))
	}
	for i := len(f.Blocks) - 1; i >= 0; i-- {
		b := f.Blocks[i]
		from, to := 2*starts[i], 2*starts[i+1]
		out[i].each(func(v mir.VReg) { get(v).addRange(from, to) })
		for k := len(b.Instrs) - 1; k >= 0; k-- {
			in := b.Instrs[k]
			pos := 2 * (starts[i] + int32(k))
			for _, d := range in.Defs {
				get(d).define(pos + 1)
			}
			end := pos + 2
			if in.Copy {
				end = pos + 1
			}
			for _, u := range in.Uses {
				get(u).addRange(from, end)
			}
		}
	}
	for _, v := range sc.touched {
		slices.Reverse(all[v].ranges)
	}
	return all
}

// linearSpilling is Spilling by linear scan.
func linearSpilling(f *mir.Func, pool *Pool, sp Spiller) (map[mir.VReg]PhysReg, error) {
	st := &spillState{fresh: map[mir.VReg]bool{}, done: map[mir.VReg]bool{}}
	sc := scratchPool.Get().(*scratch)
	defer scratchPool.Put(sc)
	for round := 0; ; round++ {
		assigned, spill, err := linearRound(f, pool, st, sc)
		if err != nil {
			return nil, err
		}
		if len(spill) == 0 {
			if verifyLinear {
				if err := verifyAssignment(f, pool, assigned); err != nil {
					return nil, err
				}
			}
			return assigned, nil
		}
		if sp == nil || round > f.NumVRegs() {
			return nil, ErrOutOfRegisters
		}
		slices.Sort(spill)
		spillAll(f, pool, sp, st, spill)
	}
}

// linearRound allocates once. It returns the assignment, or the vregs to
// spill before trying again.
func linearRound(f *mir.Func, pool *Pool, st *spillState, sc *scratch) (map[mir.VReg]PhysReg, []mir.VReg, error) {
	n := f.NumVRegs()
	all := intervals(f, n, sc)

	// Fixed intervals: every pinned vreg's ranges, one interval per
	// register of each class. A register outside the dense table (a class
	// or a number past what a Clobbers holds) is kept in a map.
	type key struct {
		c Class
		r PhysReg
	}
	var fixedKeys []key
	var overflow map[key]*ivl
	// pinnedRanges counts, per dense fixed interval, the ranges pins gave
	// it: those come in vreg order, while clobbers come in position order
	// after them, so an interval with none is sorted already.
	var pinnedRanges [len(scratch{}.fixed)]int32
	fixedFor := func(c Class, r PhysReg) (*ivl, int) {
		if c >= 0 && int(c) < mir.MaxClobberClasses && r >= 0 && r < 64 {
			i := int(c)*64 + int(r)
			fx := &sc.fixed[i]
			if !fx.fixed {
				*fx = ivl{v: -1, class: c, reg: r, has: true, fixed: true, ranges: fx.ranges[:0]}
				sc.used = append(sc.used, i)
				fixedKeys = append(fixedKeys, key{c, r})
			}
			return fx, i
		}
		k := key{c, r}
		fx := overflow[k]
		if fx == nil {
			if overflow == nil {
				overflow = map[key]*ivl{}
			}
			fx = &ivl{v: -1, class: c, reg: r, has: true, fixed: true}
			overflow[k] = fx
			fixedKeys = append(fixedKeys, k)
		}
		return fx, -1
	}
	unhandled := sc.unhandled[:0]
	for _, v := range sc.touched {
		a := all[v]
		a.class = pool.ClassOf(a.v)
		if r, ok := pool.pinned.get(a.v); ok {
			a.reg, a.has, a.fixed = r, true, true
			fx, i := fixedFor(a.class, r)
			fx.ranges = append(fx.ranges, a.ranges...)
			if i >= 0 {
				pinnedRanges[i] += int32(len(a.ranges))
			}
			continue
		}
		unhandled = append(unhandled, a)
	}
	// Clobbered registers: each is unavailable at the instant its
	// instruction writes, 2k+1, exactly as a vreg pinned there and never
	// read was.
	pos := int32(0)
	for _, b := range f.Blocks {
		for _, in := range b.Instrs {
			if in.Clobbers != nil {
				at := 2*pos + 1
				for c, mask := range in.Clobbers {
					for mask != 0 {
						r := bits.TrailingZeros64(mask)
						mask &= mask - 1
						fx, _ := fixedFor(Class(c), PhysReg(r))
						fx.ranges = append(fx.ranges, rng{at, at + 1})
					}
				}
			}
			pos++
		}
	}

	active, inactive := sc.active[:0], sc.inactive[:0]
	slices.SortFunc(fixedKeys, func(x, y key) int {
		if x.c != y.c {
			return int(x.c) - int(y.c)
		}
		return int(x.r) - int(y.r)
	})
	for _, k := range fixedKeys {
		var fx *ivl
		sorted := false
		if k.c >= 0 && int(k.c) < mir.MaxClobberClasses && k.r >= 0 && k.r < 64 {
			i := int(k.c)*64 + int(k.r)
			fx = &sc.fixed[i]
			sorted = pinnedRanges[i] == 0
		} else {
			fx = overflow[k]
		}
		if !sorted {
			slices.SortFunc(fx.ranges, func(x, y rng) int { return int(x.from) - int(y.from) })
		}
		merged := fx.ranges[:1]
		for _, r := range fx.ranges[1:] {
			last := &merged[len(merged)-1]
			if r.from < last.to {
				return nil, nil, fmt.Errorf("%w: register %v is wanted by two values at once", ErrPinConflict, k.r)
			}
			if r.from == last.to {
				last.to = r.to
				continue
			}
			merged = append(merged, r)
		}
		fx.ranges = merged
		inactive = append(inactive, fx)
	}

	slices.SortFunc(unhandled, func(x, y *ivl) int {
		if d := x.start() - y.start(); d != 0 {
			return int(d)
		}
		return int(x.v - y.v)
	})

	// Copy partners, for preferring the register a copy's other end has.
	partners := sc.partner
	pair := func(a, b mir.VReg) {
		if len(partners[a]) == 0 {
			sc.paired = append(sc.paired, a)
		}
		partners[a] = append(partners[a], b)
	}
	for _, b := range f.Blocks {
		for _, in := range b.Instrs {
			if in.Copy && len(in.Defs) > 0 && len(in.Uses) > 0 {
				d, u := in.Defs[0], in.Uses[0]
				pair(d, u)
				pair(u, d)
			}
		}
	}

	// desired is the register a pinned vreg reachable through copies
	// holds: what a value would like so that its copies vanish. Walked
	// out from every pin, nearest first.
	desired := sc.desired
	for i := range desired {
		desired[i] = -1
	}
	var wave []mir.VReg
	for v, a := range all {
		if a != nil && a.fixed {
			desired[v] = a.reg
			wave = append(wave, a.v)
		}
	}
	for depth := 0; depth < 3 && len(wave) > 0; depth++ {
		var next []mir.VReg
		for _, v := range wave {
			for _, p := range partners[v] {
				if desired[p] < 0 && all[p] != nil && all[p].class == all[v].class {
					desired[p] = desired[v]
					next = append(next, p)
				}
			}
		}
		wave = next
	}

	maxReg := PhysReg(0)
	for _, regs := range pool.free {
		for _, r := range regs {
			maxReg = max(maxReg, r)
		}
	}
	freeUntil := make([]int32, maxReg+1)
	var spill []mir.VReg
	var hints []PhysReg
	var frontierBuf []mir.VReg

	var wantedBuf []PhysReg
	mark := sc.mark // visited, by the generation below
	gen := sc.gen
	defer func() { sc.gen = gen }()
	defer func() { sc.unhandled, sc.active, sc.inactive = unhandled[:0], active[:0], inactive[:0] }()
	for ui, cur := range unhandled {
		pos := cur.start()
		keep := active[:0]
		for _, a := range active {
			switch {
			case a.end() <= pos:
			case !a.covers(pos):
				inactive = append(inactive, a)
			default:
				keep = append(keep, a)
			}
		}
		active = keep
		keep = inactive[:0]
		for _, a := range inactive {
			switch {
			case a.end() <= pos:
			case a.covers(pos):
				active = append(active, a)
			default:
				keep = append(keep, a)
			}
		}
		inactive = keep
		cur.advance(pos)

		regs := pool.regsFor(cur.v)
		for _, r := range pool.free[cur.class] {
			freeUntil[r] = math.MaxInt32
		}
		for _, a := range active {
			if a.class == cur.class && int(a.reg) < len(freeUntil) {
				freeUntil[a.reg] = 0
			}
		}
		for _, a := range inactive {
			if a.class != cur.class || int(a.reg) >= len(freeUntil) || freeUntil[a.reg] == 0 {
				continue
			}
			if x := intersect(a, cur); x < freeUntil[a.reg] {
				freeUntil[a.reg] = x
			}
		}

		end := cur.end()
		chosen, ok := PhysReg(0), false
		// The register a copy's other end has, or failing that the one
		// a copy of a copy has: a value copied into a pinned return
		// register two copies later wants that register now, before
		// the vreg between them is allocated.
		hints = hints[:0]
		gen++
		mark[cur.v] = gen
		frontier := append(frontierBuf[:0], cur.v)
		for depth := 0; depth < 3 && len(frontier) > 0 && len(hints) == 0; depth++ {
			var next []mir.VReg
			for _, v := range frontier {
				for _, p := range partners[v] {
					if mark[p] == gen {
						continue
					}
					mark[p] = gen
					pa := all[p]
					if pa == nil || pa.class != cur.class {
						continue
					}
					if pa.has {
						hints = append(hints, pa.reg)
					} else {
						next = append(next, p)
					}
				}
			}
			frontier = next
		}
		for _, h := range hints {
			if int(h) < len(freeUntil) && inPool(regs, h) && freeUntil[h] >= end {
				chosen, ok = h, true
				break
			}
		}
		if !ok {
			if d := desired[cur.v]; d >= 0 && int(d) < len(freeUntil) && inPool(regs, d) && freeUntil[d] >= end {
				chosen, ok = d, true
			}
		}
		if !ok {
			// The first free register that nothing about to start
			// inside cur wants: taking a register a later value
			// is copied into or out of would cost that value its
			// free copy.
			wanted := wantedBuf[:0]
			for j, seen := ui+1, 0; j < len(unhandled) && seen < 16 && unhandled[j].start() < end; j, seen = j+1, seen+1 {
				if d := desired[unhandled[j].v]; d >= 0 && unhandled[j].class == cur.class {
					wanted = append(wanted, d)
				}
			}
			first := PhysReg(-1)
			for _, r := range regs {
				if freeUntil[r] < end {
					continue
				}
				if first < 0 {
					first = r
				}
				if !inPool(wanted, r) {
					chosen, ok = r, true
					break
				}
			}
			if !ok && first >= 0 {
				chosen, ok = first, true
			}
		}
		if ok {
			cur.reg, cur.has = chosen, true
			active = append(active, cur)
			continue
		}

		// No register is free for the whole of cur. Either cur goes to
		// memory, or the values holding some register do, whichever
		// frees the most: the register whose blockers are all spillable
		// and live the longest.
		best, bestEnd, found := PhysReg(0), int32(-1), false
		for _, r := range regs {
			blockEnd, ok := int32(-1), true
			for _, set := range [][]*ivl{active, inactive} {
				for _, a := range set {
					if !ok || a.class != cur.class || a.reg != r {
						continue
					}
					if intersect(a, cur) == math.MaxInt32 {
						continue
					}
					if a.fixed || !st.eligible(a.v, pool.pinned) {
						ok = false
						continue
					}
					blockEnd = max(blockEnd, a.end())
				}
			}
			if ok && blockEnd > bestEnd {
				best, bestEnd, found = r, blockEnd, true
			}
		}
		curSpillable := st.eligible(cur.v, pool.pinned)
		if curSpillable && (!found || end >= bestEnd) {
			spill = append(spill, cur.v)
			continue
		}
		if !found {
			// Neither cur nor anything in its way can go to memory.
			if len(spill) > 0 {
				// A spill this round may yet make room; try again after it.
				continue
			}
			return nil, nil, ErrOutOfRegisters
		}
		evict := func(set []*ivl) []*ivl {
			keep := set[:0]
			for _, a := range set {
				if a.class == cur.class && a.reg == best && !a.fixed && intersect(a, cur) != math.MaxInt32 {
					a.has = false
					spill = append(spill, a.v)
					continue
				}
				keep = append(keep, a)
			}
			return keep
		}
		active = evict(active)
		inactive = evict(inactive)
		cur.reg, cur.has = best, true
		active = append(active, cur)
	}

	if len(spill) > 0 {
		return nil, spill, nil
	}
	assigned := make(map[mir.VReg]PhysReg, len(sc.touched))
	for _, v := range sc.touched {
		if a := all[v]; a.has {
			assigned[a.v] = a.reg
		}
	}
	return assigned, nil, nil
}

func inPool(regs []PhysReg, r PhysReg) bool {
	for _, x := range regs {
		if x == r {
			return true
		}
	}
	return false
}

// spillAll puts every vreg in vs in memory in one pass over f: the same
// rewrite spill does one vreg at a time, which walks the whole function
// per value.
func spillAll(f *mir.Func, pool *Pool, sp Spiller, st *spillState, vs []mir.VReg) {
	want := map[mir.VReg]bool{}
	for _, v := range vs {
		if st.eligible(v, pool.pinned) {
			want[v] = true
			st.done[v] = true
		}
	}
	if len(want) == 0 {
		return
	}
	// Which of them are cheaper to repeat than to store: one def, no
	// operands, and the target says so.
	rm, _ := sp.(Rematerializer)
	defs := map[mir.VReg]mir.Instr{}
	ndefs := map[mir.VReg]int{}
	for _, b := range f.Blocks {
		for _, in := range b.Instrs {
			for _, d := range in.Defs {
				if want[d] {
					defs[d] = in
					ndefs[d]++
				}
			}
		}
	}
	remat := map[mir.VReg]bool{}
	slot := map[mir.VReg]int{}
	for _, v := range vs {
		if !want[v] {
			continue
		}
		def := defs[v]
		if rm != nil && ndefs[v] == 1 && len(def.Uses) == 0 && len(def.Defs) == 1 && rm.Rematerializable(def) {
			remat[v] = true
			continue
		}
		slot[v] = sp.Slot()
	}

	for _, b := range f.Blocks {
		touches := false
		for _, in := range b.Instrs {
			for _, v := range in.Uses {
				touches = touches || want[v]
			}
			for _, v := range in.Defs {
				touches = touches || (want[v] && !remat[v])
			}
		}
		if !touches {
			continue
		}
		out := make([]mir.Instr, 0, len(b.Instrs)+8)
		for _, in := range b.Instrs {
			// Reads: one fresh vreg per spilled value, loaded (or
			// recomputed) just before.
			var read map[mir.VReg]mir.VReg
			for i, v := range in.Uses {
				if !want[v] {
					continue
				}
				if read == nil {
					read = map[mir.VReg]mir.VReg{}
					in.Uses = append([]mir.VReg(nil), in.Uses...)
				}
				w, ok := read[v]
				if !ok {
					w = st.newFresh(f, pool, pool.ClassOf(v))
					pool.inherit(w, v)
					read[v] = w
					if remat[v] {
						c := defs[v]
						c.Defs = []mir.VReg{w}
						out = append(out, c)
					} else {
						out = append(out, sp.Load(slot[v], w, pool.ClassOf(v)))
					}
				}
				in.Uses[i] = w
			}
			// Writes: stored just after. A value the instruction also
			// read keeps the same fresh vreg (see spill).
			type store struct {
				v, w mir.VReg
			}
			var stores []store
			var wrote map[mir.VReg]mir.VReg
			for i, v := range in.Defs {
				if !want[v] || remat[v] {
					continue
				}
				if wrote == nil {
					wrote = map[mir.VReg]mir.VReg{}
					in.Defs = append([]mir.VReg(nil), in.Defs...)
				}
				w, ok := wrote[v]
				if !ok {
					if r, isRead := read[v]; isRead {
						w = r
					} else {
						w = st.newFresh(f, pool, pool.ClassOf(v))
						pool.inherit(w, v)
					}
					wrote[v] = w
					stores = append(stores, store{v, w})
				}
				in.Defs[i] = w
			}
			out = append(out, in)
			for _, s := range stores {
				out = append(out, sp.Store(slot[s.v], s.w, pool.ClassOf(s.v)))
			}
		}
		b.Instrs = out
	}
}

// verify is IR_REGALLOC_VERIFY=1: the linear assignment checked against
// the graph colourer's interference, edge by edge. Two vregs that
// interfere and share a register in one class is a miscompile.
var verifyLinear = os.Getenv("IR_REGALLOC_VERIFY") == "1"

func verifyAssignment(f *mir.Func, pool *Pool, assigned map[mir.VReg]PhysReg) error {
	if err := verifyClobbers(f, pool, assigned); err != nil {
		return err
	}
	g := interference(f)
	for _, v := range g.Nodes() {
		rv, ok := assigned[v]
		if !ok {
			return fmt.Errorf("regalloc: verify: v%d has no register", v)
		}
		if allow, ok := pool.allowed[v]; ok && !inPool(allow, rv) {
			return fmt.Errorf("regalloc: verify: v%d has %v, outside the registers it may take", v, rv)
		}
		var bad error
		g.neighbours(v, func(n mir.VReg) {
			if bad != nil || pool.ClassOf(n) != pool.ClassOf(v) {
				return
			}
			if rn, ok := assigned[n]; ok && rn == rv {
				bad = fmt.Errorf("regalloc: verify: v%d and v%d interfere and both have %v", v, n, rv)
			}
		})
		if bad != nil {
			return bad
		}
	}
	return nil
}

// verifyClobbers checks that no value live across an instruction is in a
// register the instruction clobbers.
func verifyClobbers(f *mir.Func, pool *Pool, assigned map[mir.VReg]PhysReg) error {
	live := mir.Liveness(f)
	var bad error
	for _, b := range f.Blocks {
		live.LiveAfter(b, func(_ int, in mir.Instr, after map[mir.VReg]bool) {
			if bad != nil || in.Clobbers == nil {
				return
			}
			for v := range after {
				r, ok := assigned[v]
				if ok && in.Clobbers[pool.ClassOf(v)]&(1<<uint(r)) != 0 {
					bad = fmt.Errorf("regalloc: verify: v%d is live across an instruction that clobbers its %v", v, r)
					return
				}
			}
		})
	}
	return bad
}
