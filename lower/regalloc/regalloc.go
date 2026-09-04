// Package regalloc assigns mir.VRegs to physical registers.
//
// It works from an interference graph: two vregs interfere when one is
// live at a point where the other is written, and vregs that never
// interfere can share a register. That is the whole of what makes a
// function with more temporaries than the machine has registers
// compilable at all — before liveness, every vreg a function named owned
// a register for the function's entire lifetime, and nine temporaries
// was the ceiling.
//
// It is still not a full allocator, but it no longer refuses work it
// cannot colour outright: given a Spiller, a graph that needs more
// colours than the pool has is rewritten so that some value lives in
// memory, and coloured again. See spill.go, which also says what makes
// that strategy simple and what makes it bad. Without a Spiller, or when
// no value is left that may be spilled, it is still ErrOutOfRegisters,
// loudly, rather than two live vregs silently sharing one register.
//
// It colours more than one register file. A Class is a set of physical
// registers disjoint from every other class's, and a vreg in one never
// shares a register with a vreg in another — which is a fact about the
// free lists and about nothing else here. The interference graph is
// built without classes and does not need them: two vregs of different
// classes may well be live at once, and the edge between them is real
// and simply never binding, because the registers they are competing
// for are not the same registers. A target with one register file never
// mentions classes at all.
//
// The colouring is greedy in vreg order and not optimal. Optimal graph
// colouring is NP-hard, greedy is what production allocators use too, and
// the useful property is not optimality but determinism: the same MIR
// always produces the same assignment, which is what makes a compiler's
// output diffable and its tests writable.
package regalloc

import (
	"fmt"

	"github.com/vertex-language/ir/lower/mir"
)

// PhysReg is an opaque physical-register identity. regalloc never
// interprets it — a target defines what PhysReg(3) means to its own
// register file and converts back to its own type after Assign returns.
type PhysReg int

// ErrOutOfRegisters is returned when a function's interference graph
// needs more registers than the pool has.
var ErrOutOfRegisters = fmt.Errorf("regalloc: out of physical registers")

// ErrPinConflict is returned when two vregs pinned to the same physical
// register are live at the same time.
//
// A pin is not a request, it is a fact about where a value has to be —
// an incoming parameter, or an operand an instruction can only read from
// one register. Two such facts that contradict each other cannot be
// resolved by colouring, only by the caller having built something
// impossible, so this is loud rather than silently honouring one of them.
var ErrPinConflict = fmt.Errorf("regalloc: two live vregs are pinned to one register")

// A Class is a register file: a set of physical registers disjoint from
// every other class's, holding values that cannot live in each other's
// registers. An integer and a floating-point file are two classes on
// most architectures and one class on some.
//
// regalloc needs to know only that a vreg in one class never shares a
// register with a vreg in another, which is why classes touch the free
// lists and nothing else. The interference graph is built without them:
// two vregs of different classes may well be live at once, and the edge
// between them is real and simply never binding, because the registers
// they are competing for are not the same registers.
//
// It is also why a PhysReg means nothing without its class. RAX and XMM0
// are both register number zero, and a target that had to renumber one
// of them to keep them apart would be working around this package rather
// than using it.
type Class int

// DefaultClass is the class of every vreg a target does not Classify. It
// is zero so that a target with one register file never mentions classes
// at all, and NewPool fills this one.
const DefaultClass Class = 0

// Pool is a fixed set of physical registers, some pinned to specific
// virtual registers ahead of time — this is how a target's ABI
// classification places a function's incoming parameters — and the rest
// free for Assign to hand out.
type Pool struct {
	pinned map[mir.VReg]PhysReg
	free   map[Class][]PhysReg
	class  map[mir.VReg]Class
}

// NewPool builds a pool whose free list is regs, in the order Assign
// should prefer them. They are DefaultClass; a second register file is
// AddClass.
func NewPool(regs []PhysReg) *Pool {
	p := &Pool{
		pinned: map[mir.VReg]PhysReg{},
		free:   map[Class][]PhysReg{},
		class:  map[mir.VReg]Class{},
	}
	p.AddClass(DefaultClass, regs)
	return p
}

// AddClass gives the pool another register file, in the order Assign
// should prefer its registers.
func (p *Pool) AddClass(c Class, regs []PhysReg) {
	free := make([]PhysReg, len(regs))
	copy(free, regs)
	p.free[c] = free
}

// Classify records which register file v belongs in. A vreg never
// classified is in DefaultClass.
func (p *Pool) Classify(v mir.VReg, c Class) {
	if c == DefaultClass {
		return
	}
	p.class[v] = c
}

// ClassOf is v's register file.
func (p *Pool) ClassOf(v mir.VReg) Class { return p.class[v] }

// Pin fixes v to r before allocation starts. Assign never reassigns a
// pinned register, and never gives r to a vreg of the same class that
// interferes with v.
func (p *Pool) Pin(v mir.VReg, r PhysReg) { p.pinned[v] = r }

// Assign colours f's interference graph and returns the physical register
// for every VReg it names.
//
// A pinned vreg keeps its register. Every other vreg takes the register
// its copies would prefer if that one is free, and otherwise the first
// register in the pool's order that none of its already-coloured
// neighbours holds — including pinned neighbours, which is what keeps a
// temporary out of a parameter's register while that parameter is still
// live, and lets it have that register once the parameter is dead.
// Assign colours without spilling. It is Spilling with no Spiller, and
// exists because most functions never need one and saying so at the call
// is clearer than passing a nil.
func Assign(f *mir.Func, pool *Pool) (map[mir.VReg]PhysReg, error) {
	return Spilling(f, pool, nil)
}

// Spilling colours f, rewriting it through sp when the graph needs more
// registers than the pool has.
//
// A round spills every value the colouring could not place, and colours
// again. Spilling one and starting over would be the same answer and a
// great deal slower: the pressure in a large function runs a hundred over
// the register count, and a round rebuilds the interference graph, so one
// spill per round is a hundred graphs where a handful would do.
//
// It terminates because a spilled value is never chosen twice and the vregs
// the rewrite creates are never chosen at all, so the supply of candidates
// strictly shrinks; the cap below is a guard against a bug in that
// reasoning rather than the reason it ends.
func Spilling(f *mir.Func, pool *Pool, sp Spiller) (map[mir.VReg]PhysReg, error) {
	st := &spillState{fresh: map[mir.VReg]bool{}, done: map[mir.VReg]bool{}}
	for round := 0; ; round++ {
		assigned, stuck, g, err := colour(f, pool)
		if err != nil {
			return nil, err
		}
		if assigned != nil {
			return assigned, nil
		}
		if sp == nil || round > f.NumVRegs() {
			return nil, ErrOutOfRegisters
		}
		spilled := 0
		for _, s := range stuck {
			v, ok := st.pick(s, g, pool.pinned)
			if !ok {
				continue
			}
			st.spill(f, pool, sp, v)
			spilled++
		}
		if spilled == 0 {
			// Everything live at every sticking point is either pinned or
			// already one instruction long. There is nothing left to put
			// in memory.
			return nil, ErrOutOfRegisters
		}
	}
}

// colour makes one attempt. It returns the assignment, or a nil map and
// every vreg it could not place.
//
// The graph comes back with either: the caller that has to choose values to
// spill needs the same one, and building it twice is building the expensive
// thing twice.
func colour(f *mir.Func, pool *Pool) (map[mir.VReg]PhysReg, []mir.VReg, *graph, error) {
	g := interference(f)
	prefer := copyPreferences(f)
	var stuck []mir.VReg

	assigned := make(map[mir.VReg]PhysReg, len(pool.pinned))
	for v, r := range pool.pinned {
		assigned[v] = r
	}
	for v, r := range pool.pinned {
		var conflict mir.VReg
		found := false
		g.neighbours(v, func(n mir.VReg) {
			if found {
				return
			}
			other, ok := pool.pinned[n]
			if !ok || other != r || pool.ClassOf(n) != pool.ClassOf(v) {
				return
			}
			conflict, found = n, true
		})
		if found {
			return nil, nil, nil, fmt.Errorf("%w: v%d and v%d both want %v",
				ErrPinConflict, v, conflict, r)
		}
	}

	// Vreg order, so the result is a function of the MIR and not of the
	// order anything was walked in. Pinned vregs are already coloured and
	// skipped.
	for _, v := range g.Nodes() {
		if _, done := assigned[v]; done {
			continue
		}
		// Only neighbours in v's own register file can take a register
		// away from it. A neighbour in another class holds a register
		// that is not one of these, whatever number it goes by.
		class := pool.ClassOf(v)
		free := pool.free[class]
		taken := map[PhysReg]bool{}
		g.neighbours(v, func(n mir.VReg) {
			if pool.ClassOf(n) != class {
				return
			}
			if r, ok := assigned[n]; ok {
				taken[r] = true
			}
		})
		if r, ok := preferred(prefer[v], assigned, taken, free); ok {
			assigned[v] = r
			continue
		}
		r, ok := firstFree(free, taken)
		if !ok {
			// Left uncoloured and the walk continues, rather than
			// stopping here. Every vreg that finds no register is one
			// this round has to spill, and finding them all now is the
			// difference between one spill per round and one round: a
			// function whose pressure is a hundred over the register
			// count spilt a hundred times, and rebuilt and recoloured
			// the whole graph between each one.
			//
			// Continuing is safe because v holds no register: nothing
			// after it can collide with a register v did not take.
			stuck = append(stuck, v)
			continue
		}
		assigned[v] = r
	}
	if len(stuck) > 0 {
		return nil, stuck, g, nil
	}
	return assigned, nil, g, nil
}

// copyPreferences records, for each vreg, the vregs it is copied to or
// from.
//
// Giving both ends of a copy the same register makes the copy a no-op,
// which the target then does not emit at all — that is the whole payoff
// of mir.Instr.Copy, and without a preference it would only happen by
// the luck of allocation order. It is a preference and never a
// constraint: two vregs that interfere have to differ, and this is
// consulted after that has been checked.
func copyPreferences(f *mir.Func) map[mir.VReg][]mir.VReg {
	prefer := map[mir.VReg][]mir.VReg{}
	for _, b := range f.Blocks {
		for _, in := range b.Instrs {
			if !in.Copy || len(in.Defs) == 0 || len(in.Uses) == 0 {
				continue
			}
			d, u := in.Defs[0], in.Uses[0]
			prefer[d] = append(prefer[d], u)
			prefer[u] = append(prefer[u], d)
		}
	}
	return prefer
}

// preferred is the register one of v's copy partners already holds, if
// that register is one the pool offers and no neighbour of v has taken.
func preferred(partners []mir.VReg, assigned map[mir.VReg]PhysReg, taken map[PhysReg]bool, free []PhysReg) (PhysReg, bool) {
	for _, p := range partners {
		r, ok := assigned[p]
		if !ok || taken[r] {
			continue
		}
		for _, f := range free {
			if f == r {
				return r, true
			}
		}
	}
	return 0, false
}

// firstFree is the first register in the pool's order that no neighbour
// holds.
func firstFree(free []PhysReg, taken map[PhysReg]bool) (PhysReg, bool) {
	for _, r := range free {
		if !taken[r] {
			return r, true
		}
	}
	return 0, false
}

// interference builds the graph: an edge between two vregs that cannot
// share a register.
//
// Two rules, and both are about the moment an instruction writes.
//
// A destination interferes with everything live after the instruction,
// because those values are still needed and writing over one loses it.
// It does not interfere with everything live *before*: a value this
// instruction was the last reader of is dead the moment it has been read,
// and its register is exactly what a destination should reuse.
//
// A destination also interferes with the instruction's own operands, even
// the ones dying here. That is stricter than it has to be — a
// two-address add could put its result in its own first operand's
// register — but "could" is per-opcode knowledge, and mir does not
// interpret an Op. The cost is one extra move in the cases where the
// overlap would have been safe; the alternative is an allocator that has
// to be told, per architecture, which operand a destination may alias.
//
// The exception is a copy, where the destination is the operand's value
// rather than something computed from it. mir.Instr.Copy is what says
// so, and it is the one thing about an Op this package is told. Sharing
// a register there is not a hazard, it is coalescing: the move becomes a
// no-op. Note that this only removes the def-to-operand edge — a copy
// whose source outlives it still interferes through the live-after rule
// above, which is exactly the condition coalescing has to respect.
func interference(f *mir.Func) *graph {
	g := newGraph(f.NumVRegs())

	// Every vreg the function names is a node, including one whose value
	// nothing reads: it still needs a register to be written into.
	for _, b := range f.Blocks {
		for _, in := range b.Instrs {
			for _, v := range in.Defs {
				g.addNode(v)
			}
			for _, v := range in.Uses {
				g.addNode(v)
			}
		}
	}

	live := mir.Liveness(f)
	for _, b := range f.Blocks {
		live.LiveAfter(b, func(_ int, in mir.Instr, after map[mir.VReg]bool) {
			for _, d := range in.Defs {
				for v := range after {
					g.addEdge(d, v)
				}
				if !in.Copy {
					for _, u := range in.Uses {
						g.addEdge(d, u)
					}
				}
				for _, other := range in.Defs {
					g.addEdge(d, other)
				}
			}
		})
	}

	// Values live simultaneously across a block boundary interfere even
	// when no single instruction writes either: two parameters both live
	// into the same block have to be in two registers.
	//
	// Only the ones no instruction writes, though, and that is the whole
	// of the difference between this loop costing nothing and costing more
	// than everything else here put together.
	//
	// Take two values live at the same point, each of them written
	// somewhere. Whichever is written second is written while the other is
	// live — so the rule above has already drawn the edge, at that write.
	// (And if the first is written again in between, that write is itself
	// a point where the other is live, so the edge is drawn there.) A
	// value with no write anywhere is the case that argument does not
	// reach: a parameter arrives live and is never written, so two of them
	// are never simultaneously anything's live-after, and nothing but this
	// loop would separate them.
	//
	// So the pairs worth walking are the ones among values with no write,
	// which is a handful rather than the whole live set. Walking the whole
	// set is the same edges over again: a block whose live set holds a
	// hundred and thirty values is eight thousand pairs, five hundred
	// blocks of that is millions, and it is the same millions each time
	// because the set barely changes from block to block.
	defless := deflessLive(f, live)
	for _, b := range f.Blocks {
		overlap(g, defless, live.In[b])
		overlap(g, defless, live.Out[b])
	}
	g.seal()
	return g
}

// deflessLive is every vreg that is live somewhere and written nowhere.
//
// A vreg reaches a function without a write in it two ways: it is an
// incoming parameter, or it is a register an instruction reads and the
// target pinned rather than produced. Either way it has no defining
// instruction for the live-after rule to hang an edge on.
func deflessLive(f *mir.Func, live *mir.Live) map[mir.VReg]bool {
	written := map[mir.VReg]bool{}
	for _, b := range f.Blocks {
		for _, in := range b.Instrs {
			for _, v := range in.Defs {
				written[v] = true
			}
		}
	}
	out := map[mir.VReg]bool{}
	for _, b := range f.Blocks {
		for _, set := range []map[mir.VReg]bool{live.In[b], live.Out[b]} {
			for v := range set {
				if !written[v] {
					out[v] = true
				}
			}
		}
	}
	return out
}

// overlap adds an edge between every pair of simultaneously live vregs in
// set that also appears in only — the values nothing writes. See the caller
// for why the rest of the set is already covered.
func overlap(g *graph, only, set map[mir.VReg]bool) {
	if len(only) == 0 {
		return
	}
	vs := make([]mir.VReg, 0, len(only))
	for v := range set {
		if only[v] {
			vs = append(vs, v)
		}
	}
	for i := range vs {
		for j := i + 1; j < len(vs); j++ {
			g.addEdge(vs[i], vs[j])
		}
	}
}
