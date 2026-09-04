package regalloc

// Spilling: what to do when the interference graph needs more registers
// than there are.
//
// The strategy is the simplest one that is correct, and it is called
// "spill everywhere": a spilled value lives in memory, is stored the
// moment it is computed, and is loaded again immediately before each
// instruction that reads it. Every one of those loads defines a vreg
// that dies one instruction later, which is a live range short enough to
// colour in almost any graph — that is the whole reason it works.
//
// What it is not is good. A value read in a loop is reloaded on every
// iteration, where a real allocator would keep it in a register across
// the ones that have a spare and reload only where they do not. Doing
// better means splitting a live range rather than replacing it, and
// choosing where to split by how often the code there runs, which needs
// a notion of loop depth this MIR does not carry.

import "github.com/vertex-language/ir/lower/mir"

// A Spiller supplies the two instructions this package cannot write for
// itself: the one that puts a value in memory and the one that brings it
// back.
//
// It has to come from the target, because a store is an opcode and an
// addressing mode and this package knows neither. What it does know is
// where the stores and loads belong, which is the half that is the same
// on every architecture.
type Spiller interface {
	// Slot reserves stack storage for one spilled value and returns
	// whatever identifies it to the target — an offset, an index, its
	// choice. Called once per value, however many times that value is
	// stored and loaded.
	//
	// The storage has to be wide enough for the widest register the
	// target allocates, since this package does not track widths.
	Slot() int

	// Store writes v into slot; Load reads it back into v.
	//
	// The class is v's register file, because a store out of an integer
	// register and a store out of a floating-point one are different
	// instructions and the vreg number does not say which.
	Store(slot int, v mir.VReg, c Class) mir.Instr
	Load(slot int, v mir.VReg, c Class) mir.Instr
}

// spillState is what one Assign call remembers across its rounds.
type spillState struct {
	// fresh are the vregs the rewrite itself created — the ones a load
	// defines and a store reads. Spilling one of those again would
	// replace a live range one instruction long with another one
	// instruction long, and never terminate.
	fresh map[mir.VReg]bool
	// done are the vregs already in memory, so a later round does not
	// choose one of them a second time.
	done map[mir.VReg]bool
}

// pick chooses a vreg to spill, given the one that could not be
// coloured.
//
// The failing vreg first, since it is the one with no register left, and
// otherwise its most-constrained neighbour: a vreg conflicting with many
// others is the one whose removal frees the most. Neither a pinned vreg
// nor one the rewrite created is eligible — a pin is a fact about where
// a value has to be, and a rewrite's vregs are already as short as they
// can get.
func (st *spillState) pick(v mir.VReg, g *graph, pinned map[mir.VReg]PhysReg) (mir.VReg, bool) {
	if st.eligible(v, pinned) {
		return v, true
	}

	best, bestDegree, found := mir.VReg(0), -1, false
	g.neighbours(v, func(n mir.VReg) {
		if !st.eligible(n, pinned) {
			return
		}
		if d := g.degree(n); d > bestDegree {
			best, bestDegree, found = n, d, true
		}
	})
	return best, found
}

func (st *spillState) eligible(v mir.VReg, pinned map[mir.VReg]PhysReg) bool {
	if _, isPinned := pinned[v]; isPinned {
		return false
	}
	return !st.fresh[v] && !st.done[v]
}

// spill rewrites f so that v lives in memory: stored after every
// instruction that defines it, loaded before every instruction that
// reads it, and named by a fresh vreg at each of those points.
//
// An instruction that both reads and defines v — a two-address operation
// whose destination is also its source — gets both, in that order, and
// two different fresh vregs. They cannot be one vreg: the load's value
// is what the instruction reads and the store's is what it produced, and
// between those two facts the instruction ran.
func (st *spillState) spill(f *mir.Func, pool *Pool, sp Spiller, v mir.VReg) {
	slot := sp.Slot()
	class := pool.ClassOf(v)
	st.done[v] = true

	fresh := func() mir.VReg { return st.newFresh(f, pool, class) }

	for _, b := range f.Blocks {
		out := make([]mir.Instr, 0, len(b.Instrs))
		for _, in := range b.Instrs {
			reads := replace(in.Uses, v, fresh)

			// A def of the same vreg has to become the same fresh
			// one the reads did. An instruction naming one register
			// as both its source and its destination is a
			// two-address instruction — which every x86 arithmetic
			// instruction is — and giving its two ends separate
			// vregs lets the colouring put them in different
			// registers: it then reloads into one, computes into
			// another, and stores the one it did not write.
			//
			// The regression test for this is lower/i386's
			// TestRunVariadicMixed, not a unit test here. What the
			// bug needs is not just the rewrite but a colouring
			// that actually splits the pair, and constructing that
			// by hand produced a test that passed either way —
			// which is worse than none. A 32-bit backend under real
			// pressure produces it every time.
			same := fresh
			if reads.did {
				same = func() mir.VReg { return reads.with }
			}
			writes := replace(in.Defs, v, same)

			if reads.did {
				out = append(out, sp.Load(slot, reads.with, class))
				in.Uses = reads.ops
			}
			if writes.did {
				in.Defs = writes.ops
			}
			out = append(out, in)
			if writes.did {
				out = append(out, sp.Store(slot, writes.with, class))
			}
		}
		b.Instrs = out
	}
}

// newFresh makes a vreg standing in for a spilled one, in that vreg's own
// register file: a reloaded float has to be coloured out of the float
// pool, and a fresh vreg is in DefaultClass until something says so.
func (st *spillState) newFresh(f *mir.Func, pool *Pool, c Class) mir.VReg {
	v := f.NewVReg()
	pool.Classify(v, c)
	st.fresh[v] = true
	return v
}

// replaced is what one operand-list rewrite did.
type replaced struct {
	ops  []mir.VReg
	with mir.VReg
	did  bool
}

// replace swaps every occurrence of v in ops for one fresh vreg, made on
// demand so that an instruction not naming v costs nothing.
//
// One fresh vreg and not one per occurrence: an instruction that reads
// the same value twice reads one register twice, and giving it two would
// mean two loads of the same slot.
func replace(ops []mir.VReg, v mir.VReg, fresh func() mir.VReg) replaced {
	var out replaced
	for i, o := range ops {
		if o != v {
			continue
		}
		if !out.did {
			out.ops = append([]mir.VReg(nil), ops...)
			out.with = fresh()
			out.did = true
		}
		out.ops[i] = out.with
	}
	return out
}
