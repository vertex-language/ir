package mir

// Liveness: which virtual registers hold a value some later instruction
// still needs.
//
// A vreg is live at a point when some path from that point reaches a use
// of it before a redefinition. "Some path" is why this is a dataflow
// analysis over Succs rather than a scan of an instruction list: a value
// defined before a loop and used inside it is live throughout the loop
// body, including the instructions that come after its last textual use.
//
// This is the standard backward equation and nothing more:
//
//	live-in(B)  = use(B) ∪ (live-out(B) \ def(B))
//	live-out(B) = ⋃ live-in(S) for each successor S
//
// where use(B) is the vregs B reads before writing and def(B) is the ones
// it writes. Iterated to a fixed point, which it reaches because both
// sets only grow and there are finitely many vregs.

// Live is one function's liveness: which vregs are live on entry to and
// on exit from each block.
//
// Per block rather than per instruction, because a block's interior is a
// straight line: a caller that wants the live set at some instruction
// walks the block backwards from Out, which is what LiveAfter does and
// what an allocator wants anyway.
type Live struct {
	In  map[*Block]map[VReg]bool
	Out map[*Block]map[VReg]bool
}

// Liveness solves the equations above for f.
func Liveness(f *Func) *Live {
	l := &Live{
		In:  make(map[*Block]map[VReg]bool, len(f.Blocks)),
		Out: make(map[*Block]map[VReg]bool, len(f.Blocks)),
	}

	// use and def are local to a block and do not change, so they are
	// computed once rather than each round.
	uses := make(map[*Block]map[VReg]bool, len(f.Blocks))
	defs := make(map[*Block]map[VReg]bool, len(f.Blocks))
	for _, b := range f.Blocks {
		use, def := map[VReg]bool{}, map[VReg]bool{}
		for _, in := range b.Instrs {
			// Uses first: a vreg read by this instruction is an upward
			// exposed use unless an earlier instruction in this block
			// already defined it.
			for _, v := range in.Uses {
				if !def[v] {
					use[v] = true
				}
			}
			for _, v := range in.Defs {
				def[v] = true
			}
		}
		uses[b], defs[b] = use, def
		l.In[b] = map[VReg]bool{}
		l.Out[b] = map[VReg]bool{}
	}

	// Reverse block order converges fastest for a forward-ordered list,
	// which is what a target's isel produces, but the fixed point does
	// not depend on the order.
	for changed := true; changed; {
		changed = false
		for i := len(f.Blocks) - 1; i >= 0; i-- {
			b := f.Blocks[i]

			out := map[VReg]bool{}
			for _, s := range b.Succs {
				for v := range l.In[s] {
					out[v] = true
				}
			}

			in := map[VReg]bool{}
			for v := range uses[b] {
				in[v] = true
			}
			for v := range out {
				if !defs[b][v] {
					in[v] = true
				}
			}

			if !sameSet(out, l.Out[b]) || !sameSet(in, l.In[b]) {
				l.Out[b], l.In[b] = out, in
				changed = true
			}
		}
	}
	return l
}

// LiveAfter calls fn once per instruction of b, in reverse order, with
// the set of vregs live immediately after that instruction.
//
// After rather than before, because that is the set a register allocator
// asks about: an instruction's destination may not land on a register
// something still needs once the instruction has run. What it may land on
// is a register holding a value this very instruction was the last reader
// of, which is the whole benefit and the reason "after" is not "before".
//
// The set passed in is reused between calls. A caller that keeps it must
// copy it.
func (l *Live) LiveAfter(b *Block, fn func(idx int, in Instr, live map[VReg]bool)) {
	live := map[VReg]bool{}
	for v := range l.Out[b] {
		live[v] = true
	}

	for i := len(b.Instrs) - 1; i >= 0; i-- {
		in := b.Instrs[i]
		fn(i, in, live)

		// Step back over the instruction: what it defined was not live
		// before it ran, and what it reads was.
		for _, v := range in.Defs {
			delete(live, v)
		}
		for _, v := range in.Uses {
			live[v] = true
		}
	}
}

func sameSet(a, b map[VReg]bool) bool {
	if len(a) != len(b) {
		return false
	}
	for v := range a {
		if !b[v] {
			return false
		}
	}
	return true
}
