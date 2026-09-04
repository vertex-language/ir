package mir_test

import (
	"testing"

	"github.com/vertex-language/ir/lower/mir"
)

// nop is a stand-in Op. mir does not interpret one, so a test does not
// have to invent an architecture to have instructions.
type nop struct{ name string }

func def(f *mir.Func, b *mir.Block, uses ...mir.VReg) mir.VReg {
	v := f.NewVReg()
	b.Emit(mir.Instr{Op: nop{}, Defs: []mir.VReg{v}, Uses: uses})
	return v
}

func use(b *mir.Block, vs ...mir.VReg) {
	b.Emit(mir.Instr{Op: nop{}, Uses: vs})
}

func liveSet(t *testing.T, set map[mir.VReg]bool, want ...mir.VReg) {
	t.Helper()

	if len(set) != len(want) {
		t.Errorf("live set has %d vregs, want %d: %v", len(set), len(want), set)
	}
	for _, v := range want {
		if !set[v] {
			t.Errorf("v%d is not live, and should be: %v", v, set)
		}
	}
}

// The straight-line case, where liveness agrees with reading the
// instructions in order: a value is live from its definition to its last
// use and not after.
func TestLivenessStraightLine(t *testing.T) {
	f := mir.NewFunc()
	b := f.NewBlock("entry")
	x := def(f, b)
	y := def(f, b, x) // last use of x
	use(b, y)

	l := mir.Liveness(f)
	liveSet(t, l.In[b])  // nothing enters
	liveSet(t, l.Out[b]) // nothing leaves

	// And the per-instruction view the allocator uses.
	var after []map[mir.VReg]bool
	l.LiveAfter(b, func(idx int, _ mir.Instr, live map[mir.VReg]bool) {
		snap := map[mir.VReg]bool{}
		for v := range live {
			snap[v] = true
		}
		after = append([]map[mir.VReg]bool{snap}, after...)
	})
	liveSet(t, after[0], x) // after defining x, x is live
	liveSet(t, after[1], y) // after defining y, x is dead and y is live
	liveSet(t, after[2])    // after the last use, nothing
}

// The case that makes liveness a dataflow analysis rather than a scan.
//
// x is defined in @entry and used by @loop's own instruction. @body runs
// after that use in program order and jumps back, so x has to survive
// @body — even though no instruction after its use mentions it, and
// even though @body is textually last.
func TestLivenessAcrossABackEdge(t *testing.T) {
	f := mir.NewFunc()
	entry := f.NewBlock("entry")
	loop := f.NewBlock("loop")
	body := f.NewBlock("body")
	exit := f.NewBlock("exit")

	x := def(f, entry)
	f.Succ(entry, "loop")

	use(loop, x) // the compare, on every iteration
	f.Succ(loop, "body")
	f.Succ(loop, "exit")

	t1 := def(f, body)
	use(body, t1)
	f.Succ(body, "loop")

	l := mir.Liveness(f)

	if !l.Out[body][x] {
		t.Error("x is not live out of @body; the back edge reaches its use in @loop")
	}
	if !l.In[body][x] {
		t.Error("x is not live into @body")
	}
	if l.Out[exit][x] {
		t.Error("x is live out of @exit, which returns and reaches nothing")
	}

	// And so x and t1 interfere, which is the consequence that matters:
	// t1 is defined at a point where x is still live.
	l.LiveAfter(body, func(idx int, in mir.Instr, live map[mir.VReg]bool) {
		if idx == 0 && !live[x] {
			t.Error("x is not live after @body's first instruction")
		}
	})
}

// A value used on one arm of a branch and not the other is live out of
// the block that branches, because "some path" reaches a use.
func TestLivenessAcrossOneArm(t *testing.T) {
	f := mir.NewFunc()
	entry := f.NewBlock("entry")
	yes := f.NewBlock("yes")
	no := f.NewBlock("no")

	x := def(f, entry)
	f.Succ(entry, "yes")
	f.Succ(entry, "no")

	use(yes, x)

	l := mir.Liveness(f)
	if !l.Out[entry][x] {
		t.Error("x is not live out of @entry; @yes uses it")
	}
	if !l.In[yes][x] {
		t.Error("x is not live into @yes")
	}
	if l.In[no][x] {
		t.Error("x is live into @no, which never reads it")
	}
}

// A block with no successors ends the function, so nothing is live out
// of it however much the rest of the function still names.
func TestLivenessEndsAtAReturn(t *testing.T) {
	f := mir.NewFunc()
	b := f.NewBlock("entry")
	x := def(f, b)
	use(b, x)

	if l := mir.Liveness(f); len(l.Out[b]) != 0 {
		t.Errorf("live-out of a returning block = %v, want empty", l.Out[b])
	}
}

// Succ names blocks by label, which is the identity a target's isel
// already has in hand for its branch targets.
func TestSuccByLabel(t *testing.T) {
	f := mir.NewFunc()
	a := f.NewBlock("a")
	b := f.NewBlock("b")

	f.Succ(a, "b")
	if len(a.Succs) != 1 || a.Succs[0] != b {
		t.Errorf("a.Succs = %v, want [b]", a.Succs)
	}

	// A label naming no block is a branch out of the function — a tail
	// call, an unreachable trap helper — and records no edge rather than
	// panicking on a nil.
	f.Succ(a, "elsewhere")
	if len(a.Succs) != 1 {
		t.Errorf("a.Succs = %v after an unknown label, want it unchanged", a.Succs)
	}
	if f.Block("elsewhere") != nil {
		t.Error("Block returned something for a label no block has")
	}
}
