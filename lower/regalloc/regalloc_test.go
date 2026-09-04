package regalloc_test

import (
	"errors"
	"testing"

	"github.com/vertex-language/ir/lower/mir"
	"github.com/vertex-language/ir/lower/regalloc"
)

func TestAssignHonorsPins(t *testing.T) {
	f := mir.NewFunc()
	blk := f.NewBlock("entry")
	a, b := f.NewVReg(), f.NewVReg()
	result := f.NewVReg()
	blk.Emit(mir.Instr{Op: "add", Defs: []mir.VReg{result}, Uses: []mir.VReg{a, b}})

	pool := regalloc.NewPool([]regalloc.PhysReg{10, 11, 12})
	pool.Pin(a, 1)
	pool.Pin(b, 2)

	assigned, err := regalloc.Assign(f, pool)
	if err != nil {
		t.Fatalf("Assign: %v", err)
	}
	if assigned[a] != 1 || assigned[b] != 2 {
		t.Errorf("pinned vregs = %v, %v, want 1, 2", assigned[a], assigned[b])
	}
	if assigned[result] != 10 {
		t.Errorf("result = %v, want the pool's first free register (10)", assigned[result])
	}
}

func TestAssignReusesTheSameRegisterForRepeatedUses(t *testing.T) {
	f := mir.NewFunc()
	blk := f.NewBlock("entry")
	v := f.NewVReg()
	blk.Emit(mir.Instr{Op: "use", Uses: []mir.VReg{v}})
	blk.Emit(mir.Instr{Op: "use", Uses: []mir.VReg{v}})

	pool := regalloc.NewPool([]regalloc.PhysReg{10, 11})
	assigned, err := regalloc.Assign(f, pool)
	if err != nil {
		t.Fatalf("Assign: %v", err)
	}
	if assigned[v] != 10 {
		t.Errorf("v = %v, want 10", assigned[v])
	}
}

func TestAssignFailsLoudlyWhenThePoolRunsOut(t *testing.T) {
	f := mir.NewFunc()
	blk := f.NewBlock("entry")
	a, b, c := f.NewVReg(), f.NewVReg(), f.NewVReg()
	blk.Emit(mir.Instr{Op: "add3", Uses: []mir.VReg{a, b, c}})

	pool := regalloc.NewPool([]regalloc.PhysReg{10, 11})
	if _, err := regalloc.Assign(f, pool); !errors.Is(err, regalloc.ErrOutOfRegisters) {
		t.Errorf("Assign = %v, want ErrOutOfRegisters", err)
	}
}

// The point of the interference graph: two vregs whose live ranges do
// not overlap share a register, even when the pool has spares. A pool
// with two registers colours four vregs here because only ever one of
// them is live.
func TestAssignSharesARegisterBetweenDisjointRanges(t *testing.T) {
	f := mir.NewFunc()
	blk := f.NewBlock("entry")

	var last mir.VReg
	for i := 0; i < 4; i++ {
		v := f.NewVReg()
		blk.Emit(mir.Instr{Op: "def", Defs: []mir.VReg{v}})
		blk.Emit(mir.Instr{Op: "use", Uses: []mir.VReg{v}})
		last = v
	}

	pool := regalloc.NewPool([]regalloc.PhysReg{10, 11})
	assigned, err := regalloc.Assign(f, pool)
	if err != nil {
		t.Fatalf("Assign: %v", err)
	}
	if len(assigned) != 4 {
		t.Errorf("assigned %d vregs, want 4", len(assigned))
	}
	// Every one of them takes the pool's first register: each is dead
	// before the next is defined, so none of them interferes with any
	// other.
	for v, r := range assigned {
		if r != 10 {
			t.Errorf("v%d = %v, want 10 — nothing else is live", v, r)
		}
	}
	_ = last
}

// A value live across another's definition does not share with it, which
// is the same graph read the other way.
func TestAssignSeparatesOverlappingRanges(t *testing.T) {
	f := mir.NewFunc()
	blk := f.NewBlock("entry")

	a := f.NewVReg()
	b := f.NewVReg()
	blk.Emit(mir.Instr{Op: "def", Defs: []mir.VReg{a}})
	blk.Emit(mir.Instr{Op: "def", Defs: []mir.VReg{b}})
	blk.Emit(mir.Instr{Op: "use", Uses: []mir.VReg{a, b}})

	pool := regalloc.NewPool([]regalloc.PhysReg{10, 11})
	assigned, err := regalloc.Assign(f, pool)
	if err != nil {
		t.Fatalf("Assign: %v", err)
	}
	if assigned[a] == assigned[b] {
		t.Errorf("a and b both got %v; they are live at the same time", assigned[a])
	}
}

// testSpiller is a Spiller that emits recognisable placeholder
// instructions. mir does not interpret an Op, so a test does not need an
// architecture to have a store and a load.
type testSpiller struct {
	next    int
	stores  int
	loads   int
	classes []regalloc.Class // the class of every store, in order
}

type spillMark struct{ slot int }
type reloadMark struct{ slot int }

func (s *testSpiller) Slot() int { s.next++; return s.next }

func (s *testSpiller) Store(slot int, v mir.VReg, c regalloc.Class) mir.Instr {
	s.stores++
	s.classes = append(s.classes, c)
	return mir.Instr{Op: spillMark{slot}, Uses: []mir.VReg{v}}
}

func (s *testSpiller) Load(slot int, v mir.VReg, c regalloc.Class) mir.Instr {
	s.loads++
	return mir.Instr{Op: reloadMark{slot}, Defs: []mir.VReg{v}}
}

// Given a Spiller, a graph that needs more registers than the pool has
// is rewritten rather than refused.
func TestSpillingColoursWhatAssignCannot(t *testing.T) {
	f := mir.NewFunc()
	blk := f.NewBlock("entry")

	// Four values defined up front and read one at a time afterwards, so
	// all four are live across the first use, against a pool of two.
	//
	// Read one at a time and not all at once, because an instruction
	// naming four registers needs four registers and no amount of
	// spilling changes that. What spilling shortens is the distance
	// between a definition and a use, not the width of an instruction.
	var vs []mir.VReg
	for i := 0; i < 4; i++ {
		v := f.NewVReg()
		blk.Emit(mir.Instr{Op: "def", Defs: []mir.VReg{v}})
		vs = append(vs, v)
	}
	for _, v := range vs {
		blk.Emit(mir.Instr{Op: "use", Uses: []mir.VReg{v}})
	}

	pool := regalloc.NewPool([]regalloc.PhysReg{10, 11})
	if _, err := regalloc.Assign(f, pool); !errors.Is(err, regalloc.ErrOutOfRegisters) {
		t.Fatalf("Assign without a spiller = %v, want ErrOutOfRegisters", err)
	}

	sp := &testSpiller{}
	assigned, err := regalloc.Spilling(f, regalloc.NewPool([]regalloc.PhysReg{10, 11}), sp)
	if err != nil {
		t.Fatalf("Spilling: %v", err)
	}
	if sp.stores == 0 || sp.loads == 0 {
		t.Errorf("spilled with %d stores and %d loads; want both non-zero", sp.stores, sp.loads)
	}
	// Every vreg the rewritten function names has a register, including
	// the ones the rewrite itself created.
	for _, b := range f.Blocks {
		for _, in := range b.Instrs {
			for _, v := range append(append([]mir.VReg{}, in.Defs...), in.Uses...) {
				if _, ok := assigned[v]; !ok {
					t.Errorf("v%d has no register after spilling", v)
				}
			}
		}
	}
}

// Pressure spilling cannot relieve still fails, rather than looping.
//
// c is spillable, but spilling it does not help: the reload has to land
// in a register at a point where both pool registers are held by pinned
// values, and the vregs a rewrite creates are never spilled again. One
// round, no progress, and an honest error.
func TestSpillingGivesUpWhenSpillingCannotHelp(t *testing.T) {
	f := mir.NewFunc()
	blk := f.NewBlock("entry")
	a, b, c := f.NewVReg(), f.NewVReg(), f.NewVReg()
	blk.Emit(mir.Instr{Op: "def", Defs: []mir.VReg{c}})
	blk.Emit(mir.Instr{Op: "use", Uses: []mir.VReg{a, b, c}})

	pool := regalloc.NewPool([]regalloc.PhysReg{10, 11})
	pool.Pin(a, 10)
	pool.Pin(b, 11)

	if _, err := regalloc.Spilling(f, pool, &testSpiller{}); !errors.Is(err, regalloc.ErrOutOfRegisters) {
		t.Errorf("Spilling = %v, want ErrOutOfRegisters", err)
	}
}

// ── register classes ──────────────────────────────────────────────────

const floatClass regalloc.Class = 1

// Two classes with the same register numbers, which is the case the
// design is for: on AMD64 RAX and XMM0 are both register zero, and a
// target that had to renumber one of them to keep them apart would be
// working around this package rather than using it.
//
// Both vregs here are live at once and both take register zero, because
// the registers they are competing for are not the same registers.
func TestClassesShareRegisterNumbers(t *testing.T) {
	f := mir.NewFunc()
	blk := f.NewBlock("entry")
	i, x := f.NewVReg(), f.NewVReg()
	blk.Emit(mir.Instr{Op: "def", Defs: []mir.VReg{i}})
	blk.Emit(mir.Instr{Op: "def", Defs: []mir.VReg{x}})
	blk.Emit(mir.Instr{Op: "use", Uses: []mir.VReg{i, x}})

	pool := regalloc.NewPool([]regalloc.PhysReg{0, 1, 2})
	pool.AddClass(floatClass, []regalloc.PhysReg{0, 1})
	pool.Classify(x, floatClass)

	assigned, err := regalloc.Assign(f, pool)
	if err != nil {
		t.Fatalf("Assign: %v", err)
	}
	if assigned[i] != 0 || assigned[x] != 0 {
		t.Errorf("assigned = %v, %v; want both to take register 0 out of their own class",
			assigned[i], assigned[x])
	}
}

// A pin conflict is a conflict only within one class. Two vregs pinned to
// register zero in different files are two facts about two registers.
func TestPinConflictIsPerClass(t *testing.T) {
	f := mir.NewFunc()
	blk := f.NewBlock("entry")
	i, x := f.NewVReg(), f.NewVReg()
	blk.Emit(mir.Instr{Op: "use", Uses: []mir.VReg{i, x}})

	pool := regalloc.NewPool([]regalloc.PhysReg{0, 1, 2})
	pool.AddClass(floatClass, []regalloc.PhysReg{0, 1})
	pool.Classify(x, floatClass)
	pool.Pin(i, 0)
	pool.Pin(x, 0)

	if _, err := regalloc.Assign(f, pool); err != nil {
		t.Errorf("Assign = %v; two pins on register zero in different classes do not conflict", err)
	}
}

// Pressure in one file does not spend the other file's registers, and a
// spilled value comes back into its own.
func TestSpillingStaysInTheValuesOwnClass(t *testing.T) {
	f := mir.NewFunc()
	blk := f.NewBlock("entry")

	// Three floats live at once against a float pool of two, with the
	// integer pool untouched and wide open.
	var xs []mir.VReg
	for i := 0; i < 3; i++ {
		v := f.NewVReg()
		blk.Emit(mir.Instr{Op: "def", Defs: []mir.VReg{v}})
		xs = append(xs, v)
	}
	for _, v := range xs {
		blk.Emit(mir.Instr{Op: "use", Uses: []mir.VReg{v}})
	}

	pool := regalloc.NewPool([]regalloc.PhysReg{10, 11, 12, 13, 14, 15})
	pool.AddClass(floatClass, []regalloc.PhysReg{0, 1})
	for _, v := range xs {
		pool.Classify(v, floatClass)
	}

	sp := &testSpiller{}
	assigned, err := regalloc.Spilling(f, pool, sp)
	if err != nil {
		t.Fatalf("Spilling: %v", err)
	}
	if sp.stores == 0 {
		t.Fatal("nothing spilled; three live values against a pool of two")
	}
	for i, c := range sp.classes {
		if c != floatClass {
			t.Errorf("store %d had class %v, want the float class", i, c)
		}
	}
	// Every register handed out, the reloads' included, is one of the
	// float pool's — never one of the six integer registers standing
	// idle next to it.
	for v, r := range assigned {
		if r != 0 && r != 1 {
			t.Errorf("v%d got register %v, which is not in the float pool", v, r)
		}
	}
}

// Two values simultaneously live interfere whether or not an instruction
// writes either, and the graph now argues that the write-based rule covers
// every such pair except among values nothing writes. This is that argument
// checked against the thing it replaced: a live-in pair with no write
// between them still has to come out in two registers.
func TestSimultaneousLiveWithoutAWrite(t *testing.T) {
	f := mir.NewFunc()
	b := f.NewBlock("entry")

	// a and c arrive live and are never written — parameters. They are read
	// together, so they cannot share a register.
	a, c := f.NewVReg(), f.NewVReg()
	d := f.NewVReg()
	b.Emit(mir.Instr{Op: "use", Defs: []mir.VReg{d}, Uses: []mir.VReg{a, c}})
	b.Emit(mir.Instr{Op: "ret", Uses: []mir.VReg{d}})

	pool := regalloc.NewPool([]regalloc.PhysReg{0, 1, 2})
	got, err := regalloc.Assign(f, pool)
	if err != nil {
		t.Fatalf("Assign: %v", err)
	}
	if got[a] == got[c] {
		t.Errorf("a and c are both live into the use and both got %v", got[a])
	}
}

// A value that is written somewhere is separated by the write-based rule
// alone, which is what lets the pairwise walk skip all but the unwritten
// ones. Two long-lived written values that overlap must still differ.
func TestSimultaneousLiveWithAWrite(t *testing.T) {
	f := mir.NewFunc()
	entry, mid, end := f.NewBlock("entry"), f.NewBlock("mid"), f.NewBlock("end")
	entry.Succs = []*mir.Block{mid}
	mid.Succs = []*mir.Block{end}

	x, y := f.NewVReg(), f.NewVReg()
	entry.Emit(mir.Instr{Op: "def", Defs: []mir.VReg{x}})
	mid.Emit(mir.Instr{Op: "def", Defs: []mir.VReg{y}})
	// Both read after both are written: their ranges overlap across a
	// block boundary, and nothing in `end` writes either.
	end.Emit(mir.Instr{Op: "use", Uses: []mir.VReg{x, y}})

	pool := regalloc.NewPool([]regalloc.PhysReg{0, 1})
	got, err := regalloc.Assign(f, pool)
	if err != nil {
		t.Fatalf("Assign: %v", err)
	}
	if got[x] == got[y] {
		t.Errorf("x and y overlap across a block and both got %v", got[x])
	}
}
