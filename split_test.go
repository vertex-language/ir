package ir_test

import (
	"strings"
	"testing"

	"github.com/vertex-language/ir"
	"github.com/vertex-language/ir/verify"
)

// pairSlot is a struct of a pointer and a count kept in one frame slot,
// its fields reached by offset -- what a frontend makes of `let s =
// Span(base: p, count: n)` -- and read back.
func pairSlot(t *testing.T) (*ir.Module, *ir.Func) {
	m := ir.NewModule("t", ir.AArch64Linux)
	fn := m.Func("pair").Export()
	p := fn.ParamPtr("p")
	n := fn.ParamI64("n")
	fn.ReturnsI64()
	b := fn.Entry()
	slot := b.Ptr.Alloc(16, 8)
	b.Ptr.Store(p, slot)
	b.I64.Store(n, b.Ptr.Add(slot, b.I64.Const(8)))
	base := b.Ptr.Load(slot)
	count := b.I64.Load(b.Ptr.Add(slot, b.I64.Const(8)))
	b.Return(b.I64.Add(b.I64.FromPtr(base), count))
	return m, fn
}

// A slot reached by offsets is its fields, each promoted on its own.
func TestSplitSlotsThenPromote(t *testing.T) {
	m, fn := pairSlot(t)
	if got := fn.SplitSlots(); got != 1 {
		t.Fatalf("SplitSlots split %d, want 1", got)
	}
	if got := fn.PromoteSlots(); got != 2 {
		t.Fatalf("PromoteSlots promoted %d after the split, want the 2 fields", got)
	}
	if err := verify.Module(m); err != nil {
		t.Fatal(err)
	}
	if out := printedIR(t, m); strings.Contains(out, "alloc") || strings.Contains(out, "load") {
		t.Errorf("a slot is left:\n%s", out)
	}
}

// Fields that overlap are memory, and are left as they are.
func TestSplitSlotsLeavesOverlap(t *testing.T) {
	m := ir.NewModule("t", ir.AArch64Linux)
	fn := m.Func("overlap").Export()
	n := fn.ParamI64("n")
	fn.ReturnsI32()
	b := fn.Entry()
	slot := b.Ptr.Alloc(16, 8)
	b.I64.Store(n, slot)
	b.Return(b.I32.Load(b.Ptr.Add(slot, b.I64.Const(4))))
	if got := fn.SplitSlots(); got != 0 {
		t.Fatalf("split an overlapping slot")
	}
	if err := verify.Module(m); err != nil {
		t.Fatal(err)
	}
}

// A slot whose address leaves it is not split.
func TestSplitSlotsLeavesAnEscape(t *testing.T) {
	m := ir.NewModule("t", ir.AArch64Linux)
	fn := m.Func("escape").Export()
	out := fn.ParamPtr("out")
	b := fn.Entry()
	slot := b.Ptr.Alloc(16, 8)
	b.I64.Store(b.I64.Const(1), b.Ptr.Add(slot, b.I64.Const(8)))
	b.Ptr.Store(slot, out)
	b.Return()
	if got := fn.SplitSlots(); got != 0 {
		t.Fatalf("split a slot whose address is stored")
	}
	if err := verify.Module(m); err != nil {
		t.Fatal(err)
	}
}
