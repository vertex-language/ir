package ir_test

import (
	"bytes"
	"strings"
	"testing"

	"github.com/vertex-language/ir"
	"github.com/vertex-language/ir/text"
	"github.com/vertex-language/ir/verify"
)

func printedIR(t *testing.T, m *ir.Module) string {
	t.Helper()
	var buf bytes.Buffer
	if err := text.Print(&buf, m); err != nil {
		t.Fatal(err)
	}
	return buf.String()
}

// sumTo is `long sum = 0; for (long i = 0; i < n; i++) sum += i; return sum;`
// the way a frontend that does not know yet whether an address is taken
// lowers it: two allocs, a store per assignment, a load per read.
func sumTo(t *testing.T) (*ir.Module, *ir.Func) {
	m := ir.NewModule("t", ir.AArch64Linux)
	fn := m.Func("sum").Export()
	n := fn.ParamI64("n")
	fn.ReturnsI64()

	entry := fn.Entry()
	head := fn.Block("head")
	body := fn.Block("body")
	exit := fn.Block("exit")

	sum := entry.Ptr.Alloc(8, 8)
	i := entry.Ptr.Alloc(8, 8)
	entry.I64.Store(entry.I64.Const(0), sum)
	entry.I64.Store(entry.I64.Const(0), i)
	entry.Br(head.To())

	iv := head.I64.Load(i)
	head.BrIf(head.I64.SLt(iv, n), body.To(), exit.To())

	s := body.I64.Load(sum)
	iv2 := body.I64.Load(i)
	body.I64.Store(body.I64.Add(s, iv2), sum)
	body.I64.Store(body.I64.Add(iv2, body.I64.Const(1)), i)
	body.Br(head.To())

	exit.Return(exit.I64.Load(sum))
	if err := m.Err(); err != nil {
		t.Fatal(err)
	}
	return m, fn
}

func TestPromoteLoopCounters(t *testing.T) {
	m, fn := sumTo(t)
	if n := fn.PromoteSlots(); n != 2 {
		t.Fatalf("promoted %d slots, want 2", n)
	}
	if err := verify.Module(m); err != nil {
		t.Fatalf("verify after promotion: %v\n%s", err, printedIR(t, m))
	}
	got := printedIR(t, m)
	for _, gone := range []string{"alloc", "load", "store"} {
		if strings.Contains(got, gone) {
			t.Errorf("%s survived:\n%s", gone, got)
		}
	}
	if len(fn.Blocks()[1].Params()) != 2 {
		t.Errorf("the loop head takes %d parameters, want 2:\n%s", len(fn.Blocks()[1].Params()), got)
	}
}

// An alloc whose address is passed to a call is left in memory.
func TestPromoteLeavesAnEscapingSlot(t *testing.T) {
	m := ir.NewModule("t", ir.AArch64Linux)
	ext := m.Func("use")
	ext.ParamPtr("p")
	ext.Entry().Return()
	fn := m.Func("f").Export()
	fn.ReturnsI32()
	entry := fn.Entry()
	x := entry.Ptr.Alloc(4, 4)
	entry.I32.Store(entry.I32.Const(7), x)
	entry.Call(ext, x)
	entry.Return(entry.I32.Load(x))
	if err := m.Err(); err != nil {
		t.Fatal(err)
	}
	if n := fn.PromoteSlots(); n != 0 {
		t.Fatalf("promoted %d slots, want 0", n)
	}
	if err := verify.Module(m); err != nil {
		t.Fatal(err)
	}
}

// A read with nothing stored on one path reads zero there, and the verifier
// still accepts the result.
func TestPromoteUninitializedOnOnePath(t *testing.T) {
	m := ir.NewModule("t", ir.AArch64Linux)
	fn := m.Func("f").Export()
	c := fn.ParamI32("c")
	fn.ReturnsI32()
	entry := fn.Entry()
	set := fn.Block("set")
	join := fn.Block("join")
	x := entry.Ptr.Alloc(4, 4)
	entry.BrIf(entry.I32.Ne(c, entry.I32.Const(0)), set.To(), join.To())
	set.I32.Store(set.I32.Const(5), x)
	set.Br(join.To())
	join.Return(join.I32.Load(x))
	if err := m.Err(); err != nil {
		t.Fatal(err)
	}
	if n := fn.PromoteSlots(); n != 1 {
		t.Fatalf("promoted %d slots, want 1", n)
	}
	if err := verify.Module(m); err != nil {
		t.Fatalf("verify: %v\n%s", err, printedIR(t, m))
	}
	got := printedIR(t, m)
	if !strings.Contains(got, "br @join(%3)") || !strings.Contains(got, "@join(%0)") {
		t.Errorf("the branches into join do not pass the value on each path:\n%s", got)
	}
}

// A byte slot -- a C++ bool or char -- stored as the low byte of an i32 and
// read back zero- or sign-extended becomes the value masked or shifted the
// same way.
func TestPromoteSubWidth(t *testing.T) {
	m := ir.NewModule("t", ir.AArch64Linux)
	fn := m.Func("f").Export()
	c := fn.ParamI32("c")
	fn.ReturnsI32()
	entry := fn.Entry()
	set := fn.Block("set")
	join := fn.Block("join")
	x := entry.Ptr.Alloc(1, 1)
	entry.I32.Store8(entry.I32.Const(0x1ff), x)
	entry.BrIf(entry.I32.Ne(c, entry.I32.Const(0)), set.To(), join.To())
	set.I32.Store8(set.I32.Const(0x80), x)
	set.Br(join.To())
	u := join.I32.ULoad8(x)
	s := join.I32.SLoad8(x)
	join.Return(join.I32.Add(u, s))
	if err := m.Err(); err != nil {
		t.Fatal(err)
	}
	if n := fn.PromoteSlots(); n != 1 {
		t.Fatalf("promoted %d slots, want 1", n)
	}
	if err := verify.Module(m); err != nil {
		t.Fatalf("verify: %v\n%s", err, printedIR(t, m))
	}
	got := printedIR(t, m)
	for _, want := range []string{"i32.and", "i32.shl", "i32.sshr"} {
		if !strings.Contains(got, want) {
			t.Errorf("no %s:\n%s", want, got)
		}
	}
	if strings.Contains(got, "store8") || strings.Contains(got, "load8") {
		t.Errorf("the slot survived:\n%s", got)
	}
}
