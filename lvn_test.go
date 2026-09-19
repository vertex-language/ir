package ir_test

import (
	"strings"
	"testing"

	"github.com/vertex-language/ir"
	"github.com/vertex-language/ir/verify"
)

// lineEnd is `while i < to && raw[i] != 13 && raw[i] != 10 { i += 1 }`,
// bounds-checked, the way vsc lowers it: each raw[i] loads the count,
// compares, branches to a trap, and loads the byte, and each && joins
// through a block that branches on an i1 parameter.
func lineEnd(t *testing.T) (*ir.Module, *ir.Func) {
	m := ir.NewModule("t", ir.AArch64Linux)
	fn := m.Func("lineEnd").Export()
	raw := fn.ParamPtr("raw")
	from := fn.ParamI64("from")
	to := fn.ParamI64("to")
	fn.ReturnsI64()

	entry := fn.Entry()
	head := fn.Block("head")
	i := head.ParamI64("i")
	chk1 := fn.Block("chk1")
	no1 := fn.Block("no1")
	and1 := fn.Block("and1")
	c1 := and1.ParamI1("c1")
	chk2 := fn.Block("chk2")
	no2 := fn.Block("no2")
	and2 := fn.Block("and2")
	c2 := and2.ParamI1("c2")
	body := fn.Block("body")
	exit := fn.Block("exit")
	ok1 := fn.Block("ok1")
	ok2 := fn.Block("ok2")
	fatal := fn.Block("fatal")

	entry.Br(head.To(from))
	head.BrIf(head.I64.SLt(i, to), chk1.To(), no1.To())

	byteAt := func(b *ir.Block, ok *ir.Block) {
		count := b.I64.Load(b.Ptr.Add(raw, b.I64.Const(16)))
		b.BrIf(b.I64.ULe(count, i), fatal.To(), ok.To())
	}
	byteAt(chk1, ok1)
	no1.Br(and1.To(no1.I1.Const(false)))
	v1 := ok1.I32.ULoad8(ok1.Ptr.Add(ok1.Ptr.Add(raw, ok1.I64.Const(40)), i))
	ok1.Br(and1.To(ok1.I32.Ne(v1, ok1.I32.Const(13))))
	and1.BrIf(c1, chk2.To(), no2.To())

	byteAt(chk2, ok2)
	no2.Br(and2.To(no2.I1.Const(false)))
	v2 := ok2.I32.ULoad8(ok2.Ptr.Add(ok2.Ptr.Add(raw, ok2.I64.Const(40)), i))
	ok2.Br(and2.To(ok2.I32.Ne(v2, ok2.I32.Const(10))))
	and2.BrIf(c2, body.To(), exit.To())

	body.Br(head.To(body.I64.Add(i, body.I64.Const(1))))
	exit.Return(i)
	fatal.Trap()
	if err := m.Err(); err != nil {
		t.Fatal(err)
	}
	return m, fn
}

func TestNumberValuesRepeatedIndex(t *testing.T) {
	m, fn := lineEnd(t)
	if n := fn.ThreadJoins(); n != 4 {
		t.Errorf("threaded %d branches, want 4", n)
	}
	removed := fn.NumberValues()
	if err := verify.Module(m); err != nil {
		t.Fatalf("verify: %v\n%s", err, printedIR(t, m))
	}
	got := printedIR(t, m)
	// One count load, one bounds check, one byte address survive.
	if c := strings.Count(got, "i64.load"); c != 1 {
		t.Errorf("%d count loads, want 1:\n%s", c, got)
	}
	if c := strings.Count(got, "i64.ule"); c != 1 {
		t.Errorf("%d bounds checks, want 1:\n%s", c, got)
	}
	if c := strings.Count(got, "uload8"); c != 1 {
		t.Errorf("%d byte loads, want 1:\n%s", c, got)
	}
	if removed == 0 {
		t.Errorf("nothing removed")
	}
}

// A store between two loads of one address keeps both.
func TestNumberValuesStoreKillsLoads(t *testing.T) {
	m := ir.NewModule("t", ir.AArch64Linux)
	fn := m.Func("f").Export()
	p := fn.ParamPtr("p")
	fn.ReturnsI64()
	e := fn.Entry()
	a := e.I64.Load(p)
	e.I64.Store(e.I64.Const(9), p)
	b := e.I64.Load(p)
	e.Return(e.I64.Add(a, b))
	if err := m.Err(); err != nil {
		t.Fatal(err)
	}
	fn.NumberValues()
	if err := verify.Module(m); err != nil {
		t.Fatal(err)
	}
	if c := strings.Count(printedIR(t, m), "i64.load"); c != 2 {
		t.Errorf("%d loads, want 2", c)
	}
}

// A division whose answer is unused still runs: dividing by zero traps.
func TestDropDeadKeepsATrap(t *testing.T) {
	m := ir.NewModule("t", ir.AArch64Linux)
	fn := m.Func("f").Export()
	a := fn.ParamI64("a")
	b := fn.ParamI64("b")
	fn.ReturnsI64()
	e := fn.Entry()
	e.I64.SDiv(a, b)
	e.I64.Add(a, b)
	e.Return(a)
	if err := m.Err(); err != nil {
		t.Fatal(err)
	}
	fn.NumberValues()
	got := printedIR(t, m)
	if !strings.Contains(got, "sdiv") || strings.Contains(got, "i64.add") {
		t.Errorf("want the sdiv kept and the add dropped:\n%s", got)
	}
}
