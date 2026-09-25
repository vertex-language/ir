package verify_test

import (
	"testing"

	"github.com/vertex-language/ir"
	"github.com/vertex-language/ir/verify"
)

// A module using the half namespaces verifies before LegalizeHalf, which is
// when a frontend checks what it built, and after it.
func TestHalfModuleVerifies(t *testing.T) {
	m := ir.NewModule("t", ir.AArch64MacOS)
	f := m.Func("f").Export()
	x := f.ParamF16("x")
	p := f.ParamPtr("p")
	f.ReturnsBF16()
	e := f.Entry()
	h := e.F16()
	y := h.Select(h.Lt(x, h.Const(0)), h.Neg(x), h.Sqrt(x))
	h.Store(y, p)
	next := f.Block("next")
	arg := next.ParamF16("v")
	e.Br(next.To(y))
	next.Return(next.BF16().FCvtF16(arg))
	g := m.Global("tbl", ir.RO, ir.Array(2, ir.StoreBF16.FType()))
	_ = g
	if err := m.Err(); err != nil {
		t.Fatal(err)
	}
	if err := verify.Module(m); err != nil {
		t.Fatalf("before legalizing: %v", err)
	}
	m.LegalizeHalf()
	if err := verify.Module(m); err != nil {
		t.Fatalf("after legalizing: %v", err)
	}
}
