package inline_test

import (
	"strings"
	"testing"

	"github.com/vertex-language/ir"
	"github.com/vertex-language/ir/lower/inline"
	"github.com/vertex-language/ir/text"
	"github.com/vertex-language/ir/verify"
)

func format(t *testing.T, m *ir.Module) string {
	t.Helper()
	if err := verify.Module(m); err != nil {
		t.Fatalf("verify: %v", err)
	}
	b, err := text.Format(m)
	if err != nil {
		t.Fatal(err)
	}
	return string(b)
}

func same(t *testing.T, got, want string) {
	t.Helper()
	want = strings.TrimPrefix(want, "\n")
	if got != want {
		t.Errorf("got:\n%s\nwant:\n%s", got, want)
	}
}

// A helper with a branch and a block parameter, called twice from a
// kernel: each call is its own copy, the results reach the split-off
// continuation as its parameters, and the helper itself is untouched.
func TestInline(t *testing.T) {
	m := ir.NewModule("m", ir.AMDGCN)
	h := m.Func("clampd").Internal().ReturnsI32().NoUnwind()
	x := h.ParamI32("x")
	lim := h.ParamI32("lim")
	he := h.Entry()
	hout := h.Block("out")
	r := hout.ParamI32("r")
	he.BrIf(he.I32.SLt(x, lim), hout.To(x), hout.To(lim))
	hout.Return(hout.I32.Add(r, hout.I32.Const(1)))

	k := m.Func("k").Export().CallConv(ir.Kernel).NoUnwind()
	p := k.ParamPtr("p")
	ke := k.Entry()
	a := ke.I32.Load(p)
	b := ke.Call(h, a, ke.I32.Const(10)).I32(0)
	c := ke.Call(h, b, ke.I32.Const(20)).I32(0)
	ke.I32.Store(ke.I32.Add(b, c), p)
	ke.Return()

	if err := inline.Module(m, inline.Options{Into: func(f *ir.Func) bool { return f.Signature().CallConv() == ir.Kernel }}); err != nil {
		t.Fatal(err)
	}
	same(t, format(t, m), `
module m

use "amdgcn/hsa"

layout {
  abi        hsa,
  endian     little,
  ptrbits    64,
  stackalign 16,
  extfloat   none,
}

internal func @clampd(%x i32, %lim i32) i32 nounwind {
@entry:
  %0 = i32.slt %x, %lim
  brif %0, @out(%x), @out(%lim)

@out(%r i32):
  %1 = i32.const 1
  %2 = i32.add %r, %1
  return %2
}

export func @k kernel(%p ptr) nounwind {
@entry:
  %0 = i32.load %p
  %1 = i32.const 10
  br @clampd_entry_2

@clampd_ret_1(%2 i32):
  %3 = i32.const 20
  br @clampd_entry_5

@clampd_entry_2:
  %4 = i32.slt %0, %1
  brif %4, @clampd_out_3(%0), @clampd_out_3(%1)

@clampd_out_3(%r i32):
  %5 = i32.const 1
  %6 = i32.add %r, %5
  br @clampd_ret_1(%6)

@clampd_ret_4(%7 i32):
  %8 = i32.add %2, %7
  i32.store %8, %p
  return

@clampd_entry_5:
  %9 = i32.slt %2, %3
  brif %9, @clampd_out_6(%2), @clampd_out_6(%3)

@clampd_out_6(%10 i32):
  %11 = i32.const 1
  %12 = i32.add %10, %11
  br @clampd_ret_4(%12)
}
`)
}

// A helper that calls a helper: the inner call arrives with the outer
// body and is inlined in turn.
func TestInlineNested(t *testing.T) {
	m := ir.NewModule("m", ir.AMDGCN)
	inner := m.Func("inner").Internal().ReturnsI32().NoUnwind()
	ix := inner.ParamI32("x")
	inner.Entry().Return(inner.Entry().I32.Mul(ix, inner.Entry().I32.Const(2)))
	outer := m.Func("outer").Internal().ReturnsI32().NoUnwind()
	ox := outer.ParamI32("x")
	oe := outer.Entry()
	oe.Return(oe.I32.Add(oe.Call(inner, ox).I32(0), oe.I32.Const(1)))

	k := m.Func("k").Export().CallConv(ir.Kernel).NoUnwind()
	p := k.ParamPtr("p")
	ke := k.Entry()
	ke.I32.Store(ke.Call(outer, ke.I32.Load(p)).I32(0), p)
	ke.Return()

	if err := inline.Func(k, inline.Options{}); err != nil {
		t.Fatal(err)
	}
	got := format(t, m)
	if strings.Contains(got[strings.Index(got, "@k kernel"):], "call") {
		t.Errorf("a call survived:\n%s", got)
	}
}

// Recursion is refused by name.
func TestInlineRecursion(t *testing.T) {
	m := ir.NewModule("m", ir.AMDGCN)
	f := m.Func("f").Internal().ReturnsI32().NoUnwind()
	fx := f.ParamI32("x")
	fe := f.Entry()
	fe.Return(fe.Call(f, fx).I32(0))
	k := m.Func("k").Export().CallConv(ir.Kernel).NoUnwind()
	p := k.ParamPtr("p")
	ke := k.Entry()
	ke.I32.Store(ke.Call(f, ke.I32.Load(p)).I32(0), p)
	ke.Return()
	if err := verify.Module(m); err != nil {
		t.Fatal(err)
	}
	err := inline.Func(k, inline.Options{})
	if err == nil || !strings.Contains(err.Error(), "recursive") {
		t.Fatalf("err = %v", err)
	}
	if err := inline.Func(f, inline.Options{}); err == nil || !strings.Contains(err.Error(), "calls itself") {
		t.Fatalf("err = %v", err)
	}
}
