package arm64_test

// §D3's dynamic frame: an allocation whose size is a value, and the token
// that unwinds it.

import (
	"testing"

	"github.com/vertex-language/ir"
)

// An alloca written to and read back, with a call in between — which is what
// puts the outgoing argument area underneath it and proves the two do not
// overlap.
func TestRunAlloca(t *testing.T) {
	m := ir.NewModule("t", ir.AArch64Linux)
	sink := m.ImportFunc("_sink9", ir.NewSig().
		Param(ir.TypeI64).Param(ir.TypeI64).Param(ir.TypeI64).
		Param(ir.TypeI64).Param(ir.TypeI64).Param(ir.TypeI64).
		Param(ir.TypeI64).Param(ir.TypeI64).Param(ir.TypeI64).
		Ret(ir.TypeI64))

	fn := m.Func("_dyn").Export()
	n := fn.ParamI64("n")
	fn.ReturnsI64()
	entry := fn.Entry()

	// n eightbytes, filled with their own index.
	buf := entry.Ptr.Alloca(entry.I64.Mul(n, entry.I64.Const(8)), 8)

	loop := fn.Block("loop")
	body := fn.Block("body")
	out := fn.Block("out")
	i := loop.ParamI64("i")

	entry.Br(loop.To(entry.I64.Const(0)))
	loop.BrIf(loop.I64.SLt(i, n), body.To(), out.To())
	slot := body.Ptr.Add(buf, body.I64.Mul(i, body.I64.Const(8)))
	body.I64.Store(body.I64.Mul(i, body.I64.Const(3)), slot)
	body.Br(loop.To(body.I64.Add(i, body.I64.Const(1))))

	// A call with a stack argument, after the allocation. Its outgoing
	// area is at the bottom of the frame, which the allocation moved.
	nine := make([]ir.Value, 9)
	for k := range nine {
		nine[k] = out.I64.Const(int64(k))
	}
	out.Call(sink, nine...)

	// And the buffer read back, which the call must not have written on.
	total := out.I64.Const(0)
	for k := 0; k < 4; k++ {
		p := out.Ptr.Add(buf, out.I64.Const(int64(k)*8))
		total = out.I64.Add(total, out.I64.Load(p))
	}
	out.Return(total)

	got := runNative(t, m, `
#include <stdio.h>
long sink9(long a,long b,long c,long d,long e,long f,long g,long h,long i) {
    return a+b+c+d+e+f+g+h+i;
}
long dyn(long);
int main(void) { printf("%ld\n", dyn(64)); return 0; }
`)
	// 0*3 + 1*3 + 2*3 + 3*3 = 18
	if got != "18\n" {
		t.Errorf("printed %q, want %q", got, "18\n")
	}
}

// stacksave and stackrestore around a loop of allocations, which is what
// keeps the frame from growing without bound.
func TestRunStackSaveRestore(t *testing.T) {
	m := ir.NewModule("t", ir.AArch64Linux)
	fn := m.Func("_churn").Export()
	n := fn.ParamI64("n")
	fn.ReturnsI64()
	entry := fn.Entry()

	loop := fn.Block("loop")
	body := fn.Block("body")
	out := fn.Block("out")
	i := loop.ParamI64("i")
	acc := loop.ParamI64("acc")

	entry.Br(loop.To(entry.I64.Const(0), entry.I64.Const(0)))
	loop.BrIf(loop.I64.SLt(i, n), body.To(), out.To())

	// Each iteration takes a kilobyte and gives it back. Without the
	// restore this overflows the stack long before the loop ends.
	mark := body.Ptr.StackSave()
	buf := body.Ptr.Alloca(body.I64.Const(1024), 8)
	body.I64.Store(body.I64.Add(acc, i), buf)
	next := body.I64.Load(buf)
	body.Ptr.StackRestore(mark)
	body.Br(loop.To(body.I64.Add(i, body.I64.Const(1)), next))

	out.Return(acc)

	got := runNative(t, m, `
#include <stdio.h>
long churn(long);
int main(void) { printf("%ld\n", churn(100000)); return 0; }
`)
	// The sum of 0..99999.
	if got != "4999950000\n" {
		t.Errorf("printed %q, want %q", got, "4999950000\n")
	}
}

// A zeroed alloca, whose bytes §D3 guarantees read as zero.
func TestRunAllocaZeroed(t *testing.T) {
	m := ir.NewModule("t", ir.AArch64Linux)
	fn := m.Func("_zbuf").Export()
	n := fn.ParamI64("n")
	fn.ReturnsI64()
	entry := fn.Entry()

	buf := entry.Ptr.Alloca(n, 8, ir.Zeroed)

	loop := fn.Block("loop")
	body := fn.Block("body")
	out := fn.Block("out")
	i := loop.ParamI64("i")
	acc := loop.ParamI64("acc")

	entry.Br(loop.To(entry.I64.Const(0), entry.I64.Const(0)))
	loop.BrIf(loop.I64.SLt(i, n), body.To(), out.To())
	b := body.I32.ULoad8(body.Ptr.Add(buf, i))
	body.Br(loop.To(body.I64.Add(i, body.I64.Const(1)),
		body.I64.Add(acc, body.I64.ZExtI32(b))))
	out.Return(acc)

	got := runNative(t, m, `
#include <stdio.h>
long zbuf(long);
int main(void) { printf("%ld\n", zbuf(300)); return 0; }
`)
	if got != "0\n" {
		t.Errorf("printed %q, want %q", got, "0\n")
	}
}

// A zeroed ptr.alloc, whose size was known when the frame was planned. It was
// refused outright until there was a memset to call.
func TestRunAllocZeroed(t *testing.T) {
	m := ir.NewModule("t", ir.AArch64Linux)
	fn := m.Func("_zstatic").Export()
	fn.ReturnsI64()
	entry := fn.Entry()

	buf := entry.Ptr.Alloc(64, 8, ir.Zeroed)
	// One byte written, so the answer distinguishes "all zero" from
	// "never written at all".
	entry.I32.Store8(entry.I32.Const(5), entry.Ptr.Add(buf, entry.I64.Const(7)))

	total := entry.I64.Const(0)
	for k := 0; k < 64; k++ {
		b := entry.I32.ULoad8(entry.Ptr.Add(buf, entry.I64.Const(int64(k))))
		total = entry.I64.Add(total, entry.I64.ZExtI32(b))
	}
	entry.Return(total)

	got := runNative(t, m, `
#include <stdio.h>
long zstatic(void);
int main(void) { printf("%ld\n", zstatic()); return 0; }
`)
	if got != "5\n" {
		t.Errorf("printed %q, want %q", got, "5\n")
	}
}
