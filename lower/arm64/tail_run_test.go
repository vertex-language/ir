package arm64_test

// Guaranteed tail calls, across a real link.
//
// A tail call replaces this frame rather than building one on it: the
// frame comes down, the arguments are placed, and control branches. The
// callee returns to this function's caller.
//
// It is a guarantee, not an optimisation, which is why the deep test
// below matters more than the shallow one. Swift's async functions are a
// chain of these, and a chain that consumed a frame each time would be a
// recursion as long as the program's waiting.

import (
	"testing"

	"github.com/vertex-language/ir"
)

// TestRunTailCallReturnsToTheOriginalCaller: the callee's result reaches
// the caller's caller without passing through the caller.
func TestRunTailCallReturnsToTheOriginalCaller(t *testing.T) {
	m := ir.NewModule("t", ir.AArch64Linux)

	callee := m.Func("_callee")
	x := callee.ParamI64("x")
	callee.ReturnsI64()
	ce := callee.Entry()
	ce.Return(ce.I64.Add(x, ce.I64.Const(1)))

	fn := m.Func("_go").Export()
	a := fn.ParamI64("a")
	fn.ReturnsI64()
	entry := fn.Entry()
	entry.TailCall(callee, entry.I64.Mul(a, entry.I64.Const(10)))

	got := runNative(t, m, `
#include <stdio.h>
long go_(long a) __asm__("_go");
int main(void) {
    long r = go_(4);
    printf("%s %ld\n", r == 41 ? "ok" : "WRONG", r);
    return 0;
}
`)
	if got != "ok 41\n" {
		t.Errorf("printed %q, want %q", got, "ok 41\n")
	}
}

// TestRunTailCallDoesNotGrowTheStack is the whole point.
//
// A self tail call ten million deep returns. The same written as an
// ordinary call would need something like a gigabyte of stack and would
// not: this is the difference between a tail call and a call, and there
// is no way to observe it other than by depth.
func TestRunTailCallDoesNotGrowTheStack(t *testing.T) {
	m := ir.NewModule("t", ir.AArch64Linux)

	fn := m.Func("_countdown").Export()
	n := fn.ParamI64("n")
	acc := fn.ParamI64("acc")
	fn.ReturnsI64()

	entry := fn.Entry()
	done := fn.Block("done")
	more := fn.Block("more")
	entry.BrIf(entry.I64.Eq(n, entry.I64.Const(0)), done.To(), more.To())

	done.Return(acc)
	// n - 1, acc + n, and round again without a frame.
	more.TailCall(fn,
		more.I64.Sub(n, more.I64.Const(1)),
		more.I64.Add(acc, n))

	got := runNative(t, m, `
#include <stdio.h>
long countdown(long n, long acc) __asm__("_countdown");
int main(void) {
    // Ten million frames deep if these were calls.
    long r = countdown(10000000, 0);
    printf("%s %ld\n", r == 50000005000000L ? "ok" : "WRONG", r);
    return 0;
}
`)
	if got != "ok 50000005000000\n" {
		t.Errorf("printed %q, want %q", got, "ok 50000005000000\n")
	}
}

// TestRunTailCallIndirect: the same through a pointer, which has to keep
// the target somewhere the teardown does not touch.
func TestRunTailCallIndirect(t *testing.T) {
	m := ir.NewModule("t", ir.AArch64Linux)
	ft := m.FuncType("fn_i64", ir.NewSig().Param(ir.TypeI64).Ret(ir.TypeI64))

	callee := m.Func("_double")
	x := callee.ParamI64("x")
	callee.ReturnsI64()
	ce := callee.Entry()
	ce.Return(ce.I64.Mul(x, ce.I64.Const(2)))

	fn := m.Func("_go").Export()
	a := fn.ParamI64("a")
	fn.ReturnsI64()
	entry := fn.Entry()
	// Through a pointer, and with callee-saved registers in use so the
	// teardown has something to restore before the branch.
	keep := entry.I64.Add(a, entry.I64.Const(5))
	p := entry.Ptr.GetAddr(callee)
	entry.TailCallInd(p, ft, entry.I64.Add(keep, entry.I64.Const(0)))

	got := runNative(t, m, `
#include <stdio.h>
long go_(long a) __asm__("_go");
int main(void) {
    long r = go_(11);
    printf("%s %ld\n", r == 32 ? "ok" : "WRONG", r);
    return 0;
}
`)
	if got != "ok 32\n" {
		t.Errorf("printed %q, want %q", got, "ok 32\n")
	}
}

// TestRunTailCallWithAsyncContext is the shape the refactor needs: a
// chain of tail calls each handing the next its context in X22, which is
// exactly how a suspended Swift function resumes.
func TestRunTailCallWithAsyncContext(t *testing.T) {
	m := ir.NewModule("t", ir.AArch64Linux)
	sig := ir.NewSig().Param(ir.TypeI64, ir.SwiftAsync).Ret(ir.TypeI64)

	// The last funclet: returns what the context holds.
	last := m.Func("_last")
	lctx := last.ParamI64("ctx", ir.SwiftAsync)
	last.ReturnsI64()
	le := last.Entry()
	le.Return(le.I64.Add(lctx, le.I64.Const(3)))

	// The middle one: adds to the context and tail calls the last.
	mid := m.Func("_mid")
	mctx := mid.ParamI64("ctx", ir.SwiftAsync)
	mid.ReturnsI64()
	me := mid.Entry()
	me.TailCall(last, me.I64.Mul(mctx, me.I64.Const(10)))

	_ = sig
	fn := m.Func("_go").Export()
	a := fn.ParamI64("a")
	fn.ReturnsI64()
	entry := fn.Entry()
	entry.TailCall(mid, a)

	got := runNative(t, m, `
#include <stdio.h>
long go_(long a) __asm__("_go");
int main(void) {
    long r = go_(4);   // 4 -> 40 -> 43
    printf("%s %ld\n", r == 43 ? "ok" : "WRONG", r);
    return 0;
}
`)
	if got != "ok 43\n" {
		t.Errorf("printed %q, want %q", got, "ok 43\n")
	}
}
