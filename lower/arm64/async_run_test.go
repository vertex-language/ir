package arm64_test

// Swift's async context register, across a real link.
//
// An async function is handed its context -- the frame holding everything
// that has to survive a suspension -- in X22 rather than in the argument
// sequence, so its first ordinary argument is still X0. AAPCS64 has no
// such register; this is Swift's convention on top of it, and swiftc
// emits exactly this. See ir.SwiftAsync.
//
// Run tests rather than disassembly, for the reason the self register's
// are: passing the context in an argument register instead is neither a
// crash nor a link error. The callee reads whatever was in X22 and
// returns a number. The only way to catch it is to have the other end of
// the call written by something that does it properly.

import (
	"testing"

	"github.com/vertex-language/ir"
)

// TestRunAsyncContextIsRead: this backend's function is the callee, and
// the caller is assembly that puts the context in X22.
//
// The ordinary argument is what catches the other half of the mistake: if
// the context were placed in the sequence it would take X0, and `a` would
// be read out of X1.
func TestRunAsyncContextIsRead(t *testing.T) {
	m := ir.NewModule("t", ir.AArch64Linux)
	fn := m.Func("_resume").Export()
	a := fn.ParamI64("a")
	ctx := fn.ParamI64("ctx", ir.SwiftAsync)
	fn.ReturnsI64()
	entry := fn.Entry()
	// ctx * 1000 + a, so neither can be mistaken for the other.
	entry.Return(entry.I64.Add(entry.I64.Mul(ctx, entry.I64.Const(1000)), a))

	got := runNative(t, m, `
#include <stdio.h>
__asm__(
"	.text\n"
"	.globl _probe\n"
"	.p2align 2\n"
"_probe:\n"
"	sub  sp, sp, #32\n"
"	stp  x29, x30, [sp, #16]\n"
"	add  x29, sp, #16\n"
"	str  x22, [sp, #8]\n"      // the caller's X22, which _probe owes back
"	mov  x0, #9\n"             // a
"	mov  x22, #5\n"            // the async context
"	bl   _resume\n"
"	ldr  x22, [sp, #8]\n"
"	ldp  x29, x30, [sp, #16]\n"
"	add  sp, sp, #32\n"
"	ret\n"
);
long probe(void);
int main(void) {
    long r = probe();
    printf("%s %ld\n", r == 5009 ? "ok" : "WRONG", r);
    return 0;
}
`)
	if got != "ok 5009\n" {
		t.Errorf("printed %q, want %q", got, "ok 5009\n")
	}
}

// TestRunAsyncContextIsPassed: the other direction. This backend's
// function is the caller, and the callee is assembly that reads the
// context from X22 and nowhere else.
func TestRunAsyncContextIsPassed(t *testing.T) {
	m := ir.NewModule("t", ir.AArch64Linux)
	callee := m.ImportFunc("_reader", ir.NewSig().
		Param(ir.TypeI64).
		Param(ir.TypeI64, ir.SwiftAsync).
		Ret(ir.TypeI64))

	fn := m.Func("_go").Export()
	a := fn.ParamI64("a")
	ctx := fn.ParamI64("c")
	fn.ReturnsI64()
	entry := fn.Entry()
	entry.Return(entry.Call(callee, a, ctx).Value(0).(ir.I64))

	got := runNative(t, m, `
#include <stdio.h>
__asm__(
"	.text\n"
"	.globl _reader\n"
"	.p2align 2\n"
"_reader:\n"
"	mov  x1, x22\n"            // the context, from the async register
"	mov  x2, #1000\n"
"	madd x0, x1, x2, x0\n"     // ctx * 1000 + a
"	ret\n"
);
long go_(long a, long c) __asm__("_go");
int main(void) {
    long r = go_(9, 5);
    printf("%s %ld\n", r == 5009 ? "ok" : "WRONG", r);
    return 0;
}
`)
	if got != "ok 5009\n" {
		t.Errorf("printed %q, want %q", got, "ok 5009\n")
	}
}

// TestRunAsyncContextAndSelfAreDifferentRegisters: an async method has
// both -- a receiver in X20 and a context in X22 -- and neither may be
// placed where the other is, nor in the argument sequence.
func TestRunAsyncContextAndSelfAreDifferentRegisters(t *testing.T) {
	m := ir.NewModule("t", ir.AArch64Linux)
	fn := m.Func("_both").Export()
	a := fn.ParamI64("a")
	self := fn.ParamI64("self", ir.SwiftSelf)
	ctx := fn.ParamI64("ctx", ir.SwiftAsync)
	fn.ReturnsI64()
	entry := fn.Entry()
	// a + self*10 + ctx*100: every one lands in its own decimal place.
	sum := entry.I64.Add(a, entry.I64.Mul(self, entry.I64.Const(10)))
	entry.Return(entry.I64.Add(sum, entry.I64.Mul(ctx, entry.I64.Const(100))))

	got := runNative(t, m, `
#include <stdio.h>
__asm__(
"	.text\n"
"	.globl _probe\n"
"	.p2align 2\n"
"_probe:\n"
"	sub  sp, sp, #48\n"
"	stp  x29, x30, [sp, #32]\n"
"	add  x29, sp, #32\n"
"	str  x20, [sp, #8]\n"
"	str  x22, [sp, #16]\n"
"	mov  x0, #3\n"             // a
"	mov  x20, #2\n"            // self
"	mov  x22, #4\n"            // ctx
"	bl   _both\n"
"	ldr  x20, [sp, #8]\n"
"	ldr  x22, [sp, #16]\n"
"	ldp  x29, x30, [sp, #32]\n"
"	add  sp, sp, #48\n"
"	ret\n"
);
long probe(void);
int main(void) {
    long r = probe();
    printf("%s %ld\n", r == 423 ? "ok" : "WRONG", r);
    return 0;
}
`)
	if got != "ok 423\n" {
		t.Errorf("printed %q, want %q", got, "ok 423\n")
	}
}
