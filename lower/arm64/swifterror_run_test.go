package arm64_test

// Swift's error register, across a real link.
//
// A call that can fail signals it in X21: the caller clears the
// register before the call, the callee writes to it on the path that
// fails, and the caller reads it afterwards. swiftc's own code does
// exactly that -- `mov x21, #0` before the branch and `cbnz x21`
// after it -- and a function that returns normally leaves it as it
// found it.
//
// AAPCS64 has no such register, so this is a convention on top of it
// and works only because both ends say so. Which is why each test
// below has one half in hand-written assembly: a private convention
// agrees with itself, and what has to be checked is that it agrees
// with something else.

import (
	"testing"

	"github.com/vertex-language/ir"
)

// TestRunErrorIsWritten: this backend's function is the callee, and
// the caller is assembly that clears X21 and reads it back.
//
// The ordinary result comes back in X0 whether or not the error is
// set, so both are checked: an error placed in the return sequence
// would take X0 and the ordinary result would arrive in X1.
func TestRunErrorIsWritten(t *testing.T) {
	m := ir.NewModule("t", ir.AArch64Linux)
	fn := m.Func("_failing").Export()
	a := fn.ParamI64("a")
	fn.Signature().Ret(ir.TypeI64).Ret(ir.TypeI64, ir.SwiftError)
	entry := fn.Entry()
	// The value, and an error that is the value doubled -- so a test
	// cannot pass by reading one where the other belongs.
	entry.Return(a, entry.I64.Mul(a, entry.I64.Const(2)))

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
"	str  x21, [sp, #8]\n"       // the caller's X21, which _probe owes back
"	mov  x21, #0\n"             // cleared before the call
"	mov  x0, #7\n"
"	bl   _failing\n"
"	mov  x1, x21\n"             // the error
"	ldr  x21, [sp, #8]\n"
"	mov  x2, #100\n"
"	madd x0, x1, x2, x0\n"      // error * 100 + result
"	ldp  x29, x30, [sp, #16]\n"
"	add  sp, sp, #32\n"
"	ret\n"
);
long probe(void);
int main(void) {
    long r = probe();
    printf("%s %ld\n", r == 1407 ? "ok" : "WRONG", r);
    return 0;
}
`)
	if got != "ok 1407\n" {
		t.Errorf("printed %q, want %q", got, "ok 1407\n")
	}
}

// TestRunErrorIsRead: the other direction. This backend's function is
// the caller, and the callee is assembly that writes the error into
// X21 and nowhere else.
func TestRunErrorIsRead(t *testing.T) {
	m := ir.NewModule("t", ir.AArch64Linux)
	callee := m.ImportFunc("_thrower", ir.NewSig().
		Param(ir.TypeI64).
		Ret(ir.TypeI64).
		Ret(ir.TypeI64, ir.SwiftError))

	fn := m.Func("_go").Export()
	a := fn.ParamI64("a")
	fn.ReturnsI64()
	entry := fn.Entry()
	res := entry.Call(callee, a)
	value := res.Value(0).(ir.I64)
	err := res.Value(1).(ir.I64)
	entry.Return(entry.I64.Add(entry.I64.Mul(err, entry.I64.Const(100)), value))

	got := runNative(t, m, `
#include <stdio.h>
__asm__(
"	.text\n"
"	.globl _thrower\n"
"	.p2align 2\n"
"_thrower:\n"
"	lsl  x21, x0, #1\n"         // the error, in the error register
"	ret\n"                      // and the value, in X0, untouched
);
long go_(long a) __asm__("_go");
int main(void) {
    long r = go_(7);
    printf("%s %ld\n", r == 1407 ? "ok" : "WRONG", r);
    return 0;
}
`)
	if got != "ok 1407\n" {
		t.Errorf("printed %q, want %q", got, "ok 1407\n")
	}
}

// TestRunErrorIsClearedFirst: the callee writes X21 only when it
// fails, so a caller that did not clear it would read whatever was
// there and decide the call had thrown.
//
// The assembly leaves X21 alone entirely, which is what a call that
// succeeds does.
func TestRunErrorIsClearedFirst(t *testing.T) {
	m := ir.NewModule("t", ir.AArch64Linux)
	callee := m.ImportFunc("_quiet", ir.NewSig().
		Param(ir.TypeI64).
		Ret(ir.TypeI64).
		Ret(ir.TypeI64, ir.SwiftError))

	fn := m.Func("_go2").Export()
	a := fn.ParamI64("a")
	fn.ReturnsI64()
	entry := fn.Entry()
	res := entry.Call(callee, a)
	entry.Return(res.Value(1).(ir.I64))

	got := runNative(t, m, `
#include <stdio.h>
__asm__(
"	.text\n"
"	.globl _probe2\n"
"	.p2align 2\n"
"_probe2:\n"
"	sub  sp, sp, #32\n"
"	stp  x29, x30, [sp, #16]\n"
"	add  x29, sp, #16\n"
"	str  x21, [sp, #8]\n"
"	mov  x21, #4660\n"          // rubbish in the error register
"	mov  x0, #7\n"
"	bl   _go2\n"
"	ldr  x21, [sp, #8]\n"
"	ldp  x29, x30, [sp, #16]\n"
"	add  sp, sp, #32\n"
"	ret\n"
);
__asm__(
"	.text\n"
"	.globl _quiet\n"
"	.p2align 2\n"
"_quiet:\n"
"	ret\n"                      // succeeds, and leaves X21 alone
);
long probe2(void);
int main(void) {
    long r = probe2();
    printf("%s %ld\n", r == 0 ? "ok" : "STALE", r);
    return 0;
}
`)
	if got != "ok 0\n" {
		t.Errorf("printed %q, want %q", got, "ok 0\n")
	}
}
