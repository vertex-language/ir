package arm64_test

// Swift's indirect result register, across a real link.
//
// A result that a Swift function returns `@out` arrives through the
// address in X8, whatever the result's size. That is not AAPCS64's
// rule: §6.9 puts an address in X8 only where the aggregate is too
// large to come back in registers, so a four-byte result would come
// back in W0 and no address would be passed at all. Swift's generic
// functions do it by declaration instead -- the caller is the only
// one who knows how big the result became -- and Array's subscript
// getter hands an Int32 back through a four-byte slot in X8.
//
// So each test below has one half in hand-written assembly. A private
// convention agrees with itself; what has to be checked is that it
// agrees with something else.

import (
	"testing"

	"github.com/vertex-language/ir"
)

// TestRunIndirectResultIsPassed: this backend is the caller, and the
// callee is assembly that writes through X8 and returns nothing.
//
// Four bytes, which is the size AAPCS64 would have returned in W0.
// A caller that let the size decide would pass no address, and the
// assembly would store through whatever X8 happened to hold.
func TestRunIndirectResultIsPassed(t *testing.T) {
	m := ir.NewModule("t", ir.AArch64Linux)
	callee := m.ImportFunc("_writes", ir.NewSig().
		Param(ir.TypePtr, ir.SwiftIndirectResult).
		Param(ir.TypeI64))

	fn := m.Func("_go").Export()
	a := fn.ParamI64("a")
	fn.ReturnsI64()
	entry := fn.Entry()
	slot := entry.Ptr.Alloc(4, 4)
	entry.Call(callee, slot, a)
	entry.Return(entry.I64.ZExtI32(entry.I32.Load(slot)))

	got := runNative(t, m, `
#include <stdio.h>
__asm__(
"	.text\n"
"	.globl _writes\n"
"	.p2align 2\n"
"_writes:\n"
"	lsl  w9, w0, #1\n"          // the argument doubled: X0 is the first
"	str  w9, [x8]\n"            // real argument, and the slot is X8
"	ret\n"
);
long go_(long a) __asm__("_go");
int main(void) {
    long r = go_(21);
    printf("%s %ld\n", r == 42 ? "ok" : "WRONG", r);
    return 0;
}
`)
	if got != "ok 42\n" {
		t.Errorf("printed %q, want %q", got, "ok 42\n")
	}
}

// TestRunIndirectResultIsRead: the other direction. This backend's
// function takes the parameter and writes through it, and the caller
// is assembly that supplies the address in X8 and reads it back.
func TestRunIndirectResultIsPassedBack(t *testing.T) {
	m := ir.NewModule("t", ir.AArch64Linux)
	fn := m.Func("_fills").Export()
	out := fn.ParamPtr("out", ir.SwiftIndirectResult)
	a := fn.ParamI64("a")
	entry := fn.Entry()
	entry.I32.Store(entry.I32.WrapI64(entry.I64.Mul(a, entry.I64.Const(2))), out)
	entry.Return()

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
"	add  x8, sp, #4\n"          // the caller's storage, in X8
"	mov  x0, #21\n"             // and the first real argument in X0
"	bl   _fills\n"
"	ldr  w0, [sp, #4]\n"
"	ldp  x29, x30, [sp, #16]\n"
"	add  sp, sp, #32\n"
"	ret\n"
);
long probe(void);
int main(void) {
    long r = probe();
    printf("%s %ld\n", r == 42 ? "ok" : "WRONG", r);
    return 0;
}
`)
	if got != "ok 42\n" {
		t.Errorf("printed %q, want %q", got, "ok 42\n")
	}
}
