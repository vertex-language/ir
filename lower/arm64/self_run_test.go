package arm64_test

// Swift's self register, across a real link.
//
// A Swift method takes its receiver in X20 rather than in the argument
// sequence, so the first ordinary argument is still X0 and the receiver
// costs no argument register. AAPCS64 has no such register: this is a
// convention on top of it, and the only thing that makes it work is that
// both ends say so.
//
// Which is why these are run tests and not disassembly. Putting the
// receiver in an argument register instead is not a crash and not a link
// error — the callee reads whatever was in X20 and returns a number. The
// only way to catch that is to have the other end of the call written by
// something that does it properly, so each test below has one half in hand
// written assembly.

import (
	"testing"

	"github.com/vertex-language/ir"
)

// TestRunSelfIsRead: this backend's function is the callee, and the caller
// is assembly that puts the receiver in X20.
//
// The trailing argument is what catches the other half of the mistake: if
// the receiver were placed in the sequence, it would take X0 and `a` would
// be read out of X1.
func TestRunSelfIsRead(t *testing.T) {
	m := ir.NewModule("t", ir.AArch64Linux)
	fn := m.Func("_method").Export()
	a := fn.ParamI64("a")
	self := fn.ParamI64("self", ir.SwiftSelf)
	fn.ReturnsI64()
	entry := fn.Entry()
	// self * 100 + a, so the two cannot be confused for each other.
	entry.Return(entry.I64.Add(entry.I64.Mul(self, entry.I64.Const(100)), a))

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
"	str  x20, [sp, #8]\n"      // the caller's X20, which _probe owes back
"	mov  x0, #7\n"             // a
"	mov  x20, #4\n"            // self
"	bl   _method\n"
"	ldr  x20, [sp, #8]\n"
"	ldp  x29, x30, [sp, #16]\n"
"	add  sp, sp, #32\n"
"	ret\n"
);
long probe(void);
int main(void) {
    long r = probe();
    printf("%s %ld\n", r == 407 ? "ok" : "WRONG", r);
    return 0;
}
`)
	if got != "ok 407\n" {
		t.Errorf("printed %q, want %q", got, "ok 407\n")
	}
}

// TestRunSelfIsPassed: the other direction. This backend's function is the
// caller, and the callee is assembly that reads the receiver from X20 and
// nowhere else.
func TestRunSelfIsPassed(t *testing.T) {
	m := ir.NewModule("t", ir.AArch64Linux)
	callee := m.ImportFunc("_reader", ir.NewSig().
		Param(ir.TypeI64).
		Param(ir.TypeI64, ir.SwiftSelf).
		Ret(ir.TypeI64))

	fn := m.Func("_go").Export()
	a := fn.ParamI64("a")
	self := fn.ParamI64("s")
	fn.ReturnsI64()
	entry := fn.Entry()
	entry.Return(entry.Call(callee, a, self).Value(0).(ir.I64))

	got := runNative(t, m, `
#include <stdio.h>
__asm__(
"	.text\n"
"	.globl _reader\n"
"	.p2align 2\n"
"_reader:\n"
"	mov  x1, x20\n"            // the receiver, from the self register
"	mov  x2, #100\n"
"	madd x0, x1, x2, x0\n"     // self * 100 + a
"	ret\n"
);
long go_(long a, long s) __asm__("_go");
int main(void) {
    long r = go_(7, 4);
    printf("%s %ld\n", r == 407 ? "ok" : "WRONG", r);
    return 0;
}
`)
	if got != "ok 407\n" {
		t.Errorf("printed %q, want %q", got, "ok 407\n")
	}
}

// TestRunSelfSurvivesPressure: X20 is callee-saved, and a caller that puts
// the receiver there is writing a register the allocator may already be
// using. Enough live values to force it into the callee-saved half, and a
// call with a receiver in the middle of them.
func TestRunSelfSurvivesPressure(t *testing.T) {
	m := ir.NewModule("t", ir.AArch64Linux)
	callee := m.ImportFunc("_reader2", ir.NewSig().
		Param(ir.TypeI64).
		Param(ir.TypeI64, ir.SwiftSelf).
		Ret(ir.TypeI64))

	fn := m.Func("_go2").Export()
	a := fn.ParamI64("a")
	fn.ReturnsI64()
	entry := fn.Entry()

	// Twenty values live across the call, which is more than the
	// caller-saved half of the file holds.
	var live []ir.I64
	for i := 0; i < 20; i++ {
		live = append(live, entry.I64.Add(a, entry.I64.Const(int64(i))))
	}
	got := entry.Call(callee, live[0], live[1]).Value(0).(ir.I64)
	acc := got
	for _, v := range live {
		acc = entry.I64.Add(acc, v)
	}
	entry.Return(acc)

	out := runNative(t, m, `
#include <stdio.h>
__asm__(
"	.text\n"
"	.globl _reader2\n"
"	.p2align 2\n"
"_reader2:\n"
"	mov  x1, x20\n"
"	mov  x2, #100\n"
"	madd x0, x1, x2, x0\n"
"	ret\n"
);
long go2(long a) __asm__("_go2");
int main(void) {
    long r = go2(1);
    // reader2(live[0]=1, self=live[1]=2) = 2*100 + 1 = 201
    long want = 201;
    for (int i = 0; i < 20; i++) want += 1 + i;
    printf("%s %ld\n", r == want ? "ok" : "WRONG", r);
    return 0;
}
`)
	if out[:2] != "ok" {
		t.Errorf("printed %q", out)
	}
}
