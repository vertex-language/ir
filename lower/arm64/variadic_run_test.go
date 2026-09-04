package arm64_test

// §I, both ends: a variadic function this backend lowered called from C, and
// a C variadic function called from a module this backend lowered.
//
// Both directions matter and they fail differently. A wrong callee walks a
// list the C caller laid out; a wrong caller lays out a list the C callee
// walks. Only doing both proves the convention rather than a self-consistent
// misreading of it.

import (
	"testing"

	"github.com/vertex-language/ir"
	arm64lower "github.com/vertex-language/ir/lower/arm64"
)

// The callee half: va_start, va_arg at three widths, va_copy and va_end.
func TestRunVariadicCallee(t *testing.T) {
	m := ir.NewModule("t", ir.AArch64Linux)

	fn := m.Func("_vsum").Variadic().Export()
	n := fn.ParamI32("n")
	fn.ReturnsI64()
	entry := fn.Entry()

	ap := entry.Ptr.Alloc(8, 8)
	entry.VaStart(ap)

	// A copy taken before anything is read, walked afterwards, so the two
	// have to agree about where the list started.
	ap2 := entry.Ptr.Alloc(8, 8)
	entry.VaCopy(ap2, ap)

	loop := fn.Block("loop")
	body := fn.Block("body")
	out := fn.Block("out")
	i := loop.ParamI32("i")
	acc := loop.ParamI64("acc")

	entry.Br(loop.To(entry.I32.Const(0), entry.I64.Const(0)))
	loop.BrIf(loop.I32.SLt(i, n), body.To(), out.To())
	body.Br(loop.To(body.I32.Add(i, body.I32.Const(1)),
		body.I64.Add(acc, body.I64.VaArg(ap))))

	// The copy, read once, which must land on the first argument again.
	first := out.I64.VaArg(ap2)
	out.VaEnd(ap)
	out.VaEnd(ap2)
	out.Return(out.I64.Add(acc, out.I64.Mul(first, out.I64.Const(1000))))

	got := runNative(t, m, `
#include <stdio.h>
long vsum(int, ...);
int main(void) {
    // 10+20+30+40 = 100, plus the first read again times 1000.
    printf("%ld\n", vsum(4, 10L, 20L, 30L, 40L));
    return 0;
}
`)
	if got != "10100\n" {
		t.Errorf("printed %q, want %q", got, "10100\n")
	}
}

// Mixed widths through the list: every variadic argument is an eight-byte
// slot whatever its type, and a four-byte one sits in the low half of its own.
func TestRunVariadicCalleeMixed(t *testing.T) {
	m := ir.NewModule("t", ir.AArch64Linux)

	fn := m.Func("_vmix").Variadic().Export()
	fn.ParamI32("tag")
	fn.ReturnsF64()
	entry := fn.Entry()

	ap := entry.Ptr.Alloc(8, 8)
	entry.VaStart(ap)
	a := entry.I64.VaArg(ap)
	d := entry.F64.VaArg(ap)
	i := entry.I32.VaArg(ap)
	p := entry.Ptr.VaArg(ap)
	entry.VaEnd(ap)

	// The pointer dereferenced, so a wrong slot shows up as a wrong value
	// rather than as a pointer nobody looked at.
	entry.Return(entry.F64.Add(
		entry.F64.Add(d, entry.F64.SCvtI64(a)),
		entry.F64.SCvtI64(entry.I64.Add(
			entry.I64.SExtI32(i), entry.I64.Load(p)))))

	got := runNative(t, m, `
#include <stdio.h>
double vmix(int, ...);
int main(void) {
    long boxed = 500;
    printf("%g\n", vmix(0, 7L, 0.5, 30, &boxed));
    return 0;
}
`)
	// 0.5 + 7 + 30 + 500
	if got != "537.5\n" {
		t.Errorf("printed %q, want %q", got, "537.5\n")
	}
}

// The caller half: a module this backend lowered calling C's own printf-style
// variadic function, which clang compiled for the same convention.
func TestRunVariadicCaller(t *testing.T) {
	m := ir.NewModule("t", ir.AArch64Linux)
	sink := m.ImportFunc("_take", ir.NewSig().
		Param(ir.TypeI32).Variadic().Ret(ir.TypeI64))

	fn := m.Func("_feed").Export()
	fn.ReturnsI64()
	entry := fn.Entry()
	r := entry.Call(sink,
		entry.I32.Const(4),
		entry.I64.Const(1),
		entry.F64.Const(2.5),
		entry.I64.Const(3),
		entry.I64.Const(4),
	).Value(0).(ir.I64)
	entry.Return(r)

	got := runNative(t, m, `
#include <stdio.h>
#include <stdarg.h>
long take(int n, ...) {
    va_list ap; va_start(ap, n);
    long a = va_arg(ap, long);
    double d = va_arg(ap, double);
    long c = va_arg(ap, long);
    long e = va_arg(ap, long);
    va_end(ap);
    return a * 1000 + (long)(d * 10) * 100 + c * 10 + e;
}
long feed(void);
int main(void) { printf("%ld\n", feed()); return 0; }
`)
	// 1*1000 + 25*100 + 3*10 + 4
	if got != "3534\n" {
		t.Errorf("printed %q, want %q", got, "3534\n")
	}
}

// A variadic call with enough named arguments to fill the registers, so the
// variadic tail starts after named arguments that are themselves on the stack.
func TestRunVariadicAfterStackedNamed(t *testing.T) {
	m := ir.NewModule("t", ir.AArch64Linux)
	sig := ir.NewSig()
	for i := 0; i < 9; i++ {
		sig = sig.Param(ir.TypeI64)
	}
	sink := m.ImportFunc("_take9", sig.Variadic().Ret(ir.TypeI64))

	fn := m.Func("_feed9").Export()
	fn.ReturnsI64()
	entry := fn.Entry()
	args := make([]ir.Value, 0, 11)
	for i := 1; i <= 9; i++ {
		args = append(args, entry.I64.Const(int64(i)))
	}
	args = append(args, entry.I64.Const(100), entry.I64.Const(200))
	entry.Return(entry.Call(sink, args...).Value(0).(ir.I64))

	got := runNative(t, m, `
#include <stdio.h>
#include <stdarg.h>
long take9(long a,long b,long c,long d,long e,long f,long g,long h,long i, ...) {
    va_list ap; va_start(ap, i);
    long x = va_arg(ap, long), y = va_arg(ap, long);
    va_end(ap);
    return a+b+c+d+e+f+g+h+i + x*10 + y*100;
}
long feed9(void);
int main(void) { printf("%ld\n", feed9()); return 0; }
`)
	// 45 + 1000 + 20000
	if got != "21045\n" {
		t.Errorf("printed %q, want %q", got, "21045\n")
	}
}

// The base standard's convention is refused by name rather than lowered as
// Apple's, which would be a wrong call at every use.
func TestLowerRefusesBaseVariadic(t *testing.T) {
	m := ir.NewModule("t", ir.AArch64Linux)
	fn := m.Func("f").Variadic().Export()
	fn.ParamI32("n")
	fn.ReturnsI64()
	entry := fn.Entry()
	ap := entry.Ptr.Alloc(8, 8)
	entry.VaStart(ap)
	entry.Return(entry.I64.VaArg(ap))

	if _, err := arm64lower.Lower(m, arm64lower.Options{}); err == nil {
		t.Error("Lower should refuse the base standard's variadic convention")
	}
	if _, err := arm64lower.Lower(m, arm64lower.Options{Variadic: arm64lower.VariadicDarwin}); err != nil {
		t.Errorf("Darwin's variant should lower: %v", err)
	}
}
