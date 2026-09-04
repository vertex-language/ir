package i386_test

// §I, both directions: a variadic function this backend lowered called from
// C, and C's own variadic function called from a lowered module.

import (
	"testing"

	"github.com/vertex-language/ir"
)

func TestRunVariadicCallee(t *testing.T) {
	m := ir.NewModule("t", ir.I386Linux)

	fn := m.Func("vsum").Variadic().Export()
	n := fn.ParamI32("n")
	fn.ReturnsI32()
	entry := fn.Entry()

	ap := entry.Ptr.Alloc(4, 4)
	entry.VaStart(ap)
	ap2 := entry.Ptr.Alloc(4, 4)
	entry.VaCopy(ap2, ap)

	loop := fn.Block("loop")
	body := fn.Block("body")
	out := fn.Block("out")
	i := loop.ParamI32("i")
	acc := loop.ParamI32("acc")

	entry.Br(loop.To(entry.I32.Const(0), entry.I32.Const(0)))
	loop.BrIf(loop.I32.SLt(i, n), body.To(), out.To())
	body.Br(loop.To(body.I32.Add(i, body.I32.Const(1)),
		body.I32.Add(acc, body.I32.VaArg(ap))))

	// The copy, read once: it has to land on the first argument again.
	first := out.I32.VaArg(ap2)
	out.VaEnd(ap)
	out.VaEnd(ap2)
	out.Return(out.I32.Add(acc, out.I32.Mul(first, out.I32.Const(1000))))

	wantOK(t, m, `
int vsum(int, ...);
static void body(void) {
    /* 10+20+30+40 = 100, plus the first read again times 1000. */
    chk32("va callee", (unsigned)vsum(4, 10, 20, 30, 40), 10100u);
}
`)
}

// Mixed widths, where an i64 argument fills two slots and everything else
// fills one — the same rule the call side places by.
func TestRunVariadicMixed(t *testing.T) {
	m := ir.NewModule("t", ir.I386Linux)

	fn := m.Func("vmix").Variadic().Export()
	fn.ParamI32("tag")
	fn.ReturnsI64()
	entry := fn.Entry()

	ap := entry.Ptr.Alloc(4, 4)
	entry.VaStart(ap)
	a := entry.I64.VaArg(ap)
	i := entry.I32.VaArg(ap)
	b := entry.I64.VaArg(ap)
	p := entry.Ptr.VaArg(ap)
	entry.VaEnd(ap)

	entry.Return(entry.I64.Add(
		entry.I64.Add(a, b),
		entry.I64.Add(entry.I64.SExtI32(i), entry.I64.Load(p))))

	wantOK(t, m, `
long long vmix(int, ...);
static void body(void) {
    long long boxed = 500;
    chk64("va mixed", vmix(0, 7LL, 30, 0x0000000100000000LL, &boxed),
                      (unsigned long long)(7 + 30 + 0x100000000LL + 500));
}
`)
}

// The caller half: a lowered module calling C's own variadic function.
func TestRunVariadicCaller(t *testing.T) {
	m := ir.NewModule("t", ir.I386Linux)
	sink := m.ImportFunc("take", ir.NewSig().
		Param(ir.TypeI32).Variadic().Ret(ir.TypeI64))

	fn := m.Func("feed").Export()
	fn.ReturnsI64()
	entry := fn.Entry()
	entry.Return(entry.Call(sink,
		entry.I32.Const(4),
		entry.I64.Const(1),
		entry.I32.Const(2),
		entry.I64.Const(0x0000000300000000),
		entry.I32.Const(4),
	).Value(0).(ir.I64))

	wantOK(t, m, `
#include <stdarg.h>
long long take(int n, ...) {
    va_list ap; va_start(ap, n);
    long long a = va_arg(ap, long long);
    int b = va_arg(ap, int);
    long long c = va_arg(ap, long long);
    int d = va_arg(ap, int);
    va_end(ap);
    return a + b + c + d;
}
long long feed(void);
static void body(void) {
    chk64("va caller", feed(), (unsigned long long)(1 + 2 + 0x300000000LL + 4));
}
`)
}

// va_arg_ref, which is how va_arg of an aggregate is written. Every aggregate
// is by value on this psABI, so the address is the slot itself.
func TestRunVariadicAggregate(t *testing.T) {
	m := ir.NewModule("t", ir.I386Linux)
	st := m.Struct("pair")
	st.Field("a", ir.StoreI32.FType())
	st.Field("b", ir.StoreI32.FType())

	fn := m.Func("vagg").Variadic().Export()
	fn.ParamI32("tag")
	fn.ReturnsI32()
	entry := fn.Entry()

	ap := entry.Ptr.Alloc(4, 4)
	entry.VaStart(ap)
	p := entry.Ptr.VaArgRef(ap, st)
	// And an ordinary argument after it, so the cursor has to have moved
	// past both of the aggregate's slots.
	tail := entry.I32.VaArg(ap)
	entry.VaEnd(ap)

	sum := entry.I32.Add(
		entry.I32.Load(p),
		entry.I32.Load(entry.Ptr.Add(p, entry.I64.Const(4))))
	entry.Return(entry.I32.Add(sum, entry.I32.Mul(tail, entry.I32.Const(100))))

	wantOK(t, m, `
struct pair { int a, b; };
int vagg(int, ...);
static void body(void) {
    struct pair p = {7, 11};
    chk32("va_arg_ref", (unsigned)vagg(0, p, 3), (unsigned)(7 + 11 + 300));
}
`)
}
