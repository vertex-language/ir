package amd64_test

// §3.2.3's aggregates, across a real link with clang.
//
// SysV classifies an argument and a result the same way: cut the aggregate
// into eightbytes, class each one INTEGER or SSE, and anything past two
// eightbytes — or with a field straddling a boundary — is MEMORY. A result
// that is not MEMORY comes back in RAX and RDX, or XMM0 and XMM1, counted per
// file; one that is goes through storage the caller supplies in RDI and gets
// its address back in RAX.
//
// Note what is *not* here, deliberately: three doubles. AAPCS64 calls that a
// homogeneous aggregate and passes it in three registers; SysV has no such
// idea and calls twenty-four bytes MEMORY. The two ABIs genuinely differ, and
// a test asserting one shape for both would be asserting a coincidence.

import (
	"testing"

	"github.com/vertex-language/ir"
)

// Two integer eightbytes: in RDI and RSI, back in RAX and RDX.
func TestRunAggPairRoundTrip(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64MacOS)
	pair := m.Struct("Pair").
		Field("a", ir.StoreI64.FType()).
		Field("b", ir.StoreI64.FType())

	// Takes one by value and returns one, so both halves of the
	// classification are exercised in a single function.
	fn := m.Func("_swap").Export()
	ret := fn.ParamPtr("__ret", ir.SRet(pair))
	in := fn.ParamPtr("p", ir.ByVal(pair))
	tail := fn.ParamI64("t")
	entry := fn.Entry()

	a := entry.I64.Load(in)
	b := entry.I64.Load(entry.Ptr.Add(in, entry.I64.Const(8)))
	entry.I64.Store(entry.I64.Add(b, tail), ret)
	entry.I64.Store(a, entry.Ptr.Add(ret, entry.I64.Const(8)))
	entry.Return()

	got := runNative(t, m, `
#include <stdio.h>
struct Pair { long a, b; };
struct Pair swap(struct Pair, long);
int main(void) {
    struct Pair p = { 3, 4 };
    struct Pair r = swap(p, 100);
    printf("%ld %ld\n", r.a, r.b);
    return 0;
}
`)
	if want := "104 3\n"; got != want {
		t.Errorf("printed %q, want %q", got, want)
	}
}

// One SSE eightbyte: two floats share it, so the pair travels in XMM0 and
// comes back in XMM0 — not in RAX, which is what a classifier that ignored
// the field types would choose.
func TestRunAggSSEReturn(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64MacOS)
	hfa := m.Struct("F2").
		Field("x", ir.StoreF32.FType()).
		Field("y", ir.StoreF32.FType())

	fn := m.Func("_mkf2").Export()
	ret := fn.ParamPtr("__ret", ir.SRet(hfa))
	x := fn.ParamF32("x")
	y := fn.ParamF32("y")
	entry := fn.Entry()
	entry.F32.Store(x, ret)
	entry.F32.Store(y, entry.Ptr.Add(ret, entry.I64.Const(4)))
	entry.Return()

	got := runNative(t, m, `
#include <stdio.h>
struct F2 { float x, y; };
struct F2 mkf2(float, float);
int main(void) {
    struct F2 f = mkf2(1.5f, 2.5f);
    printf("%.1f %.1f\n", (double)f.x, (double)f.y);
    return 0;
}
`)
	if want := "1.5 2.5\n"; got != want {
		t.Errorf("printed %q, want %q", got, want)
	}
}

// A mixed eightbyte: a float and an int share one, and §3.2.3's merge rule
// makes the whole thing INTEGER, so it comes back in RAX rather than being
// split across the two files.
func TestRunAggMixedEightbyte(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64MacOS)
	mix := m.Struct("Mix").
		Field("f", ir.StoreF32.FType()).
		Field("i", ir.StoreI32.FType())

	fn := m.Func("_mkmix").Export()
	ret := fn.ParamPtr("__ret", ir.SRet(mix))
	f := fn.ParamF32("f")
	i := fn.ParamI32("i")
	entry := fn.Entry()
	entry.F32.Store(f, ret)
	entry.I32.Store(i, entry.Ptr.Add(ret, entry.I64.Const(4)))
	entry.Return()

	got := runNative(t, m, `
#include <stdio.h>
struct Mix { float f; int i; };
struct Mix mkmix(float, int);
int main(void) {
    struct Mix m = mkmix(2.5f, 7);
    printf("%.1f %d\n", (double)m.f, m.i);
    return 0;
}
`)
	if want := "2.5 7\n"; got != want {
		t.Errorf("printed %q, want %q", got, want)
	}
}

// Three bytes: one INTEGER eightbyte each, and the slot the callee makes for
// one has to be a whole eightbyte — the register is written back in full, so
// a three-byte slot runs over whatever is under it. Two of them, because that
// is what makes the overwrite an answer rather than a crash: the second's
// spill lands on the first's bytes, and the first then reads back as the
// second.
func TestRunAggSmallByVal(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64MacOS)
	small := m.Struct("Small").
		Field("a", ir.StoreI8.FType()).
		Field("b", ir.StoreI8.FType()).
		Field("c", ir.StoreI8.FType())

	fn := m.Func("_takesmall").Export()
	p := fn.ParamPtr("p", ir.ByVal(small))
	q := fn.ParamPtr("q", ir.ByVal(small))
	tail := fn.ParamI64("t")
	fn.ReturnsI64()
	entry := fn.Entry()

	digits := func(base ir.Ptr) ir.I64 {
		acc := entry.I64.Const(0)
		for i, mul := range []int64{100, 10, 1} {
			b := entry.I64.ULoad8(entry.Ptr.Add(base, entry.I64.Const(int64(i))))
			acc = entry.I64.Add(acc, entry.I64.Mul(b, entry.I64.Const(mul)))
		}
		return acc
	}
	// p first and q second, so a slot too small for q's spill is one that
	// has already taken p's bytes with it by the time p is read.
	sum := entry.I64.Add(entry.I64.Mul(digits(p), entry.I64.Const(1000)), digits(q))
	entry.Return(entry.I64.Add(sum, tail))

	got := runNative(t, m, `
#include <stdio.h>
struct Small { char a, b, c; };
long takesmall(struct Small, struct Small, long);
int main(void) {
    struct Small s = { 1, 2, 3 };
    struct Small u = { 4, 5, 6 };
    printf("%ld\n", takesmall(s, u, 0));
    return 0;
}
`)
	if want := "123456\n"; got != want {
		t.Errorf("printed %q, want %q", got, want)
	}
}

// Past two eightbytes: MEMORY, so the caller supplies the storage in RDI and
// nothing comes back in a register. Here so the register path cannot quietly
// capture the case that must not use it.
func TestRunAggMemoryReturn(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64MacOS)
	big := m.Struct("Big3").
		Field("a", ir.StoreI64.FType()).
		Field("b", ir.StoreI64.FType()).
		Field("c", ir.StoreI64.FType())

	fn := m.Func("_mkbig").Export()
	ret := fn.ParamPtr("__ret", ir.SRet(big))
	seed := fn.ParamI64("seed")
	entry := fn.Entry()
	for i := 0; i < 3; i++ {
		v := entry.I64.Add(seed, entry.I64.Const(int64(i)))
		entry.I64.Store(v, entry.Ptr.Add(ret, entry.I64.Const(int64(i)*8)))
	}
	entry.Return()

	got := runNative(t, m, `
#include <stdio.h>
struct Big3 { long a, b, c; };
struct Big3 mkbig(long);
int main(void) {
    struct Big3 b = mkbig(10);
    printf("%ld %ld %ld\n", b.a, b.b, b.c);
    return 0;
}
`)
	if want := "10 11 12\n"; got != want {
		t.Errorf("printed %q, want %q", got, want)
	}
}
