package arm64_test

// AAPCS64 §5.4's aggregate arguments, across a real link with clang.
//
// A struct passed by value is not passed as a pointer. §5.4 asks three
// questions in order — is it homogeneous, is it sixteen bytes or less, and
// only then is it the caller's copy by reference — and each answer puts the
// bytes somewhere different. This backend used to give the third answer to
// every aggregate, which is a private convention: self-consistent, and wrong
// against anything else on the other end of the call.
//
// clang writes the callee in each of these, so what they agree with is the
// ABI. Every one of them takes a trailing scalar as well, because the way
// this goes wrong in practice is not a garbled struct but a shifted argument
// sequence — the aggregate eating registers that belonged to what came after.

import (
	"testing"

	"github.com/vertex-language/ir"
)

// Two doublewords in X0 and X1, with the trailing argument in X2.
func TestRunByValTwoGPR(t *testing.T) {
	m := ir.NewModule("t", ir.AArch64Linux)
	pair := m.Struct("Pair").
		Field("a", ir.StoreI64.FType()).
		Field("b", ir.StoreI64.FType())

	takes := m.ImportFunc("_ctake", ir.NewSig().
		Param(ir.TypePtr, ir.ByVal(pair)).
		Param(ir.TypeI64).
		Ret(ir.TypeI64))

	fn := m.Func("_go").Export()
	a := fn.ParamI64("a")
	b := fn.ParamI64("b")
	tail := fn.ParamI64("t")
	fn.ReturnsI64()
	entry := fn.Entry()

	slot := entry.Ptr.Alloc(16, 8)
	entry.I64.Store(a, slot)
	entry.I64.Store(b, entry.Ptr.Add(slot, entry.I64.Const(8)))
	entry.Return(entry.Call(takes, slot, tail).Value(0).(ir.I64))

	got := runNative(t, m, `
#include <stdio.h>
struct Pair { long a, b; };
long ctake(struct Pair p, long t) { return p.a * 100 + p.b * 10 + t; }
long go(long, long, long);
int main(void) { printf("%ld\n", go(3, 4, 5)); return 0; }
`)
	if want := "345\n"; got != want {
		t.Errorf("printed %q, want %q", got, want)
	}
}

// Three bytes packed into one X register each, and the trailing argument
// after them. The callee stores the whole register back into its slot, so the
// slot has to be a doubleword however few bytes the struct has — three would
// take the five below it. Two of them, because that is what makes the
// overwrite an answer rather than a crash: the second's store lands on the
// first's bytes, and the first then reads back as the second.
func TestRunByValSmallGPR(t *testing.T) {
	m := ir.NewModule("t", ir.AArch64Linux)
	small := m.Struct("Small").
		Field("a", ir.StoreI8.FType()).
		Field("b", ir.StoreI8.FType()).
		Field("c", ir.StoreI8.FType())

	fn := m.Func("_gosmall").Export()
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
	sum := entry.I64.Add(entry.I64.Mul(digits(p), entry.I64.Const(1000)), digits(q))
	entry.Return(entry.I64.Add(sum, tail))

	got := runNative(t, m, `
#include <stdio.h>
struct Small { char a, b, c; };
long gosmall(struct Small, struct Small, long);
int main(void) {
    struct Small s = { 1, 2, 3 };
    struct Small u = { 4, 5, 6 };
    printf("%ld\n", gosmall(s, u, 0));
    return 0;
}
`)
	if want := "123456\n"; got != want {
		t.Errorf("printed %q, want %q", got, want)
	}
}

// A homogeneous aggregate: one SIMD register per member, four bytes apart,
// with the trailing double in D2 rather than D0.
func TestRunByValHFA(t *testing.T) {
	m := ir.NewModule("t", ir.AArch64Linux)
	hfa := m.Struct("HFA2").
		Field("x", ir.StoreF32.FType()).
		Field("y", ir.StoreF32.FType())

	takes := m.ImportFunc("_chfa", ir.NewSig().
		Param(ir.TypePtr, ir.ByVal(hfa)).
		Param(ir.TypeF64).
		Ret(ir.TypeF64))

	fn := m.Func("_go").Export()
	x := fn.ParamF32("x")
	y := fn.ParamF32("y")
	tail := fn.ParamF64("t")
	fn.ReturnsF64()
	entry := fn.Entry()

	slot := entry.Ptr.Alloc(8, 4)
	entry.F32.Store(x, slot)
	entry.F32.Store(y, entry.Ptr.Add(slot, entry.I64.Const(4)))
	entry.Return(entry.Call(takes, slot, tail).Value(0).(ir.F64))

	got := runNative(t, m, `
#include <stdio.h>
struct HFA2 { float x, y; };
double chfa(struct HFA2 h, double t) { return h.x * 100 + h.y * 10 + t; }
double go(float, float, double);
int main(void) { printf("%.1f\n", go(1.0f, 2.0f, 3.5f)); return 0; }
`)
	if want := "123.5\n"; got != want {
		t.Errorf("printed %q, want %q", got, want)
	}
}

// Twenty-four bytes and still homogeneous, which is the case a classifier
// that asks about size first gets wrong: three doubles travel in D0, D1 and
// D2, not by reference.
func TestRunByValHFAPastSixteen(t *testing.T) {
	m := ir.NewModule("t", ir.AArch64Linux)
	hfa := m.Struct("D3").
		Field("a", ir.StoreF64.FType()).
		Field("b", ir.StoreF64.FType()).
		Field("c", ir.StoreF64.FType())

	takes := m.ImportFunc("_cd3", ir.NewSig().
		Param(ir.TypePtr, ir.ByVal(hfa)).
		Param(ir.TypeF64).
		Ret(ir.TypeF64))

	fn := m.Func("_go").Export()
	tail := fn.ParamF64("t")
	fn.ReturnsF64()
	entry := fn.Entry()

	slot := entry.Ptr.Alloc(24, 8)
	for i, v := range []float64{1, 2, 3} {
		entry.F64.Store(entry.F64.Const(v), entry.Ptr.Add(slot, entry.I64.Const(int64(i)*8)))
	}
	entry.Return(entry.Call(takes, slot, tail).Value(0).(ir.F64))

	got := runNative(t, m, `
#include <stdio.h>
struct D3 { double a, b, c; };
double cd3(struct D3 d, double t) { return d.a * 1000 + d.b * 100 + d.c * 10 + t; }
double go(double);
int main(void) { printf("%.1f\n", go(4.5)); return 0; }
`)
	if want := "1234.5\n"; got != want {
		t.Errorf("printed %q, want %q", got, want)
	}
}

// The callee side of the same rules: clang calls a function this package
// wrote, so the registers have to be read where clang put them.
func TestRunByValCallee(t *testing.T) {
	m := ir.NewModule("t", ir.AArch64Linux)
	pair := m.Struct("Pair").
		Field("a", ir.StoreI64.FType()).
		Field("b", ir.StoreI64.FType())

	fn := m.Func("_gotake").Export()
	p := fn.ParamPtr("p", ir.ByVal(pair))
	tail := fn.ParamI64("t")
	fn.ReturnsI64()
	entry := fn.Entry()

	a := entry.I64.Load(p)
	b := entry.I64.Load(entry.Ptr.Add(p, entry.I64.Const(8)))
	sum := entry.I64.Add(entry.I64.Mul(a, entry.I64.Const(100)), entry.I64.Mul(b, entry.I64.Const(10)))
	entry.Return(entry.I64.Add(sum, tail))

	got := runNative(t, m, `
#include <stdio.h>
struct Pair { long a, b; };
long gotake(struct Pair, long);
int main(void) {
    struct Pair p = { 3, 4 };
    printf("%ld\n", gotake(p, 5));
    return 0;
}
`)
	if want := "345\n"; got != want {
		t.Errorf("printed %q, want %q", got, want)
	}
}

// More aggregates than the file has registers, so the last one is a copy in
// the outgoing area rather than a register sequence — and the trailing
// scalar still has to land where clang looks for it.
func TestRunByValOntoStack(t *testing.T) {
	m := ir.NewModule("t", ir.AArch64Linux)
	pair := m.Struct("Pair").
		Field("a", ir.StoreI64.FType()).
		Field("b", ir.StoreI64.FType())

	sig := ir.NewSig()
	for i := 0; i < 5; i++ {
		sig = sig.Param(ir.TypePtr, ir.ByVal(pair))
	}
	takes := m.ImportFunc("_cfive", sig.Param(ir.TypeI64).Ret(ir.TypeI64))

	fn := m.Func("_go").Export()
	tail := fn.ParamI64("t")
	fn.ReturnsI64()
	entry := fn.Entry()

	var args []ir.Value
	for i := 0; i < 5; i++ {
		slot := entry.Ptr.Alloc(16, 8)
		entry.I64.Store(entry.I64.Const(int64(i)+1), slot)
		entry.I64.Store(entry.I64.Const(int64(i)+1), entry.Ptr.Add(slot, entry.I64.Const(8)))
		args = append(args, slot)
	}
	args = append(args, tail)
	entry.Return(entry.Call(takes, args...).Value(0).(ir.I64))

	got := runNative(t, m, `
#include <stdio.h>
struct Pair { long a, b; };
long cfive(struct Pair p1, struct Pair p2, struct Pair p3,
           struct Pair p4, struct Pair p5, long t) {
    return (p1.a + p2.a + p3.a + p4.a + p5.a) * 1000
         + (p1.b + p2.b + p3.b + p4.b + p5.b) * 10 + t;
}
long go(long);
int main(void) { printf("%ld\n", go(7)); return 0; }
`)
	if want := "15157\n"; got != want {
		t.Errorf("printed %q, want %q", got, want)
	}
}
