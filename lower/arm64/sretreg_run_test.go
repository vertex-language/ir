package arm64_test

// AAPCS64 §5.5's aggregate results, across a real link with clang.
//
// A result is brought back the way an argument of the same type is passed:
// homogeneous into SIMD registers, sixteen bytes or less into X0 and X1, and
// only past that through the storage whose address arrived in X8. The front
// end writes an sret parameter for every record return because it does not
// know which of those it will be — so where the answer is registers, the
// parameter names a slot of the callee's own, the body writes through it as
// before, and the registers are loaded from it at the return.
//
// Both directions, since a caller and a callee are different halves of it:
// one where clang defines the function and this package calls it, one where
// clang calls a function this package defines.

import (
	"testing"

	"github.com/vertex-language/ir"
)

// sixteen bytes: X0 and X1, no hidden pointer anywhere.
func TestRunSRetRegsPairCallee(t *testing.T) {
	m := ir.NewModule("t", ir.AArch64Linux)
	pair := m.Struct("Pair").
		Field("a", ir.StoreI64.FType()).
		Field("b", ir.StoreI64.FType())

	fn := m.Func("_mkpair").Export()
	ret := fn.ParamPtr("__ret", ir.SRet(pair))
	a := fn.ParamI64("a")
	b := fn.ParamI64("b")
	entry := fn.Entry()
	entry.I64.Store(a, ret)
	entry.I64.Store(b, entry.Ptr.Add(ret, entry.I64.Const(8)))
	entry.Return()

	got := runNative(t, m, `
#include <stdio.h>
struct Pair { long a, b; };
struct Pair mkpair(long, long);
int main(void) {
    struct Pair p = mkpair(11, 22);
    printf("%ld %ld\n", p.a, p.b);
    return 0;
}
`)
	if want := "11 22\n"; got != want {
		t.Errorf("printed %q, want %q", got, want)
	}
}

// A homogeneous result: S0 and S1, not X0.
func TestRunSRetRegsHFACallee(t *testing.T) {
	m := ir.NewModule("t", ir.AArch64Linux)
	hfa := m.Struct("HFA2").
		Field("x", ir.StoreF32.FType()).
		Field("y", ir.StoreF32.FType())

	fn := m.Func("_mkhfa").Export()
	ret := fn.ParamPtr("__ret", ir.SRet(hfa))
	x := fn.ParamF32("x")
	y := fn.ParamF32("y")
	entry := fn.Entry()
	entry.F32.Store(x, ret)
	entry.F32.Store(y, entry.Ptr.Add(ret, entry.I64.Const(4)))
	entry.Return()

	got := runNative(t, m, `
#include <stdio.h>
struct HFA2 { float x, y; };
struct HFA2 mkhfa(float, float);
int main(void) {
    struct HFA2 h = mkhfa(1.5f, 2.5f);
    printf("%.1f %.1f\n", (double)h.x, (double)h.y);
    return 0;
}
`)
	if want := "1.5 2.5\n"; got != want {
		t.Errorf("printed %q, want %q", got, want)
	}
}

// Twenty-four bytes and homogeneous, so still registers: D0, D1 and D2. The
// size rule does not reach a homogeneous result any more than it reaches a
// homogeneous argument.
func TestRunSRetRegsHFAPastSixteenCallee(t *testing.T) {
	m := ir.NewModule("t", ir.AArch64Linux)
	hfa := m.Struct("D3").
		Field("a", ir.StoreF64.FType()).
		Field("b", ir.StoreF64.FType()).
		Field("c", ir.StoreF64.FType())

	fn := m.Func("_mkd3").Export()
	ret := fn.ParamPtr("__ret", ir.SRet(hfa))
	seed := fn.ParamF64("seed")
	entry := fn.Entry()
	for i := 0; i < 3; i++ {
		v := entry.F64.Add(seed, entry.F64.Const(float64(i)))
		entry.F64.Store(v, entry.Ptr.Add(ret, entry.I64.Const(int64(i)*8)))
	}
	entry.Return()

	got := runNative(t, m, `
#include <stdio.h>
struct D3 { double a, b, c; };
struct D3 mkd3(double);
int main(void) {
    struct D3 d = mkd3(1.5);
    printf("%.1f %.1f %.1f\n", d.a, d.b, d.c);
    return 0;
}
`)
	if want := "1.5 2.5 3.5\n"; got != want {
		t.Errorf("printed %q, want %q", got, want)
	}
}

// The caller's half: clang defines the function, this package calls it and
// reads the result out of the storage it set aside.
func TestRunSRetRegsCaller(t *testing.T) {
	m := ir.NewModule("t", ir.AArch64Linux)
	pair := m.Struct("Pair").
		Field("a", ir.StoreI64.FType()).
		Field("b", ir.StoreI64.FType())

	mk := m.ImportFunc("_cmkpair", ir.NewSig().
		Param(ir.TypePtr, ir.SRet(pair)).
		Param(ir.TypeI64).
		Param(ir.TypeI64))

	fn := m.Func("_sum").Export()
	a := fn.ParamI64("a")
	b := fn.ParamI64("b")
	fn.ReturnsI64()
	entry := fn.Entry()

	slot := entry.Ptr.Alloc(16, 8)
	entry.Call(mk, slot, a, b)
	lo := entry.I64.Load(slot)
	hi := entry.I64.Load(entry.Ptr.Add(slot, entry.I64.Const(8)))
	entry.Return(entry.I64.Add(entry.I64.Mul(lo, entry.I64.Const(100)), hi))

	got := runNative(t, m, `
#include <stdio.h>
struct Pair { long a, b; };
struct Pair cmkpair(long a, long b) { struct Pair p = { a, b }; return p; }
long sum(long, long);
int main(void) { printf("%ld\n", sum(3, 4)); return 0; }
`)
	if want := "304\n"; got != want {
		t.Errorf("printed %q, want %q", got, want)
	}
}

// A homogeneous result read back by this package's caller, so the loads have
// to come out of the SIMD file rather than X0.
func TestRunSRetRegsHFACaller(t *testing.T) {
	m := ir.NewModule("t", ir.AArch64Linux)
	hfa := m.Struct("HFA2").
		Field("x", ir.StoreF32.FType()).
		Field("y", ir.StoreF32.FType())

	mk := m.ImportFunc("_cmkhfa", ir.NewSig().
		Param(ir.TypePtr, ir.SRet(hfa)).
		Param(ir.TypeF32).
		Param(ir.TypeF32))

	fn := m.Func("_combine").Export()
	x := fn.ParamF32("x")
	y := fn.ParamF32("y")
	fn.ReturnsF64()
	entry := fn.Entry()

	slot := entry.Ptr.Alloc(8, 4)
	entry.Call(mk, slot, x, y)
	lo := entry.F32.Load(slot)
	hi := entry.F32.Load(entry.Ptr.Add(slot, entry.I64.Const(4)))
	sum := entry.F64.Add(entry.F64.FCvtF32(lo), entry.F64.Mul(entry.F64.FCvtF32(hi), entry.F64.Const(10)))
	entry.Return(sum)

	got := runNative(t, m, `
#include <stdio.h>
struct HFA2 { float x, y; };
struct HFA2 cmkhfa(float a, float b) { struct HFA2 h = { a, b }; return h; }
double combine(float, float);
int main(void) { printf("%.1f\n", combine(1.5f, 2.0f)); return 0; }
`)
	if want := "21.5\n"; got != want {
		t.Errorf("printed %q, want %q", got, want)
	}
}

// Past sixteen and not homogeneous, which still goes through memory: the
// address arrives in X8 and nothing comes back in a register. Here so that
// the register path cannot quietly capture the case that must not use it.
func TestRunSRetMemoryStillX8(t *testing.T) {
	m := ir.NewModule("t", ir.AArch64Linux)
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
