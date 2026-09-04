package arm64_test

// AAPCS64 §6.9's indirect result location register, checked against clang.
//
// A result too large to come back in X0 and X1 is written through an address
// the caller supplies, and that address travels in X8 — not in the argument
// sequence. The distinction is invisible to a program compiled entirely by
// one compiler, because a private convention is self-consistent, and it is
// the whole story when linking against anything else: passing the pointer in
// X0 shifts every real argument one register along, so the callee reads its
// first argument out of the register holding its second.
//
// Both directions are here. clang writes the caller in the first test and the
// callee in the second, so what each agrees with is the ABI rather than this
// package's opinion of it.

import (
	"testing"

	"github.com/vertex-language/ir"
)

// bigType is sixty-four bytes: eight eightbytes, far past the two §5.4
// returns in registers, so the result is memory and the pointer is X8's.
func bigType(m *ir.Module) *ir.Type {
	return m.Struct("Big").Field("v", ir.Array(8, ir.StoreI64.FType()))
}

// The callee side: this package writes the function, clang calls it. The
// scalar arguments are what catch a shifted sequence — with the pointer in
// X0 they would arrive one register late and the sum would be wrong.
func TestRunSRetCallee(t *testing.T) {
	m := ir.NewModule("t", ir.AArch64Linux)
	big := bigType(m)

	fn := m.Func("_fill").Export()
	ret := fn.ParamPtr("__ret", ir.SRet(big))
	a := fn.ParamI64("a")
	b := fn.ParamI64("b")
	c := fn.ParamI64("c")
	entry := fn.Entry()

	// v[i] = a + b*(i+1) + c, so every argument is read and a shift by one
	// register changes the answer rather than merely reordering it.
	base := entry.I64.Add(a, c)
	for i := 0; i < 8; i++ {
		off := entry.I64.Const(int64(i) * 8)
		v := entry.I64.Add(base, entry.I64.Mul(b, entry.I64.Const(int64(i+1))))
		entry.I64.Store(v, entry.Ptr.Add(ret, off))
	}
	entry.Return()

	got := runNative(t, m, `
#include <stdio.h>
struct Big { long v[8]; };
struct Big fill(long, long, long);
int main(void) {
    struct Big r = fill(1, 10, 2);
    printf("%ld %ld %ld\n", r.v[0], r.v[3], r.v[7]);
    return 0;
}
`)
	if want := "13 43 83\n"; got != want {
		t.Errorf("printed %q, want %q", got, want)
	}
}

// The caller side: clang writes the function, this package calls it and reads
// the result back out of the storage it supplied.
func TestRunSRetCaller(t *testing.T) {
	m := ir.NewModule("t", ir.AArch64Linux)
	big := bigType(m)

	fill := m.ImportFunc("_cfill", ir.NewSig().
		Param(ir.TypePtr, ir.SRet(big)).
		Param(ir.TypeI64).
		Param(ir.TypeI64).
		Param(ir.TypeI64))

	fn := m.Func("_sum").Export()
	a := fn.ParamI64("a")
	b := fn.ParamI64("b")
	c := fn.ParamI64("c")
	fn.ReturnsI64()
	entry := fn.Entry()

	slot := entry.Ptr.Alloc(64, 8)
	entry.Call(fill, slot, a, b, c)

	acc := entry.I64.Const(0)
	for i := 0; i < 8; i++ {
		off := entry.I64.Const(int64(i) * 8)
		acc = entry.I64.Add(acc, entry.I64.Load(entry.Ptr.Add(slot, off)))
	}
	entry.Return(acc)

	got := runNative(t, m, `
#include <stdio.h>
struct Big { long v[8]; };
struct Big cfill(long a, long b, long c) {
    struct Big r;
    for (int i = 0; i < 8; i++) r.v[i] = a + b * (i + 1) + c;
    return r;
}
long sum(long, long, long);
int main(void) {
    long want = 0;
    for (int i = 0; i < 8; i++) want += 1 + 10 * (i + 1) + 2;
    long got = sum(1, 10, 2);
    printf("%s %ld\n", got == want ? "ok" : "MISMATCH", got);
    return 0;
}
`)
	if want := "ok 384\n"; got != want {
		t.Errorf("printed %q, want %q", got, want)
	}
}
