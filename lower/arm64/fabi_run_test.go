package arm64_test

// AAPCS64's float half, checked across a real call boundary.
//
// The C side is the other half of every one of these: clang decides where the
// ninth float argument goes and whether a register survives a call, and a test
// that agrees with clang agrees with the ABI rather than with whatever this
// package believed about it.

import (
	"testing"

	"github.com/vertex-language/ir"
)

// Eight floats in registers and a ninth on the stack, alongside integer
// arguments that fill their own file — the two are counted separately, so a
// float never consumes an integer register or the other way round.
func TestRunFloatArguments(t *testing.T) {
	m := ir.NewModule("t", ir.AArch64Linux)
	fn := m.Func("_fargs").Export()

	var floats []ir.F64
	var ints []ir.I64
	// Interleaved on purpose: v0..v7 then the stack, x0..x7 in parallel.
	for i := 0; i < 9; i++ {
		floats = append(floats, fn.ParamF64("f"))
		ints = append(ints, fn.ParamI64("i"))
	}
	fn.ReturnsF64()
	entry := fn.Entry()

	acc := floats[0]
	for _, f := range floats[1:] {
		acc = entry.F64.Add(acc, f)
	}
	for _, n := range ints {
		acc = entry.F64.Add(acc, entry.F64.SCvtI64(n))
	}
	entry.Return(acc)

	got := runNative(t, m, `
#include <stdio.h>
double fargs(double,long,double,long,double,long,double,long,double,long,
             double,long,double,long,double,long,double,long);
int main(void) {
    double want = 0;
    for (int i = 1; i <= 9; i++) want += i * 0.5;
    for (int i = 1; i <= 9; i++) want += i * 100;
    double got = fargs(0.5,100, 1.0,200, 1.5,300, 2.0,400, 2.5,500,
                       3.0,600, 3.5,700, 4.0,800, 4.5,900);
    printf("%s\n", got == want ? "ok" : "MISMATCH");
    if (got != want) printf("got %.17g want %.17g\n", got, want);
    return 0;
}
`)
	if got != "ok\n" {
		t.Errorf("printed %q, want %q", got, "ok\n")
	}
}

// A float returned from a call this backend makes, and a float argument
// passed into one — the caller's side of the same ABI.
func TestRunFloatCall(t *testing.T) {
	m := ir.NewModule("t", ir.AArch64Linux)
	half := m.ImportFunc("_half", ir.NewSig().Param(ir.TypeF64).Ret(ir.TypeF64))

	fn := m.Func("_quarter").Export()
	a := fn.ParamF64("a")
	fn.ReturnsF64()
	entry := fn.Entry()
	once := entry.Call(half, a).Value(0).(ir.F64)
	twice := entry.Call(half, once).Value(0).(ir.F64)
	// The first result is still live across the second call, which is what
	// forces it somewhere a call does not destroy.
	entry.Return(entry.F64.Add(twice, once))

	got := runNative(t, m, `
#include <stdio.h>
double half(double x) { return x / 2.0; }
double quarter(double);
int main(void) { printf("%g\n", quarter(8.0)); return 0; }
`)
	if got != "6\n" {
		t.Errorf("printed %q, want %q", got, "6\n")
	}
}

// V8 through V15 are callee-saved in their low 64 bits, and this backend
// hands them out like any other register once nine float values are live at
// once. A function that takes one has to give it back.
//
// The caller is written in assembly rather than C. A C caller compiled at -O0
// spills its own pinned register around the call and reloads it afterwards,
// so it cannot see the clobber at all — the test would pass whether or not
// the prologue saved anything. Eleven instructions that put a known value in
// d8, call, and return what is in d8 afterwards have no such opinion.
//
// Before the prologue saved these, this reported CLOBBERED.
func TestRunCalleeSavedVector(t *testing.T) {
	m := ir.NewModule("t", ir.AArch64Linux)
	fn := m.Func("_pressure").Export()
	var ps []ir.F64
	for i := 0; i < 8; i++ {
		ps = append(ps, fn.ParamF64("p"))
	}
	fn.ReturnsF64()
	entry := fn.Entry()

	// Twelve products, every one of them live until the sum, which is more
	// values than the caller-saved half of the file holds.
	var live []ir.F64
	for i := 0; i < 12; i++ {
		live = append(live, entry.F64.Mul(ps[i%8], entry.F64.Const(float64(i)+1.5)))
	}
	acc := live[0]
	for _, v := range live[1:] {
		acc = entry.F64.Add(acc, v)
	}
	entry.Return(acc)

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
"	str  d8, [sp, #8]\n"       // the caller's d8, which _probe owes back
"	mov  x8, #4660\n"
"	fmov d8, x8\n"
"	bl   _pressure\n"
"	fmov x0, d8\n"             // what survived
"	ldr  d8, [sp, #8]\n"
"	ldp  x29, x30, [sp, #16]\n"
"	add  sp, sp, #32\n"
"	ret\n"
);
long probe(void);
double pressure(double,double,double,double,double,double,double,double);
int main(void) {
    long kept = probe();
    double want = 0;
    for (int i = 0; i < 12; i++) want += (double)((i % 8) + 1) * ((double)i + 1.5);
    double r = pressure(1,2,3,4,5,6,7,8);
    printf("%s %s\n", kept == 4660 ? "kept" : "CLOBBERED",
                      r == want ? "ok" : "MISMATCH");
    if (kept != 4660) printf("d8 came back as %ld\n", kept);
    return 0;
}
`)
	if got != "kept ok\n" {
		t.Errorf("printed %q, want %q", got, "kept ok\n")
	}
}

// A float spilled to the frame and reloaded, which is the other place a
// vector register reaches memory.
func TestRunFloatSpill(t *testing.T) {
	m := ir.NewModule("t", ir.AArch64Linux)
	sink := m.ImportFunc("_sink", ir.NewSig().Param(ir.TypeF64).Ret(ir.TypeF64))

	fn := m.Func("_spill").Export()
	a := fn.ParamF64("a")
	fn.ReturnsF64()
	entry := fn.Entry()

	// Twenty values live across a call, which is more than the file has
	// once a call is written as destroying all of it.
	var live []ir.F64
	for i := 0; i < 20; i++ {
		live = append(live, entry.F64.Mul(a, entry.F64.Const(float64(i)+1)))
	}
	entry.Call(sink, a)
	acc := live[0]
	for _, v := range live[1:] {
		acc = entry.F64.Add(acc, v)
	}
	entry.Return(acc)

	got := runNative(t, m, `
#include <stdio.h>
double sink(double x) { return x; }
double spill(double);
int main(void) {
    double want = 0;
    for (int i = 0; i < 20; i++) want += 2.5 * (double)(i + 1);
    double got = spill(2.5);
    printf("%s\n", got == want ? "ok" : "MISMATCH");
    if (got != want) printf("got %.17g want %.17g\n", got, want);
    return 0;
}
`)
	if got != "ok\n" {
		t.Errorf("printed %q, want %q", got, "ok\n")
	}
}

// f32 across the boundary, which is the same registers at another width —
// and a place a backend can quietly pass a D where the ABI says S.
func TestRunFloat32Call(t *testing.T) {
	m := ir.NewModule("t", ir.AArch64Linux)
	scale := m.ImportFunc("_scale", ir.NewSig().Param(ir.TypeF32).Param(ir.TypeF32).Ret(ir.TypeF32))

	fn := m.Func("_apply").Export()
	a := fn.ParamF32("a")
	fn.ReturnsF32()
	entry := fn.Entry()
	r := entry.Call(scale, a, entry.F32.Const(3.5)).Value(0).(ir.F32)
	entry.Return(entry.F32.Add(r, a))

	got := runNative(t, m, `
#include <stdio.h>
float scale(float x, float k) { return x * k; }
float apply(float);
int main(void) { printf("%g\n", (double)apply(2.0f)); return 0; }
`)
	if got != "9\n" {
		t.Errorf("printed %q, want %q", got, "9\n")
	}
}

// A float stored to a frame slot and loaded back through a pointer, which is
// §D at a vector width.
func TestRunFloatMemory(t *testing.T) {
	m := ir.NewModule("t", ir.AArch64Linux)
	fn := m.Func("_roundtrip").Export()
	a := fn.ParamF64("a")
	b := fn.ParamF32("b")
	fn.ReturnsF64()
	entry := fn.Entry()

	d := entry.Ptr.Alloc(8, 8)
	s := entry.Ptr.Alloc(4, 4)
	entry.F64.Store(a, d)
	entry.F32.Store(b, s)
	entry.Return(entry.F64.Add(
		entry.F64.Load(d),
		entry.F64.FCvtF32(entry.F32.Load(s)),
	))

	got := runNative(t, m, `
#include <stdio.h>
double roundtrip(double, float);
int main(void) { printf("%g\n", roundtrip(1.25, 2.5f)); return 0; }
`)
	if got != "3.75\n" {
		t.Errorf("printed %q, want %q", got, "3.75\n")
	}
}

// §G's indirect call: BLR through a pointer, whose convention comes from the
// named func type. The pointer arrives from C, so the value being called is
// one this package never saw the declaration of.
func TestRunIndirectCall(t *testing.T) {
	m := ir.NewModule("t", ir.AArch64Linux)
	ft := m.FuncType("binop", ir.NewSig().Param(ir.TypeI64).Param(ir.TypeI64).Ret(ir.TypeI64))

	fn := m.Func("_apply2").Export()
	p := fn.ParamPtr("p")
	a := fn.ParamI64("a")
	b := fn.ParamI64("b")
	fn.ReturnsI64()
	entry := fn.Entry()
	r := entry.CallInd(p, ft, a, b).Value(0).(ir.I64)
	// b is live across the call, which is what keeps the callee's address
	// and a live value from landing in the same register.
	entry.Return(entry.I64.Add(r, b))

	got := runNative(t, m, `
#include <stdio.h>
static long add(long x, long y) { return x + y; }
static long mul(long x, long y) { return x * y; }
long apply2(long (*)(long,long), long, long);
int main(void) {
    printf("%ld %ld\n", apply2(add, 10, 4), apply2(mul, 10, 4));
    return 0;
}
`)
	if got != "18 44\n" {
		t.Errorf("printed %q, want %q", got, "18 44\n")
	}
}

// An indirect call with nothing live across it, which is a function that
// looks like a leaf until you notice BLR writes X30 too. Without the frame
// record it returns to wherever the callee left the link register.
func TestRunIndirectCallBareLeaf(t *testing.T) {
	m := ir.NewModule("t", ir.AArch64Linux)
	ft := m.FuncType("thunk", ir.NewSig().Ret(ir.TypeI64))

	fn := m.Func("_viacall").Export()
	p := fn.ParamPtr("p")
	fn.ReturnsI64()
	entry := fn.Entry()
	entry.Return(entry.CallInd(p, ft).Value(0).(ir.I64))

	got := runNative(t, m, `
#include <stdio.h>
static long answer(void) { return 42; }
long viacall(long (*)(void));
int main(void) { printf("%ld\n", viacall(answer)); return 0; }
`)
	if got != "42\n" {
		t.Errorf("printed %q, want %q", got, "42\n")
	}
}

// An indirect call with a stack argument, alongside callee-saved registers of
// this function's own.
//
// The outgoing area sits at the bottom of the frame and has to be reserved
// when the frame is planned. Until an indirect call was counted there it was
// not, and SP came out one slot too high: the ninth argument went to [sp, #0],
// which was the slot holding this function's saved X21. The epilogue then
// restored the argument into it and handed that back to the caller.
//
// The caller is assembly for the reason the d8 one is — it has to actually
// hold a value in X21 across the call, and a C caller at -O0 will not.
func TestRunIndirectCallStackArgument(t *testing.T) {
	m := ir.NewModule("t", ir.AArch64Linux)
	sig := ir.NewSig()
	for i := 0; i < 9; i++ {
		sig = sig.Param(ir.TypeI64)
	}
	ft := m.FuncType("nine", sig.Ret(ir.TypeI64))

	fn := m.Func("_viacall9").Export()
	p := fn.ParamPtr("p")
	a := fn.ParamI64("a")
	fn.ReturnsI64()
	entry := fn.Entry()

	slot := entry.Ptr.Alloc(8, 8)
	entry.I64.Store(entry.I64.Const(0x5eed), slot)

	args := make([]ir.Value, 9)
	for i := range args {
		args[i] = entry.I64.Const(int64(i + 1))
	}
	r := entry.CallInd(p, ft, args...).Value(0).(ir.I64)
	entry.Return(entry.I64.Add(entry.I64.Add(r, entry.I64.Load(slot)), a))

	got := runNative(t, m, `
#include <stdio.h>
static long nine(long a,long b,long c,long d,long e,long f,long g,long h,long i) {
    return i * 1000 + a;
}
__asm__(
"	.text\n"
"	.globl _probe9\n"
"	.p2align 2\n"
"_probe9:\n"
"	sub  sp, sp, #48\n"
"	stp  x29, x30, [sp, #32]\n"
"	add  x29, sp, #32\n"
"	str  x21, [sp, #8]\n"        // the caller's x21, which _probe9 owes back
"	str  x2,  [sp, #16]\n"       // and the out-pointer, which the call destroys
"	mov  x1, #7\n"
"	mov  x21, #21845\n"
"	bl   _viacall9\n"            // x0 already holds the callee pointer
"	ldr  x2, [sp, #16]\n"
"	str  x0, [x2]\n"             // its answer, for the caller to check
"	mov  x0, x21\n"              // what survived
"	ldr  x21, [sp, #8]\n"
"	ldp  x29, x30, [sp, #32]\n"
"	add  sp, sp, #48\n"
"	ret\n"
);
long probe9(long (*)(long,long,long,long,long,long,long,long,long), long, long *);
int main(void) {
    long answer = 0;
    long kept = probe9(nine, 0, &answer);
    // 9*1000 + 1, plus the local 0x5eed, plus the 7 the probe passed.
    printf("%s %s\n", kept == 21845 ? "kept" : "CLOBBERED",
                      answer == 33309 ? "ok" : "MISMATCH");
    if (kept != 21845) printf("x21 came back as %ld\n", kept);
    if (answer != 33309) printf("answer was %ld\n", answer);
    return 0;
}
`)
	if got != "kept ok\n" {
		t.Errorf("printed %q, want %q", got, "kept ok\n")
	}
}
