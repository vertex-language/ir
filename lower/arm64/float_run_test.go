package arm64_test

// §A3, §B's float half and §C's conversions, checked by running them.
//
// Every case here computes the same expression twice — once through this
// backend and once by clang, in the C main the result is linked against — and
// compares the two. What that catches and a byte test cannot is a
// misunderstanding: an expectation written by hand carries whatever the author
// believed about the instruction, and clang's does not.

import (
	"testing"

	"github.com/vertex-language/ir"
)

// Every §A3 verb that is one instruction, against what C computes.
func TestRunFloatArithmetic(t *testing.T) {
	m := ir.NewModule("t", ir.AArch64Linux)
	fn := m.Func("_fmix").Export()
	a := fn.ParamF64("a")
	b := fn.ParamF64("b")
	fn.ReturnsF64()
	entry := fn.Entry()
	f := entry.F64

	sum := f.Add(a, b)
	dif := f.Sub(a, b)
	prod := f.Mul(sum, dif)
	quot := f.Div(prod, f.Add(b, f.Const(1.5)))
	r := f.Add(quot, f.Sqrt(f.Abs(dif)))
	r = f.Add(r, f.Neg(f.Floor(a)))
	r = f.Add(r, f.Ceil(b))
	r = f.Add(r, f.Trunc(quot))
	r = f.Add(r, f.Nearest(quot))
	r = f.Add(r, f.MinNum(a, b))
	r = f.Add(r, f.MaxNum(a, b))
	r = f.Add(r, f.FMA(a, b, dif))
	r = f.Add(r, f.CopySign(a, f.Neg(b)))
	entry.Return(r)

	got := runNative(t, m, `
#include <stdio.h>
#include <math.h>
double fmix(double, double);
int main(void) {
    double a = -7.25, b = 3.5;
    double sum = a + b, dif = a - b;
    double prod = sum * dif;
    double quot = prod / (b + 1.5);
    double want = quot + sqrt(fabs(dif));
    want += -floor(a);
    want += ceil(b);
    want += trunc(quot);
    want += nearbyint(quot);
    want += fmin(a, b);
    want += fmax(a, b);
    want += fma(a, b, dif);
    want += copysign(a, -b);
    double got = fmix(a, b);
    printf("%s\n", got == want ? "ok" : "MISMATCH");
    if (got != want) printf("got %.17g want %.17g\n", got, want);
    return 0;
}
`)
	if got != "ok\n" {
		t.Errorf("printed %q, want %q", got, "ok\n")
	}
}

// The same for f32, whose instructions are a different half of the encoding
// rather than the same one at another width.
func TestRunFloat32Arithmetic(t *testing.T) {
	m := ir.NewModule("t", ir.AArch64Linux)
	fn := m.Func("_f32mix").Export()
	a := fn.ParamF32("a")
	b := fn.ParamF32("b")
	fn.ReturnsF32()
	entry := fn.Entry()
	f := entry.F32

	r := f.Div(f.Mul(f.Add(a, b), f.Sub(a, b)), f.Const(3.25))
	r = f.Add(r, f.Sqrt(f.Abs(a)))
	r = f.Add(r, f.MaxNum(a, b))
	r = f.Add(r, f.CopySign(b, a))
	entry.Return(r)

	got := runNative(t, m, `
#include <stdio.h>
#include <math.h>
float f32mix(float, float);
int main(void) {
    float a = -7.25f, b = 3.5f;
    float want = ((a + b) * (a - b)) / 3.25f;
    want += sqrtf(fabsf(a));
    want += fmaxf(a, b);
    want += copysignf(b, a);
    printf("%s\n", f32mix(a, b) == want ? "ok" : "MISMATCH");
    return 0;
}
`)
	if got != "ok\n" {
		t.Errorf("printed %q, want %q", got, "ok\n")
	}
}

// §A3's two pairs of min and max, at the operands that tell them apart: a
// NaN, which minimum propagates and minnum discards, and a signed zero.
//
// The one case in this file whose expectation is not clang's, because C has
// no fminimum before C23 and the toolchain here may not have it either. The
// values are §A3's own words instead.
func TestRunFloatMinMaxNaN(t *testing.T) {
	m := ir.NewModule("t", ir.AArch64Linux)

	// Four functions, one per verb, so a wrong one names itself.
	for _, tc := range []struct {
		name string
		emit func(b *ir.Block, x, y ir.F64) ir.F64
	}{
		{"_vmin", func(b *ir.Block, x, y ir.F64) ir.F64 { return b.F64.Minimum(x, y) }},
		{"_vmax", func(b *ir.Block, x, y ir.F64) ir.F64 { return b.F64.Maximum(x, y) }},
		{"_vminnum", func(b *ir.Block, x, y ir.F64) ir.F64 { return b.F64.MinNum(x, y) }},
		{"_vmaxnum", func(b *ir.Block, x, y ir.F64) ir.F64 { return b.F64.MaxNum(x, y) }},
	} {
		fn := m.Func(tc.name).Export()
		x := fn.ParamF64("x")
		y := fn.ParamF64("y")
		fn.ReturnsF64()
		entry := fn.Entry()
		entry.Return(tc.emit(entry, x, y))
	}

	got := runNative(t, m, `
#include <stdio.h>
#include <math.h>
double vmin(double, double), vmax(double, double);
double vminnum(double, double), vmaxnum(double, double);
static int fail = 0;
static void chk(const char *what, double got, double want) {
    // Compared as bits, so that -0.0 and +0.0 are told apart.
    if (__builtin_memcmp(&got, &want, sizeof got) != 0) {
        printf("%s: got %.17g want %.17g\n", what, got, want);
        fail = 1;
    }
}
int main(void) {
    double nan = __builtin_nan(""), one = 1.0, two = 2.0;
    // minimum/maximum propagate a NaN.
    chk("minimum nan", isnan(vmin(nan, one)), 1.0);
    chk("minimum nan rhs", isnan(vmin(one, nan)), 1.0);
    chk("maximum nan", isnan(vmax(nan, one)), 1.0);
    // minnum/maxnum discard it.
    chk("minnum nan", vminnum(nan, one), one);
    chk("minnum nan rhs", vminnum(one, nan), one);
    chk("maxnum nan", vmaxnum(nan, two), two);
    // The ordinary answers.
    chk("minimum", vmin(one, two), one);
    chk("maximum", vmax(one, two), two);
    // Signed zero: -0 below +0 for both mins, +0 for both maxes.
    chk("minimum zero", vmin(0.0, -0.0), -0.0);
    chk("minnum zero", vminnum(0.0, -0.0), -0.0);
    chk("maximum zero", vmax(-0.0, 0.0), 0.0);
    chk("maxnum zero", vmaxnum(-0.0, 0.0), 0.0);
    printf("%s\n", fail ? "MISMATCH" : "ok");
    return 0;
}
`)
	if got != "ok\n" {
		t.Errorf("printed %q, want %q", got, "ok\n")
	}
}

// §B's five float comparisons, at ordinary operands and at a NaN — which is
// the whole reason the conditions are MI and LS rather than LT and LE.
func TestRunFloatCompare(t *testing.T) {
	m := ir.NewModule("t", ir.AArch64Linux)

	for _, tc := range []struct {
		name string
		emit func(b *ir.Block, x, y ir.F64) ir.I1
	}{
		{"_ceq", func(b *ir.Block, x, y ir.F64) ir.I1 { return b.F64.Eq(x, y) }},
		{"_cne", func(b *ir.Block, x, y ir.F64) ir.I1 { return b.F64.Ne(x, y) }},
		{"_clt", func(b *ir.Block, x, y ir.F64) ir.I1 { return b.F64.Lt(x, y) }},
		{"_cle", func(b *ir.Block, x, y ir.F64) ir.I1 { return b.F64.Le(x, y) }},
		{"_cuno", func(b *ir.Block, x, y ir.F64) ir.I1 { return b.F64.Uno(x, y) }},
	} {
		fn := m.Func(tc.name).Export()
		x := fn.ParamF64("x")
		y := fn.ParamF64("y")
		fn.ReturnsI32()
		entry := fn.Entry()
		entry.Return(entry.I32.ZExtI1(tc.emit(entry, x, y)))
	}

	got := runNative(t, m, `
#include <stdio.h>
int ceq(double,double), cne(double,double), clt(double,double);
int cle(double,double), cuno(double,double);
static int fail = 0;
static void chk(const char *what, int got, int want) {
    if (got != want) { printf("%s: got %d want %d\n", what, got, want); fail = 1; }
}
int main(void) {
    double nan = __builtin_nan(""), one = 1.0, two = 2.0;
    chk("eq",      ceq(one, one), 1);   chk("eq ne",  ceq(one, two), 0);
    chk("ne",      cne(one, two), 1);   chk("ne eq",  cne(one, one), 0);
    chk("lt",      clt(one, two), 1);   chk("lt gt",  clt(two, one), 0);
    chk("lt eq",   clt(one, one), 0);
    chk("le",      cle(one, two), 1);   chk("le eq",  cle(one, one), 1);
    chk("le gt",   cle(two, one), 0);
    chk("uno no",  cuno(one, two), 0);
    // The NaN column: every ordered verb false, ne true, uno true.
    chk("eq nan",  ceq(nan, one), 0);
    chk("ne nan",  cne(nan, one), 1);
    chk("lt nan",  clt(nan, one), 0);   chk("lt nan rhs", clt(one, nan), 0);
    chk("le nan",  cle(nan, one), 0);   chk("le nan rhs", cle(one, nan), 0);
    chk("uno nan", cuno(nan, one), 1);  chk("uno self",   cuno(nan, nan), 1);
    printf("%s\n", fail ? "MISMATCH" : "ok");
    return 0;
}
`)
	if got != "ok\n" {
		t.Errorf("printed %q, want %q", got, "ok\n")
	}
}

// §F's select with float arms, which is FCSEL rather than CSEL: the condition
// stays in the integer file and only the choice crosses.
func TestRunFloatSelect(t *testing.T) {
	m := ir.NewModule("t", ir.AArch64Linux)
	fn := m.Func("_fpick").Export()
	a := fn.ParamF64("a")
	b := fn.ParamF64("b")
	fn.ReturnsF64()
	entry := fn.Entry()
	entry.Return(entry.F64.Select(entry.F64.Lt(a, b), b, a))

	got := runNative(t, m, `
#include <stdio.h>
double fpick(double, double);
int main(void) {
    printf("%g %g\n", fpick(1.5, 9.5), fpick(9.5, 1.5));
    return 0;
}
`)
	if got != "9.5 9.5\n" {
		t.Errorf("printed %q, want %q", got, "9.5 9.5\n")
	}
}
