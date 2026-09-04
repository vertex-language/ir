package arm64_test

// §C2 and §C3, and §A's division — the three places where an A64 instruction
// exists but does not mean what the spec's verb means, so the lowering is the
// instruction plus whatever makes up the difference.

import (
	"testing"

	"github.com/vertex-language/ir"
)

// §C2's int-to-float and §C3's fcvt and bitcast, against C's own casts.
func TestRunConversions(t *testing.T) {
	m := ir.NewModule("t", ir.AArch64Linux)

	// f64 from a signed i64 and from an unsigned one, which differ for
	// any value with the top bit set.
	sc := m.Func("_s2d").Export()
	scX := sc.ParamI64("x")
	sc.ReturnsF64()
	scEntry := sc.Entry()
	scEntry.Return(scEntry.F64.SCvtI64(scX))

	uc := m.Func("_u2d").Export()
	ucX := uc.ParamI64("x")
	uc.ReturnsF64()
	ucEntry := uc.Entry()
	ucEntry.Return(ucEntry.F64.UCvtI64(ucX))

	// f32 widened to f64 and back.
	rt := m.Func("_narrow").Export()
	rtX := rt.ParamF64("x")
	rt.ReturnsF64()
	rtEntry := rt.Entry()
	rtEntry.Return(rtEntry.F64.FCvtF32(rtEntry.F32.FCvtF64(rtX)))

	// The bits of an f64 as an i64, and back.
	bc := m.Func("_bits").Export()
	bcX := bc.ParamF64("x")
	bc.ReturnsI64()
	bcEntry := bc.Entry()
	bcEntry.Return(bcEntry.I64.BitcastF64(bcX))

	un := m.Func("_unbits").Export()
	unX := un.ParamI64("x")
	un.ReturnsF64()
	unEntry := un.Entry()
	unEntry.Return(unEntry.F64.BitcastI64(unX))

	got := runNative(t, m, `
#include <stdio.h>
#include <string.h>
double s2d(long), u2d(unsigned long), narrow(double), unbits(long);
long bits(double);
static int fail = 0;
static void chkd(const char *what, double got, double want) {
    if (memcmp(&got, &want, sizeof got) != 0) {
        printf("%s: got %.17g want %.17g\n", what, got, want); fail = 1;
    }
}
int main(void) {
    chkd("s2d",      s2d(-3),  (double)(long)-3);
    chkd("s2d big",  s2d(1234567890123L), (double)(long)1234567890123L);
    // The same bits read the other way: as a signed long this is negative
    // and as an unsigned one it is nearly 2^64.
    unsigned long big = 0xffffffffffffff00UL;
    chkd("s2d neg",  s2d((long)big), (double)(long)big);
    chkd("u2d",      u2d(big), (double)big);
    chkd("narrow",   narrow(0.1), (double)(float)0.1);
    double d = -12.375;
    long b = bits(d);
    long wantb; memcpy(&wantb, &d, sizeof d);
    if (b != wantb) { printf("bits: got %ld want %ld\n", b, wantb); fail = 1; }
    chkd("unbits", unbits(wantb), d);
    printf("%s\n", fail ? "MISMATCH" : "ok");
    return 0;
}
`)
	if got != "ok\n" {
		t.Errorf("printed %q, want %q", got, "ok\n")
	}
}

// §C2's saturating float-to-int, which is one instruction here: FCVTZS and
// FCVTZU clamp to the endpoint and give zero for a NaN, which is `_sat_`'s
// specification exactly.
func TestRunSaturatingConversions(t *testing.T) {
	m := ir.NewModule("t", ir.AArch64Linux)

	s64 := m.Func("_sat_s64").Export()
	s64X := s64.ParamF64("x")
	s64.ReturnsI64()
	e := s64.Entry()
	e.Return(e.I64.SCvtSatF64(s64X))

	u64 := m.Func("_sat_u64").Export()
	u64X := u64.ParamF64("x")
	u64.ReturnsI64()
	e2 := u64.Entry()
	e2.Return(e2.I64.UCvtSatF64(u64X))

	s32 := m.Func("_sat_s32").Export()
	s32X := s32.ParamF64("x")
	s32.ReturnsI32()
	e3 := s32.Entry()
	e3.Return(e3.I32.SCvtSatF64(s32X))

	got := runNative(t, m, `
#include <stdio.h>
long sat_s64(double); unsigned long sat_u64(double); int sat_s32(double);
static int fail = 0;
static void chk(const char *what, long got, long want) {
    if (got != want) { printf("%s: got %ld want %ld\n", what, got, want); fail = 1; }
}
int main(void) {
    double nan = __builtin_nan(""), inf = __builtin_inf();
    chk("ordinary",  sat_s64(-42.9), -42);
    chk("nan",       sat_s64(nan), 0);
    chk("+inf",      sat_s64(inf), 9223372036854775807L);
    chk("-inf",      sat_s64(-inf), -9223372036854775807L - 1);
    chk("u nan",     (long)sat_u64(nan), 0);
    chk("u neg",     (long)sat_u64(-5.0), 0);
    chk("u +inf",    (long)sat_u64(inf), (long)0xffffffffffffffffUL);
    chk("s32 above", sat_s32(1e300), 2147483647);
    chk("s32 below", sat_s32(-1e300), -2147483647 - 1);
    chk("s32 nan",   sat_s32(nan), 0);
    printf("%s\n", fail ? "MISMATCH" : "ok");
    return 0;
}
`)
	if got != "ok\n" {
		t.Errorf("printed %q, want %q", got, "ok\n")
	}
}

// §C2's trapping float-to-int, on values that are in range. The bounds are
// the interesting part: a value that truncates into range has to be admitted
// even when it is outside it before truncation.
func TestRunTrappingConversionsInRange(t *testing.T) {
	m := ir.NewModule("t", ir.AArch64Linux)

	s32 := m.Func("_cvt_s32").Export()
	s32X := s32.ParamF64("x")
	s32.ReturnsI32()
	e := s32.Entry()
	e.Return(e.I32.SCvtF64(s32X))

	u32 := m.Func("_cvt_u32").Export()
	u32X := u32.ParamF64("x")
	u32.ReturnsI32()
	e2 := u32.Entry()
	e2.Return(e2.I32.UCvtF64(u32X))

	s64 := m.Func("_cvt_s64").Export()
	s64X := s64.ParamF64("x")
	s64.ReturnsI64()
	e3 := s64.Entry()
	e3.Return(e3.I64.SCvtF64(s64X))

	got := runNative(t, m, `
#include <stdio.h>
int cvt_s32(double); unsigned cvt_u32(double); long cvt_s64(double);
static int fail = 0;
static void chk(const char *what, long got, long want) {
    if (got != want) { printf("%s: got %ld want %ld\n", what, got, want); fail = 1; }
}
int main(void) {
    chk("zero",     cvt_s32(0.0), 0);
    chk("toward 0", cvt_s32(-1.9), -1);
    chk("max",      cvt_s32(2147483647.0), 2147483647);
    chk("min",      cvt_s32(-2147483648.0), -2147483648);
    // Below INT_MIN before truncation and inside it after, which the
    // strict lower bound is there to admit.
    chk("min frac", cvt_s32(-2147483648.5), -2147483648);
    chk("u zero",   cvt_u32(0.0), 0);
    chk("u neg fr", (long)cvt_u32(-0.5), 0);
    chk("u max",    (long)cvt_u32(4294967295.0), 4294967295L);
    chk("s64 max",  cvt_s64(9223372036854774784.0), 9223372036854774784L);
    chk("s64 min",  cvt_s64(-9223372036854775808.0), -9223372036854775807L - 1);
    printf("%s\n", fail ? "MISMATCH" : "ok");
    return 0;
}
`)
	if got != "ok\n" {
		t.Errorf("printed %q, want %q", got, "ok\n")
	}
}

// And that the same conversion actually traps outside the interval. One
// module per case, because a trap ends the process.
func TestRunTrappingConversionTraps(t *testing.T) {
	cases := []struct {
		name string
		arg  string
	}{
		{"NaN", `__builtin_nan("")`},
		{"above", `2147483648.0`},
		{"below", `-2147483649.0`},
		{"+inf", `__builtin_inf()`},
		{"-inf", `-__builtin_inf()`},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			m := ir.NewModule("t", ir.AArch64Linux)
			fn := m.Func("_cvt_s32").Export()
			fnX := fn.ParamF64("x")
			fn.ReturnsI32()
			e := fn.Entry()
			e.Return(e.I32.SCvtF64(fnX))

			runNativeTrap(t, m, `
#include <stdio.h>
int cvt_s32(double);
int main(void) { printf("%d\n", cvt_s32(`+tc.arg+`)); return 0; }
`)
		})
	}
}

// §A's four division verbs, against C's, at operands including the ones
// where truncation toward zero has to be the direction.
func TestRunDivision(t *testing.T) {
	m := ir.NewModule("t", ir.AArch64Linux)

	for _, tc := range []struct {
		name string
		emit func(b *ir.Block, x, y ir.I64) ir.I64
	}{
		{"_vsdiv", func(b *ir.Block, x, y ir.I64) ir.I64 { return b.I64.SDiv(x, y) }},
		{"_vudiv", func(b *ir.Block, x, y ir.I64) ir.I64 { return b.I64.UDiv(x, y) }},
		{"_vsrem", func(b *ir.Block, x, y ir.I64) ir.I64 { return b.I64.SRem(x, y) }},
		{"_vurem", func(b *ir.Block, x, y ir.I64) ir.I64 { return b.I64.URem(x, y) }},
	} {
		fn := m.Func(tc.name).Export()
		x := fn.ParamI64("x")
		y := fn.ParamI64("y")
		fn.ReturnsI64()
		entry := fn.Entry()
		entry.Return(tc.emit(entry, x, y))
	}

	// And the 32-bit width, which is a different instruction rather than
	// the same one narrowed.
	f32 := m.Func("_vsdiv32").Export()
	a := f32.ParamI32("a")
	b := f32.ParamI32("b")
	f32.ReturnsI32()
	e := f32.Entry()
	e.Return(e.I32.SDiv(a, b))

	got := runNative(t, m, `
#include <stdio.h>
long vsdiv(long,long), vsrem(long,long);
unsigned long vudiv(unsigned long,unsigned long), vurem(unsigned long,unsigned long);
int vsdiv32(int,int);
static int fail = 0;
static void chk(const char *what, long got, long want) {
    if (got != want) { printf("%s: got %ld want %ld\n", what, got, want); fail = 1; }
}
int main(void) {
    chk("sdiv",      vsdiv(100, 7), 100 / 7);
    chk("sdiv neg",  vsdiv(-100, 7), -100 / 7);
    chk("sdiv neg2", vsdiv(100, -7), 100 / -7);
    chk("sdiv both", vsdiv(-100, -7), -100 / -7);
    chk("srem",      vsrem(100, 7), 100 % 7);
    chk("srem neg",  vsrem(-100, 7), -100 % 7);
    chk("srem neg2", vsrem(100, -7), 100 % -7);
    // INT_MIN divided by anything but -1 is ordinary.
    chk("sdiv min",  vsdiv(-9223372036854775807L - 1, 2), (-9223372036854775807L - 1) / 2);
    unsigned long big = 0xffffffffffffff00UL;
    chk("udiv",      (long)vudiv(big, 7), (long)(big / 7));
    chk("urem",      (long)vurem(big, 7), (long)(big % 7));
    chk("sdiv32",    vsdiv32(-100, 7), -100 / 7);
    printf("%s\n", fail ? "MISMATCH" : "ok");
    return 0;
}
`)
	if got != "ok\n" {
		t.Errorf("printed %q, want %q", got, "ok\n")
	}
}

// Both of §A's division traps, which A64 does not raise for itself: SDIV by
// zero gives zero and INT_MIN/−1 gives INT_MIN, quietly, and §A says each is
// a trap.
func TestRunDivisionTraps(t *testing.T) {
	cases := []struct {
		name   string
		signed bool
		rem    bool
		args   string
	}{
		{"sdiv by zero", true, false, "7, 0"},
		{"udiv by zero", false, false, "7, 0"},
		{"srem by zero", true, true, "7, 0"},
		{"urem by zero", false, true, "7, 0"},
		{"sdiv overflow", true, false, "-9223372036854775807L - 1, -1"},
		{"srem overflow", true, true, "-9223372036854775807L - 1, -1"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			m := ir.NewModule("t", ir.AArch64Linux)
			fn := m.Func("_d").Export()
			x := fn.ParamI64("x")
			y := fn.ParamI64("y")
			fn.ReturnsI64()
			entry := fn.Entry()
			switch {
			case tc.signed && tc.rem:
				entry.Return(entry.I64.SRem(x, y))
			case tc.signed:
				entry.Return(entry.I64.SDiv(x, y))
			case tc.rem:
				entry.Return(entry.I64.URem(x, y))
			default:
				entry.Return(entry.I64.UDiv(x, y))
			}
			runNativeTrap(t, m, `
#include <stdio.h>
long d(long, long);
int main(void) { printf("%ld\n", d(`+tc.args+`)); return 0; }
`)
		})
	}
}

// A signed divide whose divisor is −1 and whose dividend is not INT_MIN: the
// second guard has to let it through rather than trapping on the divisor
// alone.
func TestRunDivideByMinusOne(t *testing.T) {
	m := ir.NewModule("t", ir.AArch64Linux)
	fn := m.Func("_d").Export()
	x := fn.ParamI64("x")
	y := fn.ParamI64("y")
	fn.ReturnsI64()
	entry := fn.Entry()
	entry.Return(entry.I64.SDiv(x, y))

	got := runNative(t, m, `
#include <stdio.h>
long d(long, long);
int main(void) { printf("%ld %ld\n", d(42, -1), d(-42, -1)); return 0; }
`)
	if got != "-42 42\n" {
		t.Errorf("printed %q, want %q", got, "-42 42\n")
	}
}
