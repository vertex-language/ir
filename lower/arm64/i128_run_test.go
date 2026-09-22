package arm64_test

// i128 through the legalizer and out the other side, checked against the
// __int128 clang computes for the same inputs.

import (
	"testing"

	"github.com/vertex-language/ir"
)

// wide builds a function of four i64 halves (a then b) returning one half of
// the result, which is what keeps the ABI out of the question: every
// parameter and every result here is an i64.
func wide(m *ir.Module, name string, hi bool, op func(b *ir.Block, x, y ir.I128) ir.I128) {
	fn := m.Func(name).Export()
	alo, ahi := fn.ParamI64("alo"), fn.ParamI64("ahi")
	blo, bhi := fn.ParamI64("blo"), fn.ParamI64("bhi")
	fn.ReturnsI64()
	e := fn.Entry()
	x := e.I128.Parts(ahi, alo)
	y := e.I128.Parts(bhi, blo)
	r := op(e, x, y)
	if hi {
		e.Return(e.I128.Hi(r))
	} else {
		e.Return(e.I128.Lo(r))
	}
}

// pred builds a comparison, returning 0 or 1.
func pred(m *ir.Module, name string, op func(b *ir.Block, x, y ir.I128) ir.I1) {
	fn := m.Func(name).Export()
	alo, ahi := fn.ParamI64("alo"), fn.ParamI64("ahi")
	blo, bhi := fn.ParamI64("blo"), fn.ParamI64("bhi")
	fn.ReturnsI32()
	e := fn.Entry()
	e.Return(e.I32.ZExtI1(op(e, e.I128.Parts(ahi, alo), e.I128.Parts(bhi, blo))))
}

func TestRunI128(t *testing.T) {
	m := ir.NewModule("t", ir.AArch64MacOS)

	for _, tc := range []struct {
		name string
		op   func(b *ir.Block, x, y ir.I128) ir.I128
	}{
		{"add", func(b *ir.Block, x, y ir.I128) ir.I128 { return b.I128.Add(x, y) }},
		{"sub", func(b *ir.Block, x, y ir.I128) ir.I128 { return b.I128.Sub(x, y) }},
		{"mul", func(b *ir.Block, x, y ir.I128) ir.I128 { return b.I128.Mul(x, y) }},
		{"and", func(b *ir.Block, x, y ir.I128) ir.I128 { return b.I128.And(x, y) }},
		{"or", func(b *ir.Block, x, y ir.I128) ir.I128 { return b.I128.Or(x, y) }},
		{"xor", func(b *ir.Block, x, y ir.I128) ir.I128 { return b.I128.Xor(x, y) }},
		{"neg", func(b *ir.Block, x, y ir.I128) ir.I128 { return b.I128.Neg(x) }},
		{"not", func(b *ir.Block, x, y ir.I128) ir.I128 { return b.I128.Not(x) }},
		{"shl", func(b *ir.Block, x, y ir.I128) ir.I128 { return b.I128.Shl(x, y) }},
		{"ushr", func(b *ir.Block, x, y ir.I128) ir.I128 { return b.I128.UShr(x, y) }},
		{"sshr", func(b *ir.Block, x, y ir.I128) ir.I128 { return b.I128.SShr(x, y) }},
		{"sdiv", func(b *ir.Block, x, y ir.I128) ir.I128 { return b.I128.SDiv(x, y) }},
		{"udiv", func(b *ir.Block, x, y ir.I128) ir.I128 { return b.I128.UDiv(x, y) }},
		{"srem", func(b *ir.Block, x, y ir.I128) ir.I128 { return b.I128.SRem(x, y) }},
		{"urem", func(b *ir.Block, x, y ir.I128) ir.I128 { return b.I128.URem(x, y) }},
	} {
		wide(m, "_"+tc.name+"lo", false, tc.op)
		wide(m, "_"+tc.name+"hi", true, tc.op)
	}
	pred(m, "_eq", func(b *ir.Block, x, y ir.I128) ir.I1 { return b.I128.Eq(x, y) })
	pred(m, "_ne", func(b *ir.Block, x, y ir.I128) ir.I1 { return b.I128.Ne(x, y) })
	pred(m, "_slt", func(b *ir.Block, x, y ir.I128) ir.I1 { return b.I128.SLt(x, y) })
	pred(m, "_ult", func(b *ir.Block, x, y ir.I128) ir.I1 { return b.I128.ULt(x, y) })
	pred(m, "_sle", func(b *ir.Block, x, y ir.I128) ir.I1 { return b.I128.SLe(x, y) })
	pred(m, "_ule", func(b *ir.Block, x, y ir.I128) ir.I1 { return b.I128.ULe(x, y) })

	// Memory: read a pair, multiply it, write it back.
	ld := m.Func("_roundtrip").Export()
	p := ld.ParamPtr("p")
	k := ld.ParamI64("k")
	ld.ReturnsI64()
	e := ld.Entry()
	v := e.I128.Load(p)
	e.I128.Store(e.I128.Mul(v, e.I128.SExtI64(k)), p)
	e.Return(e.I128.Lo(v))

	if err := m.Err(); err != nil {
		t.Fatalf("build: %v", err)
	}
	if !m.LegalizeI128Opts(ir.I128Options{SymbolPrefix: "_"}) {
		t.Fatal("legalizer reported no change")
	}
	if err := m.Err(); err != nil {
		t.Fatalf("legalize: %v", err)
	}

	got := runNative(t, m, `
#include <stdio.h>
typedef __int128 i128;
typedef unsigned __int128 u128;
#define DECL(n) long n##lo(long,long,long,long); long n##hi(long,long,long,long);
DECL(add) DECL(sub) DECL(mul) DECL(and) DECL(or) DECL(xor) DECL(neg) DECL(not)
DECL(shl) DECL(ushr) DECL(sshr) DECL(sdiv) DECL(udiv) DECL(srem) DECL(urem)
int eq(long,long,long,long), ne(long,long,long,long), slt(long,long,long,long);
int ult(long,long,long,long), sle(long,long,long,long), ule(long,long,long,long);
long roundtrip(void*, long);

static int fail = 0;
static void chkw(const char *what, i128 a, i128 b, long glo, long ghi, i128 want) {
    long wlo = (long)(u128)want, whi = (long)(long long)(((u128)want) >> 64);
    if (glo != wlo || ghi != whi) {
        printf("%s: got %lx:%lx want %lx:%lx\n", what, ghi, glo, whi, wlo);
        fail = 1;
    }
}
static void chki(const char *what, int got, int want) {
    if (got != want) { printf("%s: got %d want %d\n", what, got, want); fail = 1; }
}
#define LO(x) ((long)(u128)(x))
#define HI(x) ((long)(long long)(((u128)(x)) >> 64))
#define CHK(n, expr) chkw(#n, a, b, n##lo(LO(a),HI(a),LO(b),HI(b)), n##hi(LO(a),HI(a),LO(b),HI(b)), (i128)(expr))

int main(void) {
    i128 vals[] = {
        0, 1, -1, 2, -2, 12345, -12345,
        ((i128)1) << 100, -(((i128)1) << 100),
        ((i128)1) << 64, ((i128)1 << 64) + 12345,
        (i128)0x7fffffffffffffffLL, (i128)0x8000000000000000ULL,
        ~(i128)0 >> 1, (i128)1 << 127,
        ((i128)0x0123456789abcdefLL << 64) | 0xfedcba9876543210ULL,
    };
    int n = sizeof(vals)/sizeof(vals[0]);
    for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) {
        i128 a = vals[i], b = vals[j];
        CHK(add, a + b); CHK(sub, a - b); CHK(mul, (i128)((u128)a * (u128)b));
        CHK(and, a & b); CHK(or, a | b); CHK(xor, a ^ b);
        CHK(neg, (i128)(-(u128)a)); CHK(not, ~a);
        chki("eq", eq(LO(a),HI(a),LO(b),HI(b)), a == b);
        chki("ne", ne(LO(a),HI(a),LO(b),HI(b)), a != b);
        chki("slt", slt(LO(a),HI(a),LO(b),HI(b)), a < b);
        chki("sle", sle(LO(a),HI(a),LO(b),HI(b)), a <= b);
        chki("ult", ult(LO(a),HI(a),LO(b),HI(b)), (u128)a < (u128)b);
        chki("ule", ule(LO(a),HI(a),LO(b),HI(b)), (u128)a <= (u128)b);
        if (b != 0 && !(a == ((i128)1 << 127) && b == -1)) {
            CHK(sdiv, a / b); CHK(srem, a % b);
        }
        if (b != 0) { CHK(udiv, (i128)((u128)a / (u128)b)); CHK(urem, (i128)((u128)a % (u128)b)); }
    }
    // Shifts: every amount, since the halves cross at 64.
    for (int i = 0; i < n; i++) for (int s = 0; s < 128; s++) {
        i128 a = vals[i], b = (i128)s;
        CHK(shl,  (i128)((u128)a << s));
        CHK(ushr, (i128)((u128)a >> s));
        CHK(sshr, a >> s);
    }
    // Memory, and that the store wrote both halves.
    i128 slot = ((i128)0x1122334455667788LL << 64) | 0x99aabbccddeeff00ULL;
    long got = roundtrip(&slot, 3);
    if (got != LO(((i128)0x1122334455667788LL << 64) | 0x99aabbccddeeff00ULL)) {
        printf("roundtrip read %lx\n", got); fail = 1;
    }
    i128 want = (((i128)0x1122334455667788LL << 64) | 0x99aabbccddeeff00ULL) * 3;
    if (slot != want) { printf("roundtrip store mismatch\n"); fail = 1; }
    printf("%s\n", fail ? "MISMATCH" : "ok");
    return 0;
}
`)
	if got != "ok\n" {
		t.Errorf("printed %q, want %q", got, "ok\n")
	}
}
