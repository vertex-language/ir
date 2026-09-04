package i386_test

// §A's division and wide multiply, and §A2's predicates.
//
// The 64-bit divisions are calls to the helpers a C compiler would call, so
// the freestanding runtime has to supply them — there is no compiler-rt in a
// kernel. They are written out longhand below, which is fine: what is under
// test is this backend's call, not the helper's arithmetic.

import (
	"testing"

	"github.com/vertex-language/ir"
)

// divHelpersC is __divdi3 and its neighbours, by shift-and-subtract. Slow and
// obviously correct, which is what a test wants from the other side.
const divHelpersC = `
static unsigned long long udivmod(unsigned long long n, unsigned long long d,
                                  unsigned long long *rem) {
    unsigned long long q = 0, r = 0;
    for (int i = 63; i >= 0; i--) {
        r = (r << 1) | ((n >> i) & 1);
        if (r >= d) { r -= d; q |= 1ULL << i; }
    }
    if (rem) *rem = r;
    return q;
}
unsigned long long __udivdi3(unsigned long long a, unsigned long long b) {
    return udivmod(a, b, 0);
}
unsigned long long __umoddi3(unsigned long long a, unsigned long long b) {
    unsigned long long r; udivmod(a, b, &r); return r;
}
long long __divdi3(long long a, long long b) {
    int neg = 0;
    unsigned long long ua = a < 0 ? (neg ^= 1, -(unsigned long long)a) : (unsigned long long)a;
    unsigned long long ub = b < 0 ? (neg ^= 1, -(unsigned long long)b) : (unsigned long long)b;
    unsigned long long q = udivmod(ua, ub, 0);
    return neg ? -(long long)q : (long long)q;
}
long long __moddi3(long long a, long long b) {
    int neg = a < 0;
    unsigned long long ua = a < 0 ? -(unsigned long long)a : (unsigned long long)a;
    unsigned long long ub = b < 0 ? -(unsigned long long)b : (unsigned long long)b;
    unsigned long long r; udivmod(ua, ub, &r);
    return neg ? -(long long)r : (long long)r;
}
`

func TestRunDivision(t *testing.T) {
	m := ir.NewModule("t", ir.I386Linux)

	for _, tc := range []struct {
		name string
		emit func(b *ir.Block, x, y ir.I32) ir.I32
	}{
		{"d32s", func(b *ir.Block, x, y ir.I32) ir.I32 { return b.I32.SDiv(x, y) }},
		{"d32u", func(b *ir.Block, x, y ir.I32) ir.I32 { return b.I32.UDiv(x, y) }},
		{"r32s", func(b *ir.Block, x, y ir.I32) ir.I32 { return b.I32.SRem(x, y) }},
		{"r32u", func(b *ir.Block, x, y ir.I32) ir.I32 { return b.I32.URem(x, y) }},
		{"mhs", func(b *ir.Block, x, y ir.I32) ir.I32 { return b.I32.SMulHi(x, y) }},
		{"mhu", func(b *ir.Block, x, y ir.I32) ir.I32 { return b.I32.UMulHi(x, y) }},
	} {
		fn := m.Func(tc.name).Export()
		x := fn.ParamI32("x")
		y := fn.ParamI32("y")
		fn.ReturnsI32()
		e := fn.Entry()
		e.Return(tc.emit(e, x, y))
	}
	for _, tc := range []struct {
		name string
		emit func(b *ir.Block, x, y ir.I64) ir.I64
	}{
		{"d64s", func(b *ir.Block, x, y ir.I64) ir.I64 { return b.I64.SDiv(x, y) }},
		{"d64u", func(b *ir.Block, x, y ir.I64) ir.I64 { return b.I64.UDiv(x, y) }},
		{"r64s", func(b *ir.Block, x, y ir.I64) ir.I64 { return b.I64.SRem(x, y) }},
		{"m64", func(b *ir.Block, x, y ir.I64) ir.I64 { return b.I64.Mul(x, y) }},
	} {
		fn := m.Func(tc.name).Export()
		x := fn.ParamI64("x")
		y := fn.ParamI64("y")
		fn.ReturnsI64()
		e := fn.Entry()
		e.Return(tc.emit(e, x, y))
	}

	wantOK(t, m, divHelpersC+`
int d32s(int,int), r32s(int,int), mhs(int,int);
unsigned d32u(unsigned,unsigned), r32u(unsigned,unsigned), mhu(unsigned,unsigned);
long long d64s(long long,long long), r64s(long long,long long), m64(long long,long long);
unsigned long long d64u(unsigned long long,unsigned long long);
static void body(void) {
    chk32("sdiv",      (unsigned)d32s(100, 7), (unsigned)(100 / 7));
    chk32("sdiv neg",  (unsigned)d32s(-100, 7), (unsigned)(-100 / 7));
    chk32("sdiv neg2", (unsigned)d32s(100, -7), (unsigned)(100 / -7));
    chk32("srem neg",  (unsigned)r32s(-100, 7), (unsigned)(-100 % 7));
    chk32("udiv",      d32u(4000000000u, 7u), 4000000000u / 7u);
    chk32("urem",      r32u(4000000000u, 7u), 4000000000u % 7u);

    /* The high half, which the ordinary multiply throws away. */
    chk32("umulhi", (unsigned)mhu(0x80000000u, 4u), 2u);
    chk32("smulhi", (unsigned)mhs(-1, -1), 0u);
    chk32("smulhi2", (unsigned)mhs((int)0x80000000, 2), 0xffffffffu);

    /* And at sixty-four, which is a call. */
    chk64("div64",  d64s(0x123456789abcdef0LL, 1000LL), 0x123456789abcdef0LL / 1000LL);
    chk64("div64 neg", d64s(-0x123456789abcdef0LL, 1000LL), -0x123456789abcdef0LL / 1000LL);
    chk64("udiv64", d64u(0xf23456789abcdef0ULL, 1000ULL), 0xf23456789abcdef0ULL / 1000ULL);
    chk64("rem64",  r64s(0x123456789abcdef0LL, 1000LL), 0x123456789abcdef0LL % 1000LL);

    /* The 64-bit multiply, whose crossing terms are what make it three. */
    chk64("mul64 small", m64(6, 7), 42ULL);
    chk64("mul64 cross", m64(0x0000000100000000LL, 3LL), 0x0000000300000000ULL);
    chk64("mul64 big", m64(0x0123456789abcdefLL, 0x11LL),
                       (unsigned long long)(0x0123456789abcdefULL * 0x11ULL));
    chk64("mul64 wrap", m64(0x123456789abcdef0LL, 0x123456789abcdef0LL),
                        (unsigned long long)(0x123456789abcdef0ULL * 0x123456789abcdef0ULL));
}
`)
}

func TestRunOverflowPredicates(t *testing.T) {
	m := ir.NewModule("t", ir.I386Linux)

	for _, tc := range []struct {
		name string
		emit func(b *ir.Block, x, y ir.I32) ir.I1
	}{
		{"o32sa", func(b *ir.Block, x, y ir.I32) ir.I1 { return b.I32.SAddO(x, y) }},
		{"o32ua", func(b *ir.Block, x, y ir.I32) ir.I1 { return b.I32.UAddO(x, y) }},
		{"o32ss", func(b *ir.Block, x, y ir.I32) ir.I1 { return b.I32.SSubO(x, y) }},
		{"o32sm", func(b *ir.Block, x, y ir.I32) ir.I1 { return b.I32.SMulO(x, y) }},
		{"o32um", func(b *ir.Block, x, y ir.I32) ir.I1 { return b.I32.UMulO(x, y) }},
	} {
		fn := m.Func(tc.name).Export()
		x := fn.ParamI32("x")
		y := fn.ParamI32("y")
		fn.ReturnsI32()
		e := fn.Entry()
		e.Return(e.I32.ZExtI1(tc.emit(e, x, y)))
	}
	for _, tc := range []struct {
		name string
		emit func(b *ir.Block, x, y ir.I64) ir.I1
	}{
		{"o64sa", func(b *ir.Block, x, y ir.I64) ir.I1 { return b.I64.SAddO(x, y) }},
		{"o64ua", func(b *ir.Block, x, y ir.I64) ir.I1 { return b.I64.UAddO(x, y) }},
		{"o64ss", func(b *ir.Block, x, y ir.I64) ir.I1 { return b.I64.SSubO(x, y) }},
	} {
		fn := m.Func(tc.name).Export()
		x := fn.ParamI64("x")
		y := fn.ParamI64("y")
		fn.ReturnsI32()
		e := fn.Entry()
		e.Return(e.I32.ZExtI1(tc.emit(e, x, y)))
	}

	wantOK(t, m, `
int o32sa(int,int), o32ss(int,int), o32sm(int,int);
int o32ua(unsigned,unsigned), o32um(unsigned,unsigned);
int o64sa(long long,long long), o64ss(long long,long long);
int o64ua(unsigned long long,unsigned long long);
#define IMAX 2147483647
#define IMIN (-IMAX - 1)
#define LMAX 9223372036854775807LL
#define LMIN (-LMAX - 1)
static void body(void) {
    chk32("saddo no",  (unsigned)o32sa(1, 2), 0u);
    chk32("saddo yes", (unsigned)o32sa(IMAX, 1), 1u);
    chk32("saddo neg", (unsigned)o32sa(IMIN, -1), 1u);
    chk32("uaddo no",  (unsigned)o32ua(1u, 2u), 0u);
    chk32("uaddo yes", (unsigned)o32ua(0xffffffffu, 1u), 1u);
    chk32("ssubo no",  (unsigned)o32ss(1, 2), 0u);
    chk32("ssubo yes", (unsigned)o32ss(IMIN, 1), 1u);
    chk32("smulo no",  (unsigned)o32sm(1000, 1000), 0u);
    chk32("smulo yes", (unsigned)o32sm(65536, 65536), 1u);
    chk32("smulo neg", (unsigned)o32sm(IMIN, -1), 1u);
    chk32("umulo no",  (unsigned)o32um(65535u, 65535u), 0u);
    chk32("umulo yes", (unsigned)o32um(65536u, 65536u), 1u);

    chk32("saddo64 no",  (unsigned)o64sa(1, 2), 0u);
    chk32("saddo64 yes", (unsigned)o64sa(LMAX, 1), 1u);
    chk32("saddo64 neg", (unsigned)o64sa(LMIN, -1), 1u);
    /* Carrying out of the low half is not overflow; only out of the top. */
    chk32("saddo64 carry", (unsigned)o64sa(0x00000000ffffffffLL, 1), 0u);
    chk32("uaddo64 no",  (unsigned)o64ua(1ULL, 2ULL), 0u);
    chk32("uaddo64 yes", (unsigned)o64ua(0xffffffffffffffffULL, 1ULL), 1u);
    chk32("uaddo64 mid", (unsigned)o64ua(0x00000000ffffffffULL, 1ULL), 0u);
    chk32("ssubo64 no",  (unsigned)o64ss(1, 2), 0u);
    chk32("ssubo64 yes", (unsigned)o64ss(LMIN, 1), 1u);
}
`)
}
