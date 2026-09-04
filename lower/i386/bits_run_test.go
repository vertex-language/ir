package i386_test

// §A5 and §A6, which is where a 32-bit machine has the most to do: a 64-bit
// shift crosses between the halves, and neither clz nor ctz nor popcnt has an
// instruction on the 386 that answers what §A6 asks for.

import (
	"testing"

	"github.com/vertex-language/ir"
)

func TestRunShifts(t *testing.T) {
	m := ir.NewModule("t", ir.I386Linux)

	for _, tc := range []struct {
		name string
		emit func(b *ir.Block, x, n ir.I64) ir.I64
	}{
		{"s64shl", func(b *ir.Block, x, n ir.I64) ir.I64 { return b.I64.Shl(x, n) }},
		{"s64ushr", func(b *ir.Block, x, n ir.I64) ir.I64 { return b.I64.UShr(x, n) }},
		{"s64sshr", func(b *ir.Block, x, n ir.I64) ir.I64 { return b.I64.SShr(x, n) }},
		{"s64rotl", func(b *ir.Block, x, n ir.I64) ir.I64 { return b.I64.RotL(x, n) }},
		{"s64rotr", func(b *ir.Block, x, n ir.I64) ir.I64 { return b.I64.RotR(x, n) }},
	} {
		fn := m.Func(tc.name).Export()
		x := fn.ParamI64("x")
		n := fn.ParamI64("n")
		fn.ReturnsI64()
		e := fn.Entry()
		e.Return(tc.emit(e, x, n))
	}
	for _, tc := range []struct {
		name string
		emit func(b *ir.Block, x, n ir.I32) ir.I32
	}{
		{"s32shl", func(b *ir.Block, x, n ir.I32) ir.I32 { return b.I32.Shl(x, n) }},
		{"s32ushr", func(b *ir.Block, x, n ir.I32) ir.I32 { return b.I32.UShr(x, n) }},
		{"s32sshr", func(b *ir.Block, x, n ir.I32) ir.I32 { return b.I32.SShr(x, n) }},
		{"s32rotl", func(b *ir.Block, x, n ir.I32) ir.I32 { return b.I32.RotL(x, n) }},
		{"s32rotr", func(b *ir.Block, x, n ir.I32) ir.I32 { return b.I32.RotR(x, n) }},
	} {
		fn := m.Func(tc.name).Export()
		x := fn.ParamI32("x")
		n := fn.ParamI32("n")
		fn.ReturnsI32()
		e := fn.Entry()
		e.Return(tc.emit(e, x, n))
	}

	wantOK(t, m, `
long long s64shl(long long, long long), s64ushr(long long, long long);
long long s64sshr(long long, long long), s64rotl(long long, long long);
long long s64rotr(long long, long long);
int s32shl(int,int), s32ushr(int,int), s32sshr(int,int);
int s32rotl(int,int), s32rotr(int,int);

static unsigned long long shl(unsigned long long v, int n) { return n ? v << n : v; }
static unsigned long long ushr(unsigned long long v, int n) { return n ? v >> n : v; }
static unsigned long long sshr(long long v, int n) { return n ? v >> n : v; }
static unsigned long long rotl(unsigned long long v, int n) {
    n &= 63; return n ? (v << n) | (v >> (64 - n)) : v;
}
static unsigned long long rotr(unsigned long long v, int n) {
    n &= 63; return n ? (v >> n) | (v << (64 - n)) : v;
}
static void body(void) {
    unsigned long long v = 0x123456789abcdef0ULL;
    long long s = (long long)0xfedcba9876543210ULL;   /* negative */
    /* Every count from 0 to 63: the interesting ones are 0, where a naive
       shift by 32-n is undefined, and 32 and up, where the halves cross. */
    for (int n = 0; n < 64; n++) {
        chk64("shl",  s64shl((long long)v, n), shl(v, n));
        chk64("ushr", s64ushr((long long)v, n), ushr(v, n));
        chk64("sshr", s64sshr(s, n), sshr(s, n));
        chk64("rotl", s64rotl((long long)v, n), rotl(v, n));
        chk64("rotr", s64rotr((long long)v, n), rotr(v, n));
    }
    /* A count past the width, which §A5 takes modulo it. */
    chk64("shl mod",  s64shl((long long)v, 64), v);
    chk64("shl mod2", s64shl((long long)v, 65), shl(v, 1));
    chk64("rotl mod", s64rotl((long long)v, 64), v);

    unsigned w = 0x9abcdef0u;
    for (int n = 0; n < 32; n++) {
        chk32("shl32",  (unsigned)s32shl((int)w, n), n ? w << n : w);
        chk32("ushr32", (unsigned)s32ushr((int)w, n), n ? w >> n : w);
        chk32("sshr32", (unsigned)s32sshr((int)w, n), n ? (unsigned)((int)w >> n) : w);
        chk32("rotl32", (unsigned)s32rotl((int)w, n), n ? (w << n) | (w >> (32 - n)) : w);
        chk32("rotr32", (unsigned)s32rotr((int)w, n), n ? (w >> n) | (w << (32 - n)) : w);
    }
}
`)
}

func TestRunBitCounting(t *testing.T) {
	m := ir.NewModule("t", ir.I386Linux)

	for _, tc := range []struct {
		name string
		emit func(b *ir.Block, x ir.I64) ir.I64
	}{
		{"b64clz", func(b *ir.Block, x ir.I64) ir.I64 { return b.I64.Clz(x) }},
		{"b64ctz", func(b *ir.Block, x ir.I64) ir.I64 { return b.I64.Ctz(x) }},
		{"b64pop", func(b *ir.Block, x ir.I64) ir.I64 { return b.I64.Popcnt(x) }},
		{"b64bsw", func(b *ir.Block, x ir.I64) ir.I64 { return b.I64.Bswap(x) }},
	} {
		fn := m.Func(tc.name).Export()
		x := fn.ParamI64("x")
		fn.ReturnsI64()
		e := fn.Entry()
		e.Return(tc.emit(e, x))
	}
	for _, tc := range []struct {
		name string
		emit func(b *ir.Block, x ir.I32) ir.I32
	}{
		{"b32clz", func(b *ir.Block, x ir.I32) ir.I32 { return b.I32.Clz(x) }},
		{"b32ctz", func(b *ir.Block, x ir.I32) ir.I32 { return b.I32.Ctz(x) }},
		{"b32pop", func(b *ir.Block, x ir.I32) ir.I32 { return b.I32.Popcnt(x) }},
		{"b32bsw", func(b *ir.Block, x ir.I32) ir.I32 { return b.I32.Bswap(x) }},
	} {
		fn := m.Func(tc.name).Export()
		x := fn.ParamI32("x")
		fn.ReturnsI32()
		e := fn.Entry()
		e.Return(tc.emit(e, x))
	}

	wantOK(t, m, `
long long b64clz(long long), b64ctz(long long), b64pop(long long), b64bsw(long long);
int b32clz(int), b32ctz(int), b32pop(int), b32bsw(int);
static void body(void) {
    /* Zero is the case §A6 pins down and the instruction leaves undefined. */
    chk64("clz zero", b64clz(0), 64);
    chk64("ctz zero", b64ctz(0), 64);
    chk32("clz32 zero", (unsigned)b32clz(0), 32u);
    chk32("ctz32 zero", (unsigned)b32ctz(0), 32u);

    for (int i = 0; i < 64; i++) {
        unsigned long long bit = 1ULL << i;
        chk64("clz bit", b64clz((long long)bit), (unsigned long long)(63 - i));
        chk64("ctz bit", b64ctz((long long)bit), (unsigned long long)i);
        chk64("pop bit", b64pop((long long)bit), 1ULL);
    }
    for (int i = 0; i < 32; i++) {
        unsigned bit = 1u << i;
        chk32("clz32 bit", (unsigned)b32clz((int)bit), (unsigned)(31 - i));
        chk32("ctz32 bit", (unsigned)b32ctz((int)bit), (unsigned)i);
    }
    /* Set bits in both halves, so the half that answers is the near one. */
    chk64("clz both", b64clz((long long)0x0000000180000000ULL), 31);
    chk64("ctz both", b64ctz((long long)0x0000000180000000ULL), 31);
    chk64("clz lo only", b64clz((long long)0x00000000000000ffULL), 56);
    chk64("ctz hi only", b64ctz((long long)0xff00000000000000ULL), 56);

    chk64("pop all", b64pop(-1LL), 64);
    chk64("pop alt", b64pop((long long)0xaaaaaaaaaaaaaaaaULL), 32);
    chk64("pop mix", b64pop((long long)0x123456789abcdef0ULL), 32);
    chk32("pop32 all", (unsigned)b32pop(-1), 32u);
    chk32("pop32 mix", (unsigned)b32pop((int)0x9abcdef0), 19u);

    chk64("bswap", b64bsw((long long)0x0123456789abcdefULL), 0xefcdab8967452301ULL);
    chk32("bswap32", (unsigned)b32bsw((int)0x01234567), 0x67452301u);
}
`)
}

// §C4 and §D3's diff, where a 32-bit pointer meets a 64-bit integer.
func TestRunPointerInt(t *testing.T) {
	m := ir.NewModule("t", ir.I386Linux)

	fp := m.Func("fromptr").Export()
	fpP := fp.ParamPtr("p")
	fp.ReturnsI64()
	e := fp.Entry()
	e.Return(e.I64.FromPtr(fpP))

	df := m.Func("pdiff").Export()
	dfA := df.ParamPtr("a")
	dfB := df.ParamPtr("b")
	df.ReturnsI64()
	e2 := df.Entry()
	e2.Return(e2.Ptr.Diff(dfA, dfB))

	wantOK(t, m, `
long long fromptr(void *), pdiff(char *, char *);
static void body(void) {
    char buf[64];
    /* Zero-extended, per §C4: the high half is not the pointer's sign. */
    chk64("from_ptr high", fromptr((void *)0xfffffff0u), 0x00000000fffffff0ULL);
    /* And diff is sign-extended, which at 32-bit pointers is real work. */
    chk64("diff up", pdiff(buf + 40, buf + 8), 32ULL);
    chk64("diff down", pdiff(buf + 8, buf + 40), 0xffffffffffffffe0ULL);
}
`)
}

// §A's two wide multiplies and §A2's two multiplying predicates, at
// sixty-four bits — where this machine has no instruction for any of them
// and all four are the same 128-bit product underneath.
//
// Against C, which computes the same thing with __int128 and a cast. The
// operands are chosen so the carry chain is exercised: pairs whose middle
// column overflows twice, pairs where only one operand is negative, and the
// signed extremes where the correction is the whole answer.
func TestRunWideMultiply64(t *testing.T) {
	m := ir.NewModule("t", ir.I386Linux)

	bin := func(name string, ret ir.RegType, emit func(b *ir.Block, x, y ir.I64) ir.Value) {
		fn := m.Func(name).Export()
		x := fn.ParamI64("x")
		y := fn.ParamI64("y")
		if ret == ir.TypeI64 {
			fn.ReturnsI64()
		} else {
			fn.ReturnsI32()
		}
		e := fn.Entry()
		e.Return(emit(e, x, y))
	}

	bin("smh", ir.TypeI64, func(b *ir.Block, x, y ir.I64) ir.Value { return b.I64.SMulHi(x, y) })
	bin("umh", ir.TypeI64, func(b *ir.Block, x, y ir.I64) ir.Value { return b.I64.UMulHi(x, y) })
	bin("smo", ir.TypeI32, func(b *ir.Block, x, y ir.I64) ir.Value {
		return b.I32.ZExtI1(b.I64.SMulO(x, y))
	})
	bin("umo", ir.TypeI32, func(b *ir.Block, x, y ir.I64) ir.Value {
		return b.I32.ZExtI1(b.I64.UMulO(x, y))
	})

	wantOK(t, m, `
long long smh(long long, long long), umh(long long, long long);
unsigned smo(long long, long long), umo(long long, long long);

/* The reference product, computed bit by bit rather than in columns —
   there is no __int128 on a 32-bit target, and an oracle that multiplied
   in thirty-two-bit columns would be the code under test written twice.
   Shift and add is a different algorithm, and the signed case is done by
   magnitude and sign rather than by the mask correction the lowering uses. */
struct u128 { unsigned w[4]; };

static void add128(struct u128 *r, const struct u128 *x) {
    unsigned carry = 0;
    for (int i = 0; i < 4; i++) {
        unsigned s = r->w[i] + x->w[i];
        unsigned c1 = s < r->w[i];
        unsigned t = s + carry;
        unsigned c2 = t < s;
        r->w[i] = t;
        carry = c1 | c2;
    }
}
static void shl128(struct u128 *x) {
    for (int i = 3; i > 0; i--) x->w[i] = (x->w[i] << 1) | (x->w[i-1] >> 31);
    x->w[0] <<= 1;
}
static struct u128 neg128(struct u128 x) {
    struct u128 r; unsigned carry = 1;
    for (int i = 0; i < 4; i++) {
        unsigned v = ~x.w[i] + carry;
        carry = carry && v == 0;
        r.w[i] = v;
    }
    return r;
}
static struct u128 umul128(unsigned a0, unsigned a1, unsigned b0, unsigned b1) {
    struct u128 acc, add;
    for (int i = 0; i < 4; i++) acc.w[i] = 0;
    add.w[0] = a0; add.w[1] = a1; add.w[2] = 0; add.w[3] = 0;
    for (int i = 0; i < 64; i++) {
        unsigned bit = i < 32 ? (b0 >> i) & 1 : (b1 >> (i - 32)) & 1;
        if (bit) add128(&acc, &add);
        shl128(&add);
    }
    return acc;
}
static unsigned long long pair(unsigned lo, unsigned hi) {
    return ((unsigned long long)hi << 32) | lo;
}
static struct u128 smul128(unsigned a0, unsigned a1, unsigned b0, unsigned b1) {
    int neg = 0;
    if (a1 >> 31) { neg ^= 1; unsigned c = ~a0 + 1; a1 = ~a1 + (c == 0); a0 = c; }
    if (b1 >> 31) { neg ^= 1; unsigned c = ~b0 + 1; b1 = ~b1 + (c == 0); b0 = c; }
    struct u128 p = umul128(a0, a1, b0, b1);
    return neg ? neg128(p) : p;
}

static void body(void) {
    unsigned long long v[] = {
        0ULL, 1ULL, 0xffffffffffffffffULL, 2ULL, 0xfffffffffffffffeULL, 3ULL,
        0x100000000ULL, 0xffffffff00000000ULL,
        0xffffffffULL, 0x1ffffffffULL,
        0x7fffffffffffffffULL, 0x8000000000000000ULL,
        0x0123456789abcdefULL, 0xfedcba9876543211ULL,
        0x8000000000000001ULL,
    };
    unsigned n = sizeof v / sizeof *v;
    for (unsigned i = 0; i < n; i++)
        for (unsigned j = 0; j < n; j++) {
            unsigned a0 = (unsigned)v[i], a1 = (unsigned)(v[i] >> 32);
            unsigned b0 = (unsigned)v[j], b1 = (unsigned)(v[j] >> 32);
            long long a = (long long)v[i], b = (long long)v[j];

            struct u128 up = umul128(a0, a1, b0, b1);
            struct u128 sp = smul128(a0, a1, b0, b1);

            chk64("smulhi", (unsigned long long)smh(a, b), pair(sp.w[2], sp.w[3]));
            chk64("umulhi", (unsigned long long)umh(a, b), pair(up.w[2], up.w[3]));

            /* A product fits in sixty-four signed bits when its top half is
               the sign extension of its low half, and in sixty-four
               unsigned bits when its top half is nothing at all. */
            unsigned long long slo = pair(sp.w[0], sp.w[1]);
            unsigned long long sext = (slo >> 63) ? 0xffffffffffffffffULL : 0ULL;
            chk32("smulo", smo(a, b), (unsigned)(pair(sp.w[2], sp.w[3]) != sext));
            chk32("umulo", umo(a, b), (unsigned)(pair(up.w[2], up.w[3]) != 0ULL));
        }
}
`)
}
