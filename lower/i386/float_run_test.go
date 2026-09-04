package i386_test

// §A3, §B's float half and §C's conversions, run.
//
// The runtime prints hex, so the checks compare bit patterns — which for
// floats is what you want anyway: it tells −0 from +0 and one NaN from
// another, where a numeric comparison says they are equal or says nothing.

import (
	"testing"

	"github.com/vertex-language/ir"
)

// floatHelpersC is the libm this kernel does not have. Only the four
// roundings and fma, which SSE2 has no instruction for and this backend
// therefore calls out to.
//
// Written by bit manipulation rather than by arithmetic, so that they are
// obviously right rather than right by the same reasoning as the code under
// test.
const floatHelpersC = `
static double round_toward(double x, int up, int nearest) {
    unsigned long long b; __builtin_memcpy(&b, &x, 8);
    int exp = (int)((b >> 52) & 0x7ff) - 1023;
    if (exp >= 52) return x;                 /* already integral */
    if (exp < 0) {
        /* |x| < 1 */
        double zero = (b >> 63) ? -0.0 : 0.0;
        if (nearest) {
            double ax = (b >> 63) ? -x : x;
            if (ax > 0.5) return (b >> 63) ? -1.0 : 1.0;
            if (ax < 0.5) return zero;
            return zero;                      /* ties to even: 0 */
        }
        if (up && !(b >> 63) && x != 0.0) return 1.0;
        if (!up && (b >> 63) && x != 0.0) return -1.0;
        return zero;
    }
    unsigned long long mask = (1ULL << (52 - exp)) - 1;
    if ((b & mask) == 0) return x;            /* already integral */
    unsigned long long t = b & ~mask;
    double tr; __builtin_memcpy(&tr, &t, 8);  /* truncated toward zero */
    if (nearest) {
        double diff = x - tr;
        double half = (b >> 63) ? -0.5 : 0.5;
        double step = (b >> 63) ? -1.0 : 1.0;
        double away = tr + step;
        double d1 = diff < 0 ? -diff : diff;
        if (d1 > 0.5) return away;
        if (d1 < 0.5) return tr;
        /* tie: pick the even one */
        double h = tr * 0.5;
        unsigned long long hb; __builtin_memcpy(&hb, &h, 8);
        double hh = h - (double)(long)0; (void)hh; (void)hb; (void)half;
        /* tr is even iff tr/2 is integral, which at this exponent means
           the low bit of the truncated mantissa is clear */
        return ((t >> (52 - exp)) & 1) ? away : tr;
    }
    if (up)  return (b >> 63) ? tr : tr + 1.0;
    return (b >> 63) ? tr - 1.0 : tr;
}
double ceil(double x)  { return round_toward(x, 1, 0); }
double floor(double x) { return round_toward(x, 0, 0); }
double trunc(double x) {
    unsigned long long b; __builtin_memcpy(&b, &x, 8);
    int exp = (int)((b >> 52) & 0x7ff) - 1023;
    if (exp >= 52) return x;
    if (exp < 0) { double z = (b >> 63) ? -0.0 : 0.0; return z; }
    unsigned long long t = b & ~((1ULL << (52 - exp)) - 1);
    double tr; __builtin_memcpy(&tr, &t, 8); return tr;
}
double rint(double x)  { return round_toward(x, 0, 1); }
float ceilf(float x)  { return (float)ceil((double)x); }
float floorf(float x) { return (float)floor((double)x); }
float truncf(float x) { return (float)trunc((double)x); }
float rintf(float x)  { return (float)rint((double)x); }
double fma(double a, double b, double c) { return a * b + c; }
float fmaf(float a, float b, float c) { return (float)((double)a * (double)b + (double)c); }
`

// softFloatC is the compiler-rt this kernel does not link against. SSE2
// converts between a float and a *32-bit* integer and nothing wider, so the
// backend calls a helper by name for every 64-bit conversion, and a
// freestanding link has to supply one.
//
// Deliberately not the same reasoning as the code under test. The
// int-to-float direction splits the value into two halves, each of which the
// 32-bit conversion clang emits inline converts exactly, and adds them —
// which is one rounding of a sum that is itself exact. The other direction
// takes the bit pattern apart and shifts, so no float arithmetic is involved
// at all.
const softFloatC = `
double __floatdidf(long long x) {
    int hi = (int)(x >> 32);
    unsigned lo = (unsigned)x;
    return (double)hi * 4294967296.0 + (double)lo;
}
double __floatundidf(unsigned long long x) {
    unsigned hi = (unsigned)(x >> 32), lo = (unsigned)x;
    return (double)hi * 4294967296.0 + (double)lo;
}
float __floatdisf(long long x)            { return (float)__floatdidf(x); }
float __floatundisf(unsigned long long x) { return (float)__floatundidf(x); }

/* |x| truncated toward zero, and its sign. A double is mant * 2^(exp-1075)
   with the hidden bit restored, so the whole of it is a shift. */
static unsigned long long fixmag(double x, int *neg) {
    unsigned long long b; __builtin_memcpy(&b, &x, 8);
    *neg = (int)(b >> 63);
    int e = (int)((b >> 52) & 0x7ff);
    if (e == 0) return 0;               /* zero or subnormal: |x| < 1 */
    unsigned long long m = (b & 0xfffffffffffffULL) | (1ULL << 52);
    int s = e - 1075;
    if (s <= -64) return 0;             /* shifted away entirely */
    if (s < 0)  return m >> (-s);
    if (s <= 11) return m << s;
    return ~0ULL;                       /* out of range; undefined anyway */
}
long long __fixdfdi(double x) {
    int neg; unsigned long long m = fixmag(x, &neg);
    return neg ? -(long long)m : (long long)m;
}
unsigned long long __fixunsdfdi(double x) {
    int neg; unsigned long long m = fixmag(x, &neg);
    return neg ? 0ULL : m;
}
long long __fixsfdi(float x)                { return __fixdfdi((double)x); }
unsigned long long __fixunssfdi(float x)    { return __fixunsdfdi((double)x); }
`

func TestRunFloatArithmetic(t *testing.T) {
	m := ir.NewModule("t", ir.I386Linux)

	for _, tc := range []struct {
		name string
		emit func(b *ir.Block, x, y ir.F64) ir.F64
	}{
		{"fadd", func(b *ir.Block, x, y ir.F64) ir.F64 { return b.F64.Add(x, y) }},
		{"fsub", func(b *ir.Block, x, y ir.F64) ir.F64 { return b.F64.Sub(x, y) }},
		{"fmul", func(b *ir.Block, x, y ir.F64) ir.F64 { return b.F64.Mul(x, y) }},
		{"fdiv", func(b *ir.Block, x, y ir.F64) ir.F64 { return b.F64.Div(x, y) }},
		{"fcopy", func(b *ir.Block, x, y ir.F64) ir.F64 { return b.F64.CopySign(x, y) }},
	} {
		fn := m.Func(tc.name).Export()
		x := fn.ParamF64("x")
		y := fn.ParamF64("y")
		fn.ReturnsF64()
		e := fn.Entry()
		e.Return(tc.emit(e, x, y))
	}
	for _, tc := range []struct {
		name string
		emit func(b *ir.Block, x ir.F64) ir.F64
	}{
		{"fneg", func(b *ir.Block, x ir.F64) ir.F64 { return b.F64.Neg(x) }},
		{"fabs", func(b *ir.Block, x ir.F64) ir.F64 { return b.F64.Abs(x) }},
		{"fsqrt", func(b *ir.Block, x ir.F64) ir.F64 { return b.F64.Sqrt(x) }},
	} {
		fn := m.Func(tc.name).Export()
		x := fn.ParamF64("x")
		fn.ReturnsF64()
		e := fn.Entry()
		e.Return(tc.emit(e, x))
	}

	// A constant, and single precision.
	k := m.Func("fk").Export()
	k.ReturnsF64()
	ek := k.Entry()
	ek.Return(ek.F64.Const(2.5))

	s := m.Func("f32mix").Export()
	sa := s.ParamF32("a")
	sb := s.ParamF32("b")
	s.ReturnsF32()
	es := s.Entry()
	es.Return(es.F32.Add(es.F32.Mul(sa, sb), es.F32.Const(1.5)))

	wantOK(t, m, `
double fadd(double,double), fsub(double,double), fmul(double,double), fdiv(double,double);
double fcopy(double,double), fneg(double), fabs(double), fsqrt(double), fk(void);
float f32mix(float,float);
static unsigned long long bits(double v) { unsigned long long b; __builtin_memcpy(&b,&v,8); return b; }
static unsigned bits32(float v) { unsigned b; __builtin_memcpy(&b,&v,4); return b; }
static void body(void) {
    chk64("add", bits(fadd(1.5, 2.25)), bits(3.75));
    chk64("sub", bits(fsub(1.5, 2.25)), bits(-0.75));
    chk64("mul", bits(fmul(1.5, 2.25)), bits(3.375));
    chk64("div", bits(fdiv(9.0, 2.0)), bits(4.5));
    chk64("sqrt", bits(fsqrt(16.0)), bits(4.0));
    /* Sign is a bit operation, so -0 stays -0 and a NaN keeps its payload. */
    chk64("neg", bits(fneg(1.5)), bits(-1.5));
    chk64("neg zero", bits(fneg(0.0)), bits(-0.0));
    chk64("abs", bits(fabs(-1.5)), bits(1.5));
    chk64("abs zero", bits(fabs(-0.0)), bits(0.0));
    chk64("copysign", bits(fcopy(1.5, -2.0)), bits(-1.5));
    chk64("copysign zero", bits(fcopy(-1.5, 0.0)), bits(1.5));
    chk64("const", bits(fk()), bits(2.5));
    chk32("f32", bits32(f32mix(2.0f, 3.0f)), bits32(7.5f));
}
`)
}

// §A3's four min and max verbs, at the operands that tell them apart: a NaN,
// which minimum propagates and minnum discards, and a signed zero, which §A3
// pins for all four.
func TestRunFloatMinMax(t *testing.T) {
	m := ir.NewModule("t", ir.I386Linux)
	for _, tc := range []struct {
		name string
		emit func(b *ir.Block, x, y ir.F64) ir.F64
	}{
		{"vmin", func(b *ir.Block, x, y ir.F64) ir.F64 { return b.F64.Minimum(x, y) }},
		{"vmax", func(b *ir.Block, x, y ir.F64) ir.F64 { return b.F64.Maximum(x, y) }},
		{"vminnum", func(b *ir.Block, x, y ir.F64) ir.F64 { return b.F64.MinNum(x, y) }},
		{"vmaxnum", func(b *ir.Block, x, y ir.F64) ir.F64 { return b.F64.MaxNum(x, y) }},
	} {
		fn := m.Func(tc.name).Export()
		x := fn.ParamF64("x")
		y := fn.ParamF64("y")
		fn.ReturnsF64()
		e := fn.Entry()
		e.Return(tc.emit(e, x, y))
	}

	wantOK(t, m, `
double vmin(double,double), vmax(double,double), vminnum(double,double), vmaxnum(double,double);
static unsigned long long bits(double v) { unsigned long long b; __builtin_memcpy(&b,&v,8); return b; }
static int isnan_(double v) { return v != v; }
static void body(void) {
    double nan = __builtin_nan(""), one = 1.0, two = 2.0;
    chk32("minimum nan",     (unsigned)isnan_(vmin(nan, one)), 1u);
    chk32("minimum nan rhs", (unsigned)isnan_(vmin(one, nan)), 1u);
    chk32("maximum nan",     (unsigned)isnan_(vmax(nan, one)), 1u);
    chk32("maximum nan rhs", (unsigned)isnan_(vmax(one, nan)), 1u);
    chk64("minnum nan",      bits(vminnum(nan, one)), bits(one));
    chk64("minnum nan rhs",  bits(vminnum(one, nan)), bits(one));
    chk64("maxnum nan",      bits(vmaxnum(nan, two)), bits(two));
    chk64("maxnum nan rhs",  bits(vmaxnum(two, nan)), bits(two));

    chk64("minimum", bits(vmin(one, two)), bits(one));
    chk64("maximum", bits(vmax(one, two)), bits(two));
    chk64("minnum",  bits(vminnum(two, one)), bits(one));
    chk64("maxnum",  bits(vmaxnum(one, two)), bits(two));

    /* §A3 pins the signed zeroes for all four. */
    chk64("minimum zero", bits(vmin(0.0, -0.0)), bits(-0.0));
    chk64("minimum zero2", bits(vmin(-0.0, 0.0)), bits(-0.0));
    chk64("minnum zero",  bits(vminnum(0.0, -0.0)), bits(-0.0));
    chk64("maximum zero", bits(vmax(-0.0, 0.0)), bits(0.0));
    chk64("maximum zero2", bits(vmax(0.0, -0.0)), bits(0.0));
    chk64("maxnum zero",  bits(vmaxnum(-0.0, 0.0)), bits(0.0));
}
`)
}

// §B's five float comparisons, at ordinary operands and at a NaN — which is
// the whole reason eq is not one condition here.
func TestRunFloatCompare(t *testing.T) {
	m := ir.NewModule("t", ir.I386Linux)
	for _, tc := range []struct {
		name string
		emit func(b *ir.Block, x, y ir.F64) ir.I1
	}{
		{"ceq", func(b *ir.Block, x, y ir.F64) ir.I1 { return b.F64.Eq(x, y) }},
		{"cne", func(b *ir.Block, x, y ir.F64) ir.I1 { return b.F64.Ne(x, y) }},
		{"clt", func(b *ir.Block, x, y ir.F64) ir.I1 { return b.F64.Lt(x, y) }},
		{"cle", func(b *ir.Block, x, y ir.F64) ir.I1 { return b.F64.Le(x, y) }},
		{"cuno", func(b *ir.Block, x, y ir.F64) ir.I1 { return b.F64.Uno(x, y) }},
	} {
		fn := m.Func(tc.name).Export()
		x := fn.ParamF64("x")
		y := fn.ParamF64("y")
		fn.ReturnsI32()
		e := fn.Entry()
		e.Return(e.I32.ZExtI1(tc.emit(e, x, y)))
	}

	wantOK(t, m, `
int ceq(double,double), cne(double,double), clt(double,double);
int cle(double,double), cuno(double,double);
static void body(void) {
    double nan = __builtin_nan(""), one = 1.0, two = 2.0;
    chk32("eq",     (unsigned)ceq(one, one), 1u);  chk32("eq ne", (unsigned)ceq(one, two), 0u);
    chk32("ne",     (unsigned)cne(one, two), 1u);  chk32("ne eq", (unsigned)cne(one, one), 0u);
    chk32("lt",     (unsigned)clt(one, two), 1u);  chk32("lt gt", (unsigned)clt(two, one), 0u);
    chk32("lt eq",  (unsigned)clt(one, one), 0u);
    chk32("le",     (unsigned)cle(one, two), 1u);  chk32("le eq", (unsigned)cle(one, one), 1u);
    chk32("le gt",  (unsigned)cle(two, one), 0u);
    chk32("uno no", (unsigned)cuno(one, two), 0u);
    /* The NaN column: every ordered verb false, ne true, uno true. */
    chk32("eq nan",  (unsigned)ceq(nan, one), 0u);
    chk32("ne nan",  (unsigned)cne(nan, one), 1u);
    chk32("lt nan",  (unsigned)clt(nan, one), 0u);
    chk32("lt nan2", (unsigned)clt(one, nan), 0u);
    chk32("le nan",  (unsigned)cle(nan, one), 0u);
    chk32("le nan2", (unsigned)cle(one, nan), 0u);
    chk32("uno nan", (unsigned)cuno(nan, one), 1u);
    chk32("uno self",(unsigned)cuno(nan, nan), 1u);
}
`)
}

// The §A3 verbs SSE2 has no instruction for, which are calls.
func TestRunFloatRounding(t *testing.T) {
	m := ir.NewModule("t", ir.I386Linux)
	for _, tc := range []struct {
		name string
		emit func(b *ir.Block, x ir.F64) ir.F64
	}{
		{"rceil", func(b *ir.Block, x ir.F64) ir.F64 { return b.F64.Ceil(x) }},
		{"rfloor", func(b *ir.Block, x ir.F64) ir.F64 { return b.F64.Floor(x) }},
		{"rtrunc", func(b *ir.Block, x ir.F64) ir.F64 { return b.F64.Trunc(x) }},
		{"rnear", func(b *ir.Block, x ir.F64) ir.F64 { return b.F64.Nearest(x) }},
	} {
		fn := m.Func(tc.name).Export()
		x := fn.ParamF64("x")
		fn.ReturnsF64()
		e := fn.Entry()
		e.Return(tc.emit(e, x))
	}
	fma := m.Func("rfma").Export()
	fa := fma.ParamF64("a")
	fb := fma.ParamF64("b")
	fc := fma.ParamF64("c")
	fma.ReturnsF64()
	ef := fma.Entry()
	ef.Return(ef.F64.FMA(fa, fb, fc))

	wantOK(t, m, floatHelpersC+`
double rceil(double), rfloor(double), rtrunc(double), rnear(double);
double rfma(double,double,double);
static unsigned long long bits(double v) { unsigned long long b; __builtin_memcpy(&b,&v,8); return b; }
static void body(void) {
    chk64("ceil",   bits(rceil(1.25)), bits(2.0));
    chk64("ceil neg", bits(rceil(-1.25)), bits(-1.0));
    chk64("floor",  bits(rfloor(1.75)), bits(1.0));
    chk64("floor neg", bits(rfloor(-1.25)), bits(-2.0));
    chk64("trunc",  bits(rtrunc(1.75)), bits(1.0));
    chk64("trunc neg", bits(rtrunc(-1.75)), bits(-1.0));
    chk64("near",   bits(rnear(1.5)), bits(2.0));
    chk64("near even", bits(rnear(2.5)), bits(2.0));
    chk64("fma",    bits(rfma(2.0, 3.0, 1.0)), bits(7.0));
}
`)
}

// §C2 and §C3: the conversions, including the ones SSE2 cannot do.
func TestRunFloatConversions(t *testing.T) {
	m := ir.NewModule("t", ir.I386Linux)

	s32 := m.Func("s32tod").Export()
	s32a := s32.ParamI32("a")
	s32.ReturnsF64()
	e1 := s32.Entry()
	e1.Return(e1.F64.SCvtI32(s32a))

	u32 := m.Func("u32tod").Export()
	u32a := u32.ParamI32("a")
	u32.ReturnsF64()
	e2 := u32.Entry()
	e2.Return(e2.F64.UCvtI32(u32a))

	s64 := m.Func("s64tod").Export()
	s64a := s64.ParamI64("a")
	s64.ReturnsF64()
	e3 := s64.Entry()
	e3.Return(e3.F64.SCvtI64(s64a))

	dtoi := m.Func("dtoi").Export()
	dtoia := dtoi.ParamF64("a")
	dtoi.ReturnsI32()
	e4 := dtoi.Entry()
	e4.Return(e4.I32.SCvtF64(dtoia))

	dtol := m.Func("dtol").Export()
	dtola := dtol.ParamF64("a")
	dtol.ReturnsI64()
	e5 := dtol.Entry()
	e5.Return(e5.I64.SCvtF64(dtola))

	narrow := m.Func("narrow").Export()
	na := narrow.ParamF64("a")
	narrow.ReturnsF32()
	e6 := narrow.Entry()
	e6.Return(e6.F32.FCvtF64(na))

	widen := m.Func("widen").Export()
	wa := widen.ParamF32("a")
	widen.ReturnsF64()
	e7 := widen.Entry()
	e7.Return(e7.F64.FCvtF32(wa))

	bc := m.Func("dbits").Export()
	bca := bc.ParamF64("a")
	bc.ReturnsI64()
	e8 := bc.Entry()
	e8.Return(e8.I64.BitcastF64(bca))

	un := m.Func("unbits").Export()
	una := un.ParamI64("a")
	un.ReturnsF64()
	e9 := un.Entry()
	e9.Return(e9.F64.BitcastI64(una))

	fb := m.Func("fbits").Export()
	fba := fb.ParamF32("a")
	fb.ReturnsI32()
	e10 := fb.Entry()
	e10.Return(e10.I32.BitcastF32(fba))

	wantOK(t, m, softFloatC+`
double s32tod(int), u32tod(unsigned), s64tod(long long), unbits(long long), widen(float);
int dtoi(double), fbits(float);
long long dtol(double), dbits(double);
float narrow(double);
static unsigned long long bits(double v) { unsigned long long b; __builtin_memcpy(&b,&v,8); return b; }
static unsigned bits32(float v) { unsigned b; __builtin_memcpy(&b,&v,4); return b; }
static void body(void) {
    chk64("s32", bits(s32tod(-3)), bits(-3.0));
    chk64("s32 big", bits(s32tod(-2147483647-1)), bits(-2147483648.0));
    /* The unsigned correction: read as signed this is negative. */
    chk64("u32", bits(u32tod(3000000000u)), bits(3000000000.0));
    chk64("u32 small", bits(u32tod(7u)), bits(7.0));
    chk64("s64", bits(s64tod(1234567890123LL)), bits(1234567890123.0));
    chk64("s64 neg", bits(s64tod(-1234567890123LL)), bits(-1234567890123.0));

    chk32("dtoi", (unsigned)dtoi(-42.9), (unsigned)-42);
    chk32("dtoi max", (unsigned)dtoi(2147483647.0), 2147483647u);
    chk64("dtol", (unsigned long long)dtol(1234567890123.5), 1234567890123ULL);
    chk64("dtol neg", (unsigned long long)dtol(-1234567890123.5),
                      (unsigned long long)-1234567890123LL);

    chk32("narrow", bits32(narrow(0.1)), bits32((float)0.1));
    chk64("widen",  bits(widen(0.5f)), bits(0.5));
    chk64("bitcast", (unsigned long long)dbits(-12.375), bits(-12.375));
    chk64("unbitcast", bits(unbits((long long)bits(-12.375))), bits(-12.375));
    chk32("bitcast32", (unsigned)fbits(-12.375f), bits32(-12.375f));
}
`)
}

// A float across a call boundary in both directions, which is the psABI's
// ST(0) return — the one place x87 survives here.
func TestRunFloatCalls(t *testing.T) {
	m := ir.NewModule("t", ir.I386Linux)
	half := m.ImportFunc("halve", ir.NewSig().Param(ir.TypeF64).Ret(ir.TypeF64))

	fn := m.Func("quarter").Export()
	a := fn.ParamF64("a")
	fn.ReturnsF64()
	entry := fn.Entry()
	once := entry.Call(half, a).Value(0).(ir.F64)
	twice := entry.Call(half, once).Value(0).(ir.F64)
	// once live across the second call, which forces it somewhere the
	// call does not destroy — and no XMM register is such a place, so it
	// has to reach the frame.
	entry.Return(entry.F64.Add(twice, once))

	mixed := m.Func("mixed").Export()
	mi := mixed.ParamI32("i")
	md := mixed.ParamF64("d")
	mf := mixed.ParamF32("f")
	mj := mixed.ParamI64("j")
	mixed.ReturnsF64()
	em := mixed.Entry()
	em.Return(em.F64.Add(
		em.F64.Add(md, em.F64.FCvtF32(mf)),
		em.F64.Add(em.F64.SCvtI32(mi), em.F64.SCvtI64(mj))))

	wantOK(t, m, softFloatC+`
double halve(double x) { return x / 2.0; }
double quarter(double), mixed(int, double, float, long long);
static unsigned long long bits(double v) { unsigned long long b; __builtin_memcpy(&b,&v,8); return b; }
static void body(void) {
    chk64("call", bits(quarter(8.0)), bits(6.0));
    /* Interleaved widths, which is what the stack layout has to get right. */
    chk64("mixed", bits(mixed(3, 1.5, 0.5f, 100LL)), bits(105.0));
}
`)
}

// A float global, read back through §D. What §5 writes is a bit pattern and
// what a load expects is the same one at the same alignment, which is the
// only thing that can go wrong between them.
func TestRunFloatGlobals(t *testing.T) {
	m := ir.NewModule("t", ir.I386Linux)
	pi := m.Global("pi", ir.RW, ir.StoreF64.FType()).Export().
		Init(ir.Lit(ir.Float(3.5)))
	half := m.Global("half", ir.RW, ir.StoreF32.FType()).Export().
		Init(ir.Lit(ir.Float(0.5)))

	d := m.Func("readpi").Export()
	d.ReturnsF64()
	ed := d.Entry()
	ed.Return(ed.F64.Load(ed.Ptr.GetAddr(pi)))

	f := m.Func("readhalf").Export()
	f.ReturnsF32()
	ef := f.Entry()
	ef.Return(ef.F32.Load(ef.Ptr.GetAddr(half)))

	wantOK(t, m, `
double readpi(void);
float readhalf(void);
static unsigned long long bits(double v) { unsigned long long b; __builtin_memcpy(&b,&v,8); return b; }
static unsigned bits32(float v) { unsigned b; __builtin_memcpy(&b,&v,4); return b; }
static void body(void) {
    chk64("pi", bits(readpi()), bits(3.5));
    chk32("half", bits32(readhalf()), bits32(0.5f));
}
`)
}

// §C2's other seven float-to-integer rows: the unsigned destinations, whose
// range CVTTSD2SI's signed result is half of, the 64-bit ones, which are a
// call, and the saturating forms of all four.
//
// The trapping forms' traps are not run. A trap here is INT3 against a kernel
// with no IDT, which is a triple fault and a machine that prints nothing —
// so what is checked is that everything inside the interval converts, and the
// saturating rows stand in for the boundary itself.
func TestRunFloatToIntRanges(t *testing.T) {
	m := ir.NewModule("t", ir.I386Linux)

	// One argument, one result, and the verb between them. Spelled out
	// per row rather than through a table, because the return type is
	// part of what each row is testing.
	fromF64 := func(name string, wide bool, emit func(b *ir.Block, a ir.F64) ir.Value) {
		fn := m.Func(name).Export()
		a := fn.ParamF64("a")
		if wide {
			fn.ReturnsI64()
		} else {
			fn.ReturnsI32()
		}
		e := fn.Entry()
		e.Return(emit(e, a))
	}
	fromF32 := func(name string, emit func(b *ir.Block, a ir.F32) ir.Value) {
		fn := m.Func(name).Export()
		a := fn.ParamF32("a")
		fn.ReturnsI32()
		e := fn.Entry()
		e.Return(emit(e, a))
	}

	fromF64("dtou", false, func(b *ir.Block, a ir.F64) ir.Value { return b.I32.UCvtF64(a) })
	fromF64("dtoul", true, func(b *ir.Block, a ir.F64) ir.Value { return b.I64.UCvtF64(a) })
	fromF64("dtoisat", false, func(b *ir.Block, a ir.F64) ir.Value { return b.I32.SCvtSatF64(a) })
	fromF64("dtousat", false, func(b *ir.Block, a ir.F64) ir.Value { return b.I32.UCvtSatF64(a) })
	fromF64("dtolsat", true, func(b *ir.Block, a ir.F64) ir.Value { return b.I64.SCvtSatF64(a) })
	fromF64("dtoulsat", true, func(b *ir.Block, a ir.F64) ir.Value { return b.I64.UCvtSatF64(a) })
	fromF32("ftoisat", func(b *ir.Block, a ir.F32) ir.Value { return b.I32.SCvtSatF32(a) })
	fromF32("ftou", func(b *ir.Block, a ir.F32) ir.Value { return b.I32.UCvtF32(a) })

	wantOK(t, m, softFloatC+`
unsigned dtou(double), dtousat(double), ftou(float);
int dtoisat(double), ftoisat(float);
long long dtolsat(double);
unsigned long long dtoul(double), dtoulsat(double);
static void body(void) {
    double nan = __builtin_nan("");
    /* The unsigned 32-bit range, whose top half needs the bias. */
    chk32("u32 low",  dtou(7.9), 7u);
    chk32("u32 edge", dtou(2147483648.0), 2147483648u);
    chk32("u32 high", dtou(4294967295.0), 4294967295u);
    /* Truncation toward zero, so the interval opens at -1 and not at 0. */
    chk32("u32 neg",  dtou(-0.5), 0u);
    chk32("f32 u32",  ftou(3000000000.0f), 3000000000u);

    /* The 64-bit unsigned destination, which is a call. */
    chk64("u64", dtoul(1e19), 10000000000000000000ULL);
    chk64("u64 small", dtoul(7.9), 7ULL);

    /* Saturating: the bound is the destination's extreme, not the
       truncation of the interval's edge. */
    chk32("sat i32 hi", (unsigned)dtoisat(1e18), 2147483647u);
    chk32("sat i32 lo", (unsigned)dtoisat(-1e18), 2147483648u);
    chk32("sat i32 in", (unsigned)dtoisat(-42.9), (unsigned)-42);
    chk32("sat i32 nan", (unsigned)dtoisat(nan), 0u);
    /* At f32 the largest float below 2^31 is 2147483520, so a clamp in
       the float domain would answer that instead. */
    chk32("sat i32 f32", (unsigned)ftoisat(1e18f), 2147483647u);

    chk32("sat u32 hi", dtousat(1e18), 4294967295u);
    chk32("sat u32 lo", dtousat(-5.0), 0u);
    chk32("sat u32 in", dtousat(3000000000.5), 3000000000u);
    chk32("sat u32 nan", dtousat(nan), 0u);

    chk64("sat i64 hi", (unsigned long long)dtolsat(1e30), 0x7fffffffffffffffULL);
    chk64("sat i64 lo", (unsigned long long)dtolsat(-1e30), 0x8000000000000000ULL);
    chk64("sat i64 in", (unsigned long long)dtolsat(-1234567890123.5),
                        (unsigned long long)-1234567890123LL);
    chk64("sat i64 nan", (unsigned long long)dtolsat(nan), 0ULL);

    chk64("sat u64 hi", dtoulsat(1e30), 0xffffffffffffffffULL);
    chk64("sat u64 lo", dtoulsat(-1.0), 0ULL);
    chk64("sat u64 in", dtoulsat(1e19), 10000000000000000000ULL);
    chk64("sat u64 nan", dtoulsat(nan), 0ULL);
}
`)
}
