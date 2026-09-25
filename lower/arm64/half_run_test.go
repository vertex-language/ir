package arm64_test

// f16 and bf16 through LegalizeHalf and out the other side, checked bit for
// bit against a reference computed here. Every half pattern goes through
// the unary verbs and the conversions out; the binary verbs, fma and the
// conversions in run on inputs from a generator the C side shares, which
// aims at the rounding ties -- exactly half an ulp, and one bit either
// side of it -- that a conversion rounding twice gets wrong.

import (
	"bufio"
	"fmt"
	"math"
	"math/big"
	"strconv"
	"strings"
	"testing"

	"github.com/vertex-language/ir"
)

// halfNS is what F16NS and BF16NS have in common.
type halfNS[V ir.Value] interface {
	Add(a, c V) V
	Sub(a, c V) V
	Mul(a, c V) V
	Div(a, c V) V
	Minimum(a, c V) V
	Maximum(a, c V) V
	MinNum(a, c V) V
	MaxNum(a, c V) V
	CopySign(a, c V) V
	Neg(a V) V
	Abs(a V) V
	Sqrt(a V) V
	Ceil(a V) V
	Floor(a V) V
	Trunc(a V) V
	Nearest(a V) V
	FMA(a, c, d V) V
	Eq(a, c V) ir.I1
	Ne(a, c V) ir.I1
	Lt(a, c V) ir.I1
	Le(a, c V) ir.I1
	Uno(a, c V) ir.I1
	Const(v float64) V
	FCvtF32(a ir.F32) V
	FCvtF64(a ir.F64) V
	SCvtI32(a ir.I32) V
	UCvtI32(a ir.I32) V
	SCvtI64(a ir.I64) V
	Load(p ir.Ptr, a ...ir.MemAttr) V
	Store(v V, dst ir.Ptr, a ...ir.MemAttr)
}

// halfOut is how one namespace's values leave it, which the core
// namespaces spell per source type.
type halfOut[V ir.Value] struct {
	name     string
	t, other ir.RegType
	ns       func(b *ir.Block) halfNS[V]
	toF32    func(b *ir.Block, v V) ir.F32
	toF64    func(b *ir.Block, v V) ir.F64
	toI32Sat func(b *ir.Block, v V) ir.I32
	toU32Sat func(b *ir.Block, v V) ir.I32
	toI64Sat func(b *ir.Block, v V) ir.I64
	cross    func(b *ir.Block, v V) ir.Value
}

func buildHalf[V ir.Value](m *ir.Module, h halfOut[V]) {
	fn := func(name string, ret ir.RegType, params ...ir.RegType) (*ir.Block, []ir.Value) {
		f := m.Func("_" + h.name + "_" + name).Export()
		var vs []ir.Value
		for i, t := range params {
			vs = append(vs, f.ParamOf(t, "p"+strconv.Itoa(i)))
		}
		retOf(f, ret)
		return f.Entry(), vs
	}
	v := func(x ir.Value) V { return x.(V) }

	bins := map[string]func(n halfNS[V], a, c V) V{
		"add": halfNS[V].Add, "sub": halfNS[V].Sub, "mul": halfNS[V].Mul, "div": halfNS[V].Div,
		"minimum": halfNS[V].Minimum, "maximum": halfNS[V].Maximum,
		"minnum": halfNS[V].MinNum, "maxnum": halfNS[V].MaxNum, "copysign": halfNS[V].CopySign,
	}
	for name, op := range bins {
		// The third parameter is unused: it keeps every binary function's
		// C prototype the shape fma's is.
		e, p := fn(name, h.t, h.t, h.t, h.t)
		e.Return(op(h.ns(e), v(p[0]), v(p[1])))
	}
	uns := map[string]func(n halfNS[V], a V) V{
		"neg": halfNS[V].Neg, "abs": halfNS[V].Abs, "sqrt": halfNS[V].Sqrt,
		"ceil": halfNS[V].Ceil, "floor": halfNS[V].Floor, "trunc": halfNS[V].Trunc,
		"nearest": halfNS[V].Nearest,
	}
	for name, op := range uns {
		e, p := fn(name, h.t, h.t)
		e.Return(op(h.ns(e), v(p[0])))
	}
	cmps := map[string]func(n halfNS[V], a, c V) ir.I1{
		"eq": halfNS[V].Eq, "ne": halfNS[V].Ne, "lt": halfNS[V].Lt, "le": halfNS[V].Le, "uno": halfNS[V].Uno,
	}
	for name, op := range cmps {
		e, p := fn(name, ir.TypeI32, h.t, h.t)
		e.Return(e.I32.ZExtI1(op(h.ns(e), v(p[0]), v(p[1]))))
	}
	{
		e, p := fn("fma", h.t, h.t, h.t, h.t)
		e.Return(h.ns(e).FMA(v(p[0]), v(p[1]), v(p[2])))
	}
	outs := []struct {
		name string
		rt   ir.RegType
		op   func(b *ir.Block, x V) ir.Value
	}{
		{"tof32", ir.TypeF32, func(b *ir.Block, x V) ir.Value { return h.toF32(b, x) }},
		{"tof64", ir.TypeF64, func(b *ir.Block, x V) ir.Value { return h.toF64(b, x) }},
		{"toi32", ir.TypeI32, func(b *ir.Block, x V) ir.Value { return h.toI32Sat(b, x) }},
		{"tou32", ir.TypeI32, func(b *ir.Block, x V) ir.Value { return h.toU32Sat(b, x) }},
		{"toi64", ir.TypeI64, func(b *ir.Block, x V) ir.Value { return h.toI64Sat(b, x) }},
		{"cross", h.other, h.cross},
	}
	for _, o := range outs {
		e, p := fn(o.name, o.rt, h.t)
		e.Return(o.op(e, v(p[0])))
	}
	ins := []struct {
		name string
		pt   ir.RegType
		op   func(n halfNS[V], x ir.Value) V
	}{
		{"fromf32", ir.TypeF32, func(n halfNS[V], x ir.Value) V { return n.FCvtF32(x.(ir.F32)) }},
		{"fromf64", ir.TypeF64, func(n halfNS[V], x ir.Value) V { return n.FCvtF64(x.(ir.F64)) }},
		{"fromi32", ir.TypeI32, func(n halfNS[V], x ir.Value) V { return n.SCvtI32(x.(ir.I32)) }},
		{"fromu32", ir.TypeI32, func(n halfNS[V], x ir.Value) V { return n.UCvtI32(x.(ir.I32)) }},
		{"fromi64", ir.TypeI64, func(n halfNS[V], x ir.Value) V { return n.SCvtI64(x.(ir.I64)) }},
	}
	for _, in := range ins {
		e, p := fn(in.name, h.t, in.pt)
		e.Return(in.op(h.ns(e), p[0]))
	}
	{
		// Through memory: a load, a store, and a constant stored beside it.
		e, p := fn("mem", h.t, ir.TypePtr, ir.TypePtr)
		n := h.ns(e)
		x := n.Load(p[0].(ir.Ptr))
		n.Store(n.Add(x, n.Const(1.1)), p[1].(ir.Ptr))
		e.Return(x)
	}
}

func retOf(f *ir.Func, t ir.RegType) {
	switch t {
	case ir.TypeF16:
		f.ReturnsF16()
	case ir.TypeBF16:
		f.ReturnsBF16()
	case ir.TypeF32:
		f.ReturnsF32()
	case ir.TypeF64:
		f.ReturnsF64()
	case ir.TypeI32:
		f.ReturnsI32()
	case ir.TypeI64:
		f.ReturnsI64()
	}
}

const halfPairs = 40000

// halfMain is the C side. nx is the generator the Go side replays; the
// f32 and f64 inputs are built from it so that about half land on a
// rounding tie of the half format or one bit away from one.
const halfMain = `
#include <stdio.h>
#include <string.h>
static unsigned long long s = 88172645463325252ULL;
static unsigned long long nx(void) { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s; }
static unsigned fbits(float f) { unsigned u; memcpy(&u, &f, 4); return u; }
static unsigned long long dbits(double d) { unsigned long long u; memcpy(&u, &d, 8); return u; }
static float bitsf(unsigned u) { float f; memcpy(&f, &u, 4); return f; }
static double bitsd(unsigned long long u) { double d; memcpy(&d, &u, 8); return d; }

static unsigned long long wideIn(int tie, int mbits, int ebias, int f16) {
	unsigned long long r = nx(), r2 = nx();
	unsigned long long e = f16 ? 100 + (r >> 16) % 46 : 1 + (r >> 16) % 254;
	if (((r >> 8) & 0xff) == 0) e = 255;
	unsigned long long mask = (1ULL << tie) - 1, t = 1ULL << (tie - 1), m = r2 & ((1ULL << mbits) - 1);
	switch ((r >> 60) & 7) {
	case 0: m = (m & ~mask) | t; break;
	case 1: m = (m & ~mask) | t | 1; break;
	case 2: m = (m & ~mask) | (t - 1); break;
	case 3: m = (m & ~mask); break;
	}
	unsigned long long ex = e == 255 ? (ebias == 127 ? 255 : 2047) : e - 127 + ebias;
	return ((r >> 59) & 1) << (mbits + (ebias == 127 ? 8 : 11)) | ex << mbits | m;
}

#define H(p) \
int p##_add(int,int,int), p##_sub(int,int,int), p##_mul(int,int,int), p##_div(int,int,int); \
int p##_minimum(int,int,int), p##_maximum(int,int,int), p##_minnum(int,int,int), p##_maxnum(int,int,int); \
int p##_copysign(int,int,int), p##_fma(int,int,int); \
int p##_neg(int), p##_abs(int), p##_sqrt(int), p##_ceil(int), p##_floor(int), p##_trunc(int), p##_nearest(int); \
int p##_eq(int,int), p##_ne(int,int), p##_lt(int,int), p##_le(int,int), p##_uno(int,int); \
float p##_tof32(int); double p##_tof64(int); int p##_toi32(int), p##_tou32(int); long p##_toi64(int); int p##_cross(int); \
int p##_fromf32(float), p##_fromf64(double), p##_fromi32(int), p##_fromu32(int), p##_fromi64(long); \
int p##_mem(void*, void*); \
static void run_##p(int f16, int tie32, int tie64) { \
	for (int i = 0; i < 65536; i++) \
		printf("%x %x %x %x %x %x %x %x %llx %x %x %lx %x\n", p##_neg(i), p##_abs(i), p##_sqrt(i), \
			p##_ceil(i), p##_floor(i), p##_trunc(i), p##_nearest(i), fbits(p##_tof32(i)), \
			dbits(p##_tof64(i)), p##_toi32(i), p##_tou32(i), p##_toi64(i), p##_cross(i)); \
	for (int i = 0; i < PAIRS; i++) { \
		unsigned long long r = nx(); int a = r & 0xffff, b = (r >> 16) & 0xffff, c = (r >> 32) & 0xffff; \
		printf("%x %x %x %x %x %x %x %x %x %x %x %x %x %x %x\n", p##_add(a,b,0), p##_sub(a,b,0), \
			p##_mul(a,b,0), p##_div(a,b,0), p##_minimum(a,b,0), p##_maximum(a,b,0), p##_minnum(a,b,0), \
			p##_maxnum(a,b,0), p##_copysign(a,b,0), p##_fma(a,b,c), p##_eq(a,b), p##_ne(a,b), \
			p##_lt(a,b), p##_le(a,b), p##_uno(a,b)); \
	} \
	for (int i = 0; i < PAIRS; i++) { \
		unsigned x32 = wideIn(tie32, 23, 127, f16); unsigned long long x64 = wideIn(tie64, 52, 1023, f16); \
		unsigned long long r = nx(); int i32 = (int)r >> ((r >> 40) % 32); long i64 = (long)r >> ((r >> 50) % 64); \
		printf("%x %x %x %x %x\n", p##_fromf32(bitsf(x32)), p##_fromf64(bitsd(x64)), \
			p##_fromi32(i32), p##_fromu32(i32), p##_fromi64(i64)); \
	} \
	unsigned short in = f16 ? 0x3c00 : 0x3f80, out = 0; \
	printf("%x %x\n", p##_mem(&in, &out), out); \
}
H(f16) H(bf16)

int main(void) {
	run_f16(1, 13, 42);
	run_bf16(0, 16, 45);
	return 0;
}
`

func TestRunHalf(t *testing.T) {
	m := ir.NewModule("t", ir.AArch64MacOS)
	buildHalf(m, halfOut[ir.F16]{
		name: "f16", t: ir.TypeF16, other: ir.TypeBF16,
		ns:       func(b *ir.Block) halfNS[ir.F16] { return b.F16() },
		toF32:    func(b *ir.Block, v ir.F16) ir.F32 { return b.F32.FCvtF16(v) },
		toF64:    func(b *ir.Block, v ir.F16) ir.F64 { return b.F64.FCvtF16(v) },
		toI32Sat: func(b *ir.Block, v ir.F16) ir.I32 { return b.I32.SCvtSatF16(v) },
		toU32Sat: func(b *ir.Block, v ir.F16) ir.I32 { return b.I32.UCvtSatF16(v) },
		toI64Sat: func(b *ir.Block, v ir.F16) ir.I64 { return b.I64.SCvtSatF16(v) },
		cross:    func(b *ir.Block, v ir.F16) ir.Value { return b.BF16().FCvtF16(v) },
	})
	buildHalf(m, halfOut[ir.BF16]{
		name: "bf16", t: ir.TypeBF16, other: ir.TypeF16,
		ns:       func(b *ir.Block) halfNS[ir.BF16] { return b.BF16() },
		toF32:    func(b *ir.Block, v ir.BF16) ir.F32 { return b.F32.FCvtBF16(v) },
		toF64:    func(b *ir.Block, v ir.BF16) ir.F64 { return b.F64.FCvtBF16(v) },
		toI32Sat: func(b *ir.Block, v ir.BF16) ir.I32 { return b.I32.SCvtSatBF16(v) },
		toU32Sat: func(b *ir.Block, v ir.BF16) ir.I32 { return b.I32.UCvtSatBF16(v) },
		toI64Sat: func(b *ir.Block, v ir.BF16) ir.I64 { return b.I64.SCvtSatBF16(v) },
		cross:    func(b *ir.Block, v ir.BF16) ir.Value { return b.F16().FCvtBF16(v) },
	})
	if err := m.Err(); err != nil {
		t.Fatalf("build: %v", err)
	}
	if !m.LegalizeHalf() {
		t.Fatal("legalizer reported no change")
	}
	if err := m.Err(); err != nil {
		t.Fatalf("legalize: %v", err)
	}

	out := runNative(t, m, strings.Replace(halfMain, "PAIRS", strconv.Itoa(halfPairs), -1))
	sc := bufio.NewScanner(strings.NewReader(out))
	sc.Buffer(nil, 1<<20)
	c := &halfCheck{t: t, sc: sc, rng: 88172645463325252}
	c.run(ir.TypeF16, 13, 42)
	c.run(ir.TypeBF16, 16, 45)
	if c.fails > 0 {
		t.Fatalf("%d mismatches", c.fails)
	}
}

type halfCheck struct {
	t     *testing.T
	sc    *bufio.Scanner
	rng   uint64
	fails int
	line  []uint64
}

func (c *halfCheck) nx() uint64 {
	c.rng ^= c.rng << 13
	c.rng ^= c.rng >> 7
	c.rng ^= c.rng << 17
	return c.rng
}

// wideIn is the C generator's.
func (c *halfCheck) wideIn(tie, mbits, ebias uint, f16 bool) uint64 {
	r, r2 := c.nx(), c.nx()
	e := 1 + (r>>16)%254
	if f16 {
		e = 100 + (r>>16)%46
	}
	if (r>>8)&0xff == 0 {
		e = 255
	}
	mask, t, m := uint64(1)<<tie-1, uint64(1)<<(tie-1), r2&(uint64(1)<<mbits-1)
	switch (r >> 60) & 7 {
	case 0:
		m = m&^mask | t
	case 1:
		m = m&^mask | t | 1
	case 2:
		m = m&^mask | (t - 1)
	case 3:
		m = m &^ mask
	}
	ex := e - 127 + uint64(ebias)
	ew := uint(11)
	if e == 255 {
		ex = 2047
	}
	if ebias == 127 {
		ew = 8
		if e == 255 {
			ex = 255
		}
	}
	return (r>>59)&1<<(mbits+ew) | ex<<mbits | m
}

func (c *halfCheck) next(n int) bool {
	if !c.sc.Scan() {
		c.t.Fatalf("output ended early")
	}
	f := strings.Fields(c.sc.Text())
	if len(f) != n {
		c.t.Fatalf("line %q: want %d fields", c.sc.Text(), n)
	}
	c.line = c.line[:0]
	for _, s := range f {
		u, err := strconv.ParseUint(s, 16, 64)
		if err != nil {
			c.t.Fatal(err)
		}
		c.line = append(c.line, u)
	}
	return true
}

func (c *halfCheck) fail(format string, a ...any) {
	c.fails++
	if c.fails <= 40 {
		c.t.Errorf(format, a...)
	}
}

// half compares a half result: any NaN matches a NaN.
func (c *halfCheck) half(ht ir.RegType, what string, in any, got uint64, want uint16) {
	if isNaNHalf(ht, want) && got <= 0xffff && isNaNHalf(ht, uint16(got)) {
		return
	}
	if got != uint64(want) {
		c.fail("%s %s(%v): got %#x want %#x", ht, what, in, got, want)
	}
}

func (c *halfCheck) eq(ht ir.RegType, what string, in any, got, want uint64) {
	if got != want {
		c.fail("%s %s(%v): got %#x want %#x", ht, what, in, got, want)
	}
}

func isNaNHalf(t ir.RegType, h uint16) bool {
	if t == ir.TypeBF16 {
		return h&0x7f80 == 0x7f80 && h&0x7f != 0
	}
	return h&0x7c00 == 0x7c00 && h&0x3ff != 0
}

// isSNaNHalf reports a signaling NaN: the fraction's top bit clear.
func isSNaNHalf(t ir.RegType, h uint16) bool {
	q := uint16(0x200)
	if t == ir.TypeBF16 {
		q = 0x40
	}
	return isNaNHalf(t, h) && h&q == 0
}

func (c *halfCheck) run(ht ir.RegType, tie32, tie64 uint) {
	other := ir.TypeBF16
	if ht == ir.TypeBF16 {
		other = ir.TypeF16
	}
	hb := func(v float64) uint16 { return ir.HalfBits(ht, v) }
	for i := 0; i < 65536; i++ {
		c.next(13)
		h, g := uint16(i), c.line
		v := ir.HalfValue(ht, h)
		c.half(ht, "neg", h, g[0], h^0x8000)
		c.half(ht, "abs", h, g[1], h&0x7fff)
		c.half(ht, "sqrt", h, g[2], hb(math.Sqrt(v)))
		c.half(ht, "ceil", h, g[3], hb(math.Ceil(v)))
		c.half(ht, "floor", h, g[4], hb(math.Floor(v)))
		c.half(ht, "trunc", h, g[5], hb(math.Trunc(v)))
		c.half(ht, "nearest", h, g[6], hb(math.RoundToEven(v)))
		if math.IsNaN(v) {
			if f := math.Float32frombits(uint32(g[7])); !math.IsNaN(float64(f)) {
				c.fail("%s tof32(%#x): got %#x, not a NaN", ht, h, g[7])
			}
			if d := math.Float64frombits(g[8]); !math.IsNaN(d) {
				c.fail("%s tof64(%#x): got %#x, not a NaN", ht, h, g[8])
			}
		} else {
			c.eq(ht, "tof32", h, g[7], uint64(math.Float32bits(float32(v))))
			c.eq(ht, "tof64", h, g[8], math.Float64bits(v))
		}
		c.eq(ht, "toi32", h, g[9], uint64(uint32(satInt(v, math.MinInt32, math.MaxInt32))))
		c.eq(ht, "tou32", h, g[10], uint64(uint32(satInt(v, 0, math.MaxUint32))))
		c.eq(ht, "toi64", h, g[11], uint64(satInt(v, math.MinInt64, math.MaxInt64)))
		c.half(other, "cross", h, g[12], ir.HalfBits(other, v))
	}
	for i := 0; i < halfPairs; i++ {
		r := c.nx()
		a, b, cc := uint16(r), uint16(r>>16), uint16(r>>32)
		c.next(15)
		g := c.line
		x, y, z := ir.HalfValue(ht, a), ir.HalfValue(ht, b), ir.HalfValue(ht, cc)
		in := [2]uint16{a, b}
		c.half(ht, "add", in, g[0], hb(x+y))
		c.half(ht, "sub", in, g[1], hb(x-y))
		c.half(ht, "mul", in, g[2], hb(x*y))
		c.half(ht, "div", in, g[3], hb(x/y))
		c.half(ht, "minimum", in, g[4], minimum(ht, a, b, x, y, true))
		c.half(ht, "maximum", in, g[5], minimum(ht, a, b, x, y, false))
		if !(x == 0 && y == 0) { // minNum leaves the sign of a zero to the target
			c.half(ht, "minnum", in, g[6], minNum(ht, a, b, x, y, true))
			c.half(ht, "maxnum", in, g[7], minNum(ht, a, b, x, y, false))
		}
		c.half(ht, "copysign", in, g[8], a&0x7fff|b&0x8000)
		c.half(ht, "fma", [3]uint16{a, b, cc}, g[9], hb(fmaOdd(x, y, z)))
		c.eq(ht, "eq", in, g[10], b2u(x == y))
		c.eq(ht, "ne", in, g[11], b2u(x != y))
		c.eq(ht, "lt", in, g[12], b2u(x < y))
		c.eq(ht, "le", in, g[13], b2u(x <= y))
		c.eq(ht, "uno", in, g[14], b2u(math.IsNaN(x) || math.IsNaN(y)))
	}
	f16 := ht == ir.TypeF16
	for i := 0; i < halfPairs; i++ {
		x32 := uint32(c.wideIn(tie32, 23, 127, f16))
		x64 := c.wideIn(tie64, 52, 1023, f16)
		r := c.nx()
		i32 := int32(uint32(r)) >> ((r >> 40) % 32)
		i64 := int64(r) >> ((r >> 50) % 64)
		c.next(5)
		g := c.line
		c.half(ht, "fromf32", fmt.Sprintf("%#x", x32), g[0], hb(float64(math.Float32frombits(x32))))
		c.half(ht, "fromf64", fmt.Sprintf("%#x", x64), g[1], hb(math.Float64frombits(x64)))
		c.half(ht, "fromi32", i32, g[2], hb(float64(i32)))
		c.half(ht, "fromu32", uint32(i32), g[3], hb(float64(uint32(i32))))
		c.half(ht, "fromi64", i64, g[4], hb(float64(i64)))
	}
	c.next(2)
	one := uint16(0x3c00)
	if !f16 {
		one = 0x3f80
	}
	c.half(ht, "mem load", one, c.line[0], one)
	c.half(ht, "mem store", one, c.line[1], hb(1+ir.HalfValue(ht, hb(1.1))))
}

func b2u(b bool) uint64 {
	if b {
		return 1
	}
	return 0
}

// satInt is the saturating conversion: NaN to zero, clamped, truncated.
func satInt(v, lo, hi float64) int64 {
	switch {
	case math.IsNaN(v):
		return 0
	case v <= lo:
		if lo == math.MinInt64 {
			return math.MinInt64
		}
		return int64(lo)
	case v >= hi:
		if hi == math.MaxInt64 {
			return math.MaxInt64
		}
		return int64(hi)
	}
	return int64(v)
}

// minimum is IEEE-754-2019's: a NaN operand wins, and -0 is below +0.
func minimum(t ir.RegType, a, b uint16, x, y float64, min bool) uint16 {
	if math.IsNaN(x) || math.IsNaN(y) {
		return ir.HalfBits(t, math.NaN())
	}
	if x == y {
		if (a&0x8000 != 0) == min {
			return a
		}
		return b
	}
	if (x < y) == min {
		return a
	}
	return b
}

// minNum is IEEE-754-2008's: a quiet NaN operand loses, and a signaling
// one makes the result a NaN.
func minNum(t ir.RegType, a, b uint16, x, y float64, min bool) uint16 {
	switch {
	case math.IsNaN(x) && math.IsNaN(y), isSNaNHalf(t, a), isSNaNHalf(t, b):
		return ir.HalfBits(t, math.NaN())
	case math.IsNaN(x):
		return b
	case math.IsNaN(y):
		return a
	}
	return minimum(t, a, b, x, y, min)
}

// fmaOdd is a*b+c computed exactly and rounded to f64 to odd, so that
// HalfBits rounding it afterwards is the one rounding.
func fmaOdd(a, b, c float64) float64 {
	if math.IsNaN(a) || math.IsNaN(b) || math.IsNaN(c) || math.IsInf(a, 0) || math.IsInf(b, 0) || math.IsInf(c, 0) {
		return math.FMA(a, b, c)
	}
	x := new(big.Float).SetPrec(2000).SetFloat64(a)
	x.Mul(x, new(big.Float).SetFloat64(b))
	x.Add(x, new(big.Float).SetFloat64(c))
	f, acc := x.Float64()
	if acc == big.Exact {
		if f == 0 && x.Signbit() != math.Signbit(f) {
			return math.Copysign(0, -1)
		}
		return f
	}
	if (acc == big.Above && f > 0) || (acc == big.Below && f < 0) {
		f = math.Nextafter(f, 0)
	}
	return math.Float64frombits(math.Float64bits(f) | 1)
}

// A half global is encoded at build time with the rounding the
// instructions use, and read back through a half load.
func TestRunHalfGlobal(t *testing.T) {
	m := ir.NewModule("t", ir.AArch64MacOS)
	g := m.Global("_tbl", ir.RO, ir.Array(3, ir.StoreBF16.FType())).
		Init(ir.List(ir.Lit(ir.Float(1)), ir.Lit(ir.Float(1+1.0/256)), ir.Lit(ir.Float(-3.5)))).Export()
	h := m.Global("_h", ir.RO, ir.StoreF16.FType()).Init(ir.Lit(ir.Float(0.1))).Export()
	fn := m.Func("_second").Export()
	fn.ReturnsF32()
	e := fn.Entry()
	p := e.Ptr.Add(e.Ptr.GetAddr(g), e.I64.Const(4)) // element 2, in bytes
	e.Return(e.F32.FCvtBF16(e.BF16().Load(p)))
	fh := m.Func("_hbits").Export()
	fh.ReturnsF16()
	eh := fh.Entry()
	eh.Return(eh.F16().Load(eh.Ptr.GetAddr(h)))
	if err := m.Err(); err != nil {
		t.Fatal(err)
	}
	m.LegalizeHalf()
	got := runNative(t, m, `
#include <stdio.h>
extern const unsigned short tbl[3];
float second(void); int hbits(void);
int main(void) { printf("%x %x %x %g %x\n", tbl[0], tbl[1], tbl[2], second(), hbits()); return 0; }
`)
	want := fmt.Sprintf("3f80 3f80 c060 -3.5 %x\n", ir.HalfBits(ir.TypeF16, 0.1))
	if got != want {
		t.Errorf("printed %q, want %q", got, want)
	}
}
