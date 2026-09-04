package i386_test

// §H, run rather than disassembled.
//
// The contention test is the one that carries weight, and it needs more than
// one CPU to mean anything — so the kernel boots with two and runs the second
// one through the same increment loop. A retry that drops an update comes out
// short, exactly as it would with threads.

import (
	"strings"
	"testing"

	"github.com/vertex-language/ir"
)

func TestRunAtomicOps(t *testing.T) {
	m := ir.NewModule("t", ir.I386Linux)

	for _, tc := range []struct {
		name string
		emit func(b *ir.Block, v ir.I32, p ir.Ptr) ir.I32
	}{
		{"ra", func(b *ir.Block, v ir.I32, p ir.Ptr) ir.I32 { return b.I32.AtomicRmwAdd(v, p, ir.SeqCst) }},
		{"rs", func(b *ir.Block, v ir.I32, p ir.Ptr) ir.I32 { return b.I32.AtomicRmwSub(v, p, ir.SeqCst) }},
		{"rn", func(b *ir.Block, v ir.I32, p ir.Ptr) ir.I32 { return b.I32.AtomicRmwAnd(v, p, ir.AcqRel) }},
		{"ro", func(b *ir.Block, v ir.I32, p ir.Ptr) ir.I32 { return b.I32.AtomicRmwOr(v, p, ir.Release) }},
		{"rx", func(b *ir.Block, v ir.I32, p ir.Ptr) ir.I32 { return b.I32.AtomicRmwXor(v, p, ir.Acquire) }},
		{"rg", func(b *ir.Block, v ir.I32, p ir.Ptr) ir.I32 { return b.I32.AtomicRmwXchg(v, p, ir.Monotonic) }},
		{"r8a", func(b *ir.Block, v ir.I32, p ir.Ptr) ir.I32 { return b.I32.AtomicRmwAdd8(v, p, ir.SeqCst) }},
		{"r8o", func(b *ir.Block, v ir.I32, p ir.Ptr) ir.I32 { return b.I32.AtomicRmwOr8(v, p, ir.SeqCst) }},
		{"r16a", func(b *ir.Block, v ir.I32, p ir.Ptr) ir.I32 { return b.I32.AtomicRmwAdd16(v, p, ir.SeqCst) }},
		{"r16x", func(b *ir.Block, v ir.I32, p ir.Ptr) ir.I32 { return b.I32.AtomicRmwXchg16(v, p, ir.SeqCst) }},
	} {
		fn := m.Func(tc.name).Export()
		v := fn.ParamI32("v")
		p := fn.ParamPtr("p")
		fn.ReturnsI32()
		e := fn.Entry()
		e.Return(tc.emit(e, v, p))
	}

	ld := m.Func("ld").Export()
	ldP := ld.ParamPtr("p")
	ld.ReturnsI32()
	e := ld.Entry()
	e.Return(e.I32.AtomicLoad(ldP, ir.SeqCst))

	st := m.Func("st").Export()
	stV := st.ParamI32("v")
	stP := st.ParamPtr("p")
	e2 := st.Entry()
	e2.I32.AtomicStore(stV, stP, ir.SeqCst)
	e2.Return()

	str := m.Func("strel").Export()
	strV := str.ParamI32("v")
	strP := str.ParamPtr("p")
	e3 := str.Entry()
	e3.I32.AtomicStore(strV, strP, ir.Release)
	e3.Return()

	st8 := m.Func("st8").Export()
	st8V := st8.ParamI32("v")
	st8P := st8.ParamPtr("p")
	e4 := st8.Entry()
	e4.I32.AtomicStore8(st8V, st8P, ir.Release)
	e4.Return()

	st16 := m.Func("st16").Export()
	st16V := st16.ParamI32("v")
	st16P := st16.ParamPtr("p")
	e5 := st16.Entry()
	e5.I32.AtomicStore16(st16V, st16P, ir.Monotonic)
	e5.Return()

	ld8 := m.Func("ld8").Export()
	ld8P := ld8.ParamPtr("p")
	ld8.ReturnsI32()
	e6 := ld8.Entry()
	e6.Return(e6.I32.AtomicULoad8(ld8P, ir.Acquire))

	ld16 := m.Func("ld16").Export()
	ld16P := ld16.ParamPtr("p")
	ld16.ReturnsI32()
	e7 := ld16.Entry()
	e7.Return(e7.I32.AtomicULoad16(ld16P, ir.Monotonic))

	fen := m.Func("fen").Export()
	fenV := fen.ParamI32("v")
	fen.ReturnsI32()
	e8 := fen.Entry()
	e8.Fence(ir.SeqCst)
	e8.Fence(ir.Acquire)
	e8.Fence(ir.SeqCst, ir.SingleThread)
	e8.Return(fenV)

	wantOK(t, m, `
int ra(int,void*), rs(int,void*), rn(int,void*), ro(int,void*), rx(int,void*), rg(int,void*);
int r8a(int,void*), r8o(int,void*), r16a(int,void*), r16x(int,void*);
int ld(void*), ld8(void*), ld16(void*), fen(int);
void st(int,void*), strel(int,void*), st8(int,void*), st16(int,void*);
static void body(void) {
    int v;
    v = 100; chk32("rmwadd old", (unsigned)ra(5, &v), 100u);  chk32("rmwadd new", (unsigned)v, 105u);
    v = 100; chk32("rmwsub old", (unsigned)rs(5, &v), 100u);  chk32("rmwsub new", (unsigned)v, 95u);
    v = 0xff; chk32("rmwand", (unsigned)rn(0x0f, &v), 0xffu); chk32("rmwand new", (unsigned)v, 0x0fu);
    v = 0xf0; chk32("rmwor",  (unsigned)ro(0x0f, &v), 0xf0u); chk32("rmwor new",  (unsigned)v, 0xffu);
    v = 0xff; chk32("rmwxor", (unsigned)rx(0x0f, &v), 0xffu); chk32("rmwxor new", (unsigned)v, 0xf0u);
    v = 7;   chk32("rmwxchg", (unsigned)rg(42, &v), 7u);      chk32("rmwxchg new", (unsigned)v, 42u);

    /* The narrow forms touch exactly their own bytes. */
    unsigned char b[2] = {200, 111};
    chk32("rmwadd8", (unsigned)r8a(100, &b[0]), 200u);
    chk32("rmwadd8 new", b[0], (unsigned)(unsigned char)(200 + 100));
    chk32("rmwadd8 next", b[1], 111u);
    b[0] = 0xf0;
    chk32("rmwor8", (unsigned)r8o(0x0f, &b[0]), 0xf0u);
    chk32("rmwor8 new", b[0], 0xffu);

    unsigned short h[2] = {60000, 1234};
    chk32("rmwadd16", (unsigned)r16a(10000, &h[0]), 60000u);
    chk32("rmwadd16 new", h[0], (unsigned)(unsigned short)(60000 + 10000));
    chk32("rmwadd16 next", h[1], 1234u);
    h[0] = 0xbeef;
    chk32("rmwxchg16", (unsigned)r16x(0x1cafe, &h[0]), 0xbeefu);
    chk32("rmwxchg16 new", h[0], 0xcafeu);

    v = 0x12345678; chk32("load", (unsigned)ld(&v), 0x12345678u);
    v = 0; st(0x7fffffff, &v); chk32("store seqcst", (unsigned)v, 0x7fffffffu);
    v = 0; strel(0x11223344, &v); chk32("store release", (unsigned)v, 0x11223344u);
    unsigned char sb[2] = {0, 9}; st8(0x1ab, &sb[0]);
    chk32("store8", sb[0], 0xabu); chk32("store8 next", sb[1], 9u);
    unsigned short sh[2] = {0, 9}; st16(0x1beef, &sh[0]);
    chk32("store16", sh[0], 0xbeefu); chk32("store16 next", sh[1], 9u);
    unsigned char lb[2] = {0xfe, 3};   chk32("uload8", (unsigned)ld8(&lb[0]), 0xfeu);
    unsigned short lh[2] = {0xfedc, 5}; chk32("uload16", (unsigned)ld16(&lh[0]), 0xfedcu);

    chk32("fence", (unsigned)fen(1234), 1234u);
}
`)
}

// The eight-byte forms, which have no instruction on this architecture except
// CMPXCHG8B — so every one of them, including the load, is a loop around it.
func TestRunAtomic64(t *testing.T) {
	m := ir.NewModule("t", ir.I386Linux)

	ld := m.Func("ld64").Export()
	ldP := ld.ParamPtr("p")
	ld.ReturnsI64()
	e := ld.Entry()
	e.Return(e.I64.AtomicLoad(ldP, ir.SeqCst))

	st := m.Func("st64").Export()
	stV := st.ParamI64("v")
	stP := st.ParamPtr("p")
	e2 := st.Entry()
	e2.I64.AtomicStore(stV, stP, ir.SeqCst)
	e2.Return()

	for _, tc := range []struct {
		name string
		emit func(b *ir.Block, v ir.I64, p ir.Ptr) ir.I64
	}{
		{"a64add", func(b *ir.Block, v ir.I64, p ir.Ptr) ir.I64 { return b.I64.AtomicRmwAdd(v, p, ir.SeqCst) }},
		{"a64sub", func(b *ir.Block, v ir.I64, p ir.Ptr) ir.I64 { return b.I64.AtomicRmwSub(v, p, ir.SeqCst) }},
		{"a64or", func(b *ir.Block, v ir.I64, p ir.Ptr) ir.I64 { return b.I64.AtomicRmwOr(v, p, ir.SeqCst) }},
		{"a64xchg", func(b *ir.Block, v ir.I64, p ir.Ptr) ir.I64 { return b.I64.AtomicRmwXchg(v, p, ir.SeqCst) }},
	} {
		fn := m.Func(tc.name).Export()
		v := fn.ParamI64("v")
		p := fn.ParamPtr("p")
		fn.ReturnsI64()
		en := fn.Entry()
		en.Return(tc.emit(en, v, p))
	}

	cas := m.Func("cas64").Export()
	casE := cas.ParamI64("e")
	casN := cas.ParamI64("n")
	casP := cas.ParamPtr("p")
	cas.ReturnsI64()
	e3 := cas.Entry()
	e3.Return(e3.I64.AtomicCas(casE, casN, casP, ir.SeqCst, ir.SeqCst))

	cas32 := m.Func("cas32").Export()
	c32E := cas32.ParamI32("e")
	c32N := cas32.ParamI32("n")
	c32P := cas32.ParamPtr("p")
	cas32.ReturnsI32()
	e4 := cas32.Entry()
	e4.Return(e4.I32.AtomicCas(c32E, c32N, c32P, ir.SeqCst, ir.Monotonic))

	cas8 := m.Func("cas8").Export()
	c8E := cas8.ParamI32("e")
	c8N := cas8.ParamI32("n")
	c8P := cas8.ParamPtr("p")
	cas8.ReturnsI32()
	e5 := cas8.Entry()
	e5.Return(e5.I32.AtomicCas8(c8E, c8N, c8P, ir.SeqCst, ir.Monotonic))

	wantOK(t, m, `
long long ld64(void*), a64add(long long,void*), a64sub(long long,void*);
long long a64or(long long,void*), a64xchg(long long,void*);
long long cas64(long long,long long,void*);
int cas32(int,int,void*), cas8(int,int,void*);
void st64(long long,void*);
static void body(void) {
    long long v = 0x0123456789abcdefLL;
    chk64("load64", ld64(&v), 0x0123456789abcdefULL);
    /* The load stores what it read, so the value has to survive it. */
    chk64("load64 kept", (unsigned long long)v, 0x0123456789abcdefULL);

    v = 0; st64(0x123456789abcdef0LL, &v);
    chk64("store64", (unsigned long long)v, 0x123456789abcdef0ULL);

    v = 0x00000000ffffffffLL;
    chk64("add64 old", a64add(1, &v), 0x00000000ffffffffULL);
    chk64("add64 new", (unsigned long long)v, 0x0000000100000000ULL);
    chk64("sub64 old", a64sub(1, &v), 0x0000000100000000ULL);
    chk64("sub64 new", (unsigned long long)v, 0x00000000ffffffffULL);
    v = 0xf0f0f0f000000000LL;
    chk64("or64", a64or(0x0f0f0f0fLL, &v), 0xf0f0f0f000000000ULL);
    chk64("or64 new", (unsigned long long)v, 0xf0f0f0f00f0f0f0fULL);
    chk64("xchg64", a64xchg(42, &v), 0xf0f0f0f00f0f0f0fULL);
    chk64("xchg64 new", (unsigned long long)v, 42ULL);

    v = 42;
    chk64("cas64 hit",  cas64(42, 99, &v), 42ULL);
    chk64("cas64 stored", (unsigned long long)v, 99ULL);
    chk64("cas64 miss", cas64(42, 7, &v), 99ULL);
    chk64("cas64 kept", (unsigned long long)v, 99ULL);

    int w = 5;
    chk32("cas32 hit",  (unsigned)cas32(5, 6, &w), 5u);
    chk32("cas32 stored", (unsigned)w, 6u);
    chk32("cas32 miss", (unsigned)cas32(5, 8, &w), 6u);

    unsigned char b[2] = {0xab, 0x5a};
    chk32("cas8 hit", (unsigned)cas8(0xab, 0xcd, &b[0]), 0xabu);
    chk32("cas8 stored", b[0], 0xcdu);  chk32("cas8 next", b[1], 0x5au);
    chk32("cas8 miss", (unsigned)cas8(0xab, 0x11, &b[0]), 0xcdu);
}
`)
}

// The LOCK prefix is present on every read-modify-write.
//
// Weaker than the arm64 backend's contention test, and knowingly so. That one
// puts four threads on one counter and watches a non-atomic increment lose
// updates; here there are no threads and no OS, and a second CPU would have
// to be brought up through the local APIC with a real-mode trampoline — a
// hundred lines of bring-up to test one prefix. So this checks that the
// prefix is there, which is the fact that makes the instruction atomic, and
// does not claim to have observed atomicity.
func TestLowerAtomicsAreLocked(t *testing.T) {
	for _, tc := range []struct {
		name string
		emit func(b *ir.Block, v ir.I32, p ir.Ptr) ir.I32
		want string
	}{
		{"add", func(b *ir.Block, v ir.I32, p ir.Ptr) ir.I32 {
			return b.I32.AtomicRmwAdd(v, p, ir.SeqCst)
		}, "xadd"},
		{"xchg", func(b *ir.Block, v ir.I32, p ir.Ptr) ir.I32 {
			return b.I32.AtomicRmwXchg(v, p, ir.SeqCst)
		}, "xchg"},
		{"and", func(b *ir.Block, v ir.I32, p ir.Ptr) ir.I32 {
			return b.I32.AtomicRmwAnd(v, p, ir.SeqCst)
		}, "cmpxchg"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			m := ir.NewModule("t", ir.I386Linux)
			fn := m.Func("f").Export()
			v := fn.ParamI32("v")
			p := fn.ParamPtr("p")
			fn.ReturnsI32()
			e := fn.Entry()
			e.Return(tc.emit(e, v, p))

			_, raw := lowerText(t, m)
			d := objdumpText(t, "f", raw)
			hasAll(t, d, "lock", tc.want)
		})
	}
}

// A sequentially consistent store is an XCHG and a weaker one is a plain MOV.
//
// x86 orders a store after everything before it, so release costs nothing;
// what it does not order is a later load moving ahead of the store, and the
// implicit lock on XCHG is the barrier that stops it.
func TestLowerAtomicStoreOrdering(t *testing.T) {
	build := func(o ir.Ordering) string {
		m := ir.NewModule("t", ir.I386Linux)
		fn := m.Func("f").Export()
		v := fn.ParamI32("v")
		p := fn.ParamPtr("p")
		e := fn.Entry()
		e.I32.AtomicStore(v, p, o)
		e.Return()
		_, raw := lowerText(t, m)
		return objdumpText(t, "f", raw)
	}
	if d := build(ir.SeqCst); !strings.Contains(d, "xchg") {
		t.Errorf("a seq_cst store should be an xchg:\n%s", d)
	}
	if d := build(ir.Release); strings.Contains(d, "xchg") {
		t.Errorf("a release store needs no xchg on x86:\n%s", d)
	}
}
