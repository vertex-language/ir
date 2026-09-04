package arm64_test

// §H, run rather than disassembled.
//
// An atomic that is merely a load and a store passes every single-threaded
// test there is. The one that matters here is TestRunAtomicContention, which
// puts four threads on one counter: a retry loop that drops an update, or an
// exclusive pair this backend got wrong, comes out short.

import (
	"testing"

	"github.com/vertex-language/ir"
	arm64lower "github.com/vertex-language/ir/lower/arm64"
)

// Every §H family at every width, single-threaded, against what C computes.
func TestRunAtomicOps(t *testing.T) {
	m := ir.NewModule("t", ir.AArch64Linux)

	// The read-modify-writes at i64, each returning the old value.
	for _, tc := range []struct {
		name string
		emit func(b *ir.Block, v ir.I64, p ir.Ptr) ir.I64
	}{
		{"_ra", func(b *ir.Block, v ir.I64, p ir.Ptr) ir.I64 { return b.I64.AtomicRmwAdd(v, p, ir.SeqCst) }},
		{"_rs", func(b *ir.Block, v ir.I64, p ir.Ptr) ir.I64 { return b.I64.AtomicRmwSub(v, p, ir.SeqCst) }},
		{"_rn", func(b *ir.Block, v ir.I64, p ir.Ptr) ir.I64 { return b.I64.AtomicRmwAnd(v, p, ir.AcqRel) }},
		{"_ro", func(b *ir.Block, v ir.I64, p ir.Ptr) ir.I64 { return b.I64.AtomicRmwOr(v, p, ir.Release) }},
		{"_rx", func(b *ir.Block, v ir.I64, p ir.Ptr) ir.I64 { return b.I64.AtomicRmwXor(v, p, ir.Acquire) }},
		{"_rg", func(b *ir.Block, v ir.I64, p ir.Ptr) ir.I64 { return b.I64.AtomicRmwXchg(v, p, ir.Monotonic) }},
	} {
		fn := m.Func(tc.name).Export()
		v := fn.ParamI64("v")
		p := fn.ParamPtr("p")
		fn.ReturnsI64()
		e := fn.Entry()
		e.Return(tc.emit(e, v, p))
	}

	// The narrow ones, in the i32 namespace, with the old value
	// zero-extended.
	for _, tc := range []struct {
		name string
		emit func(b *ir.Block, v ir.I32, p ir.Ptr) ir.I32
	}{
		{"_r8a", func(b *ir.Block, v ir.I32, p ir.Ptr) ir.I32 { return b.I32.AtomicRmwAdd8(v, p, ir.SeqCst) }},
		{"_r8g", func(b *ir.Block, v ir.I32, p ir.Ptr) ir.I32 { return b.I32.AtomicRmwXchg8(v, p, ir.SeqCst) }},
		{"_r16a", func(b *ir.Block, v ir.I32, p ir.Ptr) ir.I32 { return b.I32.AtomicRmwAdd16(v, p, ir.SeqCst) }},
		{"_r16o", func(b *ir.Block, v ir.I32, p ir.Ptr) ir.I32 { return b.I32.AtomicRmwOr16(v, p, ir.SeqCst) }},
		{"_r32a", func(b *ir.Block, v ir.I32, p ir.Ptr) ir.I32 { return b.I32.AtomicRmwAdd(v, p, ir.SeqCst) }},
	} {
		fn := m.Func(tc.name).Export()
		v := fn.ParamI32("v")
		p := fn.ParamPtr("p")
		fn.ReturnsI32()
		e := fn.Entry()
		e.Return(tc.emit(e, v, p))
	}

	// Loads and stores, at every ordering a load and a store take.
	ld := m.Func("_ld").Export()
	ldP := ld.ParamPtr("p")
	ld.ReturnsI64()
	e := ld.Entry()
	e.Return(e.I64.AtomicLoad(ldP, ir.SeqCst))

	ldr := m.Func("_ldr").Export()
	ldrP := ldr.ParamPtr("p")
	ldr.ReturnsI64()
	e2 := ldr.Entry()
	e2.Return(e2.I64.AtomicLoad(ldrP, ir.Monotonic))

	ld8 := m.Func("_ld8").Export()
	ld8P := ld8.ParamPtr("p")
	ld8.ReturnsI32()
	e3 := ld8.Entry()
	e3.Return(e3.I32.AtomicULoad8(ld8P, ir.Acquire))

	ld16 := m.Func("_ld16").Export()
	ld16P := ld16.ParamPtr("p")
	ld16.ReturnsI32()
	e4 := ld16.Entry()
	e4.Return(e4.I32.AtomicULoad16(ld16P, ir.Monotonic))

	st := m.Func("_st").Export()
	stV := st.ParamI64("v")
	stP := st.ParamPtr("p")
	e5 := st.Entry()
	e5.I64.AtomicStore(stV, stP, ir.SeqCst)
	e5.Return()

	st8 := m.Func("_st8").Export()
	st8V := st8.ParamI32("v")
	st8P := st8.ParamPtr("p")
	e6 := st8.Entry()
	e6.I32.AtomicStore8(st8V, st8P, ir.Release)
	e6.Return()

	st16 := m.Func("_st16").Export()
	st16V := st16.ParamI32("v")
	st16P := st16.ParamPtr("p")
	e7 := st16.Entry()
	e7.I32.AtomicStore16(st16V, st16P, ir.Monotonic)
	e7.Return()

	// A fence, which is the one verb with nothing to observe: this is a
	// check that it assembles and does not disturb the value around it.
	fen := m.Func("_fen").Export()
	fenV := fen.ParamI64("v")
	fen.ReturnsI64()
	e8 := fen.Entry()
	e8.Fence(ir.SeqCst)
	e8.Fence(ir.Acquire)
	e8.Fence(ir.SeqCst, ir.SingleThread)
	e8.Return(fenV)

	got := runNative(t, m, `
#include <stdio.h>
#include <string.h>
long ra(long,void*), rs(long,void*), rn(long,void*);
long ro(long,void*), rx(long,void*), rg(long,void*);
int r8a(int,void*), r8g(int,void*), r16a(int,void*), r16o(int,void*), r32a(int,void*);
long ld(void*), ldr(void*), fen(long);
int ld8(void*), ld16(void*);
void st(long,void*), st8(int,void*), st16(int,void*);
static int fail = 0;
static void chk(const char *what, long got, long want) {
    if (got != want) { printf("%s: got %ld want %ld\n", what, got, want); fail = 1; }
}
int main(void) {
    long v;
    v = 100; chk("rmwadd old", ra(5, &v), 100);  chk("rmwadd new", v, 105);
    v = 100; chk("rmwsub old", rs(5, &v), 100);  chk("rmwsub new", v, 95);
    v = 0xff; chk("rmwand old", rn(0x0f, &v), 0xff); chk("rmwand new", v, 0x0f);
    v = 0xf0; chk("rmwor old",  ro(0x0f, &v), 0xf0); chk("rmwor new",  v, 0xff);
    v = 0xff; chk("rmwxor old", rx(0x0f, &v), 0xff); chk("rmwxor new", v, 0xf0);
    v = 7;   chk("rmwxchg old", rg(42, &v), 7);   chk("rmwxchg new", v, 42);

    // The narrow forms must touch exactly their own bytes and no others.
    unsigned char b8[2] = {200, 111};
    chk("rmwadd8 old", r8a(100, &b8[0]), 200);
    chk("rmwadd8 new", b8[0], (unsigned char)(200 + 100));
    chk("rmwadd8 next", b8[1], 111);
    b8[0] = 9;
    chk("rmwxchg8 old", r8g(0x1ff, &b8[0]), 9);
    chk("rmwxchg8 new", b8[0], 0xff);
    unsigned short h[2] = {60000, 1234};
    chk("rmwadd16 old", r16a(10000, &h[0]), 60000);
    chk("rmwadd16 new", h[0], (unsigned short)(60000 + 10000));
    chk("rmwadd16 next", h[1], 1234);
    h[0] = 0xf0f0;
    chk("rmwor16", r16o(0x0f0f, &h[0]), 0xf0f0);
    chk("rmwor16 new", h[0], 0xffff);
    unsigned w32[2] = {4000000000u, 7};
    chk("rmwadd32 old", (unsigned)r32a(1, &w32[0]), 4000000000u);
    chk("rmwadd32 new", w32[0], 4000000001u);
    chk("rmwadd32 next", w32[1], 7);

    v = 0x0123456789abcdefL;
    chk("load seqcst",  ld(&v),  0x0123456789abcdefL);
    chk("load relaxed", ldr(&v), 0x0123456789abcdefL);
    unsigned char lb[2] = {0xfe, 3};  chk("uload8",  ld8(&lb[0]), 0xfe);
    unsigned short lh[2] = {0xfedc, 5}; chk("uload16", ld16(&lh[0]), 0xfedc);

    v = 0; st(0x7fffffffffffffffL, &v); chk("store", v, 0x7fffffffffffffffL);
    unsigned char sb[2] = {0, 9}; st8(0x1ab, &sb[0]);
    chk("store8", sb[0], 0xab); chk("store8 next", sb[1], 9);
    unsigned short sh[2] = {0, 9}; st16(0x1beef, &sh[0]);
    chk("store16", sh[0], 0xbeef); chk("store16 next", sh[1], 9);

    chk("fence", fen(1234), 1234);
    printf("%s\n", fail ? "MISMATCH" : "ok");
    return 0;
}
`)
	if got != "ok\n" {
		t.Errorf("printed %q, want %q", got, "ok\n")
	}
}

// §H's compare-and-swap, on both the matching and the failing path — the
// failing one being where CLREX has to release the reservation the load took.
func TestRunAtomicCas(t *testing.T) {
	m := ir.NewModule("t", ir.AArch64Linux)

	c64 := m.Func("_cas64").Export()
	cE := c64.ParamI64("expect")
	cN := c64.ParamI64("new")
	cP := c64.ParamPtr("p")
	c64.ReturnsI64()
	e := c64.Entry()
	e.Return(e.I64.AtomicCas(cE, cN, cP, ir.SeqCst, ir.SeqCst))

	c32 := m.Func("_cas32").Export()
	c32E := c32.ParamI32("expect")
	c32N := c32.ParamI32("new")
	c32P := c32.ParamPtr("p")
	c32.ReturnsI32()
	e2 := c32.Entry()
	e2.Return(e2.I32.AtomicCas(c32E, c32N, c32P, ir.AcqRel, ir.Acquire))

	c8 := m.Func("_cas8").Export()
	c8E := c8.ParamI32("expect")
	c8N := c8.ParamI32("new")
	c8P := c8.ParamPtr("p")
	c8.ReturnsI32()
	e3 := c8.Entry()
	e3.Return(e3.I32.AtomicCas8(c8E, c8N, c8P, ir.SeqCst, ir.Monotonic))

	c16 := m.Func("_cas16").Export()
	c16E := c16.ParamI32("expect")
	c16N := c16.ParamI32("new")
	c16P := c16.ParamPtr("p")
	c16.ReturnsI32()
	e4 := c16.Entry()
	e4.Return(e4.I32.AtomicCas16(c16E, c16N, c16P, ir.SeqCst, ir.Monotonic))

	got := runNative(t, m, `
#include <stdio.h>
long cas64(long,long,void*);
int cas32(int,int,void*), cas8(int,int,void*), cas16(int,int,void*);
static int fail = 0;
static void chk(const char *what, long got, long want) {
    if (got != want) { printf("%s: got %ld want %ld\n", what, got, want); fail = 1; }
}
int main(void) {
    long v = 42;
    chk("cas64 hit",  cas64(42, 99, &v), 42);   chk("cas64 stored", v, 99);
    // A miss returns the value read and leaves memory alone. This is the
    // path that skips its store and owes the monitor a CLREX.
    chk("cas64 miss", cas64(42, 7, &v), 99);    chk("cas64 kept",   v, 99);
    // And the loop still works afterwards, which it would not if the
    // reservation from the miss were still standing in the way.
    chk("cas64 again", cas64(99, 7, &v), 99);   chk("cas64 stored2", v, 7);

    unsigned w = 5;
    chk("cas32 hit",  (unsigned)cas32(5, 6, &w), 5);  chk("cas32 stored", w, 6);
    chk("cas32 miss", (unsigned)cas32(5, 8, &w), 6);  chk("cas32 kept",   w, 6);

    // Narrow: only the accessed bytes compare and only they change.
    unsigned char b[2] = {0xab, 0x5a};
    chk("cas8 hit",  cas8(0xab, 0xcd, &b[0]), 0xab);
    chk("cas8 stored", b[0], 0xcd);  chk("cas8 next", b[1], 0x5a);
    chk("cas8 miss", cas8(0xab, 0x11, &b[0]), 0xcd);  chk("cas8 kept", b[0], 0xcd);
    // The expected value's high bits are not part of an 8-bit compare.
    b[0] = 0x77;
    chk("cas8 wide expect", cas8(0x12345677, 0x88, &b[0]), 0x77);
    chk("cas8 wide stored", b[0], 0x88);

    unsigned short h[2] = {0xbeef, 0x1234};
    chk("cas16 hit",  cas16(0xbeef, 0xcafe, &h[0]), 0xbeef);
    chk("cas16 stored", h[0], 0xcafe); chk("cas16 next", h[1], 0x1234);
    chk("cas16 miss", cas16(0xbeef, 1, &h[0]), 0xcafe);
    printf("%s\n", fail ? "MISMATCH" : "ok");
    return 0;
}
`)
	if got != "ok\n" {
		t.Errorf("printed %q, want %q", got, "ok\n")
	}
}

// Four threads and one counter.
//
// This is the test the rest of the file exists to set up. A single-threaded
// check cannot tell an atomic read-modify-write from a load, an add and a
// store; contention can, because the non-atomic version loses updates and the
// total comes out short. The compare-and-swap loop is driven the same way.
func TestRunAtomicContention(t *testing.T) {
	m := ir.NewModule("t", ir.AArch64Linux)

	inc := m.Func("_bump").Export()
	incP := inc.ParamPtr("p")
	inc.ReturnsI64()
	e := inc.Entry()
	e.Return(e.I64.AtomicRmwAdd(e.I64.Const(1), incP, ir.SeqCst))

	// The same increment written as a compare-and-swap retry, which is
	// where a wrong failure path shows up as a hang or a lost update.
	casInc := m.Func("_bumpcas").Export()
	casP := casInc.ParamPtr("p")
	casInc.ReturnsI64()
	entry := casInc.Entry()
	loop := casInc.Block("loop")
	done := casInc.Block("done")
	old := loop.ParamI64("old")

	entry.Br(loop.To(entry.I64.AtomicLoad(casP, ir.Monotonic)))
	seen := loop.I64.AtomicCas(old, loop.I64.Add(old, loop.I64.Const(1)), casP, ir.SeqCst, ir.Monotonic)
	loop.BrIf(loop.I64.Eq(seen, old), done.To(), loop.To(seen))
	done.Return(done.I64.Const(0))

	got := runNative(t, m, `
#include <stdio.h>
#include <pthread.h>
#define THREADS 4
#define ROUNDS  50000
long bump(void*), bumpcas(void*);
static long counter, casCounter;
static void *worker(void *unused) {
    (void)unused;
    for (int i = 0; i < ROUNDS; i++) { bump(&counter); bumpcas(&casCounter); }
    return 0;
}
int main(void) {
    pthread_t th[THREADS];
    for (int i = 0; i < THREADS; i++) pthread_create(&th[i], 0, worker, 0);
    for (int i = 0; i < THREADS; i++) pthread_join(th[i], 0);
    long want = (long)THREADS * ROUNDS;
    if (counter != want)    printf("rmwadd: got %ld want %ld\n", counter, want);
    if (casCounter != want) printf("cas: got %ld want %ld\n", casCounter, want);
    printf("%s\n", counter == want && casCounter == want ? "ok" : "MISMATCH");
    return 0;
}
`)
	if got != "ok\n" {
		t.Errorf("printed %q, want %q", got, "ok\n")
	}
}

// The shape of the two loops, since a running test cannot see all of it.
//
// CLREX in particular: leaving a reservation standing is not something these
// tests can catch — the next LDXR clears it anyway, and every path here
// reaches one. It is in the code because the ARM ARM says the failure path
// owes it, and this is the check that it stays.
func TestLowerAtomicShape(t *testing.T) {
	m := ir.NewModule("t", ir.AArch64Linux)
	fn := m.Func("cas").Export()
	e := fn.ParamI64("expect")
	n := fn.ParamI64("new")
	p := fn.ParamPtr("p")
	fn.ReturnsI64()
	entry := fn.Entry()
	entry.Return(entry.I64.AtomicCas(e, n, p, ir.AcqRel, ir.Monotonic))

	got, raw := lowerWords(t, m)

	const (
		clrex = 0xd5033f5f
		// LDAXR names only Rt and Rn, so everything above them is
		// fixed; STLXR also names a status register at 20:16.
		ldaxr, ldaxrMask = 0xc85ffc00, 0xfffffc00
		stlxr, stlxrMask = 0xc800fc00, 0xffe0fc00
	)
	var clrexes, loads, stores int
	for _, w := range got {
		switch {
		case w == clrex:
			clrexes++
		case w&ldaxrMask == ldaxr:
			loads++
		case w&stlxrMask == stlxr:
			stores++
		}
	}
	if clrexes != 1 {
		t.Errorf("%d CLREXes, want one on the failure path: %s", clrexes, hexWords(got))
	}
	// AcqRel success means the load acquires and the store releases.
	if loads != 1 {
		t.Errorf("%d LDAXRs, want one: %s", loads, hexWords(got))
	}
	if stores != 1 {
		t.Errorf("%d STLXRs, want one: %s", stores, hexWords(got))
	}
	objdumpHas(t, "cas", raw, "ldaxr", "stlxr", "clrex", "cbnz")
}

// A relaxed read-modify-write takes the unordered pair, which is the other
// half of the ordering mapping.
func TestLowerAtomicRelaxedRmw(t *testing.T) {
	m := ir.NewModule("t", ir.AArch64Linux)
	fn := m.Func("bump").Export()
	v := fn.ParamI64("v")
	p := fn.ParamPtr("p")
	fn.ReturnsI64()
	entry := fn.Entry()
	entry.Return(entry.I64.AtomicRmwAdd(v, p, ir.Monotonic))

	_, raw := lowerWords(t, m)
	objdumpHas(t, "bump", raw, "ldxr", "stxr", "cbnz")
}

// A load may not be given a release ordering, and a read-modify-write may not
// be unordered. Refused rather than quietly strengthened.
func TestLowerRefusesBadAtomicOrdering(t *testing.T) {
	build := func(f func(b *ir.Block, p ir.Ptr)) *ir.Module {
		m := ir.NewModule("t", ir.AArch64Linux)
		fn := m.Func("f").Export()
		p := fn.ParamPtr("p")
		entry := fn.Entry()
		f(entry, p)
		entry.Return()
		return m
	}

	m := build(func(b *ir.Block, p ir.Ptr) { b.I64.AtomicLoad(p, ir.Release) })
	if _, err := arm64lower.Lower(m, arm64lower.Options{}); err == nil {
		t.Error("Lower should refuse a release-ordered load")
	}
}
