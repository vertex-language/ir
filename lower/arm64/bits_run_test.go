package arm64_test

// The §A verbs A64 does not have an instruction for, and §A2's predicates,
// which it has flags for and no verb.

import (
	"testing"

	"github.com/vertex-language/ir"
)

// §A5's rotl and §A6's ctz and popcnt, against what C computes.
func TestRunBitVerbs(t *testing.T) {
	m := ir.NewModule("t", ir.AArch64Linux)

	rl := m.Func("_rl").Export()
	rlA := rl.ParamI64("a")
	rlN := rl.ParamI64("n")
	rl.ReturnsI64()
	e := rl.Entry()
	e.Return(e.I64.RotL(rlA, rlN))

	rl32 := m.Func("_rl32").Export()
	rl32A := rl32.ParamI32("a")
	rl32N := rl32.ParamI32("n")
	rl32.ReturnsI32()
	e2 := rl32.Entry()
	e2.Return(e2.I32.RotL(rl32A, rl32N))

	cz := m.Func("_cz").Export()
	czA := cz.ParamI64("a")
	cz.ReturnsI64()
	e3 := cz.Entry()
	e3.Return(e3.I64.Ctz(czA))

	pc := m.Func("_pc").Export()
	pcA := pc.ParamI64("a")
	pc.ReturnsI64()
	e4 := pc.Entry()
	e4.Return(e4.I64.Popcnt(pcA))

	pc32 := m.Func("_pc32").Export()
	pc32A := pc32.ParamI32("a")
	pc32.ReturnsI32()
	e5 := pc32.Entry()
	e5.Return(e5.I32.Popcnt(pc32A))

	got := runNative(t, m, `
#include <stdio.h>
long rl(long, long), cz(long), pc(long);
int rl32(int, int), pc32(int);
static int fail = 0;
static void chk(const char *what, long got, long want) {
    if (got != want) { printf("%s: got %ld want %ld\n", what, got, want); fail = 1; }
}
static unsigned long rotl64(unsigned long v, int n) {
    return n == 0 ? v : (v << n) | (v >> (64 - n));
}
static unsigned rotl32(unsigned v, int n) {
    return n == 0 ? v : (v << n) | (v >> (32 - n));
}
int main(void) {
    unsigned long v = 0x0123456789abcdefUL;
    for (int n = 0; n < 64; n++) chk("rotl", rl((long)v, n), (long)rotl64(v, n));
    unsigned v32 = 0x89abcdefU;
    for (int n = 0; n < 32; n++) chk("rotl32", rl32((int)v32, n), (int)rotl32(v32, n));

    chk("ctz 1",    cz(1), 0);
    chk("ctz 8",    cz(8), 3);
    chk("ctz 0",    cz(0), 64);
    chk("ctz top",  cz((long)0x8000000000000000UL), 63);
    chk("ctz mix",  cz((long)0x0123456789abc000UL), __builtin_ctzll(0x0123456789abc000UL));

    chk("popcnt 0",   pc(0), 0);
    chk("popcnt 1",   pc(1), 1);
    chk("popcnt all", pc(-1), 64);
    chk("popcnt mix", pc((long)v), __builtin_popcountll(v));
    chk("popcnt alt", pc((long)0xaaaaaaaaaaaaaaaaUL), 32);
    chk("popcnt32 0",   pc32(0), 0);
    chk("popcnt32 all", pc32(-1), 32);
    chk("popcnt32 mix", pc32((int)v32), __builtin_popcount(v32));
    printf("%s\n", fail ? "MISMATCH" : "ok");
    return 0;
}
`)
	if got != "ok\n" {
		t.Errorf("printed %q, want %q", got, "ok\n")
	}
}

// §A2's five predicates, against clang's __builtin_*_overflow — which is the
// same question asked of the compiler that implements it in hardware flags.
func TestRunOverflowPredicates(t *testing.T) {
	m := ir.NewModule("t", ir.AArch64Linux)

	for _, tc := range []struct {
		name string
		emit func(b *ir.Block, x, y ir.I64) ir.I1
	}{
		{"_osadd", func(b *ir.Block, x, y ir.I64) ir.I1 { return b.I64.SAddO(x, y) }},
		{"_ouadd", func(b *ir.Block, x, y ir.I64) ir.I1 { return b.I64.UAddO(x, y) }},
		{"_ossub", func(b *ir.Block, x, y ir.I64) ir.I1 { return b.I64.SSubO(x, y) }},
		{"_osmul", func(b *ir.Block, x, y ir.I64) ir.I1 { return b.I64.SMulO(x, y) }},
		{"_oumul", func(b *ir.Block, x, y ir.I64) ir.I1 { return b.I64.UMulO(x, y) }},
	} {
		fn := m.Func(tc.name).Export()
		x := fn.ParamI64("x")
		y := fn.ParamI64("y")
		fn.ReturnsI32()
		entry := fn.Entry()
		entry.Return(entry.I32.ZExtI1(tc.emit(entry, x, y)))
	}
	// And the 32-bit multiplies, which take the other path: the whole
	// product fits in an X register, so there is no high half to name.
	for _, tc := range []struct {
		name string
		emit func(b *ir.Block, x, y ir.I32) ir.I1
	}{
		{"_osmul32", func(b *ir.Block, x, y ir.I32) ir.I1 { return b.I32.SMulO(x, y) }},
		{"_oumul32", func(b *ir.Block, x, y ir.I32) ir.I1 { return b.I32.UMulO(x, y) }},
		{"_osadd32", func(b *ir.Block, x, y ir.I32) ir.I1 { return b.I32.SAddO(x, y) }},
	} {
		fn := m.Func(tc.name).Export()
		x := fn.ParamI32("x")
		y := fn.ParamI32("y")
		fn.ReturnsI32()
		entry := fn.Entry()
		entry.Return(entry.I32.ZExtI1(tc.emit(entry, x, y)))
	}

	got := runNative(t, m, `
#include <stdio.h>
int osadd(long,long), ouadd(unsigned long,unsigned long), ossub(long,long);
int osmul(long,long), oumul(unsigned long,unsigned long);
int osmul32(int,int), oumul32(unsigned,unsigned), osadd32(int,int);
static int fail = 0;
static void chk(const char *what, int got, int want) {
    if (got != want) { printf("%s: got %d want %d\n", what, got, want); fail = 1; }
}
#define MAX 9223372036854775807L
#define MIN (-MAX - 1)
int main(void) {
    long lo; unsigned long ulo; int io; unsigned uio;
    long sv[] = {0, 1, -1, 2, -2, 3, MAX, MIN, MAX-1, MIN+1, 1L<<31, -(1L<<31), 1L<<32};
    int n = sizeof sv / sizeof *sv;
    for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) {
        long a = sv[i], b = sv[j];
        chk("saddo", osadd(a,b), __builtin_add_overflow(a,b,&lo));
        chk("ssubo", ossub(a,b), __builtin_sub_overflow(a,b,&lo));
        chk("smulo", osmul(a,b), __builtin_mul_overflow(a,b,&lo));
        unsigned long ua = (unsigned long)a, ub = (unsigned long)b;
        chk("uaddo", ouadd(ua,ub), __builtin_add_overflow(ua,ub,&ulo));
        chk("umulo", oumul(ua,ub), __builtin_mul_overflow(ua,ub,&ulo));
    }
    int iv[] = {0, 1, -1, 2, -2, 65536, -65536, 2147483647, -2147483647-1, 46341, -46341};
    int m = sizeof iv / sizeof *iv;
    for (int i = 0; i < m; i++) for (int j = 0; j < m; j++) {
        int a = iv[i], b = iv[j];
        chk("smulo32", osmul32(a,b), __builtin_mul_overflow(a,b,&io));
        chk("saddo32", osadd32(a,b), __builtin_add_overflow(a,b,&io));
        unsigned ua = (unsigned)a, ub = (unsigned)b;
        chk("umulo32", oumul32(ua,ub), __builtin_mul_overflow(ua,ub,&uio));
    }
    printf("%s\n", fail ? "MISMATCH" : "ok");
    return 0;
}
`)
	if got != "ok\n" {
		t.Errorf("printed %q, want %q", got, "ok\n")
	}
}

// §C4 and §D3's diff, which on a 64-bit-pointer target are a move and a
// subtraction.
func TestRunPointerArithmetic(t *testing.T) {
	m := ir.NewModule("t", ir.AArch64Linux)
	fn := m.Func("_pdiff").Export()
	p := fn.ParamPtr("p")
	q := fn.ParamPtr("q")
	fn.ReturnsI64()
	entry := fn.Entry()
	// Round-tripped through an integer on the way, which is §C4 both ways.
	back := entry.Ptr.FromI64(entry.I64.FromPtr(p))
	entry.Return(entry.Ptr.Diff(back, q))

	got := runNative(t, m, `
#include <stdio.h>
long pdiff(char *, char *);
int main(void) {
    char buf[64];
    printf("%ld %ld\n", pdiff(buf + 40, buf + 8), pdiff(buf + 8, buf + 40));
    return 0;
}
`)
	if got != "32 -32\n" {
		t.Errorf("printed %q, want %q", got, "32 -32\n")
	}
}

// §A's two wide multiplies, at the width A64 has no instruction for.
//
// SMULH and UMULH answer the 64-bit rows outright. At thirty-two there is
// nothing to answer with and nothing needed: the whole product fits in a
// register, so this is a widening multiply and a shift — and what it has to
// get right is which extension goes with which verb, the signed high half
// being the high half of a signed product and not of the same bits read
// either way. Against C, where the same product is one cast.
func TestRunWideMultiply(t *testing.T) {
	m := ir.NewModule("t", ir.AArch64Linux)

	s32 := m.Func("_smh32").Export()
	s32a, s32b := s32.ParamI32("a"), s32.ParamI32("b")
	s32.ReturnsI32()
	e1 := s32.Entry()
	e1.Return(e1.I32.SMulHi(s32a, s32b))

	u32 := m.Func("_umh32").Export()
	u32a, u32b := u32.ParamI32("a"), u32.ParamI32("b")
	u32.ReturnsI32()
	e2 := u32.Entry()
	e2.Return(e2.I32.UMulHi(u32a, u32b))

	s64 := m.Func("_smh64").Export()
	s64a, s64b := s64.ParamI64("a"), s64.ParamI64("b")
	s64.ReturnsI64()
	e3 := s64.Entry()
	e3.Return(e3.I64.SMulHi(s64a, s64b))

	got := runNative(t, m, `
#include <stdio.h>
int smh32(int, int), umh32(int, int);
long smh64(long, long);
static int fail = 0;
static void chk(const char *what, long got, long want) {
    if (got != want) { printf("%s: got %ld want %ld\n", what, got, want); fail = 1; }
}
static int refs32(int a, int b) { return (int)(((long long)a * (long long)b) >> 32); }
static unsigned refu32(unsigned a, unsigned b) {
    return (unsigned)(((unsigned long long)a * (unsigned long long)b) >> 32);
}
int main(void) {
    /* The pairs that tell the two apart: a negative operand's high half is
       all ones under one reading and nearly all ones under the other. */
    int sv[] = {0, 1, -1, 2, -2, 65536, -65536, 2147483647, -2147483647-1,
                123456789, -123456789, 0x5a5a5a5a, (int)0xa5a5a5a5};
    for (unsigned i = 0; i < sizeof sv / sizeof *sv; i++)
        for (unsigned j = 0; j < sizeof sv / sizeof *sv; j++) {
            chk("smulhi32", smh32(sv[i], sv[j]), refs32(sv[i], sv[j]));
            chk("umulhi32", (unsigned)umh32(sv[i], sv[j]),
                            refu32((unsigned)sv[i], (unsigned)sv[j]));
        }

    chk("smulhi64 zero", smh64(0, 12345), 0);
    chk("smulhi64 small", smh64(3, 5), 0);
    chk("smulhi64 neg", smh64(-1, 1), -1);
    chk("smulhi64 big", smh64(0x100000000L, 0x100000000L), 1);
    printf("%s\n", fail ? "MISMATCH" : "ok");
    return 0;
}
`)
	if got != "ok\n" {
		t.Errorf("printed %q, want %q", got, "ok\n")
	}
}

// §2's three symbolic constants, against C's own sizeof, alignof and
// offsetof — which is the whole point of having them. A frontend that writes
// offsetof rather than a number is asking the target the same question the C
// compiler beside it asks, and this checks that the two answers agree.
//
// Both places one can appear: a const instruction, and a global's literal
// initializer.
func TestRunSymbolicConstants(t *testing.T) {
	m := ir.NewModule("t", ir.AArch64Linux)

	// { i8; i64; i32 }: padding after the byte and at the end, which is
	// where a hand-computed offset goes wrong.
	rec := m.Struct("rec")
	rec.Field("c", ir.StoreI8.FType())
	rec.Field("q", ir.StoreI64.FType())
	rec.Field("n", ir.StoreI32.FType())

	// An array of them, so an index step has a size to multiply.
	arr := m.TypeOf("arr", ir.Array(4, rec.FType()))

	for _, tc := range []struct {
		name string
		c    ir.Const
	}{
		{"_szrec", ir.SizeOf(rec)},
		{"_alrec", ir.AlignOf(rec)},
		{"_szarr", ir.SizeOf(arr)},
		{"_offc", ir.OffsetOf(rec, ir.FieldPath("c"))},
		{"_offq", ir.OffsetOf(rec, ir.FieldPath("q"))},
		{"_offn", ir.OffsetOf(rec, ir.FieldPath("n"))},
		{"_off2q", ir.OffsetOf(arr, ir.IndexPath(2), ir.FieldPath("q"))},
		// One past the end, which is the address &arr[4] names.
		{"_offend", ir.OffsetOf(arr, ir.IndexPath(4))},
	} {
		fn := m.Func(tc.name).Export()
		fn.ReturnsI64()
		e := fn.Entry()
		e.Return(e.I64.ConstOf(tc.c))
	}

	// The same question asked of an initializer, where the answer is a
	// byte pattern rather than an immediate.
	g := m.Global("gsz", ir.RW, ir.StoreI64.FType()).Export().
		Init(ir.Lit(ir.SizeOf(rec)))
	rd := m.Func("_readsz").Export()
	rd.ReturnsI64()
	er := rd.Entry()
	er.Return(er.I64.Load(er.Ptr.GetAddr(g)))

	// sizeof of a global rather than of a type, which is the other spelling.
	sym := m.Func("_szglobal").Export()
	sym.ReturnsI64()
	es := sym.Entry()
	es.Return(es.I64.ConstOf(ir.SizeOfSym(g)))

	got := runNative(t, m, `
#include <stddef.h>
#include <stdio.h>
struct rec { char c; long q; int n; };
long szrec(void), alrec(void), szarr(void), offc(void), offq(void), offn(void);
long off2q(void), offend(void), readsz(void), szglobal(void);
static int fail = 0;
static void chk(const char *what, long got, long want) {
    if (got != want) { printf("%s: got %ld want %ld\n", what, got, want); fail = 1; }
}
int main(void) {
    chk("sizeof",   szrec(), (long)sizeof(struct rec));
    chk("alignof",  alrec(), (long)_Alignof(struct rec));
    chk("sizeof[]", szarr(), (long)sizeof(struct rec[4]));
    chk("offsetof c", offc(), (long)offsetof(struct rec, c));
    chk("offsetof q", offq(), (long)offsetof(struct rec, q));
    chk("offsetof n", offn(), (long)offsetof(struct rec, n));
    chk("offsetof [2].q", off2q(),
        (long)(2 * sizeof(struct rec) + offsetof(struct rec, q)));
    chk("offsetof [4]", offend(), (long)(4 * sizeof(struct rec)));
    chk("initializer", readsz(), (long)sizeof(struct rec));
    chk("sizeof @g",   szglobal(), (long)sizeof(long));
    printf("%s\n", fail ? "MISMATCH" : "ok");
    return 0;
}
`)
	if got != "ok\n" {
		t.Errorf("printed %q, want %q", got, "ok\n")
	}
}
