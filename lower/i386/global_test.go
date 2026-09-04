package i386_test

// §5, which is now the shared walk in lower/globals plus this backend's
// layout table. What is worth testing here is the part that is this
// backend's: four-byte addresses, and an eightbyte that aligns to four.

import (
	"testing"

	"github.com/vertex-language/ir"
)

// A global read through its address, and a second global holding the first
// one's address — which is the four-byte relocation this target's §5 differs
// by.
func TestRunGlobals(t *testing.T) {
	m := ir.NewModule("t", ir.I386Linux)
	answer := m.Global("answer", ir.RW, ir.StoreI32.FType()).
		Init(ir.Lit(ir.Int(42))).Export()
	wide := m.Global("wide", ir.RW, ir.StoreI64.FType()).
		Init(ir.Lit(ir.Int(0x123456789abcdef0))).Export()
	// A pointer to another global, resolved by the linker.
	ptr := m.Global("ptr", ir.RW, ir.StorePtr.FType()).
		Init(ir.RelocInit(answer)).Export()
	// Zeroed, which belongs in .bss.
	m.Global("blank", ir.RW, ir.StoreI32.FType()).Export()

	get := m.Func("getanswer").Export()
	get.ReturnsI32()
	e := get.Entry()
	e.Return(e.I32.Load(e.Ptr.GetAddr(answer)))

	getw := m.Func("getwide").Export()
	getw.ReturnsI64()
	e2 := getw.Entry()
	e2.Return(e2.I64.Load(e2.Ptr.GetAddr(wide)))

	// Through the pointer global: load the pointer, then load through it.
	ind := m.Func("indirect").Export()
	ind.ReturnsI32()
	e3 := ind.Entry()
	e3.Return(e3.I32.Load(e3.Ptr.Load(e3.Ptr.GetAddr(ptr))))

	wantOK(t, m, `
int getanswer(void), indirect(void);
long long getwide(void);
extern int answer, blank;
extern long long wide;
static void body(void) {
    chk32("global", (unsigned)getanswer(), 42u);
    chk64("global 64", getwide(), 0x123456789abcdef0ULL);
    chk32("through pointer", (unsigned)indirect(), 42u);
    chk32("bss is zero", (unsigned)blank, 0u);
    /* And the C side agrees about where they are. */
    chk32("same object", (unsigned)answer, 42u);
}
`)
}

// An eightbyte aligns to four here, not eight — which is the one number this
// target's layout table differs by, and it changes a struct's size.
func TestLowerI386Alignment(t *testing.T) {
	m := ir.NewModule("t", ir.I386Linux)
	st := m.Struct("pair")
	st.Field("a", ir.StoreI32.FType())
	st.Field("b", ir.StoreI64.FType())

	m.Global("p", ir.RW, st.FType()).
		Init(ir.List(ir.Lit(ir.Int(1)), ir.Lit(ir.Int(2)))).Export()

	fn := m.Func("readpair").Export()
	fn.ReturnsI64()
	e := fn.Entry()
	base := e.Ptr.GetAddr(m.Lookup("p"))
	// The i64 field sits at offset four here and at eight on the other two
	// targets, which is the whole of what this checks.
	e.Return(e.I64.Load(e.Ptr.Add(base, e.I64.Const(4))))

	wantOK(t, m, `
long long readpair(void);
static void body(void) { chk64("i64 at offset 4", readpair(), 2ULL); }
`)
}

// A typedef'd type, which one backend could not lay out at all until the
// shared walk replaced its own alias resolution.
func TestLowerTypedefGlobal(t *testing.T) {
	m := ir.NewModule("t", ir.I386Linux)
	word := m.TypeOf("word", ir.StoreI32.FType())
	m.Global("w", ir.RW, word.FType()).Init(ir.Lit(ir.Int(7))).Export()

	fn := m.Func("getw").Export()
	fn.ReturnsI32()
	e := fn.Entry()
	e.Return(e.I32.Load(e.Ptr.GetAddr(m.Lookup("w"))))

	wantOK(t, m, `
int getw(void);
static void body(void) { chk32("typedef", (unsigned)getw(), 7u); }
`)
}

// §2's symbolic constants, on the target that answers them differently.
//
// This is what the constants are for. The same struct is laid out one way
// here and another on the 64-bit targets, because an eightbyte aligns to
// four in the Intel386 psABI — so a frontend that wrote the offset as a
// number would be right on one machine and wrong on this one, and one that
// writes offsetof is right on both. Checked against the C compiler's own
// answers for the same target.
func TestRunSymbolicConstants(t *testing.T) {
	m := ir.NewModule("t", ir.I386Linux)

	rec := m.Struct("rec")
	rec.Field("c", ir.StoreI8.FType())
	rec.Field("q", ir.StoreI64.FType())
	rec.Field("n", ir.StoreI32.FType())
	arr := m.TypeOf("arr", ir.Array(4, rec.FType()))

	for _, tc := range []struct {
		name string
		c    ir.Const
	}{
		{"szrec", ir.SizeOf(rec)},
		{"alrec", ir.AlignOf(rec)},
		{"offq", ir.OffsetOf(rec, ir.FieldPath("q"))},
		{"offn", ir.OffsetOf(rec, ir.FieldPath("n"))},
		{"off2q", ir.OffsetOf(arr, ir.IndexPath(2), ir.FieldPath("q"))},
	} {
		fn := m.Func(tc.name).Export()
		fn.ReturnsI32()
		e := fn.Entry()
		e.Return(e.I32.ConstOf(tc.c))
	}

	g := m.Global("gsz", ir.RW, ir.StoreI32.FType()).Export().
		Init(ir.Lit(ir.SizeOf(rec)))
	rd := m.Func("readsz").Export()
	rd.ReturnsI32()
	er := rd.Entry()
	er.Return(er.I32.Load(er.Ptr.GetAddr(g)))

	wantOK(t, m, `
struct rec { char c; long long q; int n; };
unsigned szrec(void), alrec(void), offq(void), offn(void), off2q(void), readsz(void);
static void body(void) {
    /* Four, not eight: the whole difference this target makes. */
    chk32("offsetof q", offq(), (unsigned)__builtin_offsetof(struct rec, q));
    chk32("offsetof n", offn(), (unsigned)__builtin_offsetof(struct rec, n));
    chk32("sizeof",     szrec(), (unsigned)sizeof(struct rec));
    chk32("alignof",    alrec(), (unsigned)_Alignof(struct rec));
    chk32("offsetof [2].q", off2q(),
          (unsigned)(2 * sizeof(struct rec) + __builtin_offsetof(struct rec, q)));
    chk32("initializer", readsz(), (unsigned)sizeof(struct rec));
}
`)
}

// A reloc initializer with a symbolic displacement: &arr[2], which §5 spells
// as the array's symbol plus an offsetof.
//
// This is the case Init.Plus was written for, and it used to be silently
// wrong: the walk resolved only an integer addend and answered zero for
// anything else, so a pointer meant to name the third element named the
// first. Read through end to end here, since a relocation that lands in the
// wrong place is not visible in the bytes on either side of the link.
func TestRunRelocAddend(t *testing.T) {
	m := ir.NewModule("t", ir.I386Linux)

	arrTy := m.TypeOf("arr", ir.Array(4, ir.StoreI32.FType()))
	arr := m.Global("arr", ir.RW, arrTy.FType()).Export().
		Init(ir.List(
			ir.Lit(ir.Int(10)), ir.Lit(ir.Int(20)),
			ir.Lit(ir.Int(30)), ir.Lit(ir.Int(40))))

	third := m.Global("third", ir.RW, ir.StorePtr.FType()).Export().
		Init(ir.RelocInit(arr).Plus(ir.OffsetOf(arrTy, ir.IndexPath(2))))
	plain := m.Global("first", ir.RW, ir.StorePtr.FType()).Export().
		Init(ir.RelocInit(arr))

	read := func(name string, g *ir.Global) {
		fn := m.Func(name).Export()
		fn.ReturnsI32()
		e := fn.Entry()
		e.Return(e.I32.Load(e.Ptr.Load(e.Ptr.GetAddr(g))))
	}
	read("readthird", third)
	read("readfirst", plain)

	wantOK(t, m, `
unsigned readthird(void), readfirst(void);
static void body(void) {
    chk32("no addend", readfirst(), 10u);
    chk32("offsetof addend", readthird(), 30u);
}
`)
}
