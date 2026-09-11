package arm64_test

// Compact unwind: what the backend writes into __LD,__compact_unwind, and
// whether an unwinder can walk a frame it describes.
//
// The two halves check different things and neither is enough alone. The
// encoding test pins the word, which is the part a reader can check against
// libunwind's header. The backtrace test proves the word and the frame agree
// — a correct encoding over a frame that puts its saves somewhere else is a
// register restored from the wrong eight bytes, and nothing static catches
// that.

import (
	"encoding/binary"
	"strings"
	"testing"

	arm64obj "github.com/vertex-language/arm64/obj"

	"github.com/vertex-language/ir"
	arm64lower "github.com/vertex-language/ir/lower/arm64"
	"github.com/vertex-language/ir/verify"
)

// compactUnwind lowers m for Darwin and returns one 32-byte record per
// function, in the order the functions were emitted.
func compactUnwind(t *testing.T, m *ir.Module) [][]byte {
	t.Helper()
	if err := verify.Module(m); err != nil {
		t.Fatalf("verify.Module: %v", err)
	}
	o, err := arm64lower.Lower(m, arm64lower.Options{
		LibcallPrefix: "_",
		Variadic:      arm64lower.VariadicDarwin,
	})
	if err != nil {
		t.Fatalf("Lower: %v", err)
	}
	var sec *arm64obj.Section
	for _, s := range o.Sections() {
		if strings.HasPrefix(s.Name(), "__LD,__compact_unwind") {
			sec = s
		}
	}
	if sec == nil {
		t.Fatal("no __LD,__compact_unwind section")
	}
	b := sec.Bytes()
	if len(b)%32 != 0 {
		t.Fatalf("the section is %d bytes, which is not a whole number of records", len(b))
	}
	var out [][]byte
	for i := 0; i < len(b); i += 32 {
		out = append(out, b[i:i+32])
	}
	return out
}

// enc is a record's encoding word; length is its length field.
func enc(rec []byte) uint32    { return binary.LittleEndian.Uint32(rec[12:]) }
func length(rec []byte) uint32 { return binary.LittleEndian.Uint32(rec[8:]) }

const (
	modeFrameless = 0x02000000
	modeFrame     = 0x04000000
	pairX19X20    = 0x1
	pairX21X22    = 0x2
	pairD8D9      = 0x100
)

// TestCompactUnwindEncoding: a leaf that touches nothing is frameless; a
// function holding values across a call names the integer pairs it saved; one
// holding a float across a call names a vector pair.
//
// The pairs are what the bits are, so a function that needs only X19 still
// reports X19_X20 — there is no bit for half a pair. See unwind.go.
func TestCompactUnwindEncoding(t *testing.T) {
	m := ir.NewModule("t", ir.AArch64MacOS)
	sink := m.ImportFunc("_sink", ir.NewSig().Param(ir.TypeI64).Ret(ir.TypeI64))

	// A leaf: one add and a return, no call, no frame.
	leaf := m.Func("_leaf").Export()
	a := leaf.ParamI64("a")
	leaf.ReturnsI64()
	leaf.Entry().Return(leaf.Entry().I64.Add(a, leaf.Entry().I64.Const(1)))

	// One value live across one call: X19 and therefore the X19/X20 pair.
	one := m.Func("_one").Export()
	oa := one.ParamI64("a")
	one.ReturnsI64()
	oe := one.Entry()
	oe.Return(oe.I64.Add(oe.Call(sink, oa).Value(0).(ir.I64), oa))

	// Three live across a call, which needs a second pair.
	three := m.Func("_three").Export()
	ta := three.ParamI64("a")
	three.ReturnsI64()
	te := three.Entry()
	x := te.I64.Add(ta, te.I64.Const(1))
	y := te.I64.Add(ta, te.I64.Const(2))
	z := te.I64.Add(ta, te.I64.Const(3))
	r := te.Call(sink, ta).Value(0).(ir.I64)
	te.Return(te.I64.Add(te.I64.Add(x, y), te.I64.Add(z, r)))

	// A float live across a call: the vector file's first pair.
	fsink := m.ImportFunc("_fsink", ir.NewSig().Param(ir.TypeI64).Ret(ir.TypeI64))
	flt := m.Func("_flt").Export()
	fa := flt.ParamF64("a")
	flt.ReturnsF64()
	fe := flt.Entry()
	kept := fe.F64.Add(fa, fe.F64.Const(1))
	fe.Call(fsink, fe.I64.Const(7))
	fe.Return(fe.F64.Add(kept, kept))

	recs := compactUnwind(t, m)
	if len(recs) != 4 {
		t.Fatalf("%d records, want 4", len(recs))
	}
	want := []uint32{
		modeFrameless,
		modeFrame | pairX19X20,
		modeFrame | pairX19X20 | pairX21X22,
		modeFrame | pairD8D9,
	}
	names := []string{"_leaf", "_one", "_three", "_flt"}
	for i, w := range want {
		if got := enc(recs[i]); got != w {
			t.Errorf("%s: encoding 0x%08x, want 0x%08x", names[i], got, w)
		}
		if length(recs[i]) == 0 {
			t.Errorf("%s: length 0", names[i])
		}
	}
}

// TestCompactUnwindLengthsTile: the records cover .text end to end with no
// gap, which is what makes the linker's range table a partition rather than a
// list of islands. A function with no record is not simply undescribed — the
// record before it claims its bytes.
func TestCompactUnwindLengthsTile(t *testing.T) {
	m := ir.NewModule("t", ir.AArch64MacOS)
	sink := m.ImportFunc("_sink", ir.NewSig().Param(ir.TypeI64).Ret(ir.TypeI64))
	for _, name := range []string{"_a", "_b", "_c"} {
		fn := m.Func(name).Export()
		p := fn.ParamI64("a")
		fn.ReturnsI64()
		e := fn.Entry()
		e.Return(e.I64.Add(e.Call(sink, p).Value(0).(ir.I64), p))
	}

	if err := verify.Module(m); err != nil {
		t.Fatalf("verify.Module: %v", err)
	}
	o, err := arm64lower.Lower(m, arm64lower.Options{
		LibcallPrefix: "_", Variadic: arm64lower.VariadicDarwin,
	})
	if err != nil {
		t.Fatalf("Lower: %v", err)
	}
	var total uint32
	for _, rec := range compactUnwind(t, m) {
		total += length(rec)
	}
	text := o.SectionNamed(".text")
	if text == nil {
		t.Fatal("no .text")
	}
	if int(total) != text.Size() {
		t.Errorf("the records describe %d bytes; .text is %d", total, text.Size())
	}
}

// TestNoCompactUnwindOffDarwin: the section is Mach-O's, and an ELF link has
// no use for it — __LD is a segment, and segments are a Mach-O idea.
func TestNoCompactUnwindOffDarwin(t *testing.T) {
	m := ir.NewModule("t", ir.AArch64Linux)
	fn := m.Func("f").Export()
	a := fn.ParamI64("a")
	fn.ReturnsI64()
	fn.Entry().Return(fn.Entry().I64.Add(a, a))

	o, err := arm64lower.Lower(m, arm64lower.Options{})
	if err != nil {
		t.Fatalf("Lower: %v", err)
	}
	for _, s := range o.Sections() {
		if strings.Contains(s.Name(), "compact_unwind") {
			t.Errorf("an ELF object has a %s section", s.Name())
		}
	}
}

// TestRunUnwindThroughFrame walks the stack from inside a call this backend
// made, with libunwind and nothing else.
//
// _Unwind_Backtrace is the whole point: it does not read a frame pointer
// chain, it reads __unwind_info, so a frame this backend did not describe —
// or described wrongly — stops the walk or hands back the wrong registers.
//
// The register check is the sharper half. mid() holds a sentinel in X19
// across the call, so by the time the walk reaches mid's frame the value has
// had to survive being spilled by _clobber and put back by the unwinder out
// of the slot the encoding claims it is in. An encoding that named the right
// pair over a frame that stored it somewhere else reads eight bytes of
// something else.
func TestRunUnwindThroughFrame(t *testing.T) {
	m := ir.NewModule("t", ir.AArch64MacOS)
	probe := m.ImportFunc("_probe", ir.NewSig().Param(ir.TypeI64).Ret(ir.TypeI64))

	fn := m.Func("_clobber").Export()
	a := fn.ParamI64("a")
	fn.ReturnsI64()
	e := fn.Entry()
	// Four values live across the call, so the allocator has to take the
	// first two callee-saved pairs and the prologue has to store them.
	x := e.I64.Add(a, e.I64.Const(1))
	y := e.I64.Add(a, e.I64.Const(2))
	z := e.I64.Add(a, e.I64.Const(3))
	w := e.I64.Add(a, e.I64.Const(4))
	r := e.Call(probe, a).Value(0).(ir.I64)
	e.Return(e.I64.Add(e.I64.Add(x, y), e.I64.Add(z, e.I64.Add(w, r))))

	got := runNative(t, m, `
#include <stdio.h>
#include <unwind.h>

#define GUARD 0x1919191919191919ULL

static int reached_mid, reached_main;
static unsigned long long in_mid;

long mid(long);
int main(void);

static _Unwind_Reason_Code trace(struct _Unwind_Context *ctx, void *arg) {
	(void)arg;
	void *fn = _Unwind_FindEnclosingFunction((void *)_Unwind_GetIP(ctx));
	if (fn == (void *)&mid) {
		reached_mid = 1;
		in_mid = (unsigned long long)_Unwind_GetGR(ctx, 19);
	}
	if (fn == (void *)&main) reached_main = 1;
	return _URC_NO_REASON;
}

long probe(long x) { _Unwind_Backtrace(trace, 0); return x; }

// The call is written by hand because the sentinel has to be in X19 at the
// call and C has no way to say so: a local register variable binds only at an
// asm statement, and clang is free to keep it anywhere in between.
__attribute__((noinline)) long mid(long a) {
	long r;
	unsigned long long g = GUARD;
	__asm__ volatile(
		"mov x19, %1\n\t"
		"mov x0, %2\n\t"
		"bl _clobber\n\t"
		"mov %0, x0\n\t"
		: "=r"(r)
		: "r"(g), "r"(a)
		: "x19","x0","x1","x2","x3","x4","x5","x6","x7","x8","x9","x10",
		  "x11","x12","x13","x14","x15","x16","x17","lr","memory");
	return r;
}

int main(void) {
	long r = mid(10);
	printf("%ld %d %d %d\n", r, reached_mid, reached_main, in_mid == GUARD);
	return 0;
}
`)
	// 10 + (11 + 12) + (13 + 14) = 60, and three ones for the three checks.
	if want := "60 1 1 1\n"; got != want {
		t.Errorf("printed %q, want %q", got, want)
	}
}
