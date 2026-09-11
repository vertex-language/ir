package arm64_test

// §G3, across a real link: a function this backend lowered catches an
// exception a real unwinder threw at it.
//
// The C++ ABI is the one available here. libc++abi's throw path is callable
// from C — __cxa_allocate_exception and __cxa_throw are ordinary functions,
// and typeinfo for int is a symbol — so the test throws a C++ int, catches it
// in a lowered pad block, and reads the value back out. The personality is
// __gxx_personality_v0, which is the same table reader every other Itanium
// personality is; nothing in the tables is C++'s.

import (
	"os/exec"
	"strings"
	"testing"

	"github.com/vertex-language/ir"
	arm64lower "github.com/vertex-language/ir/lower/arm64"
)

// cxaSig builds the declarations a throw and a catch need.
type cxa struct {
	beginCatch ir.Callee
	endCatch   ir.Callee
	thrower    ir.Callee
	personal   ir.Callee
	typeInfoI  ir.Symbol
}

func declareCxa(m *ir.Module) cxa {
	return cxa{
		beginCatch: m.ImportFunc("___cxa_begin_catch", ir.NewSig().Param(ir.TypePtr).Ret(ir.TypePtr)),
		endCatch:   m.ImportFunc("___cxa_end_catch", ir.NewSig()),
		thrower:    m.ImportFunc("_thrower", ir.NewSig().Param(ir.TypeI32)),
		personal:   m.ImportFunc("___gxx_personality_v0", ir.NewSig()),
		typeInfoI:  m.ImportGlobal("__ZTIi", ir.StorePtr.FType()),
	}
}

// TestRunInvokeCatches is the whole of it: invoke, a pad with one catch
// clause, the selector it reports, and resume on the path nothing matched.
func TestRunInvokeCatches(t *testing.T) {
	m := ir.NewModule("t", ir.AArch64MacOS)
	c := declareCxa(m)

	// int guarded(int a): calls thrower(a), which throws when a is odd.
	// Returns a * 10 normally and the caught value negated on a throw.
	fn := m.Func("_guarded").Export().Personality(c.personal)
	a := fn.ParamI32("a")
	fn.ReturnsI32()

	entry := fn.Entry()
	ok := fn.Block("ok")
	pad := fn.Pad("pad", ir.Catch(c.typeInfoI))

	entry.Invoke(c.thrower, []ir.Value{a}, ok.To(), pad)

	ok.Return(ok.I32.Mul(a, ok.I32.Const(10)))

	// The exception object is the C++ one; __cxa_begin_catch hands back the
	// int it wraps.
	p := pad.Call(c.beginCatch, pad.Exn()).Value(0).(ir.Ptr)
	v := pad.I32.Load(p)
	pad.Call(c.endCatch)
	pad.Return(pad.I32.Sub(pad.I32.Const(0), v))

	got := runNativeCxx(t, m, `
#include <stdio.h>

extern void *__cxa_allocate_exception(unsigned long);
extern void __cxa_throw(void *, void *, void (*)(void *));
extern void *_ZTIi;

int guarded(int);

void thrower(int a) {
	if ((a & 1) == 0) return;
	int *p = (int *)__cxa_allocate_exception(sizeof(int));
	*p = a * 100;
	__cxa_throw(p, &_ZTIi, 0);
}

int main(void) {
	printf("%d %d\n", guarded(4), guarded(7));
	return 0;
}
`)
	// 4 is even, so nothing throws: 40. 7 throws 700, caught and negated.
	if want := "40 -700\n"; got != want {
		t.Errorf("printed %q, want %q", got, want)
	}
}

// TestRunInvokeCleanupResumes: a cleanup pad runs and hands the exception
// back, and the frame above it — also lowered here — is the one that catches.
//
// Two lowered frames rather than one, because resume is only observable from
// above: what it does is decline to handle, and something has to be there to
// handle instead.
func TestRunInvokeCleanupResumes(t *testing.T) {
	m := ir.NewModule("t", ir.AArch64MacOS)
	c := declareCxa(m)
	note := m.ImportFunc("_note", ir.NewSig().Param(ir.TypeI32))

	in := m.Func("_inner").Export().Personality(c.personal)
	ia := in.ParamI32("a")
	in.ReturnsI32()
	iok := in.Block("ok")
	ipad := in.Pad("pad", ir.Cleanup)
	in.Entry().Invoke(c.thrower, []ir.Value{ia}, iok.To(), ipad)
	iok.Return(ia)
	ipad.Call(note, ipad.I32.Const(1))
	ipad.Resume(ipad.Exn())

	out := m.Func("_outer").Export().Personality(c.personal)
	oa := out.ParamI32("a")
	out.ReturnsI32()
	ook := out.Block("ok")
	// The call's result is the normal target's trailing parameter (§14).
	res := ook.ParamI32("r")
	opad := out.Pad("pad", ir.Catch(c.typeInfoI))
	out.Entry().Invoke(in, []ir.Value{oa}, ook.To(), opad)
	ook.Return(res)
	p := opad.Call(c.beginCatch, opad.Exn()).Value(0).(ir.Ptr)
	v := opad.I32.Load(p)
	opad.Call(c.endCatch)
	opad.Return(opad.I32.Sub(opad.I32.Const(0), v))

	got := runNativeCxx(t, m, `
#include <stdio.h>

extern void *__cxa_allocate_exception(unsigned long);
extern void __cxa_throw(void *, void *, void (*)(void *));
extern void *_ZTIi;

int outer(int);
static int cleanups;

void note(int n) { cleanups += n; }

void thrower(int a) {
	if ((a & 1) == 0) return;
	int *p = (int *)__cxa_allocate_exception(sizeof(int));
	*p = a * 100;
	__cxa_throw(p, &_ZTIi, 0);
}

int main(void) {
	int quiet = outer(4);
	int noisy = outer(7);
	printf("%d %d %d\n", quiet, noisy, cleanups);
	return 0;
}
`)
	if want := "4 -700 1\n"; got != want {
		t.Errorf("printed %q, want %q", got, want)
	}
}

// TestRunInvokeSelector: which clause matched, in a function with two pads.
//
// The second pad's clauses are the second half of the function's type table,
// so the raw value the personality reports is not a clause number and the pad
// has to translate it. See the selector note in lsda.go.
func TestRunInvokeSelector(t *testing.T) {
	m := ir.NewModule("t", ir.AArch64MacOS)
	c := declareCxa(m)
	tid := m.ImportGlobal("__ZTId", ir.StorePtr.FType())
	throw := m.ImportFunc("_throwKind", ir.NewSig().Param(ir.TypeI32))

	fn := m.Func("_which").Export().Personality(c.personal)
	a := fn.ParamI32("a")
	fn.ReturnsI32()

	mid := fn.Block("mid")
	ok := fn.Block("ok")
	// Two clauses each, in opposite orders, so a test that passed by
	// reading the type table's order rather than the clause list's would
	// give the other answer.
	padA := fn.Pad("padA", ir.Catch(c.typeInfoI), ir.Catch(tid))
	padB := fn.Pad("padB", ir.Catch(tid), ir.Catch(c.typeInfoI))

	fn.Entry().Invoke(throw, []ir.Value{fn.Entry().I32.Const(0)}, mid.To(), padA)
	mid.Invoke(throw, []ir.Value{a}, ok.To(), padB)
	ok.Return(ok.I32.Const(0))

	padA.Call(c.beginCatch, padA.Exn())
	padA.Call(c.endCatch)
	padA.Return(padA.I32.Add(padA.Sel(), padA.I32.Const(10)))

	padB.Call(c.beginCatch, padB.Exn())
	padB.Call(c.endCatch)
	padB.Return(padB.I32.Add(padB.Sel(), padB.I32.Const(20)))

	got := runNativeCxx(t, m, `
#include <stdio.h>

extern void *__cxa_allocate_exception(unsigned long);
extern void __cxa_throw(void *, void *, void (*)(void *));
extern void *_ZTIi;
extern void *_ZTId;

int which(int);

void throwKind(int a) {
	if (a == 1) {
		int *p = (int *)__cxa_allocate_exception(sizeof(int));
		*p = 0;
		__cxa_throw(p, &_ZTIi, 0);
	}
	if (a == 2) {
		double *p = (double *)__cxa_allocate_exception(sizeof(double));
		*p = 0;
		__cxa_throw(p, &_ZTId, 0);
	}
}

int main(void) {
	// padB catches a double with its first clause and an int with its
	// second: 21 and 22.
	printf("%d %d\n", which(2), which(1));
	return 0;
}
`)
	if want := "21 22\n"; got != want {
		t.Errorf("printed %q, want %q", got, want)
	}
}

// runNativeCxx is runNative with the C++ runtime on the link line, which is
// where the throw path and the personality routine live.
func runNativeCxx(t *testing.T, m *ir.Module, mainC string) string {
	t.Helper()
	bin := buildNative(t, m, mainC, "-lc++")
	out, err := exec.Command(bin).CombinedOutput()
	if err != nil {
		t.Fatalf("run: %v\n%s", err, out)
	}
	return string(out)
}

// TestFilterClauseRefused: C++'s exception specifications need a second table
// this does not write, and half of one is worse than none.
func TestFilterClauseRefused(t *testing.T) {
	m := ir.NewModule("t", ir.AArch64MacOS)
	c := declareCxa(m)
	fn := m.Func("_f").Export().Personality(c.personal)
	a := fn.ParamI32("a")
	fn.ReturnsI32()
	ok := fn.Block("ok")
	pad := fn.Pad("pad", ir.Filter(c.typeInfoI))
	fn.Entry().Invoke(c.thrower, []ir.Value{a}, ok.To(), pad)
	ok.Return(a)
	pad.Resume(pad.Exn())

	_, err := lowerMacOS(m)
	if err == nil || !strings.Contains(err.Error(), "filter") {
		t.Fatalf("Lower: %v; want a refusal naming the filter clause", err)
	}
}

// TestPadRefusedOffDarwin: the tables are Mach-O's, and an ELF object would
// link and then fail to unwind.
func TestPadRefusedOffDarwin(t *testing.T) {
	m := ir.NewModule("t", ir.AArch64Linux)
	c := declareCxa(m)
	fn := m.Func("_f").Export().Personality(c.personal)
	a := fn.ParamI32("a")
	fn.ReturnsI32()
	ok := fn.Block("ok")
	pad := fn.Pad("pad", ir.Cleanup)
	fn.Entry().Invoke(c.thrower, []ir.Value{a}, ok.To(), pad)
	ok.Return(a)
	pad.Resume(pad.Exn())

	_, err := arm64lower.Lower(m, arm64lower.Options{})
	if err == nil || !strings.Contains(err.Error(), "unwind tables") {
		t.Fatalf("Lower: %v; want a refusal naming the missing tables", err)
	}
}

func lowerMacOS(m *ir.Module) (any, error) {
	return arm64lower.Lower(m, arm64lower.Options{
		LibcallPrefix: "_", Variadic: arm64lower.VariadicDarwin,
	})
}
