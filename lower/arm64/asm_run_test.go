package arm64_test

// §G4, inline assembly, verified the strongest way this hardware allows: the
// template is lowered, allocated, assembled, linked against a C main and run.
//
// A byte comparison would not do here. The question an inline asm test has to
// answer is whether the register the allocator chose is the register the
// template ended up naming, and an expected-bytes test would have to encode
// the allocator's answer to check the assembler's — the same mistake twice.
// Running it asks the machine instead.

import (
	"strings"
	"testing"

	"github.com/vertex-language/ir"
	arm64lower "github.com/vertex-language/ir/lower/arm64"
	"github.com/vertex-language/ir/verify"
)

// lowerErr lowers and returns whatever went wrong, for the cases where the
// point is that something does.
func lowerErr(t *testing.T, m *ir.Module) error {
	t.Helper()
	if err := verify.Module(m); err != nil {
		t.Fatalf("verify.Module: %v", err)
	}
	_, err := arm64lower.Lower(m, arm64lower.Options{})
	return err
}

// The simplest shape: one output, no inputs, no clobbers.
func TestRunAsmOutputOnly(t *testing.T) {
	m := ir.NewModule("t", ir.AArch64MacOS)
	fn := m.Func("_answer").Export()
	fn.ReturnsI64()
	e := fn.Entry()
	r := e.Asm("mov %0, #42").Out(ir.TypeI64, ir.CStr("=r")).Emit()
	e.Return(r.I64(0))

	got := runNative(t, m, `
#include <stdio.h>
long answer(void);
int main(void) { printf("%ld\n", answer()); return 0; }
`)
	if got != "42\n" {
		t.Errorf("printed %q, want %q", got, "42\n")
	}
}

// Two inputs and one output, which is where the operand numbering has to be
// right: outputs are numbered before inputs.
func TestRunAsmAdd(t *testing.T) {
	m := ir.NewModule("t", ir.AArch64MacOS)
	fn := m.Func("_asmadd").Export()
	a := fn.ParamI64("a")
	b := fn.ParamI64("b")
	fn.ReturnsI64()
	e := fn.Entry()
	r := e.Asm("add %0, %1, %2").
		Out(ir.TypeI64, ir.CStr("=r")).
		In(a, ir.CReg).
		In(b, ir.CReg).
		Emit()
	e.Return(r.I64(0))

	got := runNative(t, m, `
#include <stdio.h>
long asmadd(long, long);
int main(void) { printf("%ld\n", asmadd(19, 23)); return 0; }
`)
	if got != "42\n" {
		t.Errorf("printed %q, want %q", got, "42\n")
	}
}

// A tied operand. One vreg stands for both the input and the output, and the
// input is copied into it first — so a value that is still live afterwards
// must survive, which is what the second use of `a` checks.
func TestRunAsmTiedOperand(t *testing.T) {
	m := ir.NewModule("t", ir.AArch64MacOS)
	fn := m.Func("_tied").Export()
	a := fn.ParamI64("a")
	fn.ReturnsI64()
	e := fn.Entry()
	r := e.Asm("add %0, %0, #1").
		Out(ir.TypeI64, ir.CStr("=r")).
		In(a, ir.CStr("0")).
		Emit()
	// a is still live: the answer is (a+1) + a, so a clobbered input shows up
	// as a wrong number rather than as a crash.
	e.Return(e.I64.Add(r.I64(0), a))

	got := runNative(t, m, `
#include <stdio.h>
long tied(long);
int main(void) { printf("%ld\n", tied(10)); return 0; }
`)
	if got != "21\n" {
		t.Errorf("printed %q, want %q", got, "21\n")
	}
}

// The %w modifier: one operand read at 32 bits.
func TestRunAsmWidthModifier(t *testing.T) {
	m := ir.NewModule("t", ir.AArch64MacOS)
	fn := m.Func("_wmod").Export()
	a := fn.ParamI32("a")
	fn.ReturnsI32()
	e := fn.Entry()
	r := e.Asm("add %w0, %w1, %w1").
		Out(ir.TypeI32, ir.CStr("=r")).
		In(a, ir.CReg).
		Emit()
	e.Return(r.I32(0))

	got := runNative(t, m, `
#include <stdio.h>
int wmod(int);
int main(void) { printf("%d\n", wmod(21)); return 0; }
`)
	if got != "42\n" {
		t.Errorf("printed %q, want %q", got, "42\n")
	}
}

// Clobbers, including a callee-saved register. x19 is callee-saved under
// AAPCS64, so naming it here has to reach the prologue: if the save is
// missing, the C caller's own use of x19 is destroyed and main returns
// garbage or crashes.
func TestRunAsmClobberCalleeSaved(t *testing.T) {
	m := ir.NewModule("t", ir.AArch64MacOS)
	fn := m.Func("_clob").Export()
	a := fn.ParamI64("a")
	fn.ReturnsI64()
	e := fn.Entry()
	r := e.Asm("mov x19, #1\n\tadd %0, %1, x19").
		Volatile().
		Out(ir.TypeI64, ir.CStr("=r")).
		In(a, ir.CReg).
		Clobber("x19").
		Emit()
	e.Return(r.I64(0))

	got := runNative(t, m, `
#include <stdio.h>
long clob(long);
long spin(long x) {
	// Keep something in a callee-saved register across the call, so a
	// missing save is observable.
	register long keep asm("x19") = 0x5eed;
	long r = clob(x);
	asm volatile("" :: "r"(keep));
	return r + (keep == 0x5eed ? 0 : 1000);
}
int main(void) { printf("%ld\n", spin(41)); return 0; }
`)
	if got != "42\n" {
		t.Errorf("printed %q, want %q", got, "42\n")
	}
}

// A local label inside a template, twice in one function. Both expansions
// contain `1:` and they are different labels; if the prefix did not
// distinguish them the second definition would collide with the first.
func TestRunAsmLocalLabelTwice(t *testing.T) {
	m := ir.NewModule("t", ir.AArch64MacOS)
	fn := m.Func("_twice").Export()
	a := fn.ParamI64("a")
	fn.ReturnsI64()
	e := fn.Entry()
	one := e.Asm("cbz %1, 1f\n\tmov %0, #10\n\tb 2f\n1:\tmov %0, #20\n2:").
		Out(ir.TypeI64, ir.CStr("=r")).In(a, ir.CReg).Emit()
	two := e.Asm("cbz %1, 1f\n\tmov %0, #1\n\tb 2f\n1:\tmov %0, #2\n2:").
		Out(ir.TypeI64, ir.CStr("=r")).In(a, ir.CReg).Emit()
	e.Return(e.I64.Add(one.I64(0), two.I64(0)))

	got := runNative(t, m, `
#include <stdio.h>
long twice(long);
int main(void) { printf("%ld %ld\n", twice(0), twice(5)); return 0; }
`)
	if got != "22 11\n" {
		t.Errorf("printed %q, want %q", got, "22 11\n")
	}
}

// One register written twice in the clobber list, once at each width. It is
// one register, and two vregs pinned to it would be two live values in one
// place — which the allocator refuses, correctly.
func TestRunAsmClobberSameRegisterTwice(t *testing.T) {
	m := ir.NewModule("t", ir.AArch64MacOS)
	fn := m.Func("_twoclob").Export()
	a := fn.ParamI64("a")
	fn.ReturnsI64()
	e := fn.Entry()
	r := e.Asm("mov w9, #2\n\tmul %0, %1, x9").
		Volatile().
		Out(ir.TypeI64, ir.CStr("=r")).
		In(a, ir.CReg).
		Clobber("x9", "w9").
		Emit()
	e.Return(r.I64(0))

	got := runNative(t, m, `
#include <stdio.h>
long twoclob(long);
int main(void) { printf("%ld\n", twoclob(21)); return 0; }
`)
	if got != "42\n" {
		t.Errorf("printed %q, want %q", got, "42\n")
	}
}

// A memory-constrained operand, spelled as an address.
func TestRunAsmMemoryOperand(t *testing.T) {
	m := ir.NewModule("t", ir.AArch64MacOS)
	fn := m.Func("_loadit").Export()
	p := fn.ParamPtr("p")
	fn.ReturnsI64()
	e := fn.Entry()
	r := e.Asm("ldr %0, %1").
		Volatile().
		Out(ir.TypeI64, ir.CStr("=r")).
		In(p, ir.CMem).
		Clobber("memory").
		Emit()
	e.Return(r.I64(0))

	got := runNative(t, m, `
#include <stdio.h>
long loadit(long *);
int main(void) { long v = 42; printf("%ld\n", loadit(&v)); return 0; }
`)
	if got != "42\n" {
		t.Errorf("printed %q, want %q", got, "42\n")
	}
}

// asm goto: the terminator form, branching into a block this package emitted.
func TestRunAsmGoto(t *testing.T) {
	m := ir.NewModule("t", ir.AArch64MacOS)
	fn := m.Func("_pick").Export()
	a := fn.ParamI64("a")
	fn.ReturnsI64()

	e := fn.Entry()
	fall := fn.Block("fall")
	taken := fn.Block("taken")

	e.AsmGoto("cbz %0, %l[taken]").In(a, ir.CReg).To(fall.To(), taken)
	fall.Return(fall.I64.Const(1))
	taken.Return(taken.I64.Const(2))

	got := runNative(t, m, `
#include <stdio.h>
long pick(long);
int main(void) { printf("%ld %ld\n", pick(0), pick(7)); return 0; }
`)
	if got != "2 1\n" {
		t.Errorf("printed %q, want %q", got, "2 1\n")
	}
}

// asm goto with outputs, which §14 binds to the fallthrough target's trailing
// parameters. The value is written by the assembled text on the edge where
// the text ran to the end, and is live only in the block that declares it.
//
// The template writes the output and then decides whether to branch, so the
// two paths differ in whether the value is ever read — which is the whole
// question the binding answers.
func TestRunAsmGotoWithOutputs(t *testing.T) {
	m := ir.NewModule("t", ir.AArch64MacOS)
	fn := m.Func("_classify").Export()
	a := fn.ParamI64("a")
	fn.ReturnsI64()

	e := fn.Entry()
	fall := fn.Block("fall")
	out := fall.ParamI64("out")
	taken := fn.Block("taken")

	e.AsmGoto("add %0, %1, #10\n\tcbz %1, %l[taken]").
		Out(ir.TypeI64, ir.CStr("=&r")).In(a, ir.CReg).To(fall.To(), taken)
	fall.Return(out)
	taken.Return(taken.I64.Const(-1))

	got := runNative(t, m, `
#include <stdio.h>
long classify(long);
int main(void) { printf("%ld %ld\n", classify(0), classify(7)); return 0; }
`)
	if got != "-1 17\n" {
		t.Errorf("printed %q, want %q", got, "-1 17\n")
	}
}

// The same, with an argument on the edge as well: the target's leading
// parameters are the branch's and the trailing ones are the asm's, and the
// two are told apart by position and nothing else.
func TestRunAsmGotoOutputAfterEdgeArgument(t *testing.T) {
	m := ir.NewModule("t", ir.AArch64MacOS)
	fn := m.Func("_both").Export()
	a := fn.ParamI64("a")
	fn.ReturnsI64()

	e := fn.Entry()
	fall := fn.Block("fall")
	carried := fall.ParamI64("carried")
	out := fall.ParamI64("out")
	taken := fn.Block("taken")

	e.AsmGoto("add %0, %1, #1\n\tcbz %1, %l[taken]").
		Out(ir.TypeI64, ir.CStr("=&r")).In(a, ir.CReg).To(fall.To(a), taken)
	fall.Return(fall.I64.Add(carried, out))
	taken.Return(taken.I64.Const(0))

	got := runNative(t, m, `
#include <stdio.h>
long both(long);
int main(void) { printf("%ld %ld\n", both(0), both(20)); return 0; }
`)
	if got != "0 41\n" {
		t.Errorf("printed %q, want %q", got, "0 41\n")
	}
}

// A bad template is a diagnostic, not a wrong object. This is the check that
// nothing in the pipeline was making.
func TestAsmBadTemplateIsRefused(t *testing.T) {
	m := ir.NewModule("t", ir.AArch64MacOS)
	fn := m.Func("_bad").Export()
	fn.ReturnsI64()
	e := fn.Entry()
	r := e.Asm("mov %0, %3").Out(ir.TypeI64, ir.CStr("=r")).Emit()
	e.Return(r.I64(0))

	err := lowerErr(t, m)
	if err == nil {
		t.Fatal("a template naming %3 with one operand lowered without complaint")
	}
	if !strings.Contains(err.Error(), "%3") {
		t.Errorf("the diagnostic does not name the reference: %v", err)
	}
}

// An instruction the assembler cannot encode is also a diagnostic.
func TestAsmBadInstructionIsRefused(t *testing.T) {
	m := ir.NewModule("t", ir.AArch64MacOS)
	fn := m.Func("_bad2").Export()
	fn.ReturnsI64()
	e := fn.Entry()
	r := e.Asm("frobnicate %0, #1").Out(ir.TypeI64, ir.CStr("=r")).Emit()
	e.Return(r.I64(0))

	err := lowerErr(t, m)
	if err == nil {
		t.Fatal("a template naming no instruction lowered without complaint")
	}
	if !strings.Contains(err.Error(), "frobnicate") {
		t.Errorf("the diagnostic does not name the mnemonic: %v", err)
	}
}
