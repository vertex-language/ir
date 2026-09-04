package arm64_test

// §3b's module-level asm and §7's asm body, verified the same way §G4 was:
// lowered, assembled, linked against a C main and run.
//
// These two forms have no operands and no allocation, so what a test has to
// establish is not that a register was chosen well. It is that the bytes land
// where the symbol says they do, that the symbol is reachable from another
// translation unit, and — for a naked function — that nothing was emitted
// around the text that the text did not ask for. A prologue this package
// added to a body ending in RET would return with a frame still on the stack,
// which is a crash and not a wrong number.

import (
	"strings"
	"testing"

	"github.com/vertex-language/ir"
)

// A naked function whose body is its whole definition. Mach-O prefixes a C
// symbol with an underscore, hence the name.
func TestRunNakedAsmBody(t *testing.T) {
	m := ir.NewModule("t", ir.AArch64MacOS)
	fn := m.Func("_naked_add").Export()
	fn.ParamI64("a")
	fn.ParamI64("b")
	fn.ReturnsI64()
	fn.AsmBody("add x0, x0, x1\n\tret")

	got := runNative(t, m, `
#include <stdio.h>
long naked_add(long, long);
int main(void) { printf("%ld\n", naked_add(40, 2)); return 0; }
`)
	if got != "42\n" {
		t.Errorf("printed %q, want %q", got, "42\n")
	}
}

// A naked body with a local label in it. The prefix that keeps two expansions
// of an inline template apart has to keep this one apart from them too.
func TestRunNakedAsmBodyLocalLabel(t *testing.T) {
	m := ir.NewModule("t", ir.AArch64MacOS)
	fn := m.Func("_naked_countdown").Export()
	fn.ParamI64("n")
	fn.ReturnsI64()
	fn.AsmBody("mov x1, #0\n" +
		"1: cbz x0, 2f\n" +
		"   add x1, x1, x0\n" +
		"   sub x0, x0, #1\n" +
		"   b 1b\n" +
		"2: mov x0, x1\n" +
		"   ret")

	got := runNative(t, m, `
#include <stdio.h>
long naked_countdown(long);
int main(void) { printf("%ld\n", naked_countdown(8)); return 0; }
`)
	if got != "36\n" {
		t.Errorf("printed %q, want %q", got, "36\n")
	}
}

// Module-level asm defining a function of its own, beside two lowered ones.
// The lowered function after it is the part that matters as much as the
// symbol: a fragment that left the section switched would put it somewhere
// other than .text.
func TestRunModuleAsmFunction(t *testing.T) {
	m := ir.NewModule("t", ir.AArch64MacOS)

	before := m.Func("_before").Export()
	before.ReturnsI64()
	e := before.Entry()
	e.Return(e.I64.Const(1))

	m.Asm(".globl _seven\n" +
		".p2align 2\n" +
		"_seven:\n" +
		"    mov x0, #7\n" +
		"    ret")

	after := m.Func("_after").Export()
	after.ReturnsI64()
	e2 := after.Entry()
	e2.Return(e2.I64.Const(2))

	got := runNative(t, m, `
#include <stdio.h>
long before(void); long seven(void); long after(void);
int main(void) { printf("%ld %ld %ld\n", before(), seven(), after()); return 0; }
`)
	if got != "1 7 2\n" {
		t.Errorf("printed %q, want %q", got, "1 7 2\n")
	}
}

// Module-level asm that switches sections and switches back. This is the
// shape a constructor table or an ELF note has, and the reason a fragment
// starts in the section it was handed rather than in .text.
func TestRunModuleAsmSectionSwitch(t *testing.T) {
	m := ir.NewModule("t", ir.AArch64MacOS)

	m.Asm(".pushsection .data\n" +
		".globl _the_answer\n" +
		".p2align 3\n" +
		"_the_answer:\n" +
		"    .quad 42\n" +
		".popsection")

	fn := m.Func("_dbl").Export()
	x := fn.ParamI64("x")
	fn.ReturnsI64()
	e := fn.Entry()
	e.Return(e.I64.Add(x, x))

	got := runNative(t, m, `
#include <stdio.h>
extern long the_answer;
long dbl(long);
int main(void) { printf("%ld %ld\n", the_answer, dbl(21)); return 0; }
`)
	if got != "42 42\n" {
		t.Errorf("printed %q, want %q", got, "42 42\n")
	}
}

// Two module-level blocks appending to one section, with a declaration
// between them. Order among them is what is being checked: the second block's
// bytes follow the first block's, wherever the declaration in between put its
// own.
func TestRunModuleAsmOrder(t *testing.T) {
	m := ir.NewModule("t", ir.AArch64MacOS)

	m.Asm(".pushsection .data\n.globl _pair\n.p2align 3\n_pair:\n    .quad 6\n.popsection")

	fn := m.Func("_unrelated").Export()
	fn.ReturnsI64()
	e := fn.Entry()
	e.Return(e.I64.Const(0))

	m.Asm(".pushsection .data\n    .quad 7\n.popsection")

	got := runNative(t, m, `
#include <stdio.h>
extern long pair[2];
long unrelated(void);
int main(void) { printf("%ld %ld %ld\n", pair[0], pair[1], unrelated()); return 0; }
`)
	if got != "6 7 0\n" {
		t.Errorf("printed %q, want %q", got, "6 7 0\n")
	}
}

// Each block is its own assembly. The section stack, the numeric local
// labels and the absolute symbols in one do not reach the next — which is
// what makes a block that opens a section and never closes it a bug local to
// itself rather than something the next block inherits.
func TestModuleAsmBlocksAreIndependent(t *testing.T) {
	m := ir.NewModule("t", ir.AArch64MacOS)
	m.Asm(".pushsection .data")
	m.Asm(".popsection")

	err := lowerErr(t, m)
	if err == nil {
		t.Fatal("lowered; the pop is in a block that pushed nothing")
	}
	if !strings.Contains(err.Error(), "no section pushed") {
		t.Errorf("error %q does not say the stack was empty", err)
	}
}

// A body the assembler refuses is the frontend's fault and has to say so with
// the position the assembler gave it, not become a module error with a
// section offset for a location.
func TestNakedAsmBodyRefused(t *testing.T) {
	m := ir.NewModule("t", ir.AArch64MacOS)
	fn := m.Func("_bad").Export()
	fn.AsmBody("this_is_not_an_instruction x0, x1")

	err := lowerErr(t, m)
	if err == nil {
		t.Fatal("lowered; the body is not assembly")
	}
	if !strings.Contains(err.Error(), "this_is_not_an_instruction") {
		t.Errorf("error %q does not name the mnemonic it refused", err)
	}
}

// The same for a module-level block.
func TestModuleAsmRefused(t *testing.T) {
	m := ir.NewModule("t", ir.AArch64MacOS)
	m.Asm(".quad")

	if err := lowerErr(t, m); err == nil {
		t.Fatal("lowered; .quad has no operand")
	}
}
