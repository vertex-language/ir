package i386_test

// §3b's module-level asm and §7's asm body, booted.
//
// The two forms have no operands and no allocation, so what is under test is
// where the bytes land and what surrounds them. Booting is still the right
// check for the second of those: a naked function that got a prologue it did
// not ask for returns with a frame still on the stack, and this harness finds
// that out the way the hardware does.

import (
	"strings"
	"testing"

	"github.com/vertex-language/ir"
)

// A naked function, in the convention its own text has to know: under cdecl
// the arguments are on the stack above the return address, and nothing has
// pushed a frame pointer because nothing was asked to.
func TestRunNakedAsmBody(t *testing.T) {
	m := ir.NewModule("t", ir.I386Linux)
	fn := m.Func("nadd").Export()
	fn.ParamI32("a")
	fn.ParamI32("b")
	fn.ReturnsI32()
	fn.AsmBody("movl 4(%esp), %eax\n\taddl 8(%esp), %eax\n\tret")

	wantOK(t, m, `
int nadd(int, int);
static void body(void) {
    chk32("naked add", (unsigned)nadd(40, 2), 42u);
}
`)
}

// A naked body with a numeric local label, and a lowered function calling the
// naked one — which is the case that says the symbol is a function like any
// other and not a special one.
func TestRunNakedAsmBodyLoop(t *testing.T) {
	m := ir.NewModule("t", ir.I386Linux)

	sum := m.Func("nsum").Export()
	sum.ParamI32("n")
	sum.ReturnsI32()
	sum.AsmBody("movl 4(%esp), %ecx\n" +
		"\txorl %eax, %eax\n" +
		"1:\ttestl %ecx, %ecx\n" +
		"\tjz 2f\n" +
		"\taddl %ecx, %eax\n" +
		"\tdecl %ecx\n" +
		"\tjmp 1b\n" +
		"2:\tret")

	// A lowered function that calls it. The naked one is an ordinary
	// symbol with an ordinary signature, so a call site needs to know
	// nothing about how its body was written.
	via := m.Func("nsum_via").Export()
	n := via.ParamI32("n")
	via.ReturnsI32()
	e := via.Entry()
	e.Return(e.Call(sum, n).Value(0).(ir.I32))

	wantOK(t, m, `
int nsum(int), nsum_via(int);
static void body(void) {
    chk32("naked loop", (unsigned)nsum(8), 36u);
    chk32("called from lowered code", (unsigned)nsum_via(8), 36u);
}
`)
}

// Module-level asm defining a function of its own, between two lowered ones.
func TestRunModuleAsmFunction(t *testing.T) {
	m := ir.NewModule("t", ir.I386Linux)

	before := m.Func("mbefore").Export().ReturnsI32()
	e := before.Entry()
	e.Return(e.I32.Const(1))

	m.Asm(".globl mseven\nmseven:\n\tmovl $7, %eax\n\tret")

	after := m.Func("mafter").Export().ReturnsI32()
	e2 := after.Entry()
	e2.Return(e2.I32.Const(2))

	wantOK(t, m, `
int mbefore(void), mseven(void), mafter(void);
static void body(void) {
    chk32("before", (unsigned)mbefore(), 1u);
    chk32("module asm", (unsigned)mseven(), 7u);
    chk32("after", (unsigned)mafter(), 2u);
}
`)
}

// A block that switches sections and switches back, which is the shape a
// constructor table or a note section has.
func TestRunModuleAsmSectionSwitch(t *testing.T) {
	m := ir.NewModule("t", ir.I386Linux)

	m.Asm(".pushsection .rodata\n" +
		".globl manswer\n" +
		".p2align 2\n" +
		"manswer:\n" +
		"\t.long 42\n" +
		".popsection")

	fn := m.Func("mdouble").Export()
	x := fn.ParamI32("x")
	fn.ReturnsI32()
	e := fn.Entry()
	e.Return(e.I32.Add(x, x))

	wantOK(t, m, `
extern const int manswer;
int mdouble(int);
static void body(void) {
    chk32("module asm data", (unsigned)manswer, 42u);
    chk32("lowered after it", (unsigned)mdouble(21), 42u);
}
`)
}

// A body the assembler refuses is the frontend's fault, and says so.
func TestAsmBodyRefused(t *testing.T) {
	m := ir.NewModule("t", ir.I386Linux)
	fn := m.Func("bad").Export()
	fn.AsmBody("nosuchinsn %eax, %ebx")

	err := lowerAsmErr(t, m)
	if err == nil {
		t.Fatal("lowered; the body is not assembly")
	}
	if !strings.Contains(err.Error(), "nosuchinsn") {
		t.Errorf("error %q does not name the mnemonic it refused", err)
	}
}

// Each module-level block is its own assembly.
func TestModuleAsmBlocksAreIndependent(t *testing.T) {
	m := ir.NewModule("t", ir.I386Linux)
	m.Asm(".pushsection .rodata")
	m.Asm(".popsection")

	err := lowerAsmErr(t, m)
	if err == nil {
		t.Fatal("lowered; the pop is in a block that pushed nothing")
	}
	if !strings.Contains(err.Error(), "no section pushed") {
		t.Errorf("error %q does not say the stack was empty", err)
	}
}
