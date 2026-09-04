package i386_test

// §G4, inline assembly, verified by booting it.
//
// This target has the strongest check in the tree available to it: the lowered
// object is linked into a multiboot kernel and run under qemu, so the answer
// comes from a processor rather than from a disassembler. That matters more
// here than anywhere else, because the question an inline asm test has to
// answer — whether the register the allocator chose is the register the
// template named — cannot be asked of an expected-bytes test without encoding
// the allocator's answer to check the assembler's.

import (
	"strings"
	"testing"

	"github.com/vertex-language/ir"
	i386lower "github.com/vertex-language/ir/lower/i386"
	"github.com/vertex-language/ir/verify"
)

func lowerAsmErr(t *testing.T, m *ir.Module) error {
	t.Helper()
	if err := verify.Module(m); err != nil {
		t.Fatalf("verify.Module: %v", err)
	}
	_, err := i386lower.Lower(m, i386lower.Options{})
	return err
}

// The operand cases, in one kernel: an output alone, several operands, a tied
// operand whose input is still live, a fixed register, and a width modifier.
func TestRunAsmOperands(t *testing.T) {
	m := ir.NewModule("t", ir.I386Linux)

	// One output, no inputs.
	just := m.Func("ajust").Export()
	just.ReturnsI32()
	je := just.Entry()
	jr := je.Asm("movl $42, %0").Out(ir.TypeI32, ir.CStr("=r")).Emit()
	je.Return(jr.I32(0))

	// Two inputs and an output; the numbering is outputs before inputs.
	add := m.Func("aadd").Export()
	ax := add.ParamI32("x")
	ay := add.ParamI32("y")
	add.ReturnsI32()
	ae := add.Entry()
	ar := ae.Asm("movl %1, %0\n\taddl %2, %0").
		Out(ir.TypeI32, ir.CStr("=r")).
		In(ax, ir.CReg).
		In(ay, ir.CReg).
		Emit()
	ae.Return(ar.I32(0))

	// A tied operand. The input is read again afterwards, so a clobbered
	// one shows up as a wrong number rather than as a crash.
	tie := m.Func("atie").Export()
	tx := tie.ParamI32("x")
	tie.ReturnsI32()
	te := tie.Entry()
	tr := te.Asm("addl $1, %0").
		Out(ir.TypeI32, ir.CStr("=r")).
		In(tx, ir.CStr("0")).
		Emit()
	te.Return(te.I32.Add(tr.I32(0), tx))

	// A fixed register: the constraint says ECX and the template writes it.
	fix := m.Func("afix").Export()
	fx := fix.ParamI32("x")
	fix.ReturnsI32()
	fe := fix.Entry()
	fr := fe.Asm("shll %%cl, %0").
		Out(ir.TypeI32, ir.CStr("=r")).
		In(fx, ir.CStr("0")).
		In(fe.I32.Const(3), ir.CStr("c")).
		Emit()
	fe.Return(fr.I32(0))

	// The %w modifier, naming the 16-bit view of a 32-bit operand.
	wid := m.Func("awid").Export()
	wid.ReturnsI32()
	we := wid.Entry()
	wr := we.Asm("xorl %0, %0\n\tmovw $7, %w0").Out(ir.TypeI32, ir.CStr("=r")).Emit()
	we.Return(wr.I32(0))

	wantOK(t, m, `
int ajust(void), aadd(int, int), atie(int), afix(int), awid(void);
static void body(void) {
    chk32("output only", (unsigned)ajust(), 42u);
    chk32("two inputs",  (unsigned)aadd(19, 23), 42u);
    chk32("tied",        (unsigned)atie(10), 21u);
    chk32("fixed ecx",   (unsigned)afix(5), 40u);
    chk32("width mod",   (unsigned)awid(), 7u);
}
`)
}

// A clobber of a callee-saved register has to reach the prologue. EBX is
// callee-saved under the Intel386 psABI, so if the save is missing the
// caller's own use of it is destroyed.
func TestRunAsmClobberCalleeSaved(t *testing.T) {
	m := ir.NewModule("t", ir.I386Linux)
	fn := m.Func("aclob").Export()
	x := fn.ParamI32("x")
	fn.ReturnsI32()
	e := fn.Entry()
	r := e.Asm("movl $1, %%ebx\n\tmovl %1, %0\n\taddl %%ebx, %0").
		Volatile().
		Out(ir.TypeI32, ir.CStr("=r")).
		In(x, ir.CReg).
		Clobber("ebx").
		Emit()
	e.Return(r.I32(0))

	wantOK(t, m, `
int aclob(int);
static void body(void) {
    register int keep asm("ebx") = 0x5eed;
    int r = aclob(41);
    asm volatile("" :: "r"(keep));
    chk32("clobber ebx", (unsigned)r, 42u);
    chk32("ebx preserved", (unsigned)keep, 0x5eedu);
}
`)
}

// One template expanded twice in one function: both contain the same 1:, and
// they are different labels.
func TestRunAsmLocalLabelTwice(t *testing.T) {
	m := ir.NewModule("t", ir.I386Linux)
	fn := m.Func("atwice").Export()
	x := fn.ParamI32("x")
	fn.ReturnsI32()
	e := fn.Entry()
	one := e.Asm("movl $10, %0\n\ttestl %1, %1\n\tjz 1f\n\tmovl $20, %0\n1:").
		Out(ir.TypeI32, ir.CStr("=r")).In(x, ir.CReg).Emit()
	two := e.Asm("movl $1, %0\n\ttestl %1, %1\n\tjz 1f\n\tmovl $2, %0\n1:").
		Out(ir.TypeI32, ir.CStr("=r")).In(x, ir.CReg).Emit()
	e.Return(e.I32.Add(one.I32(0), two.I32(0)))

	wantOK(t, m, `
int atwice(int);
static void body(void) {
    chk32("twice zero", (unsigned)atwice(0), 11u);
    chk32("twice nonzero", (unsigned)atwice(5), 22u);
}
`)
}

// asm goto: the terminator form, branching into a block this package emitted.
// asm goto with outputs, which §14 binds to the fallthrough target's trailing
// parameters: the assembled text writes the value on the edge where it ran to
// the end, and the value is live only in the block that declares it.
func TestRunAsmGotoWithOutputs(t *testing.T) {
	m := ir.NewModule("t", ir.I386Linux)
	fn := m.Func("aclassify").Export()
	x := fn.ParamI32("x")
	fn.ReturnsI32()

	e := fn.Entry()
	fall := fn.Block("fall")
	out := fall.ParamI32("out")
	taken := fn.Block("taken")

	e.AsmGoto("movl %1, %0\n\taddl $10, %0\n\ttestl %1, %1\n\tjz %l[taken]").
		Out(ir.TypeI32, ir.CStr("=&r")).In(x, ir.CReg).Clobber("cc").
		To(fall.To(), taken)
	fall.Return(out)
	taken.Return(taken.I32.Const(0xffffffff))

	wantOK(t, m, `
int aclassify(int);
static void body(void) {
    chk32("goto taken", (unsigned)aclassify(0), 0xffffffffu);
    chk32("output on the fallthrough", (unsigned)aclassify(7), 17u);
}
`)
}

func TestRunAsmGoto(t *testing.T) {
	m := ir.NewModule("t", ir.I386Linux)
	fn := m.Func("apick").Export()
	x := fn.ParamI32("x")
	fn.ReturnsI32()

	e := fn.Entry()
	fall := fn.Block("fall")
	taken := fn.Block("taken")

	e.AsmGoto("testl %0, %0\n\tjz %l[taken]").In(x, ir.CReg).Clobber("cc").
		To(fall.To(), taken)
	fall.Return(fall.I32.Const(1))
	taken.Return(taken.I32.Const(2))

	wantOK(t, m, `
int apick(int);
static void body(void) {
    chk32("goto taken", (unsigned)apick(0), 2u);
    chk32("goto fell through", (unsigned)apick(7), 1u);
}
`)
}

// A 64-bit operand is refused rather than half-substituted. An i64 is a
// register pair here, and one %-reference cannot name two registers; writing
// the low half and hoping would assemble, run, and be wrong about the top
// thirty-two bits.
func TestAsmRefusesRegisterPair(t *testing.T) {
	m := ir.NewModule("t", ir.I386Linux)
	fn := m.Func("apair").Export()
	x := fn.ParamI64("x")
	fn.ReturnsI64()
	e := fn.Entry()
	r := e.Asm("addl $1, %0").
		Out(ir.TypeI64, ir.CStr("=r")).
		In(x, ir.CStr("0")).
		Emit()
	e.Return(r.I64(0))

	err := lowerAsmErr(t, m)
	if err == nil {
		t.Fatal("an i64 asm operand lowered without complaint")
	}
	if !strings.Contains(err.Error(), "pair") {
		t.Errorf("the diagnostic does not say why: %v", err)
	}
}

func TestAsmBadTemplateIsRefused(t *testing.T) {
	m := ir.NewModule("t", ir.I386Linux)
	fn := m.Func("abad").Export()
	fn.ReturnsI32()
	e := fn.Entry()
	r := e.Asm("movl %3, %0").Out(ir.TypeI32, ir.CStr("=r")).Emit()
	e.Return(r.I32(0))

	err := lowerAsmErr(t, m)
	if err == nil {
		t.Fatal("a template naming %3 with one operand lowered without complaint")
	}
	if !strings.Contains(err.Error(), "%3") {
		t.Errorf("the diagnostic does not name the reference: %v", err)
	}
}
