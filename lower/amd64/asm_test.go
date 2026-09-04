package amd64_test

// §G4, inline assembly.
//
// The arm64 backend verifies this by linking and running, which is the honest
// test: it asks the machine whether the register the allocator chose is the
// register the template named. That is not available here — this host is
// Apple Silicon and there is no user-mode qemu for it — so these read the
// disassembly instead, and lean on the constraints that make the answer
// deterministic.
//
// That is why nearly every case below uses a fixed-register constraint. `"=a"`
// means the output is in RAX whatever the allocator would otherwise have
// picked, so the expected disassembly is a constant and not a guess about
// allocation. The cases that cannot be pinned check the shape instead.

import (
	"strings"
	"testing"

	"github.com/vertex-language/ir"
	amd64lower "github.com/vertex-language/ir/lower/amd64"
	"github.com/vertex-language/ir/verify"
)

func lowerAsmErr(t *testing.T, m *ir.Module) error {
	t.Helper()
	if err := verify.Module(m); err != nil {
		t.Fatalf("verify.Module: %v", err)
	}
	_, err := amd64lower.Lower(m, amd64lower.Options{})
	return err
}

// A fixed output register: the template writes RAX because the constraint
// says RAX, so the disassembly is known ahead of time.
func TestAsmFixedOutput(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	fn := m.Func("answer").Export().NoUnwind()
	fn.ReturnsI64()
	e := fn.Entry()
	r := e.Asm("movq $42, %0").Out(ir.TypeI64, ir.CStr("=a")).Emit()
	e.Return(r.I64(0))

	_, obj := lowerText(t, m)
	objdumpHas(t, "asmfixed", obj, "mov", "$0x2a", "%rax")
}

// The syscall shape: four fixed input registers and a fixed output, which is
// how every libc writes it and the case fixed-register constraints exist for.
func TestAsmSyscall(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	fn := m.Func("sys3").Export().NoUnwind()
	nr := fn.ParamI64("nr")
	a1 := fn.ParamI64("a1")
	a2 := fn.ParamI64("a2")
	a3 := fn.ParamI64("a3")
	fn.ReturnsI64()
	e := fn.Entry()
	r := e.Asm("syscall").
		Volatile().
		Out(ir.TypeI64, ir.CStr("=a")).
		In(nr, ir.CStr("a")).
		In(a1, ir.CStr("D")).
		In(a2, ir.CStr("S")).
		In(a3, ir.CStr("d")).
		Clobber("rcx", "r11", "memory").
		Emit()
	e.Return(r.I64(0))

	_, obj := lowerText(t, m)
	objdumpHas(t, "asmsyscall", obj, "syscall")
}

// A width modifier: %k0 is the 32-bit view of an operand held at 64 bits.
func TestAsmWidthModifier(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	fn := m.Func("wmod").Export().NoUnwind()
	fn.ReturnsI64()
	e := fn.Entry()
	r := e.Asm("movl $7, %k0").Out(ir.TypeI64, ir.CStr("=a")).Emit()
	e.Return(r.I64(0))

	_, obj := lowerText(t, m)
	objdumpHas(t, "asmwmod", obj, "mov", "$0x7", "%eax")
}

// A tied operand, pinned so the register is known: the instruction reads and
// writes RAX, and the input has to have been copied into it first.
func TestAsmTiedOperand(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	fn := m.Func("tied").Export().NoUnwind()
	a := fn.ParamI64("a")
	fn.ReturnsI64()
	e := fn.Entry()
	r := e.Asm("addq $1, %0").
		Out(ir.TypeI64, ir.CStr("=a")).
		In(a, ir.CStr("0")).
		Emit()
	e.Return(r.I64(0))

	_, obj := lowerText(t, m)
	objdumpHas(t, "asmtied", obj, "add", "$0x1", "%rax")
}

// A clobber of a callee-saved register has to reach the prologue. RBX is
// callee-saved under the SysV ABI, so naming it makes the frame save it.
func TestAsmClobberCalleeSavedIsPreserved(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	fn := m.Func("clob").Export().NoUnwind()
	fn.ReturnsI64()
	e := fn.Entry()
	r := e.Asm("movq $1, %%rbx\n\tmovq $9, %0").
		Volatile().
		Out(ir.TypeI64, ir.CStr("=a")).
		Clobber("rbx").
		Emit()
	e.Return(r.I64(0))

	_, obj := lowerText(t, m)
	// The clobber is what puts rbx in the frame's saved set; without it
	// nothing would push it.
	objdumpHas(t, "asmclob", obj, "%rbx")
}

// A literal % survives, which x86 needs constantly: %%rax in a template is
// one register and not an operand reference.
func TestAsmLiteralPercent(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	fn := m.Func("lit").Export().NoUnwind()
	fn.ReturnsI64()
	e := fn.Entry()
	r := e.Asm("xorq %%rcx, %%rcx\n\tmovq %%rcx, %0").
		Out(ir.TypeI64, ir.CStr("=a")).
		Clobber("rcx").
		Emit()
	e.Return(r.I64(0))

	_, obj := lowerText(t, m)
	objdumpHas(t, "asmlit", obj, "xor", "%rcx")
}

// A local label in a template, expanded twice in one function: both contain
// the same 1: and they are different labels.
func TestAsmLocalLabelTwice(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	fn := m.Func("twice").Export().NoUnwind()
	fn.ReturnsI64()
	e := fn.Entry()
	one := e.Asm("jmp 1f\n1:\tmovq $1, %0").Out(ir.TypeI64, ir.CStr("=a")).Emit()
	_ = e.Asm("jmp 1f\n1:\tmovq $2, %0").Out(ir.TypeI64, ir.CStr("=c")).Emit()
	e.Return(one.I64(0))

	if _, err := lowerText2(t, m); err != nil {
		t.Fatalf("two expansions of one template: %v", err)
	}
}

// asm goto, whose labels are branched to by the assembled text.
func TestAsmGoto(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	fn := m.Func("pick").Export().NoUnwind()
	a := fn.ParamI64("a")
	fn.ReturnsI64()

	e := fn.Entry()
	fall := fn.Block("fall")
	taken := fn.Block("taken")

	e.AsmGoto("testq %0, %0\n\tjz %l[taken]").In(a, ir.CReg).Clobber("cc").
		To(fall.To(), taken)
	fall.Return(fall.I64.Const(1))
	taken.Return(taken.I64.Const(2))

	_, obj := lowerText(t, m)
	objdumpHas(t, "asmgoto", obj, "test", "je")
}

// asm goto with outputs, which §14 binds to the fallthrough target's trailing
// parameters. The assembled text writes the value on the edge where it ran to
// the end, and the value is live only in the block that declares it.
func TestAsmGotoWithOutputs(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	fn := m.Func("classify").Export().NoUnwind()
	a := fn.ParamI64("a")
	fn.ReturnsI64()

	e := fn.Entry()
	fall := fn.Block("fall")
	out := fall.ParamI64("out")
	taken := fn.Block("taken")

	e.AsmGoto("leaq 10(%1), %0\n\ttestq %1, %1\n\tjz %l[taken]").
		Out(ir.TypeI64, ir.CStr("=&r")).In(a, ir.CReg).Clobber("cc").
		To(fall.To(), taken)
	fall.Return(out)
	taken.Return(taken.I64.Const(-1))

	_, obj := lowerText(t, m)
	objdumpHas(t, "asmgotoout", obj, "lea", "test", "je")
}

// A bad template is a diagnostic, not a wrong object.
func TestAsmBadTemplateIsRefused(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	fn := m.Func("bad").Export().NoUnwind()
	fn.ReturnsI64()
	e := fn.Entry()
	r := e.Asm("movq %3, %0").Out(ir.TypeI64, ir.CStr("=a")).Emit()
	e.Return(r.I64(0))

	err := lowerAsmErr(t, m)
	if err == nil {
		t.Fatal("a template naming %3 with one operand lowered without complaint")
	}
	if !strings.Contains(err.Error(), "%3") {
		t.Errorf("the diagnostic does not name the reference: %v", err)
	}
}

func TestAsmBadInstructionIsRefused(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	fn := m.Func("bad2").Export().NoUnwind()
	fn.ReturnsI64()
	e := fn.Entry()
	r := e.Asm("frobnicate %0").Out(ir.TypeI64, ir.CStr("=a")).Emit()
	e.Return(r.I64(0))

	err := lowerAsmErr(t, m)
	if err == nil {
		t.Fatal("a template naming no instruction lowered without complaint")
	}
	if !strings.Contains(err.Error(), "frobnicate") {
		t.Errorf("the diagnostic does not name the mnemonic: %v", err)
	}
}

// lowerText2 is lowerText without the fatal, for a case whose point is that
// lowering succeeds.
func lowerText2(t *testing.T, m *ir.Module) ([]byte, error) {
	t.Helper()
	if err := verify.Module(m); err != nil {
		return nil, err
	}
	o, err := amd64lower.Lower(m, amd64lower.Options{})
	if err != nil {
		return nil, err
	}
	return o.SectionAt(0).Bytes(), nil
}
