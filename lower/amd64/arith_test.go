package amd64_test

// Milestone 21: what an arithmetic instruction produces besides its
// result.
//
// §A's smul_hi and umul_hi want the half of a product a two-address
// multiply throws away, and §A2's predicates want a flag no comparison
// can reconstruct. Both are the same milestone because both are an
// instruction whose answer is not its destination register.

import (
	"bytes"
	"testing"

	"github.com/vertex-language/ir"
)

// The high half of an unsigned product. The one-operand multiply is the
// only form that keeps it: the multiplicand goes in RAX, the product
// comes back across RDX:RAX, and RDX is the answer.
func TestLowerUMulHi(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	fn := m.Func("uh").Export()
	a := fn.ParamI64("a")
	b := fn.ParamI64("b")
	fn.ReturnsI64()
	e := fn.Entry()
	e.Return(e.I64.UMulHi(a, b))

	tb, raw := lowerText(t, m)

	want := []byte{
		0x48, 0x8b, 0xc7, // mov rax, rdi
		0x48, 0xf7, 0xe6, // mul rsi
		0x48, 0x8b, 0xc2, // mov rax, rdx
		0xc3, // ret
	}
	if !bytes.Equal(tb, want) {
		t.Errorf(".text bytes = % x, want % x", tb, want)
	}
	objdumpHas(t, "uh", raw, "mulq")
}

// The signed half, at 32 bits: IMUL's one-operand form, which is a
// different instruction from the two-operand IMUL the ALU table uses and
// not merely a different encoding of it.
func TestLowerSMulHi(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	fn := m.Func("sh").Export()
	a := fn.ParamI32("a")
	b := fn.ParamI32("b")
	fn.ReturnsI32()
	e := fn.Entry()
	e.Return(e.I32.SMulHi(a, b))

	tb, raw := lowerText(t, m)

	want := []byte{
		0x8b, 0xc7, // mov eax, edi
		0xf7, 0xee, // imul esi
		0x8b, 0xc2, // mov eax, edx
		0xc3, // ret
	}
	if !bytes.Equal(tb, want) {
		t.Errorf(".text bytes = % x, want % x", tb, want)
	}
	objdumpHas(t, "sh", raw, "imull")
}

// saddo: the addition, and the overflow flag it set. The sum itself is
// discarded — §A2's predicates answer i1 and nothing else — which is why
// it lands in EDI here, on top of a parameter nothing reads again.
func TestLowerSAddO(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	fn := m.Func("ao").Export()
	a := fn.ParamI32("a")
	b := fn.ParamI32("b")
	fn.ReturnsI1()
	e := fn.Entry()
	e.Return(e.I32.SAddO(a, b))

	tb, raw := lowerText(t, m)

	want := []byte{
		0x03, 0xfe, // add edi, esi
		0x0f, 0x90, 0xc0, // seto al
		0x0f, 0xb6, 0xc0, // movzx eax, al
		0xc3, // ret
	}
	if !bytes.Equal(tb, want) {
		t.Errorf(".text bytes = % x, want % x", tb, want)
	}
	objdumpHas(t, "ao", raw, "seto")
}

// uaddo is the same addition and a different flag. Signed overflow is OF
// and unsigned carry is CF, and one ADD sets both — which is the whole
// reason these are two verbs and not one.
func TestLowerUAddO(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	fn := m.Func("uo").Export()
	a := fn.ParamI64("a")
	b := fn.ParamI64("b")
	fn.ReturnsI1()
	e := fn.Entry()
	e.Return(e.I64.UAddO(a, b))

	tb, raw := lowerText(t, m)

	want := []byte{
		0x48, 0x03, 0xfe, // add rdi, rsi
		0x0f, 0x92, 0xc0, // setb al
		0x0f, 0xb6, 0xc0, // movzx eax, al
		0xc3, // ret
	}
	if !bytes.Equal(tb, want) {
		t.Errorf(".text bytes = % x, want % x", tb, want)
	}
	objdumpHas(t, "uo", raw, "setb")
}

// umulo is the one predicate that needs the widening multiply: an
// unsigned product overflows exactly when its high half is non-zero,
// which is what MUL's CF means and what no two-address multiply reports.
func TestLowerUMulO(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	fn := m.Func("mo").Export()
	a := fn.ParamI32("a")
	b := fn.ParamI32("b")
	fn.ReturnsI1()
	e := fn.Entry()
	e.Return(e.I32.UMulO(a, b))

	tb, _ := lowerText(t, m)

	want := []byte{
		0x8b, 0xc7, // mov eax, edi
		0xf7, 0xe6, // mul esi
		0x0f, 0x92, 0xc0, // setb al
		0x0f, 0xb6, 0xc0, // movzx eax, al
		0xc3, // ret
	}
	if !bytes.Equal(tb, want) {
		t.Errorf(".text bytes = % x, want % x", tb, want)
	}
}

// The shape a frontend actually wants: multiply, trap if it overflowed,
// and use the product otherwise.
//
// The multiply happens twice, because §A2's predicate is not the
// product — it answers whether one would overflow, and the frontend
// emitting `mul` afterwards is emitting a second instruction. Fusing the
// two is a peephole and this package has nowhere to put one.
func TestLowerCheckedMultiply(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	fn := m.Func("mb").Export()
	a := fn.ParamI32("a")
	b := fn.ParamI32("b")
	fn.ReturnsI32()

	entry := fn.Entry()
	over := fn.Block("over")
	okB := fn.Block("ok")
	entry.BrIf(entry.I32.SMulO(a, b), over.To(), okB.To())
	over.Trap()
	okB.Return(okB.I32.Mul(a, b))

	tb, raw := lowerText(t, m)

	want := []byte{
		0x8b, 0xc7, // mov eax, edi
		0x0f, 0xaf, 0xc6, // imul eax, esi
		0x0f, 0x90, 0xc0, // seto al
		0x0f, 0xb6, 0xc0, // movzx eax, al
		0x84, 0xc0, // test al, al
		0x0f, 0x85, 0x05, 0x00, 0x00, 0x00, // jne over
		0xe9, 0x02, 0x00, 0x00, 0x00, // jmp ok
		0x0f, 0x0b, // ud2
		0x8b, 0xc7, // mov eax, edi
		0x0f, 0xaf, 0xc6, // imul eax, esi
		0xc3, // ret
	}
	if !bytes.Equal(tb, want) {
		t.Errorf(".text bytes = % x, want % x", tb, want)
	}
	objdumpHas(t, "mb", raw, "seto", "ud2")
}
