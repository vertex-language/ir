package amd64_test

// §D3's addresses. blockaddr is the one that matters: brind could jump
// to an address in a value and nothing here could make one.

import (
	"bytes"
	"testing"

	"github.com/vertex-language/ir"
)

// blockaddr and brind, which are only useful together: two block
// addresses, a select between them, and a jump through the answer.
//
// Both blocks reach the symbol table, because a blockaddr is a
// relocation against the label and a bare label leaves nothing to
// relocate against. That is the same promotion a jump table's targets
// get — see labeledBlocks.
func TestLowerBlockAddr(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	fn := m.Func("ba").Export()
	c := fn.ParamI1("c")
	fn.ReturnsI32()

	entry := fn.Entry()
	a := fn.Block("a")
	b := fn.Block("b")

	pa := entry.Ptr.BlockAddr(a)
	pb := entry.Ptr.BlockAddr(b)
	entry.BrInd(entry.Ptr.Select(c, pa, pb), a, b)
	a.Return(a.I32.Const(1))
	b.Return(b.I32.Const(2))

	tb, raw := lowerText(t, m)

	// No prologue: two LEAs, a select, and an indirect jump need no
	// storage, and blockaddr is not one of the verbs that forces a frame.
	want := []byte{
		0x48, 0x8d, 0x05, 0x00, 0x00, 0x00, 0x00, // lea rax, [rip + ba.a]
		0x48, 0x8d, 0x0d, 0x00, 0x00, 0x00, 0x00, // lea rcx, [rip + ba.b]
		0x40, 0x84, 0xff, // test dil, dil
		0x48, 0x8b, 0xd1, // mov rdx, rcx           (the false arm)
		0x48, 0x0f, 0x45, 0xd0, // cmovne rdx, rax  (the true arm over it)
		0xff, 0xe2, // jmp rdx
		0xb8, 0x01, 0x00, 0x00, 0x00, // ba.a: mov eax, 1
		0xc3,                         // ret
		0xb8, 0x02, 0x00, 0x00, 0x00, // ba.b: mov eax, 2
		0xc3, // ret
	}
	if !bytes.Equal(tb, want) {
		t.Errorf(".text bytes = % x, want % x", tb, want)
	}
	objdumpHasRelocs(t, "ba", raw, "R_X86_64_PC32", "ba.a", "ba.b")
}

// All three in one function, which asks how far the return address is
// from the frame pointer: eight, always, and a compiler that folded
// constants would say so. The prologue is here because frameaddr asked
// for it.
func TestLowerFrameAndReturnAddr(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	fn := m.Func("fa").Export()
	fn.ReturnsI64()

	entry := fn.Entry()
	entry.Return(entry.Ptr.Diff(entry.Ptr.FrameAddr(), entry.Ptr.ReturnAddr()))

	tb, raw := lowerText(t, m)

	want := []byte{
		0x55,             // push rbp
		0x48, 0x8b, 0xec, // mov rbp, rsp
		0x48, 0x8d, 0x45, 0x00, // lea rax, [rbp]      (frameaddr)
		0x48, 0x8b, 0x4d, 0x08, // mov rcx, [rbp+8]    (returnaddr)
		0x48, 0x8b, 0xd0, // mov rdx, rax
		0x48, 0x2b, 0xd1, // sub rdx, rcx             (diff)
		0x48, 0x8b, 0xc2, // mov rax, rdx
		0xc9, // leave
		0xc3, // ret
	}
	if !bytes.Equal(tb, want) {
		t.Errorf(".text bytes = % x, want % x", tb, want)
	}
	objdumpHas(t, "fa", raw, "leaq", "subq", "retq")
}
