package amd64_test

import (
	"bytes"
	"testing"

	"github.com/vertex-language/ir"
	amd64lower "github.com/vertex-language/ir/lower/amd64"
)

// Tests allocating a single stack slot.
func TestLowerAllocSlot(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	fn := m.Func("spill").Export()
	v := fn.ParamI32("v")
	fn.ReturnsI32()

	entry := fn.Entry()
	slot := entry.Ptr.Alloc(4, 4)
	entry.I32.Store(v, slot)
	entry.Return(entry.I32.Load(slot))

	tb, raw := lowerText(t, m)

	want := []byte{
		0x55,             // push rbp
		0x48, 0x8b, 0xec, // mov rbp, rsp
		0x48, 0x81, 0xec, 0x10, 0x00, 0x00, 0x00, // sub rsp, 16
		0x48, 0x8d, 0x45, 0xfc, // lea rax, [rbp-4]
		0x89, 0x38, // mov [rax], edi
		0x8b, 0x08, // mov ecx, [rax]
		0x8b, 0xc1, // mov eax, ecx
		0xc9, // leave
		0xc3, // ret
	}
	if !bytes.Equal(tb, want) {
		t.Errorf(".text bytes = % x, want % x", tb, want)
	}

	objdumpHas(t, "spill", raw, "pushq", "leaq", "leave", "retq")
}

// Tests alignment of multiple stack slots.
func TestLowerAllocAlignment(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	fn := m.Func("two").Export()
	a := fn.ParamI32("a")
	b := fn.ParamI64("b")
	fn.ReturnsI64()

	entry := fn.Entry()
	small := entry.Ptr.Alloc(1, 1)
	wide := entry.Ptr.Alloc(8, 8)
	entry.I32.Store8(a, small)
	entry.I64.Store(b, wide)
	entry.Return(entry.I64.Load(wide))

	tb, raw := lowerText(t, m)

	want := []byte{
		0x55,             // push rbp
		0x48, 0x8b, 0xec, // mov rbp, rsp
		0x48, 0x81, 0xec, 0x10, 0x00, 0x00, 0x00, // sub rsp, 16
		0x48, 0x8d, 0x45, 0xff, // lea rax, [rbp-1]
		0x48, 0x8d, 0x4d, 0xf0, // lea rcx, [rbp-16]
		0x40, 0x88, 0x38, // mov [rax], dil
		0x48, 0x89, 0x31, // mov [rcx], rsi
		0x48, 0x8b, 0x01, // mov rax, [rcx]
		0xc9, // leave
		0xc3, // ret
	}
	if !bytes.Equal(tb, want) {
		t.Errorf(".text bytes = % x, want % x", tb, want)
	}

	objdumpHas(t, "two", raw, "leaq", "leave")
}

// Tests that the frame is torn down on every return path.
func TestLowerFrameOnEveryReturn(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	fn := m.Func("pick").Export()
	a := fn.ParamI32("a")
	b := fn.ParamI32("b")
	fn.ReturnsI32()

	entry := fn.Entry()
	yes := fn.Block("yes")
	no := fn.Block("no")

	slot := entry.Ptr.Alloc(4, 4)
	entry.I32.Store(a, slot)
	entry.BrIf(entry.I32.SLt(a, b), yes.To(), no.To())
	yes.Return(yes.I32.Load(slot))
	no.Return(b)

	tb, _ := lowerText(t, m)

	leaves := bytes.Count(tb, []byte{0xc9, 0xc3})
	if leaves != 2 {
		t.Errorf("found %d leave;ret pairs, want 2 — one per return\n% x", leaves, tb)
	}
	if got := bytes.Count(tb, []byte{0x55, 0x48, 0x8b, 0xec}); got != 1 {
		t.Errorf("found %d prologues, want 1\n% x", got, tb)
	}
}

// Tests rejection of unsupported allocations.
//
// Only over-alignment now. The zeroed case was here too until milestone
// 33 gave it the memset it was waiting for; see TestLowerZeroedAlloc.
func TestLowerRejectsUnsupportedAllocs(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	fn := m.Func("f").Export()
	fn.ReturnsI32()

	entry := fn.Entry()
	entry.Ptr.Alloc(64, 32)
	entry.Return(entry.I32.Const(0))

	if _, err := amd64lower.Lower(m, amd64lower.Options{}); err == nil {
		t.Error("Lower should refuse a ptr.alloc asking for more alignment than the frame guarantees")
	}
}
