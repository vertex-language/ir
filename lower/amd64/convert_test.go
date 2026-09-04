package amd64_test

// Milestone 19: §C's integer conversions, and the i1 that stops being
// only a flag.
//
// The two are one milestone because one of the conversions reads an i1,
// and until a compare could be a value there was no i1 to read: an i1
// existed as the flags a branch tested and nowhere else.

import (
	"bytes"
	"testing"

	"github.com/vertex-language/ir"
)

// Sign-extension is the one §C verb that is its own instruction. The
// upper half is the source's sign bit repeated, which nothing else here
// produces.
func TestLowerSExtI32(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	fn := m.Func("s").Export()
	a := fn.ParamI32("a")
	fn.ReturnsI64()
	e := fn.Entry()
	e.Return(e.I64.SExtI32(a))

	tb, raw := lowerText(t, m)

	want := []byte{
		0x48, 0x63, 0xc7, // movsxd rax, edi
		0xc3, // ret
	}
	if !bytes.Equal(tb, want) {
		t.Errorf(".text bytes = % x, want % x", tb, want)
	}
	objdumpHas(t, "s", raw, "movslq")
}

// Zero-extension is a mov to the 32-bit view, because a write to a
// 32-bit register zeroes the upper half. It is still an instruction —
// the one place this package emits a mov it will not let a coalescer
// remove, since a source and destination in one register still need the
// upper half cleared.
func TestLowerZExtI32(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	fn := m.Func("z").Export()
	a := fn.ParamI32("a")
	fn.ReturnsI64()
	e := fn.Entry()
	e.Return(e.I64.ZExtI32(a))

	tb, _ := lowerText(t, m)

	want := []byte{
		0x8b, 0xc7, // mov eax, edi
		0xc3, // ret
	}
	if !bytes.Equal(tb, want) {
		t.Errorf(".text bytes = % x, want % x", tb, want)
	}
}

// Narrowing is free. An i32 is the low half of the register its i64 was
// in, so wrap_i64 is a copy — the mov here is the return's, out of EDI
// and into EAX, and the wrap itself coalesced into nothing.
func TestLowerWrapI64(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	fn := m.Func("w").Export()
	a := fn.ParamI64("a")
	fn.ReturnsI32()
	e := fn.Entry()
	e.Return(e.I32.WrapI64(a))

	tb, _ := lowerText(t, m)

	want := []byte{
		0x8b, 0xc7, // mov eax, edi
		0xc3, // ret
	}
	if !bytes.Equal(tb, want) {
		t.Errorf(".text bytes = % x, want % x", tb, want)
	}
}

// A compare read as a value rather than branched on: SETcc moves the
// answer out of the flags and into a byte, and the MOVZX after it makes
// that byte a whole i1.
func TestLowerCompareAsValue(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	fn := m.Func("lt").Export()
	a := fn.ParamI32("a")
	b := fn.ParamI32("b")
	fn.ReturnsI32()
	e := fn.Entry()
	e.Return(e.I32.ZExtI1(e.I32.SLt(a, b)))

	tb, raw := lowerText(t, m)

	want := []byte{
		0x3b, 0xfe, // cmp edi, esi
		0x0f, 0x9c, 0xc0, // setl al
		0x0f, 0xb6, 0xc0, // movzx eax, al
		0xc3, // ret
	}
	if !bytes.Equal(tb, want) {
		t.Errorf(".text bytes = % x, want % x", tb, want)
	}
	objdumpHas(t, "lt", raw, "setl", "movzbl")
}

// An i1 returned as itself, which needs no conversion at all: the value
// SETcc and MOVZX leave in EAX is already the whole of it.
func TestLowerI1Return(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	fn := m.Func("eq").Export()
	a := fn.ParamI64("a")
	b := fn.ParamI64("b")
	fn.ReturnsI1()
	e := fn.Entry()
	e.Return(e.I64.Eq(a, b))

	tb, _ := lowerText(t, m)

	want := []byte{
		0x48, 0x3b, 0xfe, // cmp rdi, rsi
		0x0f, 0x94, 0xc0, // sete al
		0x0f, 0xb6, 0xc0, // movzx eax, al
		0xc3, // ret
	}
	if !bytes.Equal(tb, want) {
		t.Errorf(".text bytes = % x, want % x", tb, want)
	}
}

// zext_i1 into an i64 is the ordinary zero-extension and not a copy. The
// upper half is zero here because the i1 came from a compare in this
// function, but that is a fact about this function and not about the
// instruction.
func TestLowerZExtI1ToI64(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	fn := m.Func("z1").Export()
	a := fn.ParamI32("a")
	b := fn.ParamI32("b")
	fn.ReturnsI64()
	e := fn.Entry()
	e.Return(e.I64.ZExtI1(e.I32.ULt(a, b)))

	tb, _ := lowerText(t, m)

	want := []byte{
		0x3b, 0xfe, // cmp edi, esi
		0x0f, 0x92, 0xc0, // setb al
		0x0f, 0xb6, 0xc0, // movzx eax, al
		0x8b, 0xc8, // mov ecx, eax     (the zero-extension)
		0x48, 0x8b, 0xc1, // mov rax, rcx
		0xc3, // ret
	}
	if !bytes.Equal(tb, want) {
		t.Errorf(".text bytes = % x, want % x", tb, want)
	}
}

// One compare, branched on and read as a value. Fusing is only right
// when it is every reader's answer, so this one is materialized — and
// the branch compares again, which is the price of having no pass that
// could notice the flags were still good.
func TestLowerCompareFusedAndValue(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	fn := m.Func("both1").Export()
	a := fn.ParamI32("a")
	b := fn.ParamI32("b")
	fn.ReturnsI32()

	entry := fn.Entry()
	thenB := fn.Block("then")
	elseB := fn.Block("else")

	c := entry.I32.SLt(a, b)
	flag := entry.I32.ZExtI1(c)
	entry.BrIf(c, thenB.To(), elseB.To())
	thenB.Return(flag)
	elseB.Return(elseB.I32.Const(0))

	tb, _ := lowerText(t, m)

	want := []byte{
		0x3b, 0xfe, // cmp edi, esi
		0x0f, 0x9c, 0xc0, // setl al
		0x0f, 0xb6, 0xc0, // movzx eax, al
		0x3b, 0xfe, // cmp edi, esi      (again, for the branch)
		0x0f, 0x8c, 0x05, 0x00, 0x00, 0x00, // jl then
		0xe9, 0x01, 0x00, 0x00, 0x00, // jmp else
		0xc3,                         // ret
		0xb8, 0x00, 0x00, 0x00, 0x00, // mov eax, 0
		0xc3, // ret
	}
	if !bytes.Equal(tb, want) {
		t.Errorf(".text bytes = % x, want % x", tb, want)
	}
}
