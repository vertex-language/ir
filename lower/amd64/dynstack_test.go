package amd64_test

// §D3's dynamic frame, and §C4. ptr.alloca's size arrives as a value, so
// the only place for it is below RSP — and moving RSP is what everything
// addressed from RSP has to survive.

import (
	"bytes"
	"testing"

	"github.com/vertex-language/ir"
	amd64lower "github.com/vertex-language/ir/lower/amd64"
)

// The size rounded up to the stack alignment, that much stack taken, and
// the address handed back. Rounding is not tidiness: RSP has to stay
// 16-aligned across the allocation, and nothing after it re-checks.
func TestLowerAlloca(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	fn := m.Func("dyn").Export()
	n := fn.ParamI64("n")
	fn.ReturnsPtr()
	entry := fn.Entry()
	entry.Return(entry.Ptr.Alloca(n, 16))

	tb, raw := lowerText(t, m)

	want := []byte{
		0x55,             // push rbp
		0x48, 0x8b, 0xec, // mov rbp, rsp
		0x48, 0x8d, 0x4f, 0x0f, // lea rcx, [rdi+15]
		0x48, 0x81, 0xe1, 0xf0, 0xff, 0xff, 0xff, // and rcx, -16
		0x48, 0x2b, 0xe1, // sub rsp, rcx
		0x48, 0x8d, 0x04, 0x24, // lea rax, [rsp]
		0xc9, // leave
		0xc3, // ret
	}
	if !bytes.Equal(tb, want) {
		t.Errorf(".text bytes = % x, want % x", tb, want)
	}
	objdumpHas(t, "dyn", raw, "andq", "subq", "leaq")
}

// The case the design is arranged around: an alloca in a function that
// also calls with stack arguments. The area moves down with RSP, so the
// allocation is handed the space above it — the two stack arguments land
// at RSP and RSP+8 as always, and the allocation starts at RSP+16.
func TestLowerAllocaWithStackArguments(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	sig := ir.NewSig()
	for i := 0; i < 8; i++ {
		sig = sig.Param(ir.TypeI64)
	}
	g := m.ImportFunc("g", sig)

	fn := m.Func("dyncall").Export()
	n := fn.ParamI64("n")
	fn.ReturnsPtr()
	entry := fn.Entry()
	p := entry.Ptr.Alloca(n, 16)
	entry.Call(g, n, n, n, n, n, n, n, n)
	entry.Return(p)

	tb, raw := lowerText(t, m)

	want := []byte{
		0x55,             // push rbp
		0x48, 0x8b, 0xec, // mov rbp, rsp
		0x48, 0x81, 0xec, 0x20, 0x00, 0x00, 0x00, // sub rsp, 32
		0x48, 0x89, 0x5d, 0xf8, // mov [rbp-8], rbx
		0x4c, 0x8b, 0xcf, // mov r9, rdi
		0x49, 0x8d, 0x41, 0x0f, // lea rax, [r9+15]
		0x48, 0x81, 0xe0, 0xf0, 0xff, 0xff, 0xff, // and rax, -16
		0x48, 0x2b, 0xe0, // sub rsp, rax
		0x48, 0x8d, 0x5c, 0x24, 0x10, // lea rbx, [rsp+16]   (above the outgoing area)
		0x4c, 0x89, 0x0c, 0x24, // mov [rsp], r9        (the seventh argument)
		0x4c, 0x89, 0x4c, 0x24, 0x08, // mov [rsp+8], r9      (the eighth)
		0x49, 0x8b, 0xf9, // mov rdi, r9
		0x49, 0x8b, 0xf1, // mov rsi, r9
		0x49, 0x8b, 0xd1, // mov rdx, r9
		0x49, 0x8b, 0xc9, // mov rcx, r9
		0x4d, 0x8b, 0xc1, // mov r8, r9
		0xe8, 0x00, 0x00, 0x00, 0x00, // call g
		0x48, 0x8b, 0xc3, // mov rax, rbx
		0x48, 0x8b, 0x5d, 0xf8, // mov rbx, [rbp-8]
		0xc9, // leave
		0xc3, // ret
	}
	if !bytes.Equal(tb, want) {
		t.Errorf(".text bytes = % x, want % x", tb, want)
	}
	objdumpHas(t, "dyncall", raw, "callq")
}

// stacksave and stackrestore are RSP into a value and back again, which
// is what frees every allocation made between them at once.
func TestLowerStackSaveRestore(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	fn := m.Func("bracket").Export()
	n := fn.ParamI64("n")
	fn.ReturnsPtr()
	entry := fn.Entry()
	tok := entry.Ptr.StackSave()
	p := entry.Ptr.Alloca(n, 16)
	entry.Ptr.StackRestore(tok)
	entry.Return(p)

	tb, raw := lowerText(t, m)

	want := []byte{
		0x55,             // push rbp
		0x48, 0x8b, 0xec, // mov rbp, rsp
		0x48, 0x8b, 0xc4, // mov rax, rsp        (stacksave)
		0x48, 0x8d, 0x57, 0x0f, // lea rdx, [rdi+15]
		0x48, 0x81, 0xe2, 0xf0, 0xff, 0xff, 0xff, // and rdx, -16
		0x48, 0x2b, 0xe2, // sub rsp, rdx
		0x48, 0x8d, 0x0c, 0x24, // lea rcx, [rsp]
		0x48, 0x8b, 0xe0, // mov rsp, rax        (stackrestore)
		0x48, 0x8b, 0xc1, // mov rax, rcx
		0xc9, // leave
		0xc3, // ret
	}
	if !bytes.Equal(tb, want) {
		t.Errorf(".text bytes = % x, want % x", tb, want)
	}
	objdumpHas(t, "bracket", raw, "retq")
}

// A zeroed alloca is the allocation and then the memset milestone 33
// made available, over the requested count and not the rounded one: what
// §D3 guarantees reads as zero is the allocation, and the bytes the
// rounding took are padding that is not part of it.
func TestLowerZeroedAlloca(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	fn := m.Func("zdyn").Export()
	n := fn.ParamI64("n")
	fn.ReturnsPtr()
	entry := fn.Entry()
	entry.Return(entry.Ptr.Alloca(n, 16, ir.Zeroed))

	tb, raw := lowerText(t, m)

	want := []byte{
		0x55,             // push rbp
		0x48, 0x8b, 0xec, // mov rbp, rsp
		0x48, 0x81, 0xec, 0x10, 0x00, 0x00, 0x00, // sub rsp, 16
		0x48, 0x89, 0x5d, 0xf8, // mov [rbp-8], rbx
		0x48, 0x8b, 0xd7, // mov rdx, rdi        (n, which memset also wants)
		0x48, 0x8d, 0x42, 0x0f, // lea rax, [rdx+15]
		0x48, 0x81, 0xe0, 0xf0, 0xff, 0xff, 0xff, // and rax, -16
		0x48, 0x2b, 0xe0, // sub rsp, rax
		0x48, 0x8d, 0x1c, 0x24, // lea rbx, [rsp]
		0xbe, 0x00, 0x00, 0x00, 0x00, // mov esi, 0
		0x48, 0x8b, 0xfb, // mov rdi, rbx
		0xe8, 0x00, 0x00, 0x00, 0x00, // call memset
		0x48, 0x8b, 0xc3, // mov rax, rbx
		0x48, 0x8b, 0x5d, 0xf8, // mov rbx, [rbp-8]
		0xc9, // leave
		0xc3, // ret
	}
	if !bytes.Equal(tb, want) {
		t.Errorf(".text bytes = % x, want % x", tb, want)
	}
	objdumpHasRelocs(t, "zdyn", raw, "memset")
}

// The stack guarantees sixteen, the same as a static ptr.alloc.
func TestLowerRejectsOveralignedAlloca(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	fn := m.Func("f").Export()
	n := fn.ParamI64("n")
	fn.ReturnsPtr()
	entry := fn.Entry()
	entry.Return(entry.Ptr.Alloca(n, 64))

	if _, err := amd64lower.Lower(m, amd64lower.Options{}); err == nil {
		t.Error("Lower should refuse a ptr.alloca asking for more alignment than the stack guarantees")
	}
}

// §C4, which on a 64-bit target is the identity in both directions:
// from_i64 truncates and from_ptr zero-extends only where ptrbits is
// less than 64, and checkLayout has already refused anything else. The
// round trip is one move, because both copies coalesced into it.
func TestLowerPointerIntegerCasts(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	fn := m.Func("cast").Export()
	n := fn.ParamI64("n")
	fn.ReturnsI64()
	entry := fn.Entry()
	entry.Return(entry.I64.FromPtr(entry.Ptr.FromI64(n)))

	tb, raw := lowerText(t, m)

	want := []byte{
		0x48, 0x8b, 0xc7, // mov rax, rdi
		0xc3, // ret
	}
	if !bytes.Equal(tb, want) {
		t.Errorf(".text bytes = % x, want % x", tb, want)
	}
	objdumpHas(t, "cast", raw, "retq")
}
