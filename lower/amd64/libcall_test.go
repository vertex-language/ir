package amd64_test

// §E's bulk memory, as the calls it is. The good cases are short: the
// verbs take their operands in the order and classes SysV puts them in,
// so every copy coalesces away.

import (
	"bytes"
	"testing"

	"github.com/vertex-language/ir"
	amd64lower "github.com/vertex-language/ir/lower/amd64"
)

// memcpy(dst, src, n) against three parameters. RDI, RSI and RDX are
// where they arrived and where the callee wants them, so the whole
// lowering is the call — and the prologue, which is not optional: SysV
// promises the callee a 16-byte aligned RSP and entry leaves it eight
// off that.
func TestLowerMemCpy(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	fn := m.Func("cp").Export()
	d := fn.ParamPtr("d")
	s := fn.ParamPtr("s")
	n := fn.ParamI64("n")
	entry := fn.Entry()
	entry.MemCpy(d, s, n)
	entry.Return()

	tb, raw := lowerText(t, m)

	want := []byte{
		0x55,             // push rbp
		0x48, 0x8b, 0xec, // mov rbp, rsp
		0xe8, 0x00, 0x00, 0x00, 0x00, // call memcpy
		0xc9, // leave
		0xc3, // ret
	}
	if !bytes.Equal(tb, want) {
		t.Errorf(".text bytes = % x, want % x", tb, want)
	}
	objdumpHasRelocs(t, "cp", raw, "R_X86_64_PLT32", "memcpy")
}

// memset(dst, val, n): the middle operand is an i32 where the other two
// verbs take a pointer, which is C's `int c` and §E's "writes the low
// byte of val" being the same sentence.
func TestLowerMemSet(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	fn := m.Func("ms").Export()
	d := fn.ParamPtr("d")
	v := fn.ParamI32("v")
	n := fn.ParamI64("n")
	entry := fn.Entry()
	entry.MemSet(d, v, n)
	entry.Return()

	tb, raw := lowerText(t, m)

	want := []byte{
		0x55,             // push rbp
		0x48, 0x8b, 0xec, // mov rbp, rsp
		0xe8, 0x00, 0x00, 0x00, 0x00, // call memset
		0xc9, // leave
		0xc3, // ret
	}
	if !bytes.Equal(tb, want) {
		t.Errorf(".text bytes = % x, want % x", tb, want)
	}
	objdumpHasRelocs(t, "ms", raw, "memset")
}

// memcmp is the one of the four with a result, and it comes back in EAX
// — which is also where the function returns it, so that copy coalesces
// too.
func TestLowerMemCmp(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	fn := m.Func("cm").Export()
	a := fn.ParamPtr("a")
	b := fn.ParamPtr("b")
	n := fn.ParamI64("n")
	fn.ReturnsI32()
	entry := fn.Entry()
	entry.Return(entry.MemCmp(a, b, n))

	tb, raw := lowerText(t, m)

	want := []byte{
		0x55,             // push rbp
		0x48, 0x8b, 0xec, // mov rbp, rsp
		0xe8, 0x00, 0x00, 0x00, 0x00, // call memcmp
		0xc9, // leave
		0xc3, // ret
	}
	if !bytes.Equal(tb, want) {
		t.Errorf(".text bytes = % x, want % x", tb, want)
	}
	objdumpHasRelocs(t, "cm", raw, "memcmp")
}

// A library call is a call: what it destroys is what the ABI says, not
// what its name suggests it touches.
//
// n is read after the memcpy, so it cannot stay in RDX where it arrived
// — RDX is both the third argument and a caller-saved register. It goes
// to RBX, which is what the prologue is saving, and comes back into RDX
// for the call itself.
func TestLowerLiveAcrossMemCpy(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	fn := m.Func("live").Export()
	d := fn.ParamPtr("d")
	s := fn.ParamPtr("s")
	n := fn.ParamI64("n")
	fn.ReturnsI64()
	entry := fn.Entry()
	entry.MemCpy(d, s, n)
	entry.Return(entry.I64.Add(n, n))

	tb, raw := lowerText(t, m)

	want := []byte{
		0x55,             // push rbp
		0x48, 0x8b, 0xec, // mov rbp, rsp
		0x48, 0x81, 0xec, 0x10, 0x00, 0x00, 0x00, // sub rsp, 16
		0x48, 0x89, 0x5d, 0xf8, // mov [rbp-8], rbx
		0x48, 0x8b, 0xda, // mov rbx, rdx   (n, out of the call's way)
		0x48, 0x8b, 0xd3, // mov rdx, rbx   (and back, as the argument)
		0xe8, 0x00, 0x00, 0x00, 0x00, // call memcpy
		0x48, 0x8b, 0xc3, // mov rax, rbx
		0x48, 0x03, 0xc3, // add rax, rbx
		0x48, 0x8b, 0x5d, 0xf8, // mov rbx, [rbp-8]
		0xc9, // leave
		0xc3, // ret
	}
	if !bytes.Equal(tb, want) {
		t.Errorf(".text bytes = % x, want % x", tb, want)
	}
	objdumpHas(t, "live", raw, "callq", "addq")
}

// A zeroed ptr.alloc, refused since milestone 9 for want of a memset.
//
// The frame is 48 bytes: 32 for the storage and 8 for the callee-saved
// register the address has to live in across the call, rounded up to the
// stack alignment.
func TestLowerZeroedAlloc(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	fn := m.Func("za").Export()
	fn.ReturnsPtr()
	entry := fn.Entry()
	entry.Return(entry.Ptr.Alloc(32, 8, ir.Zeroed))

	tb, raw := lowerText(t, m)

	want := []byte{
		0x55,             // push rbp
		0x48, 0x8b, 0xec, // mov rbp, rsp
		0x48, 0x81, 0xec, 0x30, 0x00, 0x00, 0x00, // sub rsp, 48
		0x48, 0x89, 0x5d, 0xd8, // mov [rbp-40], rbx
		0x48, 0x8d, 0x5d, 0xe0, // lea rbx, [rbp-32]   (the storage)
		0xbe, 0x00, 0x00, 0x00, 0x00, // mov esi, 0     (the fill byte)
		0x48, 0xba, 0x20, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // movabs rdx, 32
		0x48, 0x8b, 0xfb, // mov rdi, rbx
		0xe8, 0x00, 0x00, 0x00, 0x00, // call memset
		0x48, 0x8b, 0xc3, // mov rax, rbx   (the alloc's own value)
		0x48, 0x8b, 0x5d, 0xd8, // mov rbx, [rbp-40]
		0xc9, // leave
		0xc3, // ret
	}
	if !bytes.Equal(tb, want) {
		t.Errorf(".text bytes = % x, want % x", tb, want)
	}
	objdumpHasRelocs(t, "za", raw, "memset")
}

// A volatile bulk operation is a promise about how the bytes are
// touched. memcpy does not make it, so the call is the wrong lowering
// rather than a slow one, and there is no open-coded form to fall back
// to yet.
func TestLowerRejectsVolatileBulkMemory(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	fn := m.Func("v").Export()
	d := fn.ParamPtr("d")
	s := fn.ParamPtr("s")
	n := fn.ParamI64("n")
	entry := fn.Entry()
	entry.MemCpy(d, s, n, ir.Volatile)
	entry.Return()

	if _, err := amd64lower.Lower(m, amd64lower.Options{}); err == nil {
		t.Error("Lower should refuse a volatile memcpy")
	}
}
