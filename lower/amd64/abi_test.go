package amd64_test

// byval and sret. An aggregate reaches a signature as a pointer carrying
// byval, ir.RegType having no struct in it — and that attribute used to
// be ignored, so the callee read whatever it found.

import (
	"bytes"
	"testing"

	"github.com/vertex-language/ir"
	amd64lower "github.com/vertex-language/ir/lower/amd64"
)

func fI64() ir.FType { return ir.StoreI64.FType() }
func fF64() ir.FType { return ir.StoreF64.FType() }

// Two integer eightbytes, so the aggregate travels in two integer
// registers and the caller reads them out of the storage the pointer
// names. The pointer is parked in RAX first: it arrived in RDI, which is
// where the first eightbyte is about to go.
func TestLowerByValInRegisters(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	st := m.Struct("pair").Field("a", fI64()).Field("b", fI64())
	g := m.ImportFunc("g", ir.NewSig().Param(ir.TypePtr, ir.ByVal(st)))

	fn := m.Func("cpair").Export()
	p := fn.ParamPtr("p")
	entry := fn.Entry()
	entry.Call(g, p)
	entry.Return()

	tb, raw := lowerText(t, m)

	want := []byte{
		0x55,             // push rbp
		0x48, 0x8b, 0xec, // mov rbp, rsp
		0x48, 0x8b, 0xc7, // mov rax, rdi
		0x48, 0x8b, 0x38, // mov rdi, [rax]      (eightbyte 0)
		0x48, 0x8b, 0x70, 0x08, // mov rsi, [rax+8]   (eightbyte 1)
		0xe8, 0x00, 0x00, 0x00, 0x00, // call g
		0xc9, // leave
		0xc3, // ret
	}
	if !bytes.Equal(tb, want) {
		t.Errorf(".text bytes = % x, want % x", tb, want)
	}
	objdumpHas(t, "cpair", raw, "callq")
}

// The two eightbytes need not be in the same file. An i64 followed by an
// f64 is INTEGER then SSE, so half of one struct goes in RDI and the
// other half in XMM0.
func TestLowerByValAcrossBothFiles(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	st := m.Struct("mixed").Field("a", fI64()).Field("b", fF64())
	g := m.ImportFunc("g", ir.NewSig().Param(ir.TypePtr, ir.ByVal(st)))

	fn := m.Func("cmix").Export()
	p := fn.ParamPtr("p")
	entry := fn.Entry()
	entry.Call(g, p)
	entry.Return()

	tb, raw := lowerText(t, m)

	want := []byte{
		0x55,             // push rbp
		0x48, 0x8b, 0xec, // mov rbp, rsp
		0x48, 0x8b, 0xc7, // mov rax, rdi
		0x48, 0x8b, 0x38, // mov rdi, [rax]           (the INTEGER eightbyte)
		0xf2, 0x0f, 0x10, 0x40, 0x08, // movsd xmm0, [rax+8]  (the SSE one)
		0xe8, 0x00, 0x00, 0x00, 0x00, // call g
		0xc9, // leave
		0xc3, // ret
	}
	if !bytes.Equal(tb, want) {
		t.Errorf(".text bytes = % x, want % x", tb, want)
	}
	objdumpHas(t, "cmix", raw, "movsd", "callq")
}

// Three eightbytes is past what §3.2.3 will put in registers, so the
// aggregate goes in the outgoing area whole — and the caller is what
// copies it there, which is the memcpy milestone 33 made available.
func TestLowerByValInMemory(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	st := m.Struct("big").Field("a", fI64()).Field("b", fI64()).Field("c", fI64())
	g := m.ImportFunc("g", ir.NewSig().Param(ir.TypePtr, ir.ByVal(st)))

	fn := m.Func("cbig").Export()
	p := fn.ParamPtr("p")
	entry := fn.Entry()
	entry.Call(g, p)
	entry.Return()

	tb, raw := lowerText(t, m)

	// 32 bytes of frame: 24 for the copy, rounded up to the stack
	// alignment. Both calls see a 16-byte aligned RSP.
	want := []byte{
		0x55,             // push rbp
		0x48, 0x8b, 0xec, // mov rbp, rsp
		0x48, 0x81, 0xec, 0x20, 0x00, 0x00, 0x00, // sub rsp, 32
		0x48, 0x8b, 0xf7, // mov rsi, rdi        (the source)
		0x48, 0x8d, 0x3c, 0x24, // lea rdi, [rsp]     (the outgoing area)
		0x48, 0xba, 0x18, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // movabs rdx, 24
		0xe8, 0x00, 0x00, 0x00, 0x00, // call memcpy
		0xe8, 0x00, 0x00, 0x00, 0x00, // call g
		0xc9, // leave
		0xc3, // ret
	}
	if !bytes.Equal(tb, want) {
		t.Errorf(".text bytes = % x, want % x", tb, want)
	}
	objdumpHasRelocs(t, "cbig", raw, "memcpy", "g")
}

// The callee side of a memory-class byval, and the reason it is the
// cheap one: the caller already made the copy, at a known offset above
// RBP, so the parameter is the address of it and no byte moves.
func TestLowerByValMemoryParameter(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	st := m.Struct("big").Field("a", fI64()).Field("b", fI64()).Field("c", fI64())

	fn := m.Func("takebig").Export()
	fn.Signature().Param(ir.TypePtr, ir.ByVal(st))
	p := fn.ParamPtr("p")
	fn.ReturnsI64()
	entry := fn.Entry()
	entry.Return(entry.I64.Load(p))

	tb, raw := lowerText(t, m)

	want := []byte{
		0x55,             // push rbp
		0x48, 0x8b, 0xec, // mov rbp, rsp
		0x48, 0x8d, 0x45, 0x10, // lea rax, [rbp+16]   (the incoming copy)
		0x48, 0x8b, 0x08, // mov rcx, [rax]
		0x48, 0x8b, 0xc1, // mov rax, rcx
		0xc9, // leave
		0xc3, // ret
	}
	if !bytes.Equal(tb, want) {
		t.Errorf(".text bytes = % x, want % x", tb, want)
	}
	objdumpHas(t, "takebig", raw, "leaq")
}

// And the callee side of a register-class one, which is the opposite:
// the aggregate has no address at all until this function makes one, so
// the eightbytes are stored into a frame slot and the parameter is a lea
// of that.
//
// It is what the caller did, backwards, which is the round trip byval
// has to make: two loads there, two stores here, and the same twenty-
// four bits of layout knowledge on both sides.
func TestLowerByValRegisterParameter(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	st := m.Struct("mixed").Field("a", fI64()).Field("b", fF64())

	fn := m.Func("takemix").Export()
	fn.Signature().Param(ir.TypePtr, ir.ByVal(st))
	p := fn.ParamPtr("p")
	fn.ReturnsI64()
	entry := fn.Entry()
	entry.Return(entry.I64.Load(p))

	tb, raw := lowerText(t, m)

	want := []byte{
		0x55,             // push rbp
		0x48, 0x8b, 0xec, // mov rbp, rsp
		0x48, 0x81, 0xec, 0x10, 0x00, 0x00, 0x00, // sub rsp, 16
		0x48, 0x89, 0x7d, 0xf0, // mov [rbp-16], rdi     (the INTEGER eightbyte)
		0xf2, 0x0f, 0x11, 0x45, 0xf8, // movsd [rbp-8], xmm0   (the SSE one)
		0x48, 0x8d, 0x45, 0xf0, // lea rax, [rbp-16]
		0x48, 0x8b, 0x08, // mov rcx, [rax]
		0x48, 0x8b, 0xc1, // mov rax, rcx
		0xc9, // leave
		0xc3, // ret
	}
	if !bytes.Equal(tb, want) {
		t.Errorf(".text bytes = % x, want % x", tb, want)
	}
	objdumpHas(t, "takemix", raw, "movsd", "leaq")
}

// sret's other half. The IR models the hidden pointer as an ordinary
// first parameter, so passing it in RDI already happened; what did not
// is §3.2.3's rule that the callee returns that same pointer in RAX.
func TestLowerSRetReturnsThePointer(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	st := m.Struct("big").Field("a", fI64()).Field("b", fI64()).Field("c", fI64())

	fn := m.Func("mk").Export()
	fn.Signature().Param(ir.TypePtr, ir.SRet(st))
	out := fn.ParamPtr("out")
	entry := fn.Entry()
	entry.I64.Store(entry.I64.Const(7), out)
	entry.Return()

	tb, raw := lowerText(t, m)

	want := []byte{
		0x48, 0xb8, 0x07, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // movabs rax, 7
		0x48, 0x89, 0x07, // mov [rdi], rax
		0x48, 0x8b, 0xc7, // mov rax, rdi   (the sret pointer, returned)
		0xc3, // ret
	}
	if !bytes.Equal(tb, want) {
		t.Errorf(".text bytes = % x, want % x", tb, want)
	}
	objdumpHas(t, "mk", raw, "retq")
}

// Only when the signature declares no result of its own. A function that
// returns something as well has already claimed RAX, and what it put
// there is what it said it returns.
func TestLowerSRetWithOwnResult(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	st := m.Struct("big").Field("a", fI64()).Field("b", fI64()).Field("c", fI64())

	fn := m.Func("mk2").Export()
	fn.Signature().Param(ir.TypePtr, ir.SRet(st))
	out := fn.ParamPtr("out")
	fn.ReturnsI32()
	entry := fn.Entry()
	entry.I64.Store(entry.I64.Const(7), out)
	entry.Return(entry.I32.Const(0))

	tb, raw := lowerText(t, m)

	want := []byte{
		0x48, 0xb8, 0x07, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // movabs rax, 7
		0x48, 0x89, 0x07, // mov [rdi], rax
		0xb8, 0x00, 0x00, 0x00, 0x00, // mov eax, 0   (the declared result)
		0xc3, // ret
	}
	if !bytes.Equal(tb, want) {
		t.Errorf(".text bytes = % x, want % x", tb, want)
	}
	objdumpHas(t, "mk2", raw, "retq")
}

// An aggregate whose fields are classes this package cannot read back
// out of a register. f80 is X87 and f128's second half is SSEUP, and
// neither is a value held in a register here at all.
func TestLowerRejectsExtendedFloatAggregate(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	st := m.Struct("withF128").Field("a", ir.StoreF128.FType())
	g := m.ImportFunc("g", ir.NewSig().Param(ir.TypePtr, ir.ByVal(st)))

	fn := m.Func("f").Export()
	p := fn.ParamPtr("p")
	entry := fn.Entry()
	entry.Call(g, p)
	entry.Return()

	if _, err := amd64lower.Lower(m, amd64lower.Options{}); err == nil {
		t.Error("Lower should refuse a byval aggregate containing an f128 field")
	}
}
