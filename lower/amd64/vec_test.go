package amd64_test

// §V lowered onto SSE2.
//
// The bytes are asserted rather than the mnemonics, because the thing worth
// pinning is the two-address discipline: a lane operation writes its first
// operand, so a result that has to land somewhere else costs a MOVDQA the
// allocator could not have avoided, and a shuffle costs none because it
// reads its source instead. Which of those two shapes a verb has is not
// visible in a mnemonic list.

import (
	"bytes"
	"testing"

	"github.com/vertex-language/ir"
	amd64lower "github.com/vertex-language/ir/lower/amd64"
)

// Two vectors in, one out, through a verb that is one instruction.
func TestLowerVectorAdd(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	fn := m.Func("vadd").Export()
	p := fn.ParamPtr("p")
	q := fn.ParamPtr("q")
	fn.ReturnsI32()
	e := fn.Entry()
	v := e.V128()
	e.Return(v.I8x16Bitmask(v.I32x4Add(v.Load(p), v.Load(q))))

	tb, raw := lowerText(t, m)
	want := []byte{
		0x0f, 0x28, 0x07, // movaps xmm0, [rdi]
		0x0f, 0x28, 0x0e, // movaps xmm1, [rsi]
		0x66, 0x0f, 0x6f, 0xd0, // movdqa xmm2, xmm0
		0x66, 0x0f, 0xfe, 0xd1, // paddd xmm2, xmm1
		0x66, 0x0f, 0xd7, 0xc2, // pmovmskb eax, xmm2
		0xc3, // ret
	}
	if !bytes.Equal(tb, want) {
		t.Errorf(".text bytes = % x, want % x", tb, want)
	}
	objdumpHas(t, "vadd", raw, "paddd", "pmovmskb")
}

// A stated align below sixteen is what separates MOVDQU from MOVAPS. Both
// are correct for an aligned address and only one is correct otherwise, so
// the attribute has to reach the encoder rather than stopping at the
// optimizer — this is _mm_loadu_si128 against _mm_load_si128.
func TestUnalignedVectorAccessIsMovdqu(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	fn := m.Func("vcopy").Export()
	p := fn.ParamPtr("p")
	q := fn.ParamPtr("q")
	e := fn.Entry()
	v := e.V128()
	v.Store(v.Load(p, ir.Align(1)), q, ir.Align(1))
	e.Return()

	tb, raw := lowerText(t, m)
	want := []byte{
		0xf3, 0x0f, 0x6f, 0x07, // movdqu xmm0, [rdi]
		0xf3, 0x0f, 0x7f, 0x06, // movdqu [rsi], xmm0
		0xc3, // ret
	}
	if !bytes.Equal(tb, want) {
		t.Errorf(".text bytes = % x, want % x", tb, want)
	}
	objdumpHas(t, "vcopy", raw, "movdqu")
}

// A literal count takes the immediate form and needs no register; a
// computed one goes through MOVD, because the by-register form reads the
// count from the low quadword of a vector register and not from a general
// one. Two spellings of one verb, chosen by what the count is.
func TestVectorShiftPicksItsForm(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	fn := m.Func("vshl").Export()
	p := fn.ParamPtr("p")
	n := fn.ParamI32("n")
	fn.ReturnsI32()
	e := fn.Entry()
	v := e.V128()
	byLit := v.I32x4Shl(v.Load(p), e.I32.Const(3))
	byReg := v.I16x8ShrU(byLit, n)
	e.Return(v.I8x16Bitmask(byReg))

	_, raw := lowerText(t, m)
	objdumpHas(t, "vshl", raw, "pslld", "movd", "psrlw")
}

// The permutes read their source. That is what makes a shuffle the cheap
// way to move a lane — no copy first — and it is worth a test of its own
// because every other verb in §V is the opposite.
func TestShuffleDoesNotCopyFirst(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	fn := m.Func("vshuf").Export()
	p := fn.ParamPtr("p")
	fn.ReturnsI32()
	e := fn.Entry()
	v := e.V128()
	a := v.Load(p)
	e.Return(v.I8x16Bitmask(v.I32x4Add(a, v.I32x4Shuffle(a, 0x1b))))

	tb, raw := lowerText(t, m)
	if bytes.Count(tb, []byte{0x66, 0x0f, 0x6f}) != 1 {
		t.Errorf("expected exactly one MOVDQA — the add's — in % x", tb)
	}
	objdumpHas(t, "vshuf", raw, "pshufd")
}

// A splat is a sequence and not an instruction: SSE2 has no broadcast, so
// each width is the next one's problem solved once more.
func TestSplatWidens(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	fn := m.Func("vsplat").Export()
	n := fn.ParamI32("n")
	fn.ReturnsI32()
	e := fn.Entry()
	v := e.V128()
	e.Return(v.I8x16Bitmask(v.I8x16Splat(n)))

	_, raw := lowerText(t, m)
	objdumpHas(t, "vsplat", raw, "movd", "punpcklbw", "punpcklwd", "pshufd")
}

// An all-zero constant is a PXOR and not a load: two bytes, reading
// nothing. Anything else is sixteen bytes in .rodata.
func TestZeroConstantIsPxor(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	fn := m.Func("vzero").Export()
	fn.ReturnsI32()
	e := fn.Entry()
	v := e.V128()
	e.Return(v.I8x16Bitmask(v.Zero()))

	tb, raw := lowerText(t, m)
	want := []byte{
		0x66, 0x0f, 0xef, 0xc0, // pxor xmm0, xmm0
		0x66, 0x0f, 0xd7, 0xc0, // pmovmskb eax, xmm0
		0xc3, // ret
	}
	if !bytes.Equal(tb, want) {
		t.Errorf(".text bytes = % x, want % x", tb, want)
	}
	objdumpHas(t, "vzero", raw, "pxor")
}

// §V's andnot negates its second operand and PANDN negates its first, so
// isel hands the emitter its operands the other way round. Getting this
// backwards produces working-looking code that computes the complement of
// the intended mask, which no byte count would catch — hence the operand
// order in the expected bytes rather than a mnemonic check.
func TestAndNotSwapsItsOperands(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	fn := m.Func("vandnot").Export()
	p := fn.ParamPtr("p")
	q := fn.ParamPtr("q")
	fn.ReturnsI32()
	e := fn.Entry()
	v := e.V128()
	e.Return(v.I8x16Bitmask(v.AndNot(v.Load(p), v.Load(q))))

	tb, raw := lowerText(t, m)
	want := []byte{
		0x0f, 0x28, 0x07, // movaps xmm0, [rdi]      — a
		0x0f, 0x28, 0x0e, // movaps xmm1, [rsi]      — b
		0x66, 0x0f, 0x6f, 0xd1, // movdqa xmm2, xmm1 — b, the negated one
		0x66, 0x0f, 0xdf, 0xd0, // pandn  xmm2, xmm0 — ~b & a
		0x66, 0x0f, 0xd7, 0xc2, // pmovmskb eax, xmm2
		0xc3, // ret
	}
	if !bytes.Equal(tb, want) {
		t.Errorf(".text bytes = % x, want % x", tb, want)
	}
	objdumpHas(t, "vandnot", raw, "pandn")
}

// The Microsoft convention returns a vector in XMM0 and passes one by
// pointer. The first half is placed here; the second is the frontend's
// copy, and a v128 arriving as an argument is a frontend that skipped it.
func TestMSVectorReturnIsXMM0(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Windows)
	fn := m.Func("vret").Export()
	p := fn.ParamPtr("p")
	fn.ReturnsV128()
	e := fn.Entry()
	v := e.V128()
	e.Return(v.I32x4Add(v.Load(p), v.Load(p)))

	tb, _ := lowerText(t, m)
	// The result has to end in XMM0, whatever register the add landed in.
	if !bytes.Contains(tb, []byte{0x0f, 0x28, 0xc2}) && // movaps xmm0, xmm2
		!bytes.Contains(tb, []byte{0x66, 0x0f, 0xfe, 0xc1}) { // paddd xmm0, xmm1
		t.Errorf("no move of the result into XMM0 in % x", tb)
	}
}

func TestMSVectorArgumentIsRefused(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Windows)
	fn := m.Func("varg").Export()
	fn.ParamV128("a")
	fn.ReturnsI32()
	e := fn.Entry()
	e.Return(e.I32.Const(0))

	if _, err := amd64lower.Lower(m, amd64lower.Options{}); err == nil {
		t.Fatal("Lower = nil, want a refusal naming the by-pointer rule")
	} else if !bytes.Contains([]byte(err.Error()), []byte("by pointer")) {
		t.Errorf("Lower = %v, want it to name the by-pointer rule", err)
	}
}
