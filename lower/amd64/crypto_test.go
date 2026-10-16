package amd64_test

import (
	"bytes"
	"strings"
	"testing"

	"github.com/vertex-language/amd64/feature"
	amd64elf "github.com/vertex-language/amd64/obj/elf"
	elfobj "github.com/vertex-language/elf/obj"

	"github.com/vertex-language/ir"
	amd64lower "github.com/vertex-language/ir/lower/amd64"
	"github.com/vertex-language/ir/verify"
)

// lowerSyms lowers m for set and returns .text and every symbol name.
func lowerSyms(t *testing.T, m *ir.Module, set feature.Set) ([]byte, string) {
	t.Helper()
	if err := verify.Module(m); err != nil {
		t.Fatalf("verify.Module: %v", err)
	}
	o, err := amd64lower.Lower(m, amd64lower.Options{Features: set})
	if err != nil {
		t.Fatalf("Lower: %v", err)
	}
	var buf bytes.Buffer
	if err := amd64elf.Write(&buf, o); err != nil {
		t.Fatalf("elf.Write: %v", err)
	}
	f, err := elfobj.NewFile(bytes.NewReader(buf.Bytes()))
	if err != nil {
		t.Fatalf("NewFile: %v", err)
	}
	syms, err := f.Symbols()
	if err != nil {
		t.Fatalf("Symbols: %v", err)
	}
	var names []string
	for _, s := range syms {
		names = append(names, s.Name)
	}
	text, err := f.Section(".text").Data()
	if err != nil {
		t.Fatal(err)
	}
	return text, strings.Join(names, " ")
}

// cpu_has reads the runtime's word where the baseline cannot answer, is 1
// where it can, and is 0 for a capability x86-64 has no instruction for;
// and a function that requires such a capability calls for its verbs.
func TestCpuHasAndUncoveredCapabilities(t *testing.T) {
	build := func() *ir.Module {
		m := ir.NewModule("t", ir.X86_64Linux)
		f := m.Func("f").Export().Requires(ir.CapAES, ir.CapSHA512)
		p := f.ParamPtr("p")
		f.ReturnsI32()
		e := f.Entry()
		v := e.V128()
		a := v.Load(p)
		v.Store(v.Sha512Msg1(v.AesEnc(a, a), a), p)
		has := e.I1.CpuHas(ir.CapAES)
		e.Return(e.I32.Select(e.I1.And(has, e.I1.CpuHas(ir.CapSHA512)), e.I32.Const(1), e.I32.Const(0)))
		return m
	}
	text, syms := lowerSyms(t, build(), feature.Default())
	for _, want := range []string{"vertex_cpu_has", "vertex_sha512_msg1"} {
		if !strings.Contains(syms, want) {
			t.Errorf("no %s among %s", want, syms)
		}
	}
	if !bytes.Contains(text, []byte{0x66, 0x0f, 0x38, 0xdc}) {
		t.Errorf("no aesenc in % x", text)
	}

	// With AES in the baseline the runtime is not asked at all: aes is 1
	// and sha512 0.
	_, syms = lowerSyms(t, build(), feature.Default().Add(feature.AES))
	if strings.Contains(syms, "vertex_cpu_has") {
		t.Errorf("cpu_has asked the runtime with AES in the baseline: %s", syms)
	}
}

// The two SHA-256 round verbs over the same state are one pair of
// SHA256RNDS2s, not two; and a v128 parameter arriving in XMM0, live
// across them, is moved out of the register the instruction reads.
func TestSHA256RoundsPairAndXMM0(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	f := m.Func("f").Export().Requires(ir.CapSHA256)
	abcd := f.ParamV128("abcd")
	efgh := f.ParamV128("efgh")
	wk := f.ParamV128("wk")
	f.ReturnsV128()
	e := f.Entry()
	v := e.V128()
	x := v.Sha256RoundsAbcd(abcd, efgh, wk)
	y := v.Sha256RoundsEfgh(efgh, abcd, wk)
	e.Return(v.Xor3(x, y, abcd))

	text, _ := lowerSyms(t, m, feature.Default())
	if n := bytes.Count(text, []byte{0x0f, 0x38, 0xcb}); n != 2 {
		t.Errorf("%d sha256rnds2, want 2:\n% x", n, text)
	}
}

// Without SSSE3 the byte swaps are shuffles and the swizzle a call; a
// function that requires aes has SSSE3, and they are PSHUFB.
func TestByteSwapBySSSE3(t *testing.T) {
	build := func(caps ...ir.Capability) *ir.Module {
		m := ir.NewModule("t", ir.X86_64Linux)
		f := m.Func("f").Export().Requires(caps...)
		p := f.ParamPtr("p")
		e := f.Entry()
		v := e.V128()
		a := v.Load(p)
		v.Store(v.I8x16Swizzle(v.I32x4Bswap(a), a), p)
		e.Return()
		return m
	}
	pshufb := []byte{0x66, 0x0f, 0x38, 0x00}
	text, syms := lowerSyms(t, build(), feature.Default())
	if bytes.Contains(text, pshufb) || !strings.Contains(syms, "vertex_i8x16_swizzle") {
		t.Errorf("baseline: want no pshufb and a swizzle call; symbols %s", syms)
	}
	text, syms = lowerSyms(t, build(ir.CapAES), feature.Default())
	if n := bytes.Count(text, pshufb); n != 2 || strings.Contains(syms, "vertex_i8x16_swizzle") {
		t.Errorf("requires aes: %d pshufb, want 2, and no call; symbols %s", n, syms)
	}
}

// A baseline helper's swizzle inlined at O1 into a function that requires
// clmul is that function's PSHUFB: expansion waits for inlining, so GCM's
// byte reversal is not a runtime call per block.
func TestSwizzleInlinedIntoRequires(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	rev := m.Func("rev").Internal()
	x := rev.ParamV128("x")
	rev.ReturnsV128()
	re := rev.Entry()
	re.Return(re.V128().I8x16Swizzle(x, re.V128().Const([16]byte{15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0})))

	f := m.Func("f").Export().Requires(ir.CapClmul)
	p := f.ParamPtr("p")
	e := f.Entry()
	v := e.V128()
	a := e.Call(rev, v.Load(p)).V128(0)
	v.Store(v.I64x2Clmul(a, a, 0), p)
	e.Return()

	if err := verify.Module(m); err != nil {
		t.Fatal(err)
	}
	o, err := amd64lower.Lower(m, amd64lower.Options{OptLevel: amd64lower.O1})
	if err != nil {
		t.Fatalf("Lower: %v", err)
	}
	var buf bytes.Buffer
	if err := amd64elf.Write(&buf, o); err != nil {
		t.Fatal(err)
	}
	// rev's own copy, at the baseline, still calls; f's is PSHUFB.
	if !bytes.Contains(buf.Bytes(), []byte{0x66, 0x0f, 0x38, 0x00}) {
		t.Error("no pshufb: the swizzle inlined into a requires function is still a call")
	}
}
