package ir_test

import (
	"strings"
	"testing"

	"github.com/vertex-language/ir"
	"github.com/vertex-language/ir/text"
	"github.com/vertex-language/ir/verify"
)

func cryptoModule(requires bool) *ir.Module {
	m := ir.NewModule("t", ir.AArch64Linux)
	fn := m.Func("f").Export()
	if requires {
		fn.Requires(ir.CapAES)
	}
	p := fn.ParamPtr("p")
	fn.ReturnsI32()
	e := fn.Entry()
	v := e.V128()
	a := v.Load(p)
	x := v.AesEnc(a, a)
	x = v.AesKeygenAssist(x, 0x1b)
	x = v.Sha1RoundsCh(x, e.I32.Const(5), a)
	v.Store(x, p)
	c := e.I32.Crc32cU64(e.I32.Const(-1), e.I64.Const(7))
	has := e.I1.CpuHas(ir.CapAES)
	other := e.I1.CpuHas(ir.CapSHA512)
	e.Return(e.I32.Select(e.I1.And(has, other), c, e.I32.Const(0)))
	return m
}

func printed(t *testing.T, m *ir.Module) string {
	t.Helper()
	if err := m.Err(); err != nil {
		t.Fatalf("Err = %v", err)
	}
	if err := verify.Module(m); err != nil {
		t.Fatalf("verify: %v", err)
	}
	var buf strings.Builder
	if err := text.Print(&buf, m); err != nil {
		t.Fatal(err)
	}
	return buf.String()
}

// With nothing in the baseline every gated verb is a call, and cpu_has
// reads the runtime's word.
func TestLegalizeCryptoCalls(t *testing.T) {
	m := cryptoModule(false)
	if !m.LegalizeCrypto(0) {
		t.Fatal("LegalizeCrypto changed nothing")
	}
	got := printed(t, m)
	for _, want := range []string{
		"call @vertex_aes_enc", "call @vertex_aes_keygen_assist", "call @vertex_sha1_rounds_ch",
		"call @vertex_crc32c_u64", "call @vertex_cpu_has", "ptr.alloc 64 align 16",
	} {
		if !strings.Contains(got, want) {
			t.Errorf("no %q in:\n%s", want, got)
		}
	}
	for _, gone := range []string{"v128.aes_enc ", "i32.crc32c_u64", "i1.cpu_has"} {
		if strings.Contains(got, gone) {
			t.Errorf("%q survived:\n%s", gone, got)
		}
	}
	if strings.Count(got, "ptr.alloc") != 1 {
		t.Errorf("want one scratch slot:\n%s", got)
	}
}

// A function that requires aes keeps its AES verbs; a baseline that has a
// capability answers cpu_has for it.
func TestLegalizeCryptoKeeps(t *testing.T) {
	m := cryptoModule(true)
	m.LegalizeCrypto(ir.Capabilities(0).With(ir.CapAES, ir.CapCRC32C))
	got := printed(t, m)
	for _, want := range []string{"v128.aes_enc ", "v128.aes_keygen_assist", "i32.crc32c_u64",
		"call @vertex_sha1_rounds_ch", "i1.const 1", "call @vertex_cpu_has"} {
		if !strings.Contains(got, want) {
			t.Errorf("no %q in:\n%s", want, got)
		}
	}
}

// The verbs a baseline SSE2 target lacks become the ones it has, and the
// one no shift reaches becomes a call.
func TestExpandVectorVerbs(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	fn := m.Func("f").Export()
	p := fn.ParamPtr("p")
	e := fn.Entry()
	v := e.V128()
	a := v.Load(p)
	x := v.Xor3(a, a, a)
	x = v.XorAndNot(x, a, a)
	x = v.I32x4Rotl(x, 7)
	x = v.I64x2Rotl(x, 13)
	x = v.I16x8Bswap(v.I32x4Bswap(v.I64x2Bswap(x)))
	x = v.I8x16Swizzle(x, a)
	v.Store(x, p)
	e.Return()
	if !m.ExpandVectorVerbs(func(ir.Verb) bool { return true }) {
		t.Fatal("nothing expanded")
	}
	got := printed(t, m)
	for _, gone := range []string{"xor3", "xor_andnot", "_rotl", "_bswap", "v128.i8x16_swizzle"} {
		if strings.Contains(got, gone) {
			t.Errorf("%s survived:\n%s", gone, got)
		}
	}
	for _, want := range []string{"v128.i16x8_shuffle_low", "v128.i32x4_shl", "call @vertex_i8x16_swizzle"} {
		if !strings.Contains(got, want) {
			t.Errorf("no %s in:\n%s", want, got)
		}
	}
}

// A backend that lowers none of the verbs answers cpu_has with 0, requires
// or not, so a program keeps to the path that does not need them rather
// than one whose every instruction is a call.
func TestLegalizeCryptoAllFoldsCpuHas(t *testing.T) {
	m := cryptoModule(true)
	if !m.LegalizeCryptoAll() {
		t.Fatal("nothing legalized")
	}
	got := printed(t, m)
	if strings.Contains(got, ir.CPUHasSymbol) {
		t.Errorf("cpu_has still asks the runtime:\n%s", got)
	}
	if !strings.Contains(got, "call @vertex_aes_enc") {
		t.Errorf("aes_enc kept as an instruction despite requires:\n%s", got)
	}
}
