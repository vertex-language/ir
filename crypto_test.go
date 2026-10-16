package ir_test

import (
	"strings"
	"testing"

	"github.com/vertex-language/ir"
	"github.com/vertex-language/ir/text"
	"github.com/vertex-language/ir/verify"
)

// Every §V5 and §A8 verb, built, verified and printed, in a function that
// requires the capabilities and chooses them with i1.cpu_has.
func TestCryptoVerbs(t *testing.T) {
	for _, tgt := range []ir.Target{ir.X86_64Linux, ir.AArch64Linux} {
		m := ir.NewModule("t", tgt)
		fn := m.Func("f").Export().Requires(ir.CapAES, ir.CapClmul, ir.CapSHA256)
		p := fn.ParamPtr("p")
		fn.ReturnsI32()

		e := fn.Entry()
		v := e.V128()
		a := v.Load(p)
		k := v.Load(p, ir.Align(1))
		x := v.AesEnc(a, k)
		x = v.AesEncLast(x, k)
		x = v.AesDec(x, v.AesInvMixColumns(k))
		x = v.AesDecLast(x, v.AesKeygenAssist(k, 0x36))
		x = v.I64x2Clmul(x, k, 3)
		x = v.I8x16Swizzle(x, v.I32x4Bswap(k))
		x = v.I16x8Bswap(v.I64x2Bswap(x))
		e32 := e.I32.Const(7)
		x = v.Sha1RoundsCh(x, e32, k)
		x = v.Sha1RoundsParity(x, e32, k)
		x = v.Sha1RoundsMaj(x, e32, k)
		x = v.Sha1Msg2(v.Sha1Msg1(x, k, a), k)
		x = v.Sha256RoundsAbcd(x, k, a)
		x = v.Sha256RoundsEfgh(k, x, a)
		x = v.Sha256Msg2(v.Sha256Msg1(x, k), k, a)
		x = v.Sha512Rounds1(x, k, a)
		x = v.Sha512Rounds2(x, k, a)
		x = v.Sha512Msg2(v.Sha512Msg1(x, k), k, a)
		x = v.Xor3(x, k, a)
		x = v.XorAndNot(x, k, a)
		x = v.I64x2Rotl(v.I32x4Rotl(x, 7), 62)
		x = v.Sm3Ss1(x, k, a)
		x = v.Sm3Tt1a(x, k, a, 1)
		x = v.Sm3Tt1b(x, k, a, 2)
		x = v.Sm3Tt2a(x, k, a, 3)
		x = v.Sm3Tt2b(x, k, a, 0)
		x = v.Sm3Partw2(v.Sm3Partw1(x, k, a), k, a)
		x = v.Sm4Key(v.Sm4Rounds(x, k), a)
		v.Store(x, p)

		c := e.I32.Crc32cU8(e32, e32)
		c = e.I32.Crc32cU16(c, e32)
		c = e.I32.Crc32cU32(c, e32)
		c = e.I32.Crc32cU64(c, e.I64.Const(9))
		c = e.I32.Crc32U8(c, e32)
		c = e.I32.Crc32U16(c, e32)
		c = e.I32.Crc32U32(c, e32)
		c = e.I32.Crc32U64(c, e.I64.Const(9))
		has := e.I1.CpuHas(ir.CapSHA512)
		e.Return(e.I32.Select(has, c, e32))

		if err := m.Err(); err != nil {
			t.Fatalf("%v: Err = %v", tgt, err)
		}
		if err := verify.Module(m); err != nil {
			t.Fatalf("%v: verify: %v", tgt, err)
		}
		var buf strings.Builder
		if err := text.Print(&buf, m); err != nil {
			t.Fatalf("print: %v", err)
		}
		got := buf.String()
		for _, want := range []string{
			"requires (aes, clmul, sha256)", "v128.aes_enc ", "v128.aes_keygen_assist",
			"v128.i64x2_clmul", "v128.i8x16_swizzle", "v128.sha1_rounds_maj",
			"v128.sha256_rounds_efgh", "v128.sha512_msg2", "v128.xor_andnot",
			"v128.i64x2_rotl", "v128.i32x4_rotl", "v128.sm3_tt2b", "v128.sm4_key",
			"i32.crc32c_u64", "i32.crc32_u8", "i1.cpu_has sha512",
		} {
			if !strings.Contains(got, want) {
				t.Errorf("%v: no %q in:\n%s", tgt, want, got)
			}
		}
	}
}

// Each verb names its capability, and every capability has a verb.
func TestCapabilityOfVerb(t *testing.T) {
	seen := ir.Capabilities(0)
	for _, v := range []ir.Verb{ir.VAesEnc, ir.VI64x2Clmul, ir.VSha1Msg1, ir.VSha256Msg2,
		ir.VSha512Msg1, ir.VSm3Ss1, ir.VSm4Key, ir.VCrc32U8, ir.VCrc32cU64} {
		c, ok := v.Capability()
		if !ok {
			t.Fatalf("%s has no capability", v)
		}
		seen = seen.With(c)
	}
	if got := len(seen.List()); got != 9 {
		t.Errorf("capabilities covered = %v, want all nine", seen)
	}
	for _, v := range []ir.Verb{ir.VI32x4Add, ir.VVecXor3, ir.VVecXorAndNot, ir.VI64x2Rotl} {
		if _, ok := v.Capability(); ok {
			t.Errorf("%s has a capability", v)
		}
	}
	for _, c := range seen.List() {
		if d, ok := ir.ParseCapability(c.String()); !ok || d != c {
			t.Errorf("ParseCapability(%q) = %v, %v", c, d, ok)
		}
	}
}

// Literals out of range are refused when built, not at lowering.
func TestCryptoLiteralRanges(t *testing.T) {
	m := ir.NewModule("t", ir.AArch64Linux)
	fn := m.Func("f").Export()
	p := fn.ParamPtr("p")
	e := fn.Entry()
	v := e.V128()
	a := v.Load(p)
	v.I64x2Clmul(a, a, 4)
	if m.Err() == nil {
		t.Fatal("clmul selector 4 was accepted")
	}
}
