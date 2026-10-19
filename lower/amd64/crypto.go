package amd64

// §V5 cryptographic and §A8 checksum instructions lowered to AES-NI, PCLMULQDQ,
// SHA extensions, and SSE4.2 (CRC-32C).
// Unsupported hardware capabilities are legalized into runtime helper calls before isel.

import (
	"fmt"

	amd64asm "github.com/vertex-language/amd64"
	"github.com/vertex-language/amd64/feature"
	"github.com/vertex-language/amd64/reg"
	"github.com/vertex-language/ir"
	"github.com/vertex-language/ir/lower/mir"
	"github.com/vertex-language/ir/lower/regalloc"
)

// seqOp is one instruction, or a short fixed sequence of them, with its
// registers Defs then Uses.
type seqOp struct {
	name string
	emit func(t *amd64asm.Section, r []regalloc.PhysReg)
}

func (o seqOp) String() string { return o.name }

func x(r regalloc.PhysReg) reg.Xmm { return reg.Xmm(r) }

// lowered is the capabilities this package selects instructions for.
var lowered = ir.Capabilities(0).With(ir.CapAES, ir.CapClmul, ir.CapSHA1, ir.CapSHA256, ir.CapCRC32C)

// capabilitiesOf is the IR capabilities a feature set gives: what
// LegalizeCryptoOn may leave as instructions in any function.
func capabilitiesOf(f feature.Set) ir.Capabilities {
	var c ir.Capabilities
	if f.Has(feature.AES) {
		c = c.With(ir.CapAES)
	}
	if f.Has(feature.PCLMULQDQ) {
		c = c.With(ir.CapClmul)
	}
	if f.Has(feature.SHA) {
		c = c.With(ir.CapSHA1, ir.CapSHA256)
	}
	if f.Has(feature.SSE42) {
		c = c.With(ir.CapCRC32C)
	}
	return c
}

// featuresOf is the features a function that may use cs has. Each brings
// SSSE3 with it: no processor has AES-NI, PCLMULQDQ or SHA without it, and
// SSE4.2 requires it architecturally, so a function that requires one of
// them may permute bytes with PSHUFB -- which is how a hash loads its block
// and GHASH reverses one.
func featuresOf(cs ir.Capabilities) []feature.Feature {
	var out []feature.Feature
	for _, c := range cs.List() {
		switch c {
		case ir.CapAES:
			out = append(out, feature.AES, feature.SSSE3)
		case ir.CapClmul:
			out = append(out, feature.PCLMULQDQ, feature.SSSE3)
		case ir.CapSHA1, ir.CapSHA256:
			out = append(out, feature.SHA, feature.SSSE3)
		case ir.CapCRC32C:
			out = append(out, feature.SSE42)
		}
	}
	return out
}

// funcFeatures is the processor fn's body may assume: the module's, and
// what the capabilities it may use imply.
func funcFeatures(module feature.Set, fn *ir.Func) feature.Set {
	return module.Add(featuresOf(capabilitiesOf(module) | fn.RequiresAttr()&lowered)...)
}

// expandFor is ExpandVectorVerbs' question for a function with feat: the
// verbs with no instruction there. Only the byte permutes need SSSE3; the
// rotates and the three-operand xors are sequences here on every x86-64.
func expandFor(feat feature.Set) func(ir.Verb) bool {
	return func(v ir.Verb) bool {
		switch v {
		case ir.VI16x8Bswap, ir.VI32x4Bswap, ir.VI64x2Bswap, ir.VI8x16Swizzle:
			return !feat.Has(feature.SSSE3)
		}
		return false
	}
}

// cryptoVerb reports whether this file lowers in.
func cryptoVerb(in *ir.Inst) bool {
	op := in.Op()
	if _, ok := op.Verb.Capability(); ok {
		return true
	}
	if op.Type != ir.TypeV128 {
		return false
	}
	switch op.Verb {
	case ir.VVecXor3, ir.VVecXorAndNot, ir.VI32x4Rotl, ir.VI64x2Rotl,
		ir.VI16x8Bswap, ir.VI32x4Bswap, ir.VI64x2Bswap, ir.VI8x16Swizzle:
		return true
	}
	return false
}

// xsel is the selection state of one instruction.
type xsel struct {
	c  *cursor
	vr *vregs
	in *ir.Inst
}

func (s *xsel) fail(why string) error { return fmt.Errorf("%s: %s", s.in.Op(), why) }

func (s *xsel) args() ([]mir.VReg, error) {
	out := make([]mir.VReg, s.in.NumArgs())
	for i := range out {
		r, ok := s.vr.lookup(s.in.Arg(i))
		if !ok {
			return nil, s.fail(fmt.Sprintf("operand %d defined outside the function", i))
		}
		out[i] = r
	}
	return out, nil
}

func (s *xsel) lit() (int64, error) {
	k, ok := s.in.Lit()
	if !ok || k.Kind() != ir.ConstInt {
		return 0, s.fail("needs its literal")
	}
	return k.Int(), nil
}

func (s *xsel) result() (mir.VReg, error) {
	r, err := s.vr.define(s.in.Result(0))
	if err != nil {
		return 0, s.fail(err.Error())
	}
	return r, nil
}

func (s *xsel) temp() mir.VReg { return s.vr.temp(wv128) }

func (s *xsel) op(name string, defs, uses []mir.VReg, f func(t *amd64asm.Section, r []regalloc.PhysReg)) {
	s.c.Emit(mir.Instr{Op: seqOp{name: name, emit: f}, Defs: defs, Uses: uses})
}

// two is the two-address shape: r[0] = r[1] op r[2].
func (s *xsel) two(name string, d, a, b mir.VReg, f func(*amd64asm.Section, reg.Xmm, reg.Xmm)) {
	s.op(name, []mir.VReg{d}, []mir.VReg{a, b}, func(t *amd64asm.Section, r []regalloc.PhysReg) {
		t.MovdqaXmmRM128(x(r[0]), x(r[1]))
		f(t, x(r[0]), x(r[2]))
	})
}

// bytes is a fresh vreg holding sixteen bytes, from .rodata.
func (s *xsel) bytes(b [16]byte) mir.VReg {
	var lo, hi uint64
	for i := 0; i < 8; i++ {
		lo |= uint64(b[i]) << (8 * i)
		hi |= uint64(b[8+i]) << (8 * i)
	}
	d := s.temp()
	s.c.Emit(mir.Instr{Op: wideConstOp{lo: lo, hi: hi}, Defs: []mir.VReg{d}})
	return d
}

// splat32 is a fresh vreg holding k in every doubleword.
func (s *xsel) splat32(k uint32) mir.VReg {
	var b [16]byte
	for i := range b {
		b[i] = byte(k >> (8 * (i % 4)))
	}
	return s.bytes(b)
}

// permute is PSHUFB by a pattern: byte i of the result is byte idx[i].
func (s *xsel) permute(d, a mir.VReg, idx [16]byte) {
	s.two("pshufb", d, a, s.bytes(idx), func(t *amd64asm.Section, d, m reg.Xmm) { t.PshufbXmmRM128(d, m) })
}

// laneBytes is the PSHUFB pattern that moves byte j of every lane of size
// bytes to byte to(j).
func laneBytes(size int, from func(j int) int) [16]byte {
	var idx [16]byte
	for i := range idx {
		lane := i / size * size
		idx[i] = byte(lane + from(i%size))
	}
	return idx
}

// iselCrypto lowers one instruction cryptoVerb accepts.
func iselCrypto(c *cursor, vr *vregs, in *ir.Inst) error {
	s := &xsel{c: c, vr: vr, in: in}
	verb := in.Op().Verb
	if in.Op().Type == ir.TypeI32 {
		return s.crc()
	}
	switch verb {
	case ir.VVecXor3, ir.VVecXorAndNot:
		return s.xor3(verb)
	case ir.VI32x4Rotl, ir.VI64x2Rotl:
		return s.rotl(verb == ir.VI64x2Rotl)
	case ir.VI16x8Bswap, ir.VI32x4Bswap, ir.VI64x2Bswap:
		return s.bswap(verb)
	case ir.VI8x16Swizzle:
		return s.swizzle()
	case ir.VSha256RoundsAbcd, ir.VSha256RoundsEfgh:
		return s.sha256Rounds(verb == ir.VSha256RoundsEfgh)
	case ir.VSha1RoundsCh, ir.VSha1RoundsParity, ir.VSha1RoundsMaj:
		return s.sha1Rounds(verb)
	}
	return s.crypto()
}

// ---- §V -------------------------------------------------------------------

// xor3 is two PXORs; xor_andnot is PANDN, whose destination is the operand
// it negates, then PXOR.
func (s *xsel) xor3(verb ir.Verb) error {
	a, err := s.args()
	if err != nil {
		return err
	}
	d, err := s.result()
	if err != nil {
		return err
	}
	if verb == ir.VVecXor3 {
		s.op("xor3", []mir.VReg{d}, a, func(t *amd64asm.Section, r []regalloc.PhysReg) {
			t.MovdqaXmmRM128(x(r[0]), x(r[1]))
			t.PxorXmmRM128(x(r[0]), x(r[2]))
			t.PxorXmmRM128(x(r[0]), x(r[3]))
		})
		return nil
	}
	s.op("xor_andnot", []mir.VReg{d, s.temp()}, a, func(t *amd64asm.Section, r []regalloc.PhysReg) {
		d, tmp := x(r[0]), x(r[1])
		t.MovdqaXmmRM128(tmp, x(r[4]))
		t.PandnXmmRM128(tmp, x(r[3]))
		t.MovdqaXmmRM128(d, x(r[2]))
		t.PxorXmmRM128(d, tmp)
	})
	return nil
}

// rotl is a lane rotate by a literal: a word or doubleword shuffle where
// the count is half the lane, PSHUFB where it is whole bytes and the
// function has SSSE3, and two shifts and an OR otherwise. ChaCha20's 16
// and 8 are the first two; its 12 and 7 are the third.
func (s *xsel) rotl(wide bool) error {
	args, err := s.args()
	if err != nil {
		return err
	}
	a := args[0]
	k, err := s.lit()
	if err != nil {
		return err
	}
	d, err := s.result()
	if err != nil {
		return err
	}
	bits := int64(32)
	if wide {
		bits = 64
	}
	k %= bits
	switch {
	case k == 0:
		emitCopy(s.c, d, a, wv128)
	case wide && k == 32:
		s.c.Emit(mir.Instr{Op: vecShuffleOp{verb: ir.VI32x4Shuffle, k: 0xb1}, Defs: []mir.VReg{d}, Uses: []mir.VReg{a}})
	case !wide && k == 16:
		s.op("rotl 16", []mir.VReg{d}, []mir.VReg{a}, func(t *amd64asm.Section, r []regalloc.PhysReg) {
			t.PshuflwXmmRM128Imm8(x(r[0]), x(r[1]), 0xb1)
			t.PshufhwXmmRM128Imm8(x(r[0]), x(r[0]), 0xb1)
		})
	case k%8 == 0 && s.c.features.Has(feature.SSSE3):
		size, n := int(bits/8), int(k/8)
		s.permute(d, a, laneBytes(size, func(j int) int { return (j - n + size) % size }))
	default:
		shl, shr := (*amd64asm.Section).PslldXmmImm8, (*amd64asm.Section).PsrldXmmImm8
		if wide {
			shl, shr = (*amd64asm.Section).PsllqXmmImm8, (*amd64asm.Section).PsrlqXmmImm8
		}
		s.op("rotl", []mir.VReg{d, s.temp()}, []mir.VReg{a}, func(t *amd64asm.Section, r []regalloc.PhysReg) {
			d, tmp, a := x(r[0]), x(r[1]), x(r[2])
			t.MovdqaXmmRM128(tmp, a)
			shl(t, tmp, k)
			t.MovdqaXmmRM128(d, a)
			shr(t, d, bits-k)
			t.PorXmmRM128(d, tmp)
		})
	}
	return nil
}

func (s *xsel) bswap(verb ir.Verb) error {
	a, err := s.args()
	if err != nil {
		return err
	}
	d, err := s.result()
	if err != nil {
		return err
	}
	size := map[ir.Verb]int{ir.VI16x8Bswap: 2, ir.VI32x4Bswap: 4, ir.VI64x2Bswap: 8}[verb]
	s.permute(d, a[0], laneBytes(size, func(j int) int { return size - 1 - j }))
	return nil
}

// swizzle is PSHUFB, which zeroes a byte whose index has its top bit set
// and otherwise reads the index's low four bits. A saturating add of 0x70
// gives every index of 16 or more its top bit and leaves the low four bits
// of the rest, which is §V4's rule.
func (s *xsel) swizzle() error {
	a, err := s.args()
	if err != nil {
		return err
	}
	d, err := s.result()
	if err != nil {
		return err
	}
	var k [16]byte
	for i := range k {
		k[i] = 0x70
	}
	s.op("swizzle", []mir.VReg{d, s.temp()}, []mir.VReg{a[0], a[1], s.bytes(k)}, func(t *amd64asm.Section, r []regalloc.PhysReg) {
		d, idx := x(r[0]), x(r[1])
		t.MovdqaXmmRM128(idx, x(r[3]))
		t.PaddusbXmmRM128(idx, x(r[4]))
		t.MovdqaXmmRM128(d, x(r[2]))
		t.PshufbXmmRM128(d, idx)
	})
	return nil
}

// ---- §A8 ------------------------------------------------------------------

// crc is CRC32 at the data's width. It is the Castagnoli polynomial and
// only that one, so the IEEE verbs never reach here.
func (s *xsel) crc() error {
	a, err := s.args()
	if err != nil {
		return err
	}
	d, err := s.result()
	if err != nil {
		return err
	}
	verb := s.in.Op().Verb
	switch verb {
	case ir.VCrc32cU8, ir.VCrc32cU16, ir.VCrc32cU32, ir.VCrc32cU64:
	default:
		return s.fail("is not lowered for x86-64")
	}
	s.c.Emit(mir.Instr{Op: seqOp{name: string(verb), emit: func(t *amd64asm.Section, r []regalloc.PhysReg) {
		dst := reg.R32(r[0])
		t.MovR32RM32(dst, reg.R32(r[1]))
		switch verb {
		case ir.VCrc32cU8:
			t.Crc32R32RM8(dst, reg.R8(r[2]))
		case ir.VCrc32cU16:
			t.Crc32R32RM16(dst, reg.R16(r[2]))
		case ir.VCrc32cU32:
			t.Crc32R32RM32(dst, reg.R32(r[2]))
		default:
			t.Crc32R64RM64(reg.R64(r[0]), reg.R64(r[2]))
		}
	}}, Defs: []mir.VReg{d}, Uses: a})
	return nil
}

// ---- §V5 ------------------------------------------------------------------

func (s *xsel) crypto() error {
	verb := s.in.Op().Verb
	a, err := s.args()
	if err != nil {
		return err
	}
	d, err := s.result()
	if err != nil {
		return err
	}
	switch verb {
	case ir.VAesEnc, ir.VAesEncLast, ir.VAesDec, ir.VAesDecLast:
		// x86's rounds are the IR's: the key XORed after MixColumns.
		f := map[ir.Verb]func(*amd64asm.Section, reg.Xmm, reg.Xmm){
			ir.VAesEnc:     func(t *amd64asm.Section, d, k reg.Xmm) { t.AesencXmmRM128(d, k) },
			ir.VAesEncLast: func(t *amd64asm.Section, d, k reg.Xmm) { t.AesenclastXmmRM128(d, k) },
			ir.VAesDec:     func(t *amd64asm.Section, d, k reg.Xmm) { t.AesdecXmmRM128(d, k) },
			ir.VAesDecLast: func(t *amd64asm.Section, d, k reg.Xmm) { t.AesdeclastXmmRM128(d, k) },
		}[verb]
		s.two(string(verb), d, a[0], a[1], f)
	case ir.VAesInvMixColumns:
		s.op("aesimc", []mir.VReg{d}, a, func(t *amd64asm.Section, r []regalloc.PhysReg) { t.AesimcXmmRM128(x(r[0]), x(r[1])) })
	case ir.VAesKeygenAssist:
		rcon, err := s.lit()
		if err != nil {
			return err
		}
		s.op("aeskeygenassist", []mir.VReg{d}, a, func(t *amd64asm.Section, r []regalloc.PhysReg) {
			t.AeskeygenassistXmmRM128Imm8(x(r[0]), x(r[1]), rcon&0xff)
		})
	case ir.VI64x2Clmul:
		sel, err := s.lit()
		if err != nil {
			return err
		}
		// The IR's bit 1 is b's lane; PCLMULQDQ's is bit 4.
		imm := sel&1 | (sel>>1&1)<<4
		s.two("pclmulqdq", d, a[0], a[1], func(t *amd64asm.Section, d, b reg.Xmm) { t.PclmulqdqXmmRM128Imm8(d, b, imm) })
	case ir.VSha1Msg1:
		// W[t+2..t+5] is w0's high half and w4's low: SHUFPS. SHA1MSG1
		// computes only the first two terms and in the other lane order.
		s.op("sha1_msg1", []mir.VReg{d}, a, func(t *amd64asm.Section, r []regalloc.PhysReg) {
			t.MovdqaXmmRM128(x(r[0]), x(r[1]))
			t.ShufpsXmmRM128Imm8(x(r[0]), x(r[2]), 0x4e)
			t.PxorXmmRM128(x(r[0]), x(r[1]))
			t.PxorXmmRM128(x(r[0]), x(r[3]))
		})
	case ir.VSha1Msg2:
		// SHA1MSG2 keeps W[t] in the high lane; the IR keeps it in lane 0.
		s.op("sha1_msg2", []mir.VReg{d, s.temp()}, a, func(t *amd64asm.Section, r []regalloc.PhysReg) {
			d, w := x(r[0]), x(r[1])
			t.PshufdXmmRM128Imm8(d, x(r[2]), 0x1b)
			t.PshufdXmmRM128Imm8(w, x(r[3]), 0x1b)
			t.Sha1msg2XmmRM128(d, w)
			t.PshufdXmmRM128Imm8(d, d, 0x1b)
		})
	case ir.VSha256Msg1:
		s.two("sha256msg1", d, a[0], a[1], func(t *amd64asm.Section, d, b reg.Xmm) { t.Sha256msg1XmmRM128(d, b) })
	case ir.VSha256Msg2:
		// SHA256MSG2 takes x + W[t+9..t+12] -- PALIGNR of w12 and w8 -- and
		// adds the σ1 terms, the last two of which depend on the first two.
		s.op("sha256_msg2", []mir.VReg{d}, a, func(t *amd64asm.Section, r []regalloc.PhysReg) {
			d, xx, w8, w12 := x(r[0]), x(r[1]), x(r[2]), x(r[3])
			t.MovdqaXmmRM128(d, w12)
			t.PalignrXmmRM128Imm8(d, w8, 4)
			t.PadddXmmRM128(d, xx)
			t.Sha256msg2XmmRM128(d, w12)
		})
	default:
		return s.fail("is not lowered for x86-64")
	}
	return nil
}

// sha1K is the constant SHA1RNDS4 adds for each of its functions, which
// the IR's wk has added already.
var sha1K = map[ir.Verb]struct {
	fn int64
	k  uint32
}{
	ir.VSha1RoundsCh:     {0, 0x5a827999},
	ir.VSha1RoundsParity: {1, 0x6ed9eba1},
	ir.VSha1RoundsMaj:    {2, 0x8f1bbcdc},
}

// sha1Rounds is SHA1RNDS4, which differs from the IR's verb three ways:
// the state is a in the high lane rather than lane 0; the round constant is
// added by the instruction rather than in wk; and e arrives added to the
// first word of the schedule rather than on its own. So wk less the
// constant, with e added to its first word, is reversed into the
// instruction's order, as is abcd, and the result reversed back. The
// constant subtracted is the one the instruction adds, whichever constant
// the caller added: the parity rounds share an encoding between 20-39 and
// 60-79.
func (s *xsel) sha1Rounds(verb ir.Verb) error {
	a, err := s.args()
	if err != nil {
		return err
	}
	d, err := s.result()
	if err != nil {
		return err
	}
	k := sha1K[verb]
	s.op(string(verb), []mir.VReg{d, s.temp()}, []mir.VReg{a[0], a[1], a[2], s.splat32(k.k)},
		func(t *amd64asm.Section, r []regalloc.PhysReg) {
			d, w := x(r[0]), x(r[1])
			abcd, e, wk, kk := x(r[2]), reg.R32(r[3]), x(r[4]), x(r[5])
			t.MovdXmmRM32(d, e)
			t.PadddXmmRM128(d, wk)
			t.PsubdXmmRM128(d, kk)
			t.PshufdXmmRM128Imm8(w, d, 0x1b)
			t.PshufdXmmRM128Imm8(d, abcd, 0x1b)
			t.Sha1rnds4XmmRM128Imm8(d, w, k.fn)
			t.PshufdXmmRM128Imm8(d, d, 0x1b)
		})
	return nil
}

// sha256Rounds is the two round verbs together, as §V5 says a lowering
// that sees both computes them. SHA256RNDS2 holds the state as ABEF and
// CDGH, each with its first letter in the high lane, does two rounds, and
// reads W+K from XMM0's low quadword: the halves are SHUFPSed into that
// order, the instruction is run twice -- the second time on the high
// quadword of wk -- and the two halves SHUFPSed back. After two rounds the
// CDGH half is the ABEF half that went in, which is why the second round
// reads the first's input.
//
// Whichever of the pair comes first in the block computes both, and leaves
// the other's answer for it to copy; one with no sibling computes both and
// keeps its own.
func (s *xsel) sha256Rounds(efghFirst bool) error {
	d, err := s.result()
	if err != nil {
		return err
	}
	if v, ok := s.c.sha256[s.in]; ok {
		delete(s.c.sha256, s.in)
		emitCopy(s.c, d, v, wv128)
		return nil
	}
	a, err := s.args()
	if err != nil {
		return err
	}
	abcd, efgh, wk := a[0], a[1], a[2]
	if efghFirst {
		abcd, efgh = efgh, abcd
	}

	var sibling *ir.Inst
	want := ir.VSha256RoundsEfgh
	if efghFirst {
		want = ir.VSha256RoundsAbcd
	}
	for _, o := range s.in.Block().Insts() {
		if o != s.in && o.Op().Type == ir.TypeV128 && o.Op().Verb == want &&
			o.Arg(0) == s.in.Arg(1) && o.Arg(1) == s.in.Arg(0) && o.Arg(2) == s.in.Arg(2) {
			if _, taken := s.c.sha256[o]; !taken {
				sibling = o
				break
			}
		}
	}

	abef, cdgh := s.temp(), s.temp()
	s.op("sha256 abef cdgh", []mir.VReg{abef, cdgh}, []mir.VReg{abcd, efgh}, func(t *amd64asm.Section, r []regalloc.PhysReg) {
		t.MovdqaXmmRM128(x(r[0]), x(r[3]))
		t.ShufpsXmmRM128Imm8(x(r[0]), x(r[2]), 0x11)
		t.MovdqaXmmRM128(x(r[1]), x(r[3]))
		t.ShufpsXmmRM128Imm8(x(r[1]), x(r[2]), 0xbb)
	})
	rnds2 := func(t *amd64asm.Section, r []regalloc.PhysReg) {
		t.MovdqaXmmRM128(x(r[0]), x(r[1]))
		t.Sha256rnds2XmmRM128Xmm0(x(r[0]), x(r[2]))
	}
	k0 := s.vr.physicalXmm(reg.XMM0, wv128)
	emitCopy(s.c, k0, wk, wv128)
	s1 := s.temp()
	s.op("sha256rnds2", []mir.VReg{s1}, []mir.VReg{cdgh, abef, k0}, rnds2)
	k1 := s.vr.physicalXmm(reg.XMM0, wv128)
	s.c.Emit(mir.Instr{Op: vecShuffleOp{verb: ir.VI32x4Shuffle, k: 0x0e}, Defs: []mir.VReg{k1}, Uses: []mir.VReg{wk}})
	s2 := s.temp()
	s.op("sha256rnds2", []mir.VReg{s2}, []mir.VReg{abef, s1, k1}, rnds2)

	half := func(dst mir.VReg, efgh bool) {
		imm := int64(0xbb)
		if efgh {
			imm = 0x11
		}
		s.op("sha256 half", []mir.VReg{dst}, []mir.VReg{s2, s1}, func(t *amd64asm.Section, r []regalloc.PhysReg) {
			t.MovdqaXmmRM128(x(r[0]), x(r[1]))
			t.ShufpsXmmRM128Imm8(x(r[0]), x(r[2]), imm)
		})
	}
	half(d, efghFirst)
	if sibling != nil {
		other := s.temp()
		half(other, !efghFirst)
		if s.c.sha256 == nil {
			s.c.sha256 = map[*ir.Inst]mir.VReg{}
		}
		s.c.sha256[sibling] = other
	}
	return nil
}
