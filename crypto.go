package ir

import "strings"

// §P's capabilities, §V5's cryptographic verbs and §A8's checksums.
//
// A capability names a group of verbs by what they compute, not by a
// vendor's feature bit: aes is AES-NI on one target and FEAT_AES on the
// other, and the module says aes on both. Every verb here is defined on
// every target that has v128 (§0): on a processor without the instruction,
// lowering makes it a call that answers the same, in constant time. A
// function that Requires a capability may use its instructions directly
// whatever the module's baseline is, and i1.cpu_has is how a caller picks
// between that function and the portable one.

// A Capability is one group of §A8 and §V5 verbs.
type Capability uint8

const (
	CapAES Capability = iota + 1
	CapClmul
	CapSHA1
	CapSHA256
	CapSHA512
	CapSM3
	CapSM4
	CapCRC32
	CapCRC32C

	capCount
)

var capNames = [capCount]string{
	CapAES: "aes", CapClmul: "clmul", CapSHA1: "sha1", CapSHA256: "sha256",
	CapSHA512: "sha512", CapSM3: "sm3", CapSM4: "sm4",
	CapCRC32: "crc32", CapCRC32C: "crc32c",
}

// Valid reports whether c is one of the capabilities.
func (c Capability) Valid() bool { return c > 0 && c < capCount }

func (c Capability) String() string {
	if !c.Valid() {
		return "<invalid capability>"
	}
	return capNames[c]
}

// ParseCapability is the capability spelled s, the way the text spells it.
func ParseCapability(s string) (Capability, bool) {
	for c := CapAES; c < capCount; c++ {
		if capNames[c] == s {
			return c, true
		}
	}
	return 0, false
}

// Capabilities is a set of them. The zero value is empty.
type Capabilities uint16

// With is cs added to the set.
func (s Capabilities) With(cs ...Capability) Capabilities {
	for _, c := range cs {
		if c.Valid() {
			s |= 1 << c
		}
	}
	return s
}

// Has reports whether c is in the set.
func (s Capabilities) Has(c Capability) bool { return c.Valid() && s&(1<<c) != 0 }

// List is the set's members, in the order the spec lists them.
func (s Capabilities) List() []Capability {
	var out []Capability
	for c := CapAES; c < capCount; c++ {
		if s.Has(c) {
			out = append(out, c)
		}
	}
	return out
}

func (s Capabilities) String() string {
	names := make([]string, 0, capCount)
	for _, c := range s.List() {
		names = append(names, c.String())
	}
	return strings.Join(names, ", ")
}

// Capability is the capability a verb of §A8 or §V5 belongs to, and
// whether it belongs to one.
func (v Verb) Capability() (Capability, bool) {
	switch v {
	case VAesEnc, VAesEncLast, VAesDec, VAesDecLast, VAesInvMixColumns, VAesKeygenAssist:
		return CapAES, true
	case VI64x2Clmul:
		return CapClmul, true
	case VSha1RoundsCh, VSha1RoundsParity, VSha1RoundsMaj, VSha1Msg1, VSha1Msg2:
		return CapSHA1, true
	case VSha256RoundsAbcd, VSha256RoundsEfgh, VSha256Msg1, VSha256Msg2:
		return CapSHA256, true
	case VSha512Rounds1, VSha512Rounds2, VSha512Msg1, VSha512Msg2:
		return CapSHA512, true
	case VSm3Ss1, VSm3Tt1a, VSm3Tt1b, VSm3Tt2a, VSm3Tt2b, VSm3Partw1, VSm3Partw2:
		return CapSM3, true
	case VSm4Rounds, VSm4Key:
		return CapSM4, true
	case VCrc32U8, VCrc32U16, VCrc32U32, VCrc32U64:
		return CapCRC32, true
	case VCrc32cU8, VCrc32cU16, VCrc32cU32, VCrc32cU64:
		return CapCRC32C, true
	}
	return 0, false
}

// CpuHas is whether the processor running this has c: 1 at lowering where
// the module's baseline includes it, otherwise the runtime's answer, asked
// of the processor the first time.
func (n I1NS) CpuHas(c Capability) I1 {
	op := Op{TypeI1, VCpuHas}
	if !c.Valid() {
		n.b.fail(op, ErrPlacement, "capability %d; see §P for the list", c)
		return I1{}
	}
	return I1{n.b.def1i(op, TypeI1, nil, &imm{capa: c, hasCapa: true})}
}

// ---- §A8 ------------------------------------------------------------------

func (n I32NS) crc(v Verb, crc I32, data *Def) I32 {
	return I32{n.b.def1(Op{TypeI32, v}, TypeI32, crc.d, data)}
}

// Crc32cU8, U16 and U32 fold the low 8, 16 or 32 bits of data into crc,
// by the Castagnoli polynomial; Crc32cU64 folds all of an i64. No
// inversion either side: that is the protocol's, not the polynomial's.
func (n I32NS) Crc32cU8(crc, data I32) I32  { return n.crc(VCrc32cU8, crc, data.d) }
func (n I32NS) Crc32cU16(crc, data I32) I32 { return n.crc(VCrc32cU16, crc, data.d) }
func (n I32NS) Crc32cU32(crc, data I32) I32 { return n.crc(VCrc32cU32, crc, data.d) }
func (n I32NS) Crc32cU64(crc I32, data I64) I32 {
	return n.crc(VCrc32cU64, crc, data.d)
}

// Crc32U8 to Crc32U64 are the same by the IEEE 802.3 polynomial.
func (n I32NS) Crc32U8(crc, data I32) I32      { return n.crc(VCrc32U8, crc, data.d) }
func (n I32NS) Crc32U16(crc, data I32) I32     { return n.crc(VCrc32U16, crc, data.d) }
func (n I32NS) Crc32U32(crc, data I32) I32     { return n.crc(VCrc32U32, crc, data.d) }
func (n I32NS) Crc32U64(crc I32, data I64) I32 { return n.crc(VCrc32U64, crc, data.d) }

// ---- §V5 ------------------------------------------------------------------

func (n V128NS) tern(v Verb, a, c, d V128) V128 {
	return V128{n.b.def1(Op{TypeV128, v}, TypeV128, a.d, c.d, d.d)}
}

// binImm is the shape of a two-operand verb with a literal: clmul's lane
// selector, xar's rotation.
func (n V128NS) binImm(v Verb, a, c V128, k int64) V128 {
	return V128{n.b.def1i(Op{TypeV128, v}, TypeV128, []*Def{a.d, c.d}, &imm{lit: Int(k), hasLit: true})}
}

func (n V128NS) ternImm(v Verb, a, c, d V128, k int64) V128 {
	return V128{n.b.def1i(Op{TypeV128, v}, TypeV128, []*Def{a.d, c.d, d.d}, &imm{lit: Int(k), hasLit: true})}
}

// literal checks a verb's literal against its range before it is built.
func (n V128NS) literal(v Verb, k, max int64) bool {
	if k < 0 || k > max {
		n.b.fail(Op{TypeV128, v}, ErrPlacement, "literal %d; the range is 0 to %d", k, max)
		return false
	}
	return true
}

// I8x16Swizzle is table[index[i]] for each byte, zero where an index is 16
// or more.
func (n V128NS) I8x16Swizzle(table, index V128) V128 { return n.bin(VI8x16Swizzle, table, index) }

// I16x8Bswap, I32x4Bswap and I64x2Bswap reverse each lane's bytes: the
// big-endian load a hash starts every block with.
func (n V128NS) I16x8Bswap(a V128) V128 { return n.un(VI16x8Bswap, a) }
func (n V128NS) I32x4Bswap(a V128) V128 { return n.un(VI32x4Bswap, a) }
func (n V128NS) I64x2Bswap(a V128) V128 { return n.un(VI64x2Bswap, a) }

// AesEnc is one AES round: SubBytes, ShiftRows, MixColumns, then xor key.
func (n V128NS) AesEnc(state, key V128) V128 { return n.bin(VAesEnc, state, key) }

// AesEncLast is the final round, which has no MixColumns.
func (n V128NS) AesEncLast(state, key V128) V128 { return n.bin(VAesEncLast, state, key) }

// AesDec is a round of the equivalent inverse cipher: InvShiftRows,
// InvSubBytes, InvMixColumns, then xor key -- a key that is
// AesInvMixColumns of the encryption round key.
func (n V128NS) AesDec(state, key V128) V128 { return n.bin(VAesDec, state, key) }

// AesDecLast is the final inverse round, which has no InvMixColumns.
func (n V128NS) AesDecLast(state, key V128) V128 { return n.bin(VAesDecLast, state, key) }

func (n V128NS) AesInvMixColumns(a V128) V128 { return n.un(VAesInvMixColumns, a) }

// AesKeygenAssist is SubWord(x1), RotWord(SubWord(x1))^rcon, SubWord(x3),
// RotWord(SubWord(x3))^rcon, lane 0 first.
func (n V128NS) AesKeygenAssist(a V128, rcon int64) V128 {
	if !n.literal(VAesKeygenAssist, rcon, 255) {
		return V128{}
	}
	return n.immOf(VAesKeygenAssist, a, rcon)
}

// I64x2Clmul is the 128-bit carry-less product of one lane of a and one of
// b: bit 0 of sel picks a's high lane, bit 1 b's.
func (n V128NS) I64x2Clmul(a, b V128, sel int64) V128 {
	if !n.literal(VI64x2Clmul, sel, 3) {
		return V128{}
	}
	return n.binImm(VI64x2Clmul, a, b, sel)
}

func (n V128NS) sha1(v Verb, abcd V128, e I32, wk V128) V128 {
	return V128{n.b.def1(Op{TypeV128, v}, TypeV128, abcd.d, e.d, wk.d)}
}

// Sha1RoundsCh, Parity and Maj are four SHA-1 rounds of that function: abcd
// after them, given e and wk = W[t..t+3] + K.
func (n V128NS) Sha1RoundsCh(abcd V128, e I32, wk V128) V128 {
	return n.sha1(VSha1RoundsCh, abcd, e, wk)
}
func (n V128NS) Sha1RoundsParity(abcd V128, e I32, wk V128) V128 {
	return n.sha1(VSha1RoundsParity, abcd, e, wk)
}
func (n V128NS) Sha1RoundsMaj(abcd V128, e I32, wk V128) V128 {
	return n.sha1(VSha1RoundsMaj, abcd, e, wk)
}

// Sha1Msg1 is W[t..t+3] ^ W[t+2..t+5] ^ W[t+8..t+11]; Sha1Msg2 finishes
// W[t+16..t+19] from it and W[t+12..t+15].
func (n V128NS) Sha1Msg1(w0, w4, w8 V128) V128 { return n.tern(VSha1Msg1, w0, w4, w8) }
func (n V128NS) Sha1Msg2(x, w12 V128) V128     { return n.bin(VSha1Msg2, x, w12) }

// Sha256RoundsAbcd and Sha256RoundsEfgh are the two halves of the state
// after the same four rounds, from the state before them.
func (n V128NS) Sha256RoundsAbcd(abcd, efgh, wk V128) V128 {
	return n.tern(VSha256RoundsAbcd, abcd, efgh, wk)
}
func (n V128NS) Sha256RoundsEfgh(efgh, abcd, wk V128) V128 {
	return n.tern(VSha256RoundsEfgh, efgh, abcd, wk)
}

// Sha256Msg1 is W[t..t+3] + σ0(W[t+1..t+4]); Sha256Msg2 finishes
// W[t+16..t+19] from it, W[t+8..t+11] and W[t+12..t+15].
func (n V128NS) Sha256Msg1(w0, w4 V128) V128     { return n.bin(VSha256Msg1, w0, w4) }
func (n V128NS) Sha256Msg2(x, w8, w12 V128) V128 { return n.tern(VSha256Msg2, x, w8, w12) }

// Sha512Rounds1, Sha512Rounds2, Sha512Msg1 and Sha512Msg2 are Arm's
// SHA512H, SHA512H2, SHA512SU0 and SHA512SU1, operand for operand.
func (n V128NS) Sha512Rounds1(a, b, c V128) V128 { return n.tern(VSha512Rounds1, a, b, c) }
func (n V128NS) Sha512Rounds2(a, b, c V128) V128 { return n.tern(VSha512Rounds2, a, b, c) }
func (n V128NS) Sha512Msg1(a, b V128) V128       { return n.bin(VSha512Msg1, a, b) }
func (n V128NS) Sha512Msg2(a, b, c V128) V128    { return n.tern(VSha512Msg2, a, b, c) }

// Xor3 is a ^ b ^ c; XorAndNot is a ^ (b AND NOT c).
func (n V128NS) Xor3(a, b, c V128) V128      { return n.tern(VVecXor3, a, b, c) }
func (n V128NS) XorAndNot(a, b, c V128) V128 { return n.tern(VVecXorAndNot, a, b, c) }

// I32x4Rotl and I64x2Rotl rotate each lane left by a literal count.
func (n V128NS) I32x4Rotl(a V128, k int64) V128 {
	if !n.literal(VI32x4Rotl, k, 31) {
		return V128{}
	}
	return n.immOf(VI32x4Rotl, a, k)
}
func (n V128NS) I64x2Rotl(a V128, k int64) V128 {
	if !n.literal(VI64x2Rotl, k, 63) {
		return V128{}
	}
	return n.immOf(VI64x2Rotl, a, k)
}

// The SM3 verbs are Arm's SM3SS1, SM3TT1A/B, SM3TT2A/B and SM3PARTW1/2;
// the TT verbs take the lane of their third operand as a literal.
func (n V128NS) Sm3Ss1(a, b, c V128) V128    { return n.tern(VSm3Ss1, a, b, c) }
func (n V128NS) Sm3Partw1(a, b, c V128) V128 { return n.tern(VSm3Partw1, a, b, c) }
func (n V128NS) Sm3Partw2(a, b, c V128) V128 { return n.tern(VSm3Partw2, a, b, c) }
func (n V128NS) sm3tt(v Verb, a, b, c V128, lane int64) V128 {
	if !n.literal(v, lane, 3) {
		return V128{}
	}
	return n.ternImm(v, a, b, c, lane)
}
func (n V128NS) Sm3Tt1a(a, b, c V128, lane int64) V128 { return n.sm3tt(VSm3Tt1a, a, b, c, lane) }
func (n V128NS) Sm3Tt1b(a, b, c V128, lane int64) V128 { return n.sm3tt(VSm3Tt1b, a, b, c, lane) }
func (n V128NS) Sm3Tt2a(a, b, c V128, lane int64) V128 { return n.sm3tt(VSm3Tt2a, a, b, c, lane) }
func (n V128NS) Sm3Tt2b(a, b, c V128, lane int64) V128 { return n.sm3tt(VSm3Tt2b, a, b, c, lane) }

// Sm4Rounds is four SM4 rounds under four round keys; Sm4Key is the next
// four round keys from these four and four constants.
func (n V128NS) Sm4Rounds(state, rk V128) V128 { return n.bin(VSm4Rounds, state, rk) }
func (n V128NS) Sm4Key(rk, ck V128) V128       { return n.bin(VSm4Key, rk, ck) }
