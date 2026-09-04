package ir

// V128NS is the v128 namespace, reached through Builder.V128 — §V.
//
// One register type, many lane shapes. A value here is sixteen bytes and
// nothing more; what those bytes mean is the verb's business, which is why
// there is no bitcast in this namespace and nothing to bitcast between. A
// frontend whose source language has several vector types — C's __m128i,
// __m128 and __m128d are one register spelled three ways — maps all of them
// here and loses nothing, because the operations already say which is meant.
//
// Availability is the layout block's, under the same rule as f80 and f128: a
// target with no vector register file records ErrLayout here rather than
// emulating sixteen bytes in two general registers, which would be slower
// than the scalar code the caller would have written instead.
type V128NS struct{ b *Builder }

func (n V128NS) un(v Verb, a V128) V128 {
	return V128{n.b.def1(Op{TypeV128, v}, TypeV128, a.d)}
}

func (n V128NS) bin(v Verb, a, c V128) V128 {
	return V128{n.b.def1(Op{TypeV128, v}, TypeV128, a.d, c.d)}
}

// immOf is the shape of every verb whose second operand the hardware will
// only take as a literal: the lane index of an extract, the pattern of a
// shuffle, the byte count of a whole-register shift.
func (n V128NS) immOf(v Verb, a V128, k int64) V128 {
	return V128{n.b.def1i(Op{TypeV128, v}, TypeV128, []*Def{a.d}, &imm{lit: Int(k), hasLit: true})}
}

// shift is the shape of the lane shifts: one count for the whole register,
// in a general register or a literal, never one count per lane.
func (n V128NS) shift(v Verb, a V128, amt I32) V128 {
	return V128{n.b.def1(Op{TypeV128, v}, TypeV128, a.d, amt.d)}
}

// ---- shape-free -----------------------------------------------------------

// Const emits v128.const. The literal is the register's sixteen bytes in
// memory order, so a caller that has lane values converts them once, here,
// rather than leaving the byte order to be rediscovered per target.
func (n V128NS) Const(b [16]byte) V128 {
	return V128{n.b.def1i(Op{TypeV128, VConst}, TypeV128, nil, &imm{lit: Bytes(b[:]), hasLit: true})}
}

// Zero is the all-zero register, which every target has a cheaper way to
// make than loading sixteen bytes of it.
func (n V128NS) Zero() V128 { return n.Const([16]byte{}) }

func (n V128NS) And(a, c V128) V128    { return n.bin(VAnd, a, c) }
func (n V128NS) Or(a, c V128) V128     { return n.bin(VOr, a, c) }
func (n V128NS) Xor(a, c V128) V128    { return n.bin(VXor, a, c) }
func (n V128NS) Not(a V128) V128       { return n.un(VNot, a) }
func (n V128NS) AndNot(a, c V128) V128 { return n.bin(VVecAndNot, a, c) }

// ShlBytes and ShrBytes shift the whole register by a count in bytes, zeroes
// shifted in. A count of sixteen or more yields zero.
func (n V128NS) ShlBytes(a V128, k int64) V128 { return n.immOf(VVecShlBytes, a, k) }
func (n V128NS) ShrBytes(a V128, k int64) V128 { return n.immOf(VVecShrBytes, a, k) }

// ZExtI32 and ZExtI64 put a scalar in the low lane and zero the rest. This
// is not a splat, and the difference is load-bearing: every use of it is
// building a vector out of one scalar without filling the lanes a splat
// would fill.
func (n V128NS) ZExtI32(a I32) V128 {
	return V128{n.b.def1(Op{TypeV128, VVecZExtI32}, TypeV128, a.d)}
}

func (n V128NS) ZExtI64(a I64) V128 {
	return V128{n.b.def1(Op{TypeV128, VVecZExtI64}, TypeV128, a.d)}
}

// ---- i8x16 ----------------------------------------------------------------

func (n V128NS) I8x16Add(a, c V128) V128     { return n.bin(VI8x16Add, a, c) }
func (n V128NS) I8x16Sub(a, c V128) V128     { return n.bin(VI8x16Sub, a, c) }
func (n V128NS) I8x16AddSatS(a, c V128) V128 { return n.bin(VI8x16AddSatS, a, c) }
func (n V128NS) I8x16AddSatU(a, c V128) V128 { return n.bin(VI8x16AddSatU, a, c) }
func (n V128NS) I8x16SubSatS(a, c V128) V128 { return n.bin(VI8x16SubSatS, a, c) }
func (n V128NS) I8x16SubSatU(a, c V128) V128 { return n.bin(VI8x16SubSatU, a, c) }
func (n V128NS) I8x16MinU(a, c V128) V128    { return n.bin(VI8x16MinU, a, c) }
func (n V128NS) I8x16MaxU(a, c V128) V128    { return n.bin(VI8x16MaxU, a, c) }
func (n V128NS) I8x16AvgrU(a, c V128) V128   { return n.bin(VI8x16AvgrU, a, c) }

// I8x16Eq and I8x16GtS produce a mask: a lane is all-ones where the
// comparison holds and all-zero where it does not. That is what makes the
// result a v128 and not the i1 a scalar comparison yields — there are
// sixteen answers, and they are used as operands to the bitwise verbs.
func (n V128NS) I8x16Eq(a, c V128) V128  { return n.bin(VI8x16Eq, a, c) }
func (n V128NS) I8x16GtS(a, c V128) V128 { return n.bin(VI8x16GtS, a, c) }

// I8x16Splat fills all sixteen lanes with the low byte of a.
func (n V128NS) I8x16Splat(a I32) V128 {
	return V128{n.b.def1(Op{TypeV128, VI8x16Splat}, TypeV128, a.d)}
}

// I8x16Bitmask gathers the top bit of each lane into the low sixteen bits of
// an i32, lane zero in bit zero. It is how a mask leaves the vector file and
// becomes something a branch can test.
func (n V128NS) I8x16Bitmask(a V128) I32 {
	return I32{n.b.def1(Op{TypeV128, VI8x16Bitmask}, TypeI32, a.d)}
}

// I8x16NarrowS and I8x16NarrowU take two i16x8 and produce one i8x16, a's
// lanes low, saturating to the signed or unsigned byte range.
func (n V128NS) I8x16NarrowS(a, c V128) V128 { return n.bin(VI8x16NarrowS, a, c) }
func (n V128NS) I8x16NarrowU(a, c V128) V128 { return n.bin(VI8x16NarrowU, a, c) }

// I8x16UnpackLow interleaves the low eight bytes of each operand, a's byte
// first; UnpackHigh does the same to the high eight. Widening a vector is
// unpacking it against zero, and duplicating a lane is unpacking it against
// itself.
func (n V128NS) I8x16UnpackLow(a, c V128) V128  { return n.bin(VI8x16UnpackLow, a, c) }
func (n V128NS) I8x16UnpackHigh(a, c V128) V128 { return n.bin(VI8x16UnpackHi, a, c) }

// I8x16SadU sums the absolute differences of each half's eight bytes into
// the low word of the corresponding quadword, the rest zero.
func (n V128NS) I8x16SadU(a, c V128) V128 { return n.bin(VI8x16SadU, a, c) }

// ---- i16x8 ----------------------------------------------------------------

func (n V128NS) I16x8Add(a, c V128) V128     { return n.bin(VI16x8Add, a, c) }
func (n V128NS) I16x8Sub(a, c V128) V128     { return n.bin(VI16x8Sub, a, c) }
func (n V128NS) I16x8Mul(a, c V128) V128     { return n.bin(VI16x8Mul, a, c) }
func (n V128NS) I16x8MulHiS(a, c V128) V128  { return n.bin(VI16x8MulHiS, a, c) }
func (n V128NS) I16x8MulHiU(a, c V128) V128  { return n.bin(VI16x8MulHiU, a, c) }
func (n V128NS) I16x8AddSatS(a, c V128) V128 { return n.bin(VI16x8AddSatS, a, c) }
func (n V128NS) I16x8AddSatU(a, c V128) V128 { return n.bin(VI16x8AddSatU, a, c) }
func (n V128NS) I16x8SubSatS(a, c V128) V128 { return n.bin(VI16x8SubSatS, a, c) }
func (n V128NS) I16x8SubSatU(a, c V128) V128 { return n.bin(VI16x8SubSatU, a, c) }
func (n V128NS) I16x8MinS(a, c V128) V128    { return n.bin(VI16x8MinS, a, c) }
func (n V128NS) I16x8MaxS(a, c V128) V128    { return n.bin(VI16x8MaxS, a, c) }
func (n V128NS) I16x8AvgrU(a, c V128) V128   { return n.bin(VI16x8AvgrU, a, c) }
func (n V128NS) I16x8Eq(a, c V128) V128      { return n.bin(VI16x8Eq, a, c) }
func (n V128NS) I16x8GtS(a, c V128) V128     { return n.bin(VI16x8GtS, a, c) }

// The lane shifts take one count for the whole register. A count at or past
// the lane's width yields zero lanes, and for ShrS lanes of the sign bit —
// which is the useful answer and not the scalar namespaces' rule, where the
// count is taken modulo the width instead.
func (n V128NS) I16x8Shl(a V128, amt I32) V128  { return n.shift(VI16x8Shl, a, amt) }
func (n V128NS) I16x8ShrS(a V128, amt I32) V128 { return n.shift(VI16x8ShrS, a, amt) }
func (n V128NS) I16x8ShrU(a V128, amt I32) V128 { return n.shift(VI16x8ShrU, a, amt) }

func (n V128NS) I16x8Splat(a I32) V128 {
	return V128{n.b.def1(Op{TypeV128, VI16x8Splat}, TypeV128, a.d)}
}

// I16x8ExtractLaneU reads one lane out, zero-extended into an i32. The index
// is a literal because the instruction's is.
func (n V128NS) I16x8ExtractLaneU(a V128, lane int64) I32 {
	return I32{n.b.def1i(Op{TypeV128, VI16x8ExtractU}, TypeI32, []*Def{a.d},
		&imm{lit: Int(lane), hasLit: true})}
}

// I16x8ReplaceLane writes one lane and leaves the other seven.
func (n V128NS) I16x8ReplaceLane(a V128, v I32, lane int64) V128 {
	return V128{n.b.def1i(Op{TypeV128, VI16x8Replace}, TypeV128, []*Def{a.d, v.d},
		&imm{lit: Int(lane), hasLit: true})}
}

func (n V128NS) I16x8NarrowS(a, c V128) V128    { return n.bin(VI16x8NarrowS, a, c) }
func (n V128NS) I16x8UnpackLow(a, c V128) V128  { return n.bin(VI16x8UnpackLow, a, c) }
func (n V128NS) I16x8UnpackHigh(a, c V128) V128 { return n.bin(VI16x8UnpackHi, a, c) }

// The word shuffles permute one half and copy the other through, which is
// the only word-granularity permute the hardware has.
func (n V128NS) I16x8ShuffleLow(a V128, p int64) V128  { return n.immOf(VI16x8ShuffleLow, a, p) }
func (n V128NS) I16x8ShuffleHigh(a V128, p int64) V128 { return n.immOf(VI16x8ShuffleHi, a, p) }

// I16x8MaddS multiplies each pair of adjacent signed words and adds the two
// products, producing four doublewords.
func (n V128NS) I16x8MaddS(a, c V128) V128 { return n.bin(VI16x8MaddS, a, c) }

// ---- i32x4 ----------------------------------------------------------------

func (n V128NS) I32x4Add(a, c V128) V128 { return n.bin(VI32x4Add, a, c) }
func (n V128NS) I32x4Sub(a, c V128) V128 { return n.bin(VI32x4Sub, a, c) }
func (n V128NS) I32x4Eq(a, c V128) V128  { return n.bin(VI32x4Eq, a, c) }
func (n V128NS) I32x4GtS(a, c V128) V128 { return n.bin(VI32x4GtS, a, c) }

func (n V128NS) I32x4Shl(a V128, amt I32) V128  { return n.shift(VI32x4Shl, a, amt) }
func (n V128NS) I32x4ShrS(a V128, amt I32) V128 { return n.shift(VI32x4ShrS, a, amt) }
func (n V128NS) I32x4ShrU(a V128, amt I32) V128 { return n.shift(VI32x4ShrU, a, amt) }

func (n V128NS) I32x4Splat(a I32) V128 {
	return V128{n.b.def1(Op{TypeV128, VI32x4Splat}, TypeV128, a.d)}
}

// I32x4Shuffle permutes the four doublewords by four two-bit fields of p,
// the low field selecting the low result lane.
func (n V128NS) I32x4Shuffle(a V128, p int64) V128 { return n.immOf(VI32x4Shuffle, a, p) }

func (n V128NS) I32x4UnpackLow(a, c V128) V128  { return n.bin(VI32x4UnpackLow, a, c) }
func (n V128NS) I32x4UnpackHigh(a, c V128) V128 { return n.bin(VI32x4UnpackHi, a, c) }

// I32x4MulEvenU multiplies the two even doubleword lanes of each operand as
// unsigned and produces two quadwords. The odd lanes are not read.
func (n V128NS) I32x4MulEvenU(a, c V128) V128 { return n.bin(VI32x4MulEvenU, a, c) }

// ---- i64x2 ----------------------------------------------------------------

func (n V128NS) I64x2Add(a, c V128) V128 { return n.bin(VI64x2Add, a, c) }
func (n V128NS) I64x2Sub(a, c V128) V128 { return n.bin(VI64x2Sub, a, c) }

func (n V128NS) I64x2Shl(a V128, amt I32) V128  { return n.shift(VI64x2Shl, a, amt) }
func (n V128NS) I64x2ShrU(a V128, amt I32) V128 { return n.shift(VI64x2ShrU, a, amt) }

// There is no I64x2ShrS. No SSE2 instruction shifts a quadword arithmetically
// — the first one to do so is AVX-512's — and a verb that quietly lowered to
// five instructions would hide that from the one caller who cares.

func (n V128NS) I64x2Splat(a I64) V128 {
	return V128{n.b.def1(Op{TypeV128, VI64x2Splat}, TypeV128, a.d)}
}

func (n V128NS) I64x2UnpackLow(a, c V128) V128  { return n.bin(VI64x2UnpackLow, a, c) }
func (n V128NS) I64x2UnpackHigh(a, c V128) V128 { return n.bin(VI64x2UnpackHi, a, c) }

// ---- lanes out of the register --------------------------------------------

// I32x4ExtractLane and I64x2ExtractLane read one lane into a general
// register. Lane zero is one instruction on every target; a higher lane is a
// permute and then that instruction, which is why the index is a literal —
// a variable one is a store and a scalar load, and the frontend writes that
// where it means it.
func (n V128NS) I32x4ExtractLane(a V128, lane int64) I32 {
	return I32{n.b.def1i(Op{TypeV128, VI32x4Extract}, TypeI32, []*Def{a.d},
		&imm{lit: Int(lane), hasLit: true})}
}

func (n V128NS) I64x2ExtractLane(a V128, lane int64) I64 {
	return I64{n.b.def1i(Op{TypeV128, VI64x2Extract}, TypeI64, []*Def{a.d},
		&imm{lit: Int(lane), hasLit: true})}
}
