package ir

// A Verb is the operation half of a mnemonic. A mnemonic is either Type.Verb,
// parameterized over a machine type, or bare — §17.
type Verb string

// An Op is a mnemonic: a reg-type namespace and a verb. Type is TypeNone for
// the bare set. A dot in the text form means exactly one thing: this opcode is
// parameterized over a machine type.
type Op struct {
	Type RegType
	Verb Verb
}

func (o Op) String() string {
	if o.Type == TypeNone {
		return string(o.Verb)
	}
	return o.Type.String() + "." + string(o.Verb)
}

// IsBare reports whether o is a member of §17's bare set.
func (o Op) IsBare() bool { return o.Type == TypeNone }

// IsTerminator reports whether o ends a block (§G2, §G3, §G4).
func (o Op) IsTerminator() bool {
	if o.Type != TypeNone {
		return false
	}
	switch o.Verb {
	case VBr, VBrIf, VBrTable, VBrInd, VReturn, VTrap,
		VInvoke, VInvokeInd, VResume, VAsmGoto:
		return true
	}
	return false
}

// IsCall reports whether o transfers control to a callee.
func (o Op) IsCall() bool {
	if o.Type != TypeNone {
		return false
	}
	switch o.Verb {
	case VCall, VCallInd, VInvoke, VInvokeInd:
		return true
	}
	return false
}

// The verb table. This is the mnemonic index of §17, less the namespaces that
// parameterize it.
const (
	// §A integer arithmetic
	VAdd    Verb = "add"
	VSub    Verb = "sub"
	VMul    Verb = "mul"
	VSMulHi Verb = "smulhi"
	VUMulHi Verb = "umulhi"
	VSDiv   Verb = "sdiv"
	VUDiv   Verb = "udiv"
	VSRem   Verb = "srem"
	VURem   Verb = "urem"
	VNeg    Verb = "neg"

	// §A2 overflow predicates. There is no usubo; see §L.
	VSAddO Verb = "saddo"
	VUAddO Verb = "uaddo"
	VSSubO Verb = "ssubo"
	VSMulO Verb = "smulo"
	VUMulO Verb = "umulo"

	// §A3 float arithmetic
	VDiv      Verb = "div"
	VFMA      Verb = "fma"
	VAbs      Verb = "abs"
	VSqrt     Verb = "sqrt"
	VMinimum  Verb = "minimum"
	VMaximum  Verb = "maximum"
	VMinNum   Verb = "minnum"
	VMaxNum   Verb = "maxnum"
	VCopySign Verb = "copysign"
	VCeil     Verb = "ceil"
	VFloor    Verb = "floor"
	VTrunc    Verb = "trunc"
	VNearest  Verb = "nearest"

	// §A4 bitwise
	VNot Verb = "not"
	VAnd Verb = "and"
	VOr  Verb = "or"
	VXor Verb = "xor"

	// §A5 shifts and rotates
	VShl  Verb = "shl"
	VSShr Verb = "sshr"
	VUShr Verb = "ushr"
	VRotL Verb = "rotl"
	VRotR Verb = "rotr"

	// §A6 bit counting and byte swap
	VClz    Verb = "clz"
	VCtz    Verb = "ctz"
	VPopcnt Verb = "popcnt"
	VBswap  Verb = "bswap"

	// §A7 constants
	VConst Verb = "const"

	// §B comparisons. No gt or ge in any namespace; swap the operands.
	VEq  Verb = "eq"
	VNe  Verb = "ne"
	VSLt Verb = "slt"
	VULt Verb = "ult"
	VSLe Verb = "sle"
	VULe Verb = "ule"
	VLt  Verb = "lt"
	VLe  Verb = "le"
	VUno Verb = "uno"

	// §C integer conversions
	VWrapI64 Verb = "wrap_i64"
	VSExtI32 Verb = "sext_i32"
	VZExtI32 Verb = "zext_i32"
	VZExtI1  Verb = "zext_i1"

	// §C2 int to float
	VSCvtI32 Verb = "scvt_i32"
	VSCvtI64 Verb = "scvt_i64"
	VUCvtI32 Verb = "ucvt_i32"
	VUCvtI64 Verb = "ucvt_i64"

	// §C2 float to int, trapping
	VSCvtF32  Verb = "scvt_f32"
	VSCvtF64  Verb = "scvt_f64"
	VSCvtF80  Verb = "scvt_f80"
	VSCvtF128 Verb = "scvt_f128"
	VUCvtF32  Verb = "ucvt_f32"
	VUCvtF64  Verb = "ucvt_f64"
	VUCvtF80  Verb = "ucvt_f80"
	VUCvtF128 Verb = "ucvt_f128"

	// §C2 float to int, saturating
	VSCvtSatF32  Verb = "scvt_sat_f32"
	VSCvtSatF64  Verb = "scvt_sat_f64"
	VSCvtSatF80  Verb = "scvt_sat_f80"
	VSCvtSatF128 Verb = "scvt_sat_f128"
	VUCvtSatF32  Verb = "ucvt_sat_f32"
	VUCvtSatF64  Verb = "ucvt_sat_f64"
	VUCvtSatF80  Verb = "ucvt_sat_f80"
	VUCvtSatF128 Verb = "ucvt_sat_f128"

	// §C3 float width and bitcast
	VFCvtF32    Verb = "fcvt_f32"
	VFCvtF64    Verb = "fcvt_f64"
	VFCvtF80    Verb = "fcvt_f80"
	VFCvtF128   Verb = "fcvt_f128"
	VBitcastF32 Verb = "bitcast_f32"
	VBitcastI32 Verb = "bitcast_i32"
	VBitcastF64 Verb = "bitcast_f64"
	VBitcastI64 Verb = "bitcast_i64"

	// §C4 pointer conversions
	VFromI64 Verb = "from_i64"
	VFromPtr Verb = "from_ptr"

	// §D memory
	VLoad  Verb = "load"
	VStore Verb = "store"

	// §D2 sub-width memory
	VSLoad8  Verb = "sload8"
	VSLoad16 Verb = "sload16"
	VSLoad32 Verb = "sload32"
	VULoad8  Verb = "uload8"
	VULoad16 Verb = "uload16"
	VULoad32 Verb = "uload32"
	VStore8  Verb = "store8"
	VStore16 Verb = "store16"
	VStore32 Verb = "store32"

	// §D3 pointer ops
	VAlloc        Verb = "alloc"
	VAlloca       Verb = "alloca"
	VStackSave    Verb = "stacksave"
	VStackRestore Verb = "stackrestore"
	VGetAddr      Verb = "getaddr"
	VTLSAddr      Verb = "tlsaddr"
	VBlockAddr    Verb = "blockaddr"
	VFrameAddr    Verb = "frameaddr"
	VReturnAddr   Verb = "returnaddr"
	VDiff         Verb = "diff"

	// §E bulk memory, bare
	VMemCpy  Verb = "memcpy"
	VMemMove Verb = "memmove"
	VMemSet  Verb = "memset"
	VMemCmp  Verb = "memcmp"

	// §F select
	VSelect Verb = "select"

	// §G calls, bare
	VCall    Verb = "call"
	VCallInd Verb = "callind"

	// §G2 terminators, bare
	VBr      Verb = "br"
	VBrIf    Verb = "brif"
	VBrTable Verb = "br_table"
	VBrInd   Verb = "brind"
	VReturn  Verb = "return"
	VTrap    Verb = "trap"

	// §G3 unwinding, bare
	VInvoke    Verb = "invoke"
	VInvokeInd Verb = "invokeind"
	VResume    Verb = "resume"

	// §G4 inline assembly, bare
	VAsm     Verb = "asm"
	VAsmGoto Verb = "asm goto"

	// §H atomics
	VAtomicLoad    Verb = "atomic_load"
	VAtomicULoad8  Verb = "atomic_uload8"
	VAtomicULoad16 Verb = "atomic_uload16"
	VAtomicStore   Verb = "atomic_store"
	VAtomicStore8  Verb = "atomic_store8"
	VAtomicStore16 Verb = "atomic_store16"

	VAtomicRmwAdd  Verb = "atomic_rmwadd"
	VAtomicRmwSub  Verb = "atomic_rmwsub"
	VAtomicRmwAnd  Verb = "atomic_rmwand"
	VAtomicRmwOr   Verb = "atomic_rmwor"
	VAtomicRmwXor  Verb = "atomic_rmwxor"
	VAtomicRmwXchg Verb = "atomic_rmwxchg"

	VAtomicRmwAdd8   Verb = "atomic_rmwadd8"
	VAtomicRmwSub8   Verb = "atomic_rmwsub8"
	VAtomicRmwAnd8   Verb = "atomic_rmwand8"
	VAtomicRmwOr8    Verb = "atomic_rmwor8"
	VAtomicRmwXor8   Verb = "atomic_rmwxor8"
	VAtomicRmwXchg8  Verb = "atomic_rmwxchg8"
	VAtomicRmwAdd16  Verb = "atomic_rmwadd16"
	VAtomicRmwSub16  Verb = "atomic_rmwsub16"
	VAtomicRmwAnd16  Verb = "atomic_rmwand16"
	VAtomicRmwOr16   Verb = "atomic_rmwor16"
	VAtomicRmwXor16  Verb = "atomic_rmwxor16"
	VAtomicRmwXchg16 Verb = "atomic_rmwxchg16"

	VAtomicCas   Verb = "atomic_cas"
	VAtomicCas8  Verb = "atomic_cas8"
	VAtomicCas16 Verb = "atomic_cas16"

	VFence Verb = "fence" // bare

	// §I variadics
	VVaStart  Verb = "va_start" // bare
	VVaEnd    Verb = "va_end"   // bare
	VVaCopy   Verb = "va_copy"  // bare
	VVaArg    Verb = "va_arg"
	VVaArgRef Verb = "va_arg_ref"
)

// The §V verb set: the v128 namespace's own verbs, spelled shape-first
// because that is how the operation is named everywhere it is written —
// i16x8_add reads as "add, as eight words", and the shape is the half a
// reader needs first. The bitwise verbs and the memory verbs carry no shape
// at all, because sixteen bytes ANDed are sixteen bytes ANDed however the
// lanes are drawn.
//
// Signedness is in the verb, per §1, and so is saturation: a saturating add
// is a different circuit from a wrapping one, not a mode on it.
const (
	// The five bitwise verbs are §A4's, unchanged: v128.and is spelled
	// with the same verb i32.and is, because it is the same operation on a
	// wider register and a second name for it would be a second name for
	// nothing. Only ANDNOT is new, and only because the scalar namespaces
	// have no use for a fused one while every vector mask does.
	VVecAndNot Verb = "andnot" // a AND NOT b, in that order

	// Whole-register byte shifts, zeroes shifted in. The count is a
	// literal because the hardware has no other form.
	VVecShlBytes Verb = "shl_bytes"
	VVecShrBytes Verb = "shr_bytes"

	// Scalar in, vector out: the value in the low lane and zeroes above
	// it. This is not a splat, and the difference is load-bearing — every
	// use of it is building a vector from one scalar without disturbing
	// what a splat would fill.
	VVecZExtI32 Verb = "zext_i32"
	VVecZExtI64 Verb = "zext_i64"

	// i8x16.
	VI8x16Add       Verb = "i8x16_add"
	VI8x16Sub       Verb = "i8x16_sub"
	VI8x16AddSatS   Verb = "i8x16_add_sat_s"
	VI8x16AddSatU   Verb = "i8x16_add_sat_u"
	VI8x16SubSatS   Verb = "i8x16_sub_sat_s"
	VI8x16SubSatU   Verb = "i8x16_sub_sat_u"
	VI8x16MinU      Verb = "i8x16_min_u"
	VI8x16MaxU      Verb = "i8x16_max_u"
	VI8x16AvgrU     Verb = "i8x16_avgr_u"
	VI8x16Eq        Verb = "i8x16_eq"
	VI8x16GtS       Verb = "i8x16_gt_s"
	VI8x16Splat     Verb = "i8x16_splat"
	VI8x16Bitmask   Verb = "i8x16_bitmask"
	VI8x16NarrowS   Verb = "i8x16_narrow_s" // two i16x8 in, saturating
	VI8x16NarrowU   Verb = "i8x16_narrow_u"
	VI8x16UnpackLow Verb = "i8x16_unpack_low"
	VI8x16UnpackHi  Verb = "i8x16_unpack_high"
	VI8x16SadU      Verb = "i8x16_sad_u" // sums of absolute differences

	// i16x8.
	VI16x8Add        Verb = "i16x8_add"
	VI16x8Sub        Verb = "i16x8_sub"
	VI16x8Mul        Verb = "i16x8_mul"
	VI16x8MulHiS     Verb = "i16x8_mulhi_s"
	VI16x8MulHiU     Verb = "i16x8_mulhi_u"
	VI16x8AddSatS    Verb = "i16x8_add_sat_s"
	VI16x8AddSatU    Verb = "i16x8_add_sat_u"
	VI16x8SubSatS    Verb = "i16x8_sub_sat_s"
	VI16x8SubSatU    Verb = "i16x8_sub_sat_u"
	VI16x8MinS       Verb = "i16x8_min_s"
	VI16x8MaxS       Verb = "i16x8_max_s"
	VI16x8AvgrU      Verb = "i16x8_avgr_u"
	VI16x8Eq         Verb = "i16x8_eq"
	VI16x8GtS        Verb = "i16x8_gt_s"
	VI16x8Shl        Verb = "i16x8_shl"
	VI16x8ShrS       Verb = "i16x8_shr_s"
	VI16x8ShrU       Verb = "i16x8_shr_u"
	VI16x8Splat      Verb = "i16x8_splat"
	VI16x8ExtractU   Verb = "i16x8_extract_lane_u"
	VI16x8Replace    Verb = "i16x8_replace_lane"
	VI16x8NarrowS    Verb = "i16x8_narrow_s" // two i32x4 in, saturating
	VI16x8UnpackLow  Verb = "i16x8_unpack_low"
	VI16x8UnpackHi   Verb = "i16x8_unpack_high"
	VI16x8ShuffleLow Verb = "i16x8_shuffle_low"
	VI16x8ShuffleHi  Verb = "i16x8_shuffle_high"
	VI16x8MaddS      Verb = "i16x8_madd_s" // pairwise multiply-add into i32x4

	// i32x4.
	VI32x4Add       Verb = "i32x4_add"
	VI32x4Sub       Verb = "i32x4_sub"
	VI32x4Eq        Verb = "i32x4_eq"
	VI32x4GtS       Verb = "i32x4_gt_s"
	VI32x4Shl       Verb = "i32x4_shl"
	VI32x4ShrS      Verb = "i32x4_shr_s"
	VI32x4ShrU      Verb = "i32x4_shr_u"
	VI32x4Splat     Verb = "i32x4_splat"
	VI32x4Shuffle   Verb = "i32x4_shuffle"
	VI32x4UnpackLow Verb = "i32x4_unpack_low"
	VI32x4UnpackHi  Verb = "i32x4_unpack_high"
	VI32x4Extract   Verb = "i32x4_extract_lane"
	VI32x4MulEvenU  Verb = "i32x4_mul_even_u" // even lanes widened into i64x2

	// i64x2.
	VI64x2Add       Verb = "i64x2_add"
	VI64x2Sub       Verb = "i64x2_sub"
	VI64x2Shl       Verb = "i64x2_shl"
	VI64x2ShrU      Verb = "i64x2_shr_u"
	VI64x2Extract   Verb = "i64x2_extract_lane"
	VI64x2Splat     Verb = "i64x2_splat"
	VI64x2UnpackLow Verb = "i64x2_unpack_low"
	VI64x2UnpackHi  Verb = "i64x2_unpack_high"
)
