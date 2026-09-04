package ir

// Conversions name the destination in the namespace and the source in the verb,
// so a conversion and its inverse share a verb.
//
// This file is the one combinatorial axis in the IR. The overview names
// conversions the standing exception to additive extension: each new register
// type adds a verb per existing type it converts with, in both directions. When
// §K's lane-width conversions arrive, this should be one file's diff.

// —— §C. Integer ——

// WrapI64 discards the high 32 bits.
func (n I32NS) WrapI64(a I64) I32 {
	return I32{n.b.def1(Op{TypeI32, VWrapI64}, TypeI32, a.d)}
}

// ZExtI1 widens a one-bit value to 0 or 1.
func (n I32NS) ZExtI1(a I1) I32 { return I32{n.b.def1(Op{TypeI32, VZExtI1}, TypeI32, a.d)} }

// SExtI32 sign-extends.
func (n I64NS) SExtI32(a I32) I64 { return I64{n.b.def1(Op{TypeI64, VSExtI32}, TypeI64, a.d)} }

// ZExtI32 zero-extends.
func (n I64NS) ZExtI32(a I32) I64 { return I64{n.b.def1(Op{TypeI64, VZExtI32}, TypeI64, a.d)} }

// ZExtI1 widens a one-bit value to 0 or 1.
func (n I64NS) ZExtI1(a I1) I64 { return I64{n.b.def1(Op{TypeI64, VZExtI1}, TypeI64, a.d)} }

// Narrowing to i8 or i16 is a store and widening from them is a sub-width load.
// There is no register-to-register form, because i8 and i16 are not register
// types.

// —— §C2. Int to float ——

func (n F32NS) SCvtI32(a I32) F32 { return F32{n.b.def1(Op{TypeF32, VSCvtI32}, TypeF32, a.d)} }
func (n F32NS) SCvtI64(a I64) F32 { return F32{n.b.def1(Op{TypeF32, VSCvtI64}, TypeF32, a.d)} }
func (n F32NS) UCvtI32(a I32) F32 { return F32{n.b.def1(Op{TypeF32, VUCvtI32}, TypeF32, a.d)} }
func (n F32NS) UCvtI64(a I64) F32 { return F32{n.b.def1(Op{TypeF32, VUCvtI64}, TypeF32, a.d)} }

func (n F64NS) SCvtI32(a I32) F64 { return F64{n.b.def1(Op{TypeF64, VSCvtI32}, TypeF64, a.d)} }
func (n F64NS) SCvtI64(a I64) F64 { return F64{n.b.def1(Op{TypeF64, VSCvtI64}, TypeF64, a.d)} }
func (n F64NS) UCvtI32(a I32) F64 { return F64{n.b.def1(Op{TypeF64, VUCvtI32}, TypeF64, a.d)} }
func (n F64NS) UCvtI64(a I64) F64 { return F64{n.b.def1(Op{TypeF64, VUCvtI64}, TypeF64, a.d)} }

func (n F80NS) SCvtI32(a I32) F80 { return F80{n.b.def1(Op{TypeF80, VSCvtI32}, TypeF80, a.d)} }
func (n F80NS) SCvtI64(a I64) F80 { return F80{n.b.def1(Op{TypeF80, VSCvtI64}, TypeF80, a.d)} }
func (n F80NS) UCvtI32(a I32) F80 { return F80{n.b.def1(Op{TypeF80, VUCvtI32}, TypeF80, a.d)} }
func (n F80NS) UCvtI64(a I64) F80 { return F80{n.b.def1(Op{TypeF80, VUCvtI64}, TypeF80, a.d)} }

func (n F128NS) SCvtI32(a I32) F128 { return F128{n.b.def1(Op{TypeF128, VSCvtI32}, TypeF128, a.d)} }
func (n F128NS) SCvtI64(a I64) F128 { return F128{n.b.def1(Op{TypeF128, VSCvtI64}, TypeF128, a.d)} }
func (n F128NS) UCvtI32(a I32) F128 { return F128{n.b.def1(Op{TypeF128, VUCvtI32}, TypeF128, a.d)} }
func (n F128NS) UCvtI64(a I64) F128 { return F128{n.b.def1(Op{TypeF128, VUCvtI64}, TypeF128, a.d)} }

// —— §C2. Float to int ——
//
// The trapping forms are the default, per the no-UB commitment. A frontend
// emits the Sat forms only where it has proven the conversion in range or has
// chosen to define the out-of-range case; those clamp, and turn NaN into zero.

func (n I32NS) SCvtF32(a F32) I32   { return I32{n.b.def1(Op{TypeI32, VSCvtF32}, TypeI32, a.d)} }
func (n I32NS) SCvtF64(a F64) I32   { return I32{n.b.def1(Op{TypeI32, VSCvtF64}, TypeI32, a.d)} }
func (n I32NS) SCvtF80(a F80) I32   { return I32{n.b.def1(Op{TypeI32, VSCvtF80}, TypeI32, a.d)} }
func (n I32NS) SCvtF128(a F128) I32 { return I32{n.b.def1(Op{TypeI32, VSCvtF128}, TypeI32, a.d)} }
func (n I32NS) UCvtF32(a F32) I32   { return I32{n.b.def1(Op{TypeI32, VUCvtF32}, TypeI32, a.d)} }
func (n I32NS) UCvtF64(a F64) I32   { return I32{n.b.def1(Op{TypeI32, VUCvtF64}, TypeI32, a.d)} }
func (n I32NS) UCvtF80(a F80) I32   { return I32{n.b.def1(Op{TypeI32, VUCvtF80}, TypeI32, a.d)} }
func (n I32NS) UCvtF128(a F128) I32 { return I32{n.b.def1(Op{TypeI32, VUCvtF128}, TypeI32, a.d)} }

func (n I32NS) SCvtSatF32(a F32) I32 {
	return I32{n.b.def1(Op{TypeI32, VSCvtSatF32}, TypeI32, a.d)}
}
func (n I32NS) SCvtSatF64(a F64) I32 {
	return I32{n.b.def1(Op{TypeI32, VSCvtSatF64}, TypeI32, a.d)}
}
func (n I32NS) SCvtSatF80(a F80) I32 {
	return I32{n.b.def1(Op{TypeI32, VSCvtSatF80}, TypeI32, a.d)}
}
func (n I32NS) SCvtSatF128(a F128) I32 {
	return I32{n.b.def1(Op{TypeI32, VSCvtSatF128}, TypeI32, a.d)}
}
func (n I32NS) UCvtSatF32(a F32) I32 {
	return I32{n.b.def1(Op{TypeI32, VUCvtSatF32}, TypeI32, a.d)}
}
func (n I32NS) UCvtSatF64(a F64) I32 {
	return I32{n.b.def1(Op{TypeI32, VUCvtSatF64}, TypeI32, a.d)}
}
func (n I32NS) UCvtSatF80(a F80) I32 {
	return I32{n.b.def1(Op{TypeI32, VUCvtSatF80}, TypeI32, a.d)}
}
func (n I32NS) UCvtSatF128(a F128) I32 {
	return I32{n.b.def1(Op{TypeI32, VUCvtSatF128}, TypeI32, a.d)}
}

func (n I64NS) SCvtF32(a F32) I64   { return I64{n.b.def1(Op{TypeI64, VSCvtF32}, TypeI64, a.d)} }
func (n I64NS) SCvtF64(a F64) I64   { return I64{n.b.def1(Op{TypeI64, VSCvtF64}, TypeI64, a.d)} }
func (n I64NS) SCvtF80(a F80) I64   { return I64{n.b.def1(Op{TypeI64, VSCvtF80}, TypeI64, a.d)} }
func (n I64NS) SCvtF128(a F128) I64 { return I64{n.b.def1(Op{TypeI64, VSCvtF128}, TypeI64, a.d)} }
func (n I64NS) UCvtF32(a F32) I64   { return I64{n.b.def1(Op{TypeI64, VUCvtF32}, TypeI64, a.d)} }
func (n I64NS) UCvtF64(a F64) I64   { return I64{n.b.def1(Op{TypeI64, VUCvtF64}, TypeI64, a.d)} }
func (n I64NS) UCvtF80(a F80) I64   { return I64{n.b.def1(Op{TypeI64, VUCvtF80}, TypeI64, a.d)} }
func (n I64NS) UCvtF128(a F128) I64 { return I64{n.b.def1(Op{TypeI64, VUCvtF128}, TypeI64, a.d)} }

func (n I64NS) SCvtSatF32(a F32) I64 {
	return I64{n.b.def1(Op{TypeI64, VSCvtSatF32}, TypeI64, a.d)}
}
func (n I64NS) SCvtSatF64(a F64) I64 {
	return I64{n.b.def1(Op{TypeI64, VSCvtSatF64}, TypeI64, a.d)}
}
func (n I64NS) SCvtSatF80(a F80) I64 {
	return I64{n.b.def1(Op{TypeI64, VSCvtSatF80}, TypeI64, a.d)}
}
func (n I64NS) SCvtSatF128(a F128) I64 {
	return I64{n.b.def1(Op{TypeI64, VSCvtSatF128}, TypeI64, a.d)}
}
func (n I64NS) UCvtSatF32(a F32) I64 {
	return I64{n.b.def1(Op{TypeI64, VUCvtSatF32}, TypeI64, a.d)}
}
func (n I64NS) UCvtSatF64(a F64) I64 {
	return I64{n.b.def1(Op{TypeI64, VUCvtSatF64}, TypeI64, a.d)}
}
func (n I64NS) UCvtSatF80(a F80) I64 {
	return I64{n.b.def1(Op{TypeI64, VUCvtSatF80}, TypeI64, a.d)}
}
func (n I64NS) UCvtSatF128(a F128) I64 {
	return I64{n.b.def1(Op{TypeI64, VUCvtSatF128}, TypeI64, a.d)}
}

// —— §C3. Float width ——

// FCvtF32 widens, exactly.
func (n F64NS) FCvtF32(a F32) F64 { return F64{n.b.def1(Op{TypeF64, VFCvtF32}, TypeF64, a.d)} }

// FCvtF64 narrows, round-to-nearest.
func (n F32NS) FCvtF64(a F64) F32 { return F32{n.b.def1(Op{TypeF32, VFCvtF64}, TypeF32, a.d)} }

func (n F80NS) FCvtF32(a F32) F80 { return F80{n.b.def1(Op{TypeF80, VFCvtF32}, TypeF80, a.d)} }
func (n F80NS) FCvtF64(a F64) F80 { return F80{n.b.def1(Op{TypeF80, VFCvtF64}, TypeF80, a.d)} }
func (n F32NS) FCvtF80(a F80) F32 { return F32{n.b.def1(Op{TypeF32, VFCvtF80}, TypeF32, a.d)} }
func (n F64NS) FCvtF80(a F80) F64 { return F64{n.b.def1(Op{TypeF64, VFCvtF80}, TypeF64, a.d)} }

func (n F128NS) FCvtF32(a F32) F128 { return F128{n.b.def1(Op{TypeF128, VFCvtF32}, TypeF128, a.d)} }
func (n F128NS) FCvtF64(a F64) F128 { return F128{n.b.def1(Op{TypeF128, VFCvtF64}, TypeF128, a.d)} }
func (n F32NS) FCvtF128(a F128) F32 { return F32{n.b.def1(Op{TypeF32, VFCvtF128}, TypeF32, a.d)} }
func (n F64NS) FCvtF128(a F128) F64 { return F64{n.b.def1(Op{TypeF64, VFCvtF128}, TypeF64, a.d)} }

// FCvtF80 widens f80 to f128. Both namespaces must be admitted by the layout
// block.
func (n F128NS) FCvtF80(a F80) F128 {
	if !n.b.requireExtFloat(TypeF80) {
		return F128{}
	}
	return F128{n.b.def1(Op{TypeF128, VFCvtF80}, TypeF128, a.d)}
}

// FCvtF128 narrows f128 to f80. Both namespaces must be admitted.
func (n F80NS) FCvtF128(a F128) F80 {
	if !n.b.requireExtFloat(TypeF128) {
		return F80{}
	}
	return F80{n.b.def1(Op{TypeF80, VFCvtF128}, TypeF80, a.d)}
}

// —— §C3. Bitcast ——
//
// There is no bitcast for f80 or f128: neither has an integer register type of
// matching width. Reach their representation through memory.

func (n I32NS) BitcastF32(a F32) I32 {
	return I32{n.b.def1(Op{TypeI32, VBitcastF32}, TypeI32, a.d)}
}
func (n F32NS) BitcastI32(a I32) F32 {
	return F32{n.b.def1(Op{TypeF32, VBitcastI32}, TypeF32, a.d)}
}
func (n I64NS) BitcastF64(a F64) I64 {
	return I64{n.b.def1(Op{TypeI64, VBitcastF64}, TypeI64, a.d)}
}
func (n F64NS) BitcastI64(a I64) F64 {
	return F64{n.b.def1(Op{TypeF64, VBitcastI64}, TypeF64, a.d)}
}

// —— §C4. Pointer ↔ integer ——
//
// There is no i32 pair. On a ptrbits 32 module (intptr_t)p is FromPtr then
// WrapI64, and back is ZExtI32 then FromI64; both are lossless and both fold to
// a register move in lowering.

// FromI64 truncates where ptrbits < 64.
func (n PtrNS) FromI64(a I64) Ptr { return Ptr{n.b.def1(Op{TypePtr, VFromI64}, TypePtr, a.d)} }

// FromPtr zero-extends where ptrbits < 64.
func (n I64NS) FromPtr(a Ptr) I64 { return I64{n.b.def1(Op{TypeI64, VFromPtr}, TypeI64, a.d)} }
