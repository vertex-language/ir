package ir

// select evaluates both arms. It is not a short-circuit: C's &&, || and ?: lower
// to branches unless the frontend has proven both arms safe.
//
// select plus a comparison is also how integer min and max are spelled — §L
// declines to give them verbs, since every target has a conditional move.
// The atomic min and max §K reserves are a different matter: there the
// operation is indivisible and a select-based expansion is not equivalent.

func (n I1NS) Select(c I1, a, b I1) I1 {
	return I1{n.b.def1(Op{TypeI1, VSelect}, TypeI1, c.d, a.d, b.d)}
}

func (n I32NS) Select(c I1, a, b I32) I32 {
	return I32{n.b.def1(Op{TypeI32, VSelect}, TypeI32, c.d, a.d, b.d)}
}

func (n I64NS) Select(c I1, a, b I64) I64 {
	return I64{n.b.def1(Op{TypeI64, VSelect}, TypeI64, c.d, a.d, b.d)}
}

func (n F32NS) Select(c I1, a, b F32) F32 {
	return F32{n.b.def1(Op{TypeF32, VSelect}, TypeF32, c.d, a.d, b.d)}
}

func (n F64NS) Select(c I1, a, b F64) F64 {
	return F64{n.b.def1(Op{TypeF64, VSelect}, TypeF64, c.d, a.d, b.d)}
}

func (n F80NS) Select(c I1, a, b F80) F80 {
	return F80{n.b.def1(Op{TypeF80, VSelect}, TypeF80, c.d, a.d, b.d)}
}

func (n V128NS) Select(c I1, a, b V128) V128 {
	return V128{n.b.def1(Op{TypeV128, VSelect}, TypeV128, c.d, a.d, b.d)}
}

func (n F128NS) Select(c I1, a, b F128) F128 {
	return F128{n.b.def1(Op{TypeF128, VSelect}, TypeF128, c.d, a.d, b.d)}
}

func (n PtrNS) Select(c I1, a, b Ptr) Ptr {
	return Ptr{n.b.def1(Op{TypePtr, VSelect}, TypePtr, c.d, a.d, b.d)}
}
