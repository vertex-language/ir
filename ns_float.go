package ir

// The float namespaces. §A3's verb set is identical across f32, f64, and
// whichever ext-float namespaces the layout block admits; the Go types differ
// because the values do.
//
// minimum/maximum are IEEE-754-2019: any NaN operand yields NaN. minnum/maxnum
// are IEEE-754-2008, discarding a NaN operand in favour of the other, which is
// what C's fmin/fmax require — lowering fmin to minimum is silently wrong.
//
// There is no float remainder verb; see §L.

// F32NS is the f32 namespace.
type F32NS struct{ b *Builder }

func (n F32NS) un(v Verb, a F32) F32     { return F32{n.b.def1(Op{TypeF32, v}, TypeF32, a.d)} }
func (n F32NS) bin(v Verb, a, c F32) F32 { return F32{n.b.def1(Op{TypeF32, v}, TypeF32, a.d, c.d)} }
func (n F32NS) pred(v Verb, a, c F32) I1 { return I1{n.b.def1(Op{TypeF32, v}, TypeI1, a.d, c.d)} }

func (n F32NS) Const(v float64) F32 { return n.ConstOf(Float(v)) }
func (n F32NS) ConstOf(c Const) F32 {
	return F32{n.b.def1i(Op{TypeF32, VConst}, TypeF32, nil, &imm{lit: c, hasLit: true})}
}

func (n F32NS) Add(a, c F32) F32      { return n.bin(VAdd, a, c) }
func (n F32NS) Sub(a, c F32) F32      { return n.bin(VSub, a, c) }
func (n F32NS) Mul(a, c F32) F32      { return n.bin(VMul, a, c) }
func (n F32NS) Div(a, c F32) F32      { return n.bin(VDiv, a, c) }
func (n F32NS) Neg(a F32) F32         { return n.un(VNeg, a) }
func (n F32NS) Abs(a F32) F32         { return n.un(VAbs, a) }
func (n F32NS) Sqrt(a F32) F32        { return n.un(VSqrt, a) }
func (n F32NS) Ceil(a F32) F32        { return n.un(VCeil, a) }
func (n F32NS) Floor(a F32) F32       { return n.un(VFloor, a) }
func (n F32NS) Trunc(a F32) F32       { return n.un(VTrunc, a) }
func (n F32NS) Nearest(a F32) F32     { return n.un(VNearest, a) }
func (n F32NS) Minimum(a, c F32) F32  { return n.bin(VMinimum, a, c) }
func (n F32NS) Maximum(a, c F32) F32  { return n.bin(VMaximum, a, c) }
func (n F32NS) MinNum(a, c F32) F32   { return n.bin(VMinNum, a, c) }
func (n F32NS) MaxNum(a, c F32) F32   { return n.bin(VMaxNum, a, c) }
func (n F32NS) CopySign(a, c F32) F32 { return n.bin(VCopySign, a, c) }

// FMA is a*b+c with one rounding.
func (n F32NS) FMA(a, c, d F32) F32 {
	return F32{n.b.def1(Op{TypeF32, VFMA}, TypeF32, a.d, c.d, d.d)}
}

// §B. Ne is the exact negation of Eq, which is what C's != means; Uno exists
// because unordered-ness is not reachable from Eq and Ne alone.
func (n F32NS) Eq(a, c F32) I1  { return n.pred(VEq, a, c) }
func (n F32NS) Ne(a, c F32) I1  { return n.pred(VNe, a, c) }
func (n F32NS) Lt(a, c F32) I1  { return n.pred(VLt, a, c) }
func (n F32NS) Le(a, c F32) I1  { return n.pred(VLe, a, c) }
func (n F32NS) Uno(a, c F32) I1 { return n.pred(VUno, a, c) }

// F64NS is the f64 namespace.
type F64NS struct{ b *Builder }

func (n F64NS) un(v Verb, a F64) F64     { return F64{n.b.def1(Op{TypeF64, v}, TypeF64, a.d)} }
func (n F64NS) bin(v Verb, a, c F64) F64 { return F64{n.b.def1(Op{TypeF64, v}, TypeF64, a.d, c.d)} }
func (n F64NS) pred(v Verb, a, c F64) I1 { return I1{n.b.def1(Op{TypeF64, v}, TypeI1, a.d, c.d)} }

func (n F64NS) Const(v float64) F64 { return n.ConstOf(Float(v)) }
func (n F64NS) ConstOf(c Const) F64 {
	return F64{n.b.def1i(Op{TypeF64, VConst}, TypeF64, nil, &imm{lit: c, hasLit: true})}
}

func (n F64NS) Add(a, c F64) F64      { return n.bin(VAdd, a, c) }
func (n F64NS) Sub(a, c F64) F64      { return n.bin(VSub, a, c) }
func (n F64NS) Mul(a, c F64) F64      { return n.bin(VMul, a, c) }
func (n F64NS) Div(a, c F64) F64      { return n.bin(VDiv, a, c) }
func (n F64NS) Neg(a F64) F64         { return n.un(VNeg, a) }
func (n F64NS) Abs(a F64) F64         { return n.un(VAbs, a) }
func (n F64NS) Sqrt(a F64) F64        { return n.un(VSqrt, a) }
func (n F64NS) Ceil(a F64) F64        { return n.un(VCeil, a) }
func (n F64NS) Floor(a F64) F64       { return n.un(VFloor, a) }
func (n F64NS) Trunc(a F64) F64       { return n.un(VTrunc, a) }
func (n F64NS) Nearest(a F64) F64     { return n.un(VNearest, a) }
func (n F64NS) Minimum(a, c F64) F64  { return n.bin(VMinimum, a, c) }
func (n F64NS) Maximum(a, c F64) F64  { return n.bin(VMaximum, a, c) }
func (n F64NS) MinNum(a, c F64) F64   { return n.bin(VMinNum, a, c) }
func (n F64NS) MaxNum(a, c F64) F64   { return n.bin(VMaxNum, a, c) }
func (n F64NS) CopySign(a, c F64) F64 { return n.bin(VCopySign, a, c) }

func (n F64NS) FMA(a, c, d F64) F64 {
	return F64{n.b.def1(Op{TypeF64, VFMA}, TypeF64, a.d, c.d, d.d)}
}

func (n F64NS) Eq(a, c F64) I1  { return n.pred(VEq, a, c) }
func (n F64NS) Ne(a, c F64) I1  { return n.pred(VNe, a, c) }
func (n F64NS) Lt(a, c F64) I1  { return n.pred(VLt, a, c) }
func (n F64NS) Le(a, c F64) I1  { return n.pred(VLe, a, c) }
func (n F64NS) Uno(a, c F64) I1 { return n.pred(VUno, a, c) }

// F80NS is the f80 namespace, reached through Builder.F80.
type F80NS struct{ b *Builder }

func (n F80NS) un(v Verb, a F80) F80     { return F80{n.b.def1(Op{TypeF80, v}, TypeF80, a.d)} }
func (n F80NS) bin(v Verb, a, c F80) F80 { return F80{n.b.def1(Op{TypeF80, v}, TypeF80, a.d, c.d)} }
func (n F80NS) pred(v Verb, a, c F80) I1 { return I1{n.b.def1(Op{TypeF80, v}, TypeI1, a.d, c.d)} }

func (n F80NS) Const(v float64) F80 { return n.ConstOf(Float(v)) }
func (n F80NS) ConstOf(c Const) F80 {
	return F80{n.b.def1i(Op{TypeF80, VConst}, TypeF80, nil, &imm{lit: c, hasLit: true})}
}

func (n F80NS) Add(a, c F80) F80      { return n.bin(VAdd, a, c) }
func (n F80NS) Sub(a, c F80) F80      { return n.bin(VSub, a, c) }
func (n F80NS) Mul(a, c F80) F80      { return n.bin(VMul, a, c) }
func (n F80NS) Div(a, c F80) F80      { return n.bin(VDiv, a, c) }
func (n F80NS) Neg(a F80) F80         { return n.un(VNeg, a) }
func (n F80NS) Abs(a F80) F80         { return n.un(VAbs, a) }
func (n F80NS) Sqrt(a F80) F80        { return n.un(VSqrt, a) }
func (n F80NS) Ceil(a F80) F80        { return n.un(VCeil, a) }
func (n F80NS) Floor(a F80) F80       { return n.un(VFloor, a) }
func (n F80NS) Trunc(a F80) F80       { return n.un(VTrunc, a) }
func (n F80NS) Nearest(a F80) F80     { return n.un(VNearest, a) }
func (n F80NS) Minimum(a, c F80) F80  { return n.bin(VMinimum, a, c) }
func (n F80NS) Maximum(a, c F80) F80  { return n.bin(VMaximum, a, c) }
func (n F80NS) MinNum(a, c F80) F80   { return n.bin(VMinNum, a, c) }
func (n F80NS) MaxNum(a, c F80) F80   { return n.bin(VMaxNum, a, c) }
func (n F80NS) CopySign(a, c F80) F80 { return n.bin(VCopySign, a, c) }

func (n F80NS) FMA(a, c, d F80) F80 {
	return F80{n.b.def1(Op{TypeF80, VFMA}, TypeF80, a.d, c.d, d.d)}
}

func (n F80NS) Eq(a, c F80) I1  { return n.pred(VEq, a, c) }
func (n F80NS) Ne(a, c F80) I1  { return n.pred(VNe, a, c) }
func (n F80NS) Lt(a, c F80) I1  { return n.pred(VLt, a, c) }
func (n F80NS) Le(a, c F80) I1  { return n.pred(VLe, a, c) }
func (n F80NS) Uno(a, c F80) I1 { return n.pred(VUno, a, c) }

// F128NS is the f128 namespace, reached through Builder.F128.
type F128NS struct{ b *Builder }

func (n F128NS) un(v Verb, a F128) F128 { return F128{n.b.def1(Op{TypeF128, v}, TypeF128, a.d)} }
func (n F128NS) bin(v Verb, a, c F128) F128 {
	return F128{n.b.def1(Op{TypeF128, v}, TypeF128, a.d, c.d)}
}
func (n F128NS) pred(v Verb, a, c F128) I1 {
	return I1{n.b.def1(Op{TypeF128, v}, TypeI1, a.d, c.d)}
}

func (n F128NS) Const(v float64) F128 { return n.ConstOf(Float(v)) }
func (n F128NS) ConstOf(c Const) F128 {
	return F128{n.b.def1i(Op{TypeF128, VConst}, TypeF128, nil, &imm{lit: c, hasLit: true})}
}

func (n F128NS) Add(a, c F128) F128      { return n.bin(VAdd, a, c) }
func (n F128NS) Sub(a, c F128) F128      { return n.bin(VSub, a, c) }
func (n F128NS) Mul(a, c F128) F128      { return n.bin(VMul, a, c) }
func (n F128NS) Div(a, c F128) F128      { return n.bin(VDiv, a, c) }
func (n F128NS) Neg(a F128) F128         { return n.un(VNeg, a) }
func (n F128NS) Abs(a F128) F128         { return n.un(VAbs, a) }
func (n F128NS) Sqrt(a F128) F128        { return n.un(VSqrt, a) }
func (n F128NS) Ceil(a F128) F128        { return n.un(VCeil, a) }
func (n F128NS) Floor(a F128) F128       { return n.un(VFloor, a) }
func (n F128NS) Trunc(a F128) F128       { return n.un(VTrunc, a) }
func (n F128NS) Nearest(a F128) F128     { return n.un(VNearest, a) }
func (n F128NS) Minimum(a, c F128) F128  { return n.bin(VMinimum, a, c) }
func (n F128NS) Maximum(a, c F128) F128  { return n.bin(VMaximum, a, c) }
func (n F128NS) MinNum(a, c F128) F128   { return n.bin(VMinNum, a, c) }
func (n F128NS) MaxNum(a, c F128) F128   { return n.bin(VMaxNum, a, c) }
func (n F128NS) CopySign(a, c F128) F128 { return n.bin(VCopySign, a, c) }

func (n F128NS) FMA(a, c, d F128) F128 {
	return F128{n.b.def1(Op{TypeF128, VFMA}, TypeF128, a.d, c.d, d.d)}
}

func (n F128NS) Eq(a, c F128) I1  { return n.pred(VEq, a, c) }
func (n F128NS) Ne(a, c F128) I1  { return n.pred(VNe, a, c) }
func (n F128NS) Lt(a, c F128) I1  { return n.pred(VLt, a, c) }
func (n F128NS) Le(a, c F128) I1  { return n.pred(VLe, a, c) }
func (n F128NS) Uno(a, c F128) I1 { return n.pred(VUno, a, c) }
