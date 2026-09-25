package ir

// The half-float namespaces: f16 is IEEE binary16 and bf16 is bfloat16,
// binary32 cut to its top sixteen bits. They are §K's reservation landed,
// admitted by the layout block's halffloat list under the same rule as
// ext-float, and §A3's verb set is the same over them as over f32.
//
// They exist for the GPU and for storage: weights are bf16, activations
// are f16, and a kernel that has to widen every element to f32 by hand
// both doubles its memory traffic and hides from the backend the packed
// half instructions every GPU in scope has. The approximate six are f32's
// alone; a kernel wanting one widens, as CUDA's hexp2 does.
//
// A target with no half arithmetic does not see these namespaces:
// LegalizeHalf carries each value in an i32 and does the work in f32.

// F16NS is the f16 namespace, reached through Builder.F16.
type F16NS struct{ b *Builder }

func (n F16NS) un(v Verb, a F16) F16 { return F16{n.b.def1(Op{TypeF16, v}, TypeF16, a.d)} }
func (n F16NS) bin(v Verb, a, c F16) F16 {
	return F16{n.b.def1(Op{TypeF16, v}, TypeF16, a.d, c.d)}
}
func (n F16NS) pred(v Verb, a, c F16) I1 {
	return I1{n.b.def1(Op{TypeF16, v}, TypeI1, a.d, c.d)}
}

// Const is the literal rounded to f16, to nearest even.
func (n F16NS) Const(v float64) F16 { return n.ConstOf(Float(v)) }
func (n F16NS) ConstOf(c Const) F16 {
	return F16{n.b.def1i(Op{TypeF16, VConst}, TypeF16, nil, &imm{lit: c, hasLit: true})}
}

func (n F16NS) Add(a, c F16) F16      { return n.bin(VAdd, a, c) }
func (n F16NS) Sub(a, c F16) F16      { return n.bin(VSub, a, c) }
func (n F16NS) Mul(a, c F16) F16      { return n.bin(VMul, a, c) }
func (n F16NS) Div(a, c F16) F16      { return n.bin(VDiv, a, c) }
func (n F16NS) Neg(a F16) F16         { return n.un(VNeg, a) }
func (n F16NS) Abs(a F16) F16         { return n.un(VAbs, a) }
func (n F16NS) Sqrt(a F16) F16        { return n.un(VSqrt, a) }
func (n F16NS) Ceil(a F16) F16        { return n.un(VCeil, a) }
func (n F16NS) Floor(a F16) F16       { return n.un(VFloor, a) }
func (n F16NS) Trunc(a F16) F16       { return n.un(VTrunc, a) }
func (n F16NS) Nearest(a F16) F16     { return n.un(VNearest, a) }
func (n F16NS) Minimum(a, c F16) F16  { return n.bin(VMinimum, a, c) }
func (n F16NS) Maximum(a, c F16) F16  { return n.bin(VMaximum, a, c) }
func (n F16NS) MinNum(a, c F16) F16   { return n.bin(VMinNum, a, c) }
func (n F16NS) MaxNum(a, c F16) F16   { return n.bin(VMaxNum, a, c) }
func (n F16NS) CopySign(a, c F16) F16 { return n.bin(VCopySign, a, c) }

// FMA is a*b+c with one rounding.
func (n F16NS) FMA(a, c, d F16) F16 {
	return F16{n.b.def1(Op{TypeF16, VFMA}, TypeF16, a.d, c.d, d.d)}
}

func (n F16NS) Eq(a, c F16) I1  { return n.pred(VEq, a, c) }
func (n F16NS) Ne(a, c F16) I1  { return n.pred(VNe, a, c) }
func (n F16NS) Lt(a, c F16) I1  { return n.pred(VLt, a, c) }
func (n F16NS) Le(a, c F16) I1  { return n.pred(VLe, a, c) }
func (n F16NS) Uno(a, c F16) I1 { return n.pred(VUno, a, c) }

// BF16NS is the bf16 namespace, reached through Builder.BF16.
type BF16NS struct{ b *Builder }

func (n BF16NS) un(v Verb, a BF16) BF16 { return BF16{n.b.def1(Op{TypeBF16, v}, TypeBF16, a.d)} }
func (n BF16NS) bin(v Verb, a, c BF16) BF16 {
	return BF16{n.b.def1(Op{TypeBF16, v}, TypeBF16, a.d, c.d)}
}
func (n BF16NS) pred(v Verb, a, c BF16) I1 {
	return I1{n.b.def1(Op{TypeBF16, v}, TypeI1, a.d, c.d)}
}

// Const is the literal rounded to bf16, to nearest even.
func (n BF16NS) Const(v float64) BF16 { return n.ConstOf(Float(v)) }
func (n BF16NS) ConstOf(c Const) BF16 {
	return BF16{n.b.def1i(Op{TypeBF16, VConst}, TypeBF16, nil, &imm{lit: c, hasLit: true})}
}

func (n BF16NS) Add(a, c BF16) BF16      { return n.bin(VAdd, a, c) }
func (n BF16NS) Sub(a, c BF16) BF16      { return n.bin(VSub, a, c) }
func (n BF16NS) Mul(a, c BF16) BF16      { return n.bin(VMul, a, c) }
func (n BF16NS) Div(a, c BF16) BF16      { return n.bin(VDiv, a, c) }
func (n BF16NS) Neg(a BF16) BF16         { return n.un(VNeg, a) }
func (n BF16NS) Abs(a BF16) BF16         { return n.un(VAbs, a) }
func (n BF16NS) Sqrt(a BF16) BF16        { return n.un(VSqrt, a) }
func (n BF16NS) Ceil(a BF16) BF16        { return n.un(VCeil, a) }
func (n BF16NS) Floor(a BF16) BF16       { return n.un(VFloor, a) }
func (n BF16NS) Trunc(a BF16) BF16       { return n.un(VTrunc, a) }
func (n BF16NS) Nearest(a BF16) BF16     { return n.un(VNearest, a) }
func (n BF16NS) Minimum(a, c BF16) BF16  { return n.bin(VMinimum, a, c) }
func (n BF16NS) Maximum(a, c BF16) BF16  { return n.bin(VMaximum, a, c) }
func (n BF16NS) MinNum(a, c BF16) BF16   { return n.bin(VMinNum, a, c) }
func (n BF16NS) MaxNum(a, c BF16) BF16   { return n.bin(VMaxNum, a, c) }
func (n BF16NS) CopySign(a, c BF16) BF16 { return n.bin(VCopySign, a, c) }

// FMA is a*b+c with one rounding.
func (n BF16NS) FMA(a, c, d BF16) BF16 {
	return BF16{n.b.def1(Op{TypeBF16, VFMA}, TypeBF16, a.d, c.d, d.d)}
}

func (n BF16NS) Eq(a, c BF16) I1  { return n.pred(VEq, a, c) }
func (n BF16NS) Ne(a, c BF16) I1  { return n.pred(VNe, a, c) }
func (n BF16NS) Lt(a, c BF16) I1  { return n.pred(VLt, a, c) }
func (n BF16NS) Le(a, c BF16) I1  { return n.pred(VLe, a, c) }
func (n BF16NS) Uno(a, c BF16) I1 { return n.pred(VUno, a, c) }
