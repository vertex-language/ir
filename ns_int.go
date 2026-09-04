package ir

// I1NS is the i1 namespace: five bitwise verbs, const, and select, and no
// comparisons — i1.ne is i1.xor and i1.eq is i1.not of that. See §L.
type I1NS struct{ b *Builder }

// Const emits i1.const.
func (n I1NS) Const(v bool) I1 {
	i := int64(0)
	if v {
		i = 1
	}
	return I1{n.b.def1i(Op{TypeI1, VConst}, TypeI1, nil, &imm{lit: Int(i), hasLit: true})}
}

func (n I1NS) Not(a I1) I1    { return I1{n.b.def1(Op{TypeI1, VNot}, TypeI1, a.d)} }
func (n I1NS) And(a, c I1) I1 { return I1{n.b.def1(Op{TypeI1, VAnd}, TypeI1, a.d, c.d)} }
func (n I1NS) Or(a, c I1) I1  { return I1{n.b.def1(Op{TypeI1, VOr}, TypeI1, a.d, c.d)} }
func (n I1NS) Xor(a, c I1) I1 { return I1{n.b.def1(Op{TypeI1, VXor}, TypeI1, a.d, c.d)} }

// I32NS is the i32 namespace.
type I32NS struct{ b *Builder }

func (n I32NS) un(v Verb, a I32) I32     { return I32{n.b.def1(Op{TypeI32, v}, TypeI32, a.d)} }
func (n I32NS) bin(v Verb, a, c I32) I32 { return I32{n.b.def1(Op{TypeI32, v}, TypeI32, a.d, c.d)} }
func (n I32NS) pred(v Verb, a, c I32) I1 { return I1{n.b.def1(Op{TypeI32, v}, TypeI1, a.d, c.d)} }

// Const emits i32.const.
func (n I32NS) Const(v int64) I32 { return n.ConstOf(Int(v)) }

// ConstOf emits i32.const with a symbolic constant: sizeof, alignof, offsetof.
func (n I32NS) ConstOf(c Const) I32 {
	return I32{n.b.def1i(Op{TypeI32, VConst}, TypeI32, nil, &imm{lit: c, hasLit: true})}
}

// §A. Integer arithmetic wraps; overflow is detected by §A2, never by the
// arithmetic itself. The division verbs trap rather than leaving a case
// undefined.
func (n I32NS) Add(a, c I32) I32    { return n.bin(VAdd, a, c) }
func (n I32NS) Sub(a, c I32) I32    { return n.bin(VSub, a, c) }
func (n I32NS) Mul(a, c I32) I32    { return n.bin(VMul, a, c) }
func (n I32NS) SMulHi(a, c I32) I32 { return n.bin(VSMulHi, a, c) }
func (n I32NS) UMulHi(a, c I32) I32 { return n.bin(VUMulHi, a, c) }
func (n I32NS) SDiv(a, c I32) I32   { return n.bin(VSDiv, a, c) }
func (n I32NS) UDiv(a, c I32) I32   { return n.bin(VUDiv, a, c) }
func (n I32NS) SRem(a, c I32) I32   { return n.bin(VSRem, a, c) }
func (n I32NS) URem(a, c I32) I32   { return n.bin(VURem, a, c) }
func (n I32NS) Neg(a I32) I32       { return n.un(VNeg, a) }

// §A2. Each predicate pairs with the corresponding wrapping verb: the flag and
// the truncated result together are the full widened answer. There is no
// USubO — unsigned subtract borrow is ULt of the inputs alone.
func (n I32NS) SAddO(a, c I32) I1 { return n.pred(VSAddO, a, c) }
func (n I32NS) UAddO(a, c I32) I1 { return n.pred(VUAddO, a, c) }
func (n I32NS) SSubO(a, c I32) I1 { return n.pred(VSSubO, a, c) }
func (n I32NS) SMulO(a, c I32) I1 { return n.pred(VSMulO, a, c) }
func (n I32NS) UMulO(a, c I32) I1 { return n.pred(VUMulO, a, c) }

// §A4.
func (n I32NS) Not(a I32) I32    { return n.un(VNot, a) }
func (n I32NS) And(a, c I32) I32 { return n.bin(VAnd, a, c) }
func (n I32NS) Or(a, c I32) I32  { return n.bin(VOr, a, c) }
func (n I32NS) Xor(a, c I32) I32 { return n.bin(VXor, a, c) }

// §A5. Shift amounts are taken modulo 32. No form traps.
func (n I32NS) Shl(a, amt I32) I32  { return n.bin(VShl, a, amt) }
func (n I32NS) SShr(a, amt I32) I32 { return n.bin(VSShr, a, amt) }
func (n I32NS) UShr(a, amt I32) I32 { return n.bin(VUShr, a, amt) }
func (n I32NS) RotL(a, amt I32) I32 { return n.bin(VRotL, a, amt) }
func (n I32NS) RotR(a, amt I32) I32 { return n.bin(VRotR, a, amt) }

// §A6. The zero-input results of Clz and Ctz are 32, specified rather than
// target-defined.
func (n I32NS) Clz(a I32) I32    { return n.un(VClz, a) }
func (n I32NS) Ctz(a I32) I32    { return n.un(VCtz, a) }
func (n I32NS) Popcnt(a I32) I32 { return n.un(VPopcnt, a) }
func (n I32NS) Bswap(a I32) I32  { return n.un(VBswap, a) }

// §B. There is no Gt or Ge; swap the operands.
func (n I32NS) Eq(a, c I32) I1  { return n.pred(VEq, a, c) }
func (n I32NS) Ne(a, c I32) I1  { return n.pred(VNe, a, c) }
func (n I32NS) SLt(a, c I32) I1 { return n.pred(VSLt, a, c) }
func (n I32NS) ULt(a, c I32) I1 { return n.pred(VULt, a, c) }
func (n I32NS) SLe(a, c I32) I1 { return n.pred(VSLe, a, c) }
func (n I32NS) ULe(a, c I32) I1 { return n.pred(VULe, a, c) }

// I64NS is the i64 namespace.
type I64NS struct{ b *Builder }

func (n I64NS) un(v Verb, a I64) I64     { return I64{n.b.def1(Op{TypeI64, v}, TypeI64, a.d)} }
func (n I64NS) bin(v Verb, a, c I64) I64 { return I64{n.b.def1(Op{TypeI64, v}, TypeI64, a.d, c.d)} }
func (n I64NS) pred(v Verb, a, c I64) I1 { return I1{n.b.def1(Op{TypeI64, v}, TypeI1, a.d, c.d)} }

// Const emits i64.const.
func (n I64NS) Const(v int64) I64 { return n.ConstOf(Int(v)) }

// ConstOf emits i64.const with a symbolic constant.
func (n I64NS) ConstOf(c Const) I64 {
	return I64{n.b.def1i(Op{TypeI64, VConst}, TypeI64, nil, &imm{lit: c, hasLit: true})}
}

func (n I64NS) Add(a, c I64) I64    { return n.bin(VAdd, a, c) }
func (n I64NS) Sub(a, c I64) I64    { return n.bin(VSub, a, c) }
func (n I64NS) Mul(a, c I64) I64    { return n.bin(VMul, a, c) }
func (n I64NS) SMulHi(a, c I64) I64 { return n.bin(VSMulHi, a, c) }
func (n I64NS) UMulHi(a, c I64) I64 { return n.bin(VUMulHi, a, c) }
func (n I64NS) SDiv(a, c I64) I64   { return n.bin(VSDiv, a, c) }
func (n I64NS) UDiv(a, c I64) I64   { return n.bin(VUDiv, a, c) }
func (n I64NS) SRem(a, c I64) I64   { return n.bin(VSRem, a, c) }
func (n I64NS) URem(a, c I64) I64   { return n.bin(VURem, a, c) }
func (n I64NS) Neg(a I64) I64       { return n.un(VNeg, a) }

func (n I64NS) SAddO(a, c I64) I1 { return n.pred(VSAddO, a, c) }
func (n I64NS) UAddO(a, c I64) I1 { return n.pred(VUAddO, a, c) }
func (n I64NS) SSubO(a, c I64) I1 { return n.pred(VSSubO, a, c) }
func (n I64NS) SMulO(a, c I64) I1 { return n.pred(VSMulO, a, c) }
func (n I64NS) UMulO(a, c I64) I1 { return n.pred(VUMulO, a, c) }

func (n I64NS) Not(a I64) I64    { return n.un(VNot, a) }
func (n I64NS) And(a, c I64) I64 { return n.bin(VAnd, a, c) }
func (n I64NS) Or(a, c I64) I64  { return n.bin(VOr, a, c) }
func (n I64NS) Xor(a, c I64) I64 { return n.bin(VXor, a, c) }

func (n I64NS) Shl(a, amt I64) I64  { return n.bin(VShl, a, amt) }
func (n I64NS) SShr(a, amt I64) I64 { return n.bin(VSShr, a, amt) }
func (n I64NS) UShr(a, amt I64) I64 { return n.bin(VUShr, a, amt) }
func (n I64NS) RotL(a, amt I64) I64 { return n.bin(VRotL, a, amt) }
func (n I64NS) RotR(a, amt I64) I64 { return n.bin(VRotR, a, amt) }

func (n I64NS) Clz(a I64) I64    { return n.un(VClz, a) }
func (n I64NS) Ctz(a I64) I64    { return n.un(VCtz, a) }
func (n I64NS) Popcnt(a I64) I64 { return n.un(VPopcnt, a) }
func (n I64NS) Bswap(a I64) I64  { return n.un(VBswap, a) }

func (n I64NS) Eq(a, c I64) I1  { return n.pred(VEq, a, c) }
func (n I64NS) Ne(a, c I64) I1  { return n.pred(VNe, a, c) }
func (n I64NS) SLt(a, c I64) I1 { return n.pred(VSLt, a, c) }
func (n I64NS) ULt(a, c I64) I1 { return n.pred(VULt, a, c) }
func (n I64NS) SLe(a, c I64) I1 { return n.pred(VSLe, a, c) }
func (n I64NS) ULe(a, c I64) I1 { return n.pred(VULe, a, c) }
