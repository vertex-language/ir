package ir

// I128NS is the i128 namespace.
//
// No target here has a 128-bit general register. The width is carried as a
// pair of i64 halves and the arithmetic is done over the two, the way i386
// already carries i64 on a machine with 32-bit registers -- so the verbs
// are §L's, unchanged, and what differs is only where the halves live.
// The four divisions are the exception: a pair cannot do them in line, and
// they become calls to the helpers a C compiler calls, as §E does for
// every width a target is short of.
type I128NS struct{ b *Builder }

func (n I128NS) un(v Verb, a I128) I128 { return I128{n.b.def1(Op{TypeI128, v}, TypeI128, a.d)} }
func (n I128NS) bin(v Verb, a, c I128) I128 {
	return I128{n.b.def1(Op{TypeI128, v}, TypeI128, a.d, c.d)}
}
func (n I128NS) pred(v Verb, a, c I128) I1 { return I1{n.b.def1(Op{TypeI128, v}, TypeI1, a.d, c.d)} }

// Const emits i128.const. A literal wider than int64 is built from its
// halves with Parts; this is the common small case, sign-extended.
func (n I128NS) Const(v int64) I128 { return n.ConstOf(Int(v)) }

// ConstOf emits i128.const with a symbolic constant.
func (n I128NS) ConstOf(c Const) I128 {
	return I128{n.b.def1i(Op{TypeI128, VConst}, TypeI128, nil, &imm{lit: c, hasLit: true})}
}

// Parts assembles a value from its halves: hi the more significant i64, lo
// the less. It is the way to write a literal no int64 holds, and the
// inverse of Lo and Hi.
func (n I128NS) Parts(hi, lo I64) I128 {
	return n.Or(n.Shl(n.ZExtI64(hi), n.Const(64)), n.ZExtI64(lo))
}

// Lo and Hi are the halves of a value: Lo is i64.wrap_i128, Hi the same of
// the value shifted down.
func (n I128NS) Lo(a I128) I64 { return n.b.I64.WrapI128(a) }
func (n I128NS) Hi(a I128) I64 { return n.b.I64.WrapI128(n.UShr(a, n.Const(64))) }

// §A. Integer arithmetic wraps; overflow is detected by §A2, never by the
// arithmetic itself. The division verbs trap rather than leaving a case
// undefined.
func (n I128NS) Add(a, c I128) I128  { return n.bin(VAdd, a, c) }
func (n I128NS) Sub(a, c I128) I128  { return n.bin(VSub, a, c) }
func (n I128NS) Mul(a, c I128) I128  { return n.bin(VMul, a, c) }
func (n I128NS) SDiv(a, c I128) I128 { return n.bin(VSDiv, a, c) }
func (n I128NS) UDiv(a, c I128) I128 { return n.bin(VUDiv, a, c) }
func (n I128NS) SRem(a, c I128) I128 { return n.bin(VSRem, a, c) }
func (n I128NS) URem(a, c I128) I128 { return n.bin(VURem, a, c) }
func (n I128NS) Neg(a I128) I128     { return n.un(VNeg, a) }

// §A4.
func (n I128NS) Not(a I128) I128    { return n.un(VNot, a) }
func (n I128NS) And(a, c I128) I128 { return n.bin(VAnd, a, c) }
func (n I128NS) Or(a, c I128) I128  { return n.bin(VOr, a, c) }
func (n I128NS) Xor(a, c I128) I128 { return n.bin(VXor, a, c) }

// §A5. Shift amounts are taken modulo 128. No form traps.
func (n I128NS) Shl(a, amt I128) I128  { return n.bin(VShl, a, amt) }
func (n I128NS) SShr(a, amt I128) I128 { return n.bin(VSShr, a, amt) }
func (n I128NS) UShr(a, amt I128) I128 { return n.bin(VUShr, a, amt) }

// §A6. The zero-input results of Clz and Ctz are 128.
func (n I128NS) Clz(a I128) I128    { return n.un(VClz, a) }
func (n I128NS) Ctz(a I128) I128    { return n.un(VCtz, a) }
func (n I128NS) Popcnt(a I128) I128 { return n.un(VPopcnt, a) }
func (n I128NS) Bswap(a I128) I128  { return n.un(VBswap, a) }

// §B. There is no Gt or Ge; swap the operands.
func (n I128NS) Eq(a, c I128) I1  { return n.pred(VEq, a, c) }
func (n I128NS) Ne(a, c I128) I1  { return n.pred(VNe, a, c) }
func (n I128NS) SLt(a, c I128) I1 { return n.pred(VSLt, a, c) }
func (n I128NS) ULt(a, c I128) I1 { return n.pred(VULt, a, c) }
func (n I128NS) SLe(a, c I128) I1 { return n.pred(VSLe, a, c) }
func (n I128NS) ULe(a, c I128) I1 { return n.pred(VULe, a, c) }
