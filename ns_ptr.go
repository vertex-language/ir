package ir

// PtrNS is the ptr namespace. Pointers are opaque: the IR tracks addresses, not
// pointee types. ptr.add and ptr.sub wrap modulo 2^ptrbits.
type PtrNS struct{ b *Builder }

// Const emits ptr.const. Null is the only pointer literal; a non-null absolute
// address is FromI64 of an i64.const.
func (n PtrNS) Const() Ptr {
	return Ptr{n.b.def1i(Op{TypePtr, VConst}, TypePtr, nil, &imm{lit: Int(0), hasLit: true})}
}

// Add offsets a pointer by a byte count.
func (n PtrNS) Add(p Ptr, off I64) Ptr {
	return Ptr{n.b.def1(Op{TypePtr, VAdd}, TypePtr, p.d, off.d)}
}

// Sub offsets a pointer backwards by a byte count.
func (n PtrNS) Sub(p Ptr, off I64) Ptr {
	return Ptr{n.b.def1(Op{TypePtr, VSub}, TypePtr, p.d, off.d)}
}

// Diff is the signed byte distance between two pointers, sign-extended from
// ptrbits.
func (n PtrNS) Diff(a, b Ptr) I64 {
	return I64{n.b.def1(Op{TypePtr, VDiff}, TypeI64, a.d, b.d)}
}

// §B's four pointer comparisons. Lt and Le are unsigned address comparisons;
// there is no Gt or Ge, in this namespace or any other.
func (n PtrNS) Eq(a, b Ptr) I1 { return I1{n.b.def1(Op{TypePtr, VEq}, TypeI1, a.d, b.d)} }
func (n PtrNS) Ne(a, b Ptr) I1 { return I1{n.b.def1(Op{TypePtr, VNe}, TypeI1, a.d, b.d)} }
func (n PtrNS) Lt(a, b Ptr) I1 { return I1{n.b.def1(Op{TypePtr, VLt}, TypeI1, a.d, b.d)} }
func (n PtrNS) Le(a, b Ptr) I1 { return I1{n.b.def1(Op{TypePtr, VLe}, TypeI1, a.d, b.d)} }
