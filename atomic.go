package ir

// An Ordering is §8's memory ordering.
type Ordering uint8

const (
	Unordered Ordering = iota + 1
	Monotonic
	Acquire
	Release
	AcqRel
	SeqCst
)

var orderingText = [...]string{
	Unordered: "unordered", Monotonic: "monotonic", Acquire: "acquire",
	Release: "release", AcqRel: "acq_rel", SeqCst: "seq_cst",
}

func (o Ordering) String() string {
	if int(o) < len(orderingText) {
		return orderingText[o]
	}
	return "<invalid ordering>"
}

// strength ranks orderings for §19.9's no-stronger-than rule.
func (o Ordering) strength() int {
	switch o {
	case Unordered:
		return 0
	case Monotonic:
		return 1
	case Acquire, Release:
		return 2
	case AcqRel:
		return 3
	case SeqCst:
		return 4
	}
	return -1
}

// A FenceOpt is an option on fence.
type FenceOpt uint8

// SingleThread makes a fence a compiler barrier and nothing else. It orders this
// thread's accesses against this thread's own interrupted execution — a signal
// handler, a scheduler on the same core — and emits no machine barrier. It is
// C11's atomic_signal_fence, which is otherwise inexpressible: lowering it as an
// ordinary fence is a correctness-preserving pessimization at every use, and the
// alternative frontends reach for instead, asm volatile ("" ::: "memory"), is
// opaque to the optimizer in ways a fence is not.
const SingleThread FenceOpt = 1

// Atomic accesses assume natural alignment for their width and trap otherwise.
// An align attribute does not weaken this — a misaligned atomic is not atomic on
// any target the IR addresses — so only Volatile is admitted here.
func (b *Builder) atomicAttrs(op Op, o Ordering, attrs []MemAttr) *imm {
	if o == 0 {
		b.fail(op, ErrOrdering, "no ordering given")
		return nil
	}
	im := &imm{}
	im.ord[0] = o
	im.nord = 1
	for _, a := range attrs {
		if a.volatile {
			im.volatile = true
			continue
		}
		if a.align != 0 {
			b.fail(op, ErrAlign, "align on an atomic access")
			return nil
		}
	}
	return im
}

// rmwOrdering enforces §19.9: read-modify-write orderings are not unordered.
func (b *Builder) rmwOrdering(op Op, o Ordering) bool {
	if o == Unordered {
		b.fail(op, ErrOrdering, "read-modify-write ordering is unordered")
		return false
	}
	return true
}

// casOrderings enforces §19.9's two rules on compare-and-swap.
func (b *Builder) casOrderings(op Op, succ, fail Ordering) bool {
	if succ == Unordered || fail == Unordered {
		b.fail(op, ErrOrdering, "compare-and-swap ordering is unordered")
		return false
	}
	if fail == Release || fail == AcqRel {
		b.fail(op, ErrOrdering, "failure ordering is %s", fail)
		return false
	}
	if fail.strength() > succ.strength() {
		b.fail(op, ErrOrdering, "failure ordering %s is stronger than success ordering %s", fail, succ)
		return false
	}
	return true
}

func (b *Builder) atomicLoad(t RegType, v Verb, p Ptr, o Ordering, attrs []MemAttr) *Def {
	op := Op{t, v}
	im := b.atomicAttrs(op, o, attrs)
	if im == nil {
		return nil
	}
	return b.def1i(op, t, []*Def{p.d}, im)
}

func (b *Builder) atomicStore(t RegType, v Verb, val, addr *Def, o Ordering, attrs []MemAttr) {
	op := Op{t, v}
	im := b.atomicAttrs(op, o, attrs)
	if im == nil {
		return
	}
	b.voidi(op, []*Def{val, addr}, im)
}

func (b *Builder) atomicRmw(t RegType, v Verb, val, addr *Def, o Ordering, attrs []MemAttr) *Def {
	op := Op{t, v}
	if !b.rmwOrdering(op, o) {
		return nil
	}
	im := b.atomicAttrs(op, o, attrs)
	if im == nil {
		return nil
	}
	return b.def1i(op, t, []*Def{val, addr}, im)
}

func (b *Builder) atomicCas(t RegType, v Verb, exp, new, addr *Def, succ, fail Ordering, attrs []MemAttr) *Def {
	op := Op{t, v}
	if !b.casOrderings(op, succ, fail) {
		return nil
	}
	im := b.atomicAttrs(op, succ, attrs)
	if im == nil {
		return nil
	}
	im.ord[1] = fail
	im.nord = 2
	return b.def1i(op, t, []*Def{exp, new, addr}, im)
}

// —— i32 ——
//
// Narrow atomics live in the i32 namespace only. An i64 set would be reachable
// by zero-extension with no hardware distinction. _Atomic(_Bool) uses the 8-bit
// forms, since i1 has no storage width.

func (n I32NS) AtomicLoad(p Ptr, o Ordering, a ...MemAttr) I32 {
	return I32{n.b.atomicLoad(TypeI32, VAtomicLoad, p, o, a)}
}
func (n I32NS) AtomicULoad8(p Ptr, o Ordering, a ...MemAttr) I32 {
	return I32{n.b.atomicLoad(TypeI32, VAtomicULoad8, p, o, a)}
}
func (n I32NS) AtomicULoad16(p Ptr, o Ordering, a ...MemAttr) I32 {
	return I32{n.b.atomicLoad(TypeI32, VAtomicULoad16, p, o, a)}
}
func (n I32NS) AtomicStore(v I32, dst Ptr, o Ordering, a ...MemAttr) {
	n.b.atomicStore(TypeI32, VAtomicStore, v.d, dst.d, o, a)
}
func (n I32NS) AtomicStore8(v I32, dst Ptr, o Ordering, a ...MemAttr) {
	n.b.atomicStore(TypeI32, VAtomicStore8, v.d, dst.d, o, a)
}
func (n I32NS) AtomicStore16(v I32, dst Ptr, o Ordering, a ...MemAttr) {
	n.b.atomicStore(TypeI32, VAtomicStore16, v.d, dst.d, o, a)
}

// Each read-modify-write returns the old value.
func (n I32NS) AtomicRmwAdd(v I32, dst Ptr, o Ordering, a ...MemAttr) I32 {
	return I32{n.b.atomicRmw(TypeI32, VAtomicRmwAdd, v.d, dst.d, o, a)}
}
func (n I32NS) AtomicRmwSub(v I32, dst Ptr, o Ordering, a ...MemAttr) I32 {
	return I32{n.b.atomicRmw(TypeI32, VAtomicRmwSub, v.d, dst.d, o, a)}
}
func (n I32NS) AtomicRmwAnd(v I32, dst Ptr, o Ordering, a ...MemAttr) I32 {
	return I32{n.b.atomicRmw(TypeI32, VAtomicRmwAnd, v.d, dst.d, o, a)}
}
func (n I32NS) AtomicRmwOr(v I32, dst Ptr, o Ordering, a ...MemAttr) I32 {
	return I32{n.b.atomicRmw(TypeI32, VAtomicRmwOr, v.d, dst.d, o, a)}
}
func (n I32NS) AtomicRmwXor(v I32, dst Ptr, o Ordering, a ...MemAttr) I32 {
	return I32{n.b.atomicRmw(TypeI32, VAtomicRmwXor, v.d, dst.d, o, a)}
}
func (n I32NS) AtomicRmwXchg(v I32, dst Ptr, o Ordering, a ...MemAttr) I32 {
	return I32{n.b.atomicRmw(TypeI32, VAtomicRmwXchg, v.d, dst.d, o, a)}
}

// The narrow forms return the old value, zero-extended.
func (n I32NS) AtomicRmwAdd8(v I32, dst Ptr, o Ordering, a ...MemAttr) I32 {
	return I32{n.b.atomicRmw(TypeI32, VAtomicRmwAdd8, v.d, dst.d, o, a)}
}
func (n I32NS) AtomicRmwSub8(v I32, dst Ptr, o Ordering, a ...MemAttr) I32 {
	return I32{n.b.atomicRmw(TypeI32, VAtomicRmwSub8, v.d, dst.d, o, a)}
}
func (n I32NS) AtomicRmwAnd8(v I32, dst Ptr, o Ordering, a ...MemAttr) I32 {
	return I32{n.b.atomicRmw(TypeI32, VAtomicRmwAnd8, v.d, dst.d, o, a)}
}
func (n I32NS) AtomicRmwOr8(v I32, dst Ptr, o Ordering, a ...MemAttr) I32 {
	return I32{n.b.atomicRmw(TypeI32, VAtomicRmwOr8, v.d, dst.d, o, a)}
}
func (n I32NS) AtomicRmwXor8(v I32, dst Ptr, o Ordering, a ...MemAttr) I32 {
	return I32{n.b.atomicRmw(TypeI32, VAtomicRmwXor8, v.d, dst.d, o, a)}
}
func (n I32NS) AtomicRmwXchg8(v I32, dst Ptr, o Ordering, a ...MemAttr) I32 {
	return I32{n.b.atomicRmw(TypeI32, VAtomicRmwXchg8, v.d, dst.d, o, a)}
}
func (n I32NS) AtomicRmwAdd16(v I32, dst Ptr, o Ordering, a ...MemAttr) I32 {
	return I32{n.b.atomicRmw(TypeI32, VAtomicRmwAdd16, v.d, dst.d, o, a)}
}
func (n I32NS) AtomicRmwSub16(v I32, dst Ptr, o Ordering, a ...MemAttr) I32 {
	return I32{n.b.atomicRmw(TypeI32, VAtomicRmwSub16, v.d, dst.d, o, a)}
}
func (n I32NS) AtomicRmwAnd16(v I32, dst Ptr, o Ordering, a ...MemAttr) I32 {
	return I32{n.b.atomicRmw(TypeI32, VAtomicRmwAnd16, v.d, dst.d, o, a)}
}
func (n I32NS) AtomicRmwOr16(v I32, dst Ptr, o Ordering, a ...MemAttr) I32 {
	return I32{n.b.atomicRmw(TypeI32, VAtomicRmwOr16, v.d, dst.d, o, a)}
}
func (n I32NS) AtomicRmwXor16(v I32, dst Ptr, o Ordering, a ...MemAttr) I32 {
	return I32{n.b.atomicRmw(TypeI32, VAtomicRmwXor16, v.d, dst.d, o, a)}
}
func (n I32NS) AtomicRmwXchg16(v I32, dst Ptr, o Ordering, a ...MemAttr) I32 {
	return I32{n.b.atomicRmw(TypeI32, VAtomicRmwXchg16, v.d, dst.d, o, a)}
}

// AtomicCas returns the value read; success is Eq against expect.
func (n I32NS) AtomicCas(expect, new I32, dst Ptr, succ, fail Ordering, a ...MemAttr) I32 {
	return I32{n.b.atomicCas(TypeI32, VAtomicCas, expect.d, new.d, dst.d, succ, fail, a)}
}
func (n I32NS) AtomicCas8(expect, new I32, dst Ptr, succ, fail Ordering, a ...MemAttr) I32 {
	return I32{n.b.atomicCas(TypeI32, VAtomicCas8, expect.d, new.d, dst.d, succ, fail, a)}
}
func (n I32NS) AtomicCas16(expect, new I32, dst Ptr, succ, fail Ordering, a ...MemAttr) I32 {
	return I32{n.b.atomicCas(TypeI32, VAtomicCas16, expect.d, new.d, dst.d, succ, fail, a)}
}

// —— i64 ——

func (n I64NS) AtomicLoad(p Ptr, o Ordering, a ...MemAttr) I64 {
	return I64{n.b.atomicLoad(TypeI64, VAtomicLoad, p, o, a)}
}
func (n I64NS) AtomicStore(v I64, dst Ptr, o Ordering, a ...MemAttr) {
	n.b.atomicStore(TypeI64, VAtomicStore, v.d, dst.d, o, a)
}
func (n I64NS) AtomicRmwAdd(v I64, dst Ptr, o Ordering, a ...MemAttr) I64 {
	return I64{n.b.atomicRmw(TypeI64, VAtomicRmwAdd, v.d, dst.d, o, a)}
}
func (n I64NS) AtomicRmwSub(v I64, dst Ptr, o Ordering, a ...MemAttr) I64 {
	return I64{n.b.atomicRmw(TypeI64, VAtomicRmwSub, v.d, dst.d, o, a)}
}
func (n I64NS) AtomicRmwAnd(v I64, dst Ptr, o Ordering, a ...MemAttr) I64 {
	return I64{n.b.atomicRmw(TypeI64, VAtomicRmwAnd, v.d, dst.d, o, a)}
}
func (n I64NS) AtomicRmwOr(v I64, dst Ptr, o Ordering, a ...MemAttr) I64 {
	return I64{n.b.atomicRmw(TypeI64, VAtomicRmwOr, v.d, dst.d, o, a)}
}
func (n I64NS) AtomicRmwXor(v I64, dst Ptr, o Ordering, a ...MemAttr) I64 {
	return I64{n.b.atomicRmw(TypeI64, VAtomicRmwXor, v.d, dst.d, o, a)}
}
func (n I64NS) AtomicRmwXchg(v I64, dst Ptr, o Ordering, a ...MemAttr) I64 {
	return I64{n.b.atomicRmw(TypeI64, VAtomicRmwXchg, v.d, dst.d, o, a)}
}
func (n I64NS) AtomicCas(expect, new I64, dst Ptr, succ, fail Ordering, a ...MemAttr) I64 {
	return I64{n.b.atomicCas(TypeI64, VAtomicCas, expect.d, new.d, dst.d, succ, fail, a)}
}

// —— ptr ——

func (n PtrNS) AtomicLoad(p Ptr, o Ordering, a ...MemAttr) Ptr {
	return Ptr{n.b.atomicLoad(TypePtr, VAtomicLoad, p, o, a)}
}
func (n PtrNS) AtomicStore(v Ptr, dst Ptr, o Ordering, a ...MemAttr) {
	n.b.atomicStore(TypePtr, VAtomicStore, v.d, dst.d, o, a)
}

// AtomicRmwAdd takes an i64 delta and returns the old pointer.
func (n PtrNS) AtomicRmwAdd(delta I64, dst Ptr, o Ordering, a ...MemAttr) Ptr {
	return Ptr{n.b.atomicRmw(TypePtr, VAtomicRmwAdd, delta.d, dst.d, o, a)}
}
func (n PtrNS) AtomicRmwSub(delta I64, dst Ptr, o Ordering, a ...MemAttr) Ptr {
	return Ptr{n.b.atomicRmw(TypePtr, VAtomicRmwSub, delta.d, dst.d, o, a)}
}
func (n PtrNS) AtomicRmwXchg(v Ptr, dst Ptr, o Ordering, a ...MemAttr) Ptr {
	return Ptr{n.b.atomicRmw(TypePtr, VAtomicRmwXchg, v.d, dst.d, o, a)}
}
func (n PtrNS) AtomicCas(expect, new Ptr, dst Ptr, succ, fail Ordering, a ...MemAttr) Ptr {
	return Ptr{n.b.atomicCas(TypePtr, VAtomicCas, expect.d, new.d, dst.d, succ, fail, a)}
}

// Fence is bare: no type is involved. No access carries the singlethread token;
// fence alone does.
func (b *Builder) Fence(o Ordering, opts ...FenceOpt) {
	op := Op{TypeNone, VFence}
	if o == 0 {
		b.fail(op, ErrOrdering, "no ordering given")
		return
	}
	im := &imm{}
	im.ord[0] = o
	im.nord = 1
	for _, x := range opts {
		if x == SingleThread {
			im.single = true
		}
	}
	b.voidi(op, nil, im)
}
