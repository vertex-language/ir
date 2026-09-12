package ptx

// §H. PTX's memory model has the orderings VIR has and the scopes §H
// gained, from sm_70 up. Below sm_70 there are no scope or semantic
// qualifiers: an atomic is atom, a load or store is volatile, and an
// acquire or release is a membar on the right side of it.
//
// seq_cst has no atom qualifier of its own. The mapping is LLVM's: a
// fence.sc at the scope ahead of the access, then the acq_rel form.

import (
	"fmt"

	"github.com/vertex-language/ir"
	"github.com/vertex-language/ptx"
)

// scopeOf is a VIR scope as PTX spells it; absent means system.
func scopeOf(s ir.Scope) ptx.Scope {
	switch s {
	case ir.Workgroup:
		return ptx.ScopeCTA
	case ir.Device:
		return ptx.ScopeGPU
	}
	return ptx.ScopeSys
}

// levelOf is the same scope as a pre-sm_70 membar level.
func levelOf(s ir.Scope) ptx.Level {
	switch s {
	case ir.Workgroup:
		return ptx.LevelCTA
	case ir.Device:
		return ptx.LevelGL
	}
	return ptx.LevelSys
}

func (x *fn) scoped() bool { return x.l.opts.sm().SM >= 70 }

// seqCstFence is the fence.sc that precedes a sequentially consistent
// access on a scoped target, and membar on an unscoped one.
func (x *fn) seqCstFence(s ir.Scope) {
	if x.scoped() {
		x.b.Fence(ptx.SemSC, scopeOf(s))
	} else {
		x.b.Membar(levelOf(s))
	}
}

func (x *fn) fence(in *ir.Inst) error {
	if in.SingleThread() {
		// A compiler barrier. Nothing here reorders across it, so
		// nothing is emitted; the token is honoured by being read.
		return nil
	}
	o := in.Orderings()[0]
	s := in.Scope()
	if o == ir.Unordered || o == ir.Monotonic {
		return nil
	}
	if !x.scoped() {
		x.b.Membar(levelOf(s))
		return nil
	}
	if o == ir.SeqCst {
		x.b.Fence(ptx.SemSC, scopeOf(s))
	} else {
		x.b.Fence(ptx.SemAcqRel, scopeOf(s))
	}
	return nil
}

func (x *fn) atomic(in *ir.Inst) error {
	op := in.Op()
	t := op.Type
	ords := in.Orderings()
	o := ords[0]
	s := in.Scope()
	if s == ir.Workgroup && !x.scoped() {
		return fmt.Errorf("workgroup scope needs sm_70; widening it is correct and not yet a decision")
	}
	if t.IsFloat() {
		if err := x.l.needSM(op, 20, "atom.add.f32"); err != nil {
			return err
		}
		if t == ir.TypeF64 {
			if err := x.l.needSM(op, 60, "atom.add.f64"); err != nil {
				return err
			}
		}
	}

	switch op.Verb {
	case ir.VAtomicLoad:
		return x.atomicLoad(in, o, s)
	case ir.VAtomicStore:
		return x.atomicStore(in, o, s)
	case ir.VAtomicCas:
		return x.atomicRmw(in, ptx.AtomCAS, bitsT(t), o, s, x.arg(in, 0), x.arg(in, 1))
	}

	v := x.arg(in, 0)
	var aop ptx.AtomOp
	var ty ptx.Type
	switch op.Verb {
	case ir.VAtomicRmwAdd:
		aop = ptx.AtomAdd
		ty = unsignedT(t)
		if t.IsFloat() {
			ty = floatT(t)
		}
	case ir.VAtomicRmwSub:
		// atom has no sub; add the negation, which wraps as §0 says.
		neg := x.temp(bitsT(t))
		if t == ir.TypePtr {
			x.b.Neg(ptx.S64, neg, v)
		} else {
			x.b.Neg(signedT(t), neg, v)
		}
		v = neg
		aop, ty = ptx.AtomAdd, unsignedT(t)
	case ir.VAtomicRmwAnd:
		aop, ty = ptx.AtomAnd, bitsT(t)
	case ir.VAtomicRmwOr:
		aop, ty = ptx.AtomOr, bitsT(t)
	case ir.VAtomicRmwXor:
		aop, ty = ptx.AtomXor, bitsT(t)
	case ir.VAtomicRmwXchg:
		aop, ty = ptx.AtomExch, bitsT(t)
	case ir.VAtomicRmwSMin:
		aop, ty = ptx.AtomMin, signedT(t)
	case ir.VAtomicRmwSMax:
		aop, ty = ptx.AtomMax, signedT(t)
	case ir.VAtomicRmwUMin:
		aop, ty = ptx.AtomMin, unsignedT(t)
	case ir.VAtomicRmwUMax:
		aop, ty = ptx.AtomMax, unsignedT(t)
	}
	return x.atomicRmw(in, aop, ty, o, s, v)
}

// atomicRmw is atom with the semantic and scope the ordering asks for.
// addr is the last register operand; vals are the value operands in
// PTX's order.
func (x *fn) atomicRmw(in *ir.Inst, aop ptx.AtomOp, ty ptx.Type, o ir.Ordering, s ir.Scope, vals ...ptx.Reg) error {
	d := x.res(in)
	addr := x.arg(in, in.NumArgs()-1)
	ops := make([]ptx.Operand, len(vals))
	for i, v := range vals {
		ops[i] = v
	}
	if !x.scoped() {
		if o == ir.Release || o == ir.AcqRel || o == ir.SeqCst {
			x.b.Membar(levelOf(s))
		}
		x.b.Atom(ty, d, ptx.At(addr), ops, aop)
		if o == ir.Acquire || o == ir.AcqRel || o == ir.SeqCst {
			x.b.Membar(levelOf(s))
		}
		return nil
	}
	var sem ptx.Sem
	switch o {
	case ir.Monotonic:
		sem = ptx.SemRelaxed
	case ir.Acquire:
		sem = ptx.SemAcquire
	case ir.Release:
		sem = ptx.SemRelease
	case ir.AcqRel:
		sem = ptx.SemAcqRel
	case ir.SeqCst:
		x.seqCstFence(s)
		sem = ptx.SemAcqRel
	}
	x.b.Atom(ty, d, ptx.At(addr), ops, aop, sem, scopeOf(s))
	return nil
}

func (x *fn) atomicLoad(in *ir.Inst, o ir.Ordering, s ir.Scope) error {
	t := in.Op().Type
	d, addr := x.res(in), x.arg(in, 0)
	ty := memT(t)
	if !x.scoped() {
		x.b.Emit("ld.volatile"+ty.String(), []ptx.Operand{d}, []ptx.Operand{ptx.At(addr)})
		if o == ir.Acquire || o == ir.SeqCst {
			x.b.Membar(levelOf(s))
		}
		return nil
	}
	switch o {
	case ir.Unordered, ir.Monotonic:
		x.b.Ld(ty, d, ptx.At(addr), ptx.SemRelaxed, scopeOf(s))
	case ir.Acquire:
		x.b.Ld(ty, d, ptx.At(addr), ptx.SemAcquire, scopeOf(s))
	case ir.SeqCst:
		x.seqCstFence(s)
		x.b.Ld(ty, d, ptx.At(addr), ptx.SemAcquire, scopeOf(s))
	default:
		return fmt.Errorf("a load with %s ordering", o)
	}
	return nil
}

func (x *fn) atomicStore(in *ir.Inst, o ir.Ordering, s ir.Scope) error {
	t := in.Op().Type
	v, addr := x.arg(in, 0), x.arg(in, 1)
	ty := memT(t)
	if !x.scoped() {
		if o == ir.Release || o == ir.SeqCst {
			x.b.Membar(levelOf(s))
		}
		x.b.Emit("st.volatile"+ty.String(), nil, []ptx.Operand{ptx.At(addr), v})
		return nil
	}
	switch o {
	case ir.Unordered, ir.Monotonic:
		x.b.St(ty, ptx.At(addr), v, ptx.SemRelaxed, scopeOf(s))
	case ir.Release:
		x.b.St(ty, ptx.At(addr), v, ptx.SemRelease, scopeOf(s))
	case ir.SeqCst:
		x.seqCstFence(s)
		x.b.St(ty, ptx.At(addr), v, ptx.SemRelease, scopeOf(s))
	default:
		return fmt.Errorf("a store with %s ordering", o)
	}
	return nil
}
