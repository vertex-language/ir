package air

// §W and §H: the work-item and wave verbs, barriers, fences and atomics.

import (
	"fmt"

	am "github.com/vertex-language/air"
	"github.com/vertex-language/ir"
)

// workitem reads the built-in argument that answers the verb: a uint3 by
// axis, or a uint.
func (x *fn) workitem(in *ir.Inst) {
	p := x.builtin[in.Op().Verb]
	if axis, ok := in.Axis(); ok {
		x.def(in, x.cur.ExtractElement(p, am.ConstInt(am.I32, int64(axis))))
		return
	}
	x.vals[in.Result(0)] = p
}

// wave is §W's wave verbs over SIMD-group functions. An Apple GPU's SIMD
// group is 32 wide, as a warp is, and its functions act on the active
// lanes, so the member mask is applied where the verb's answer depends on
// it -- the votes -- and not needed where it doesn't -- the shuffles, which
// read from a lane the caller already chose.
func (x *fn) wave(in *ir.Inst) error {
	b := x.cur
	switch in.Op().Verb {
	case ir.VWaveShflIdx:
		x.def(in, b.SimdFn(am.SimdOpShuffle, false, x.arg(in, 0), x.arg(in, 1)))
	case ir.VWaveShflUp:
		// A lane with no source lane below reads its own: simd_shuffle_up
		// leaves the lowest delta lanes unchanged.
		x.def(in, b.SimdFn(am.SimdOpShuffleUp, false, x.arg(in, 0), x.arg(in, 1)))
	case ir.VWaveShflDown:
		x.def(in, b.SimdFn(am.SimdOpShuffleDown, false, x.arg(in, 0), x.arg(in, 1)))
	case ir.VWaveShflXor:
		x.def(in, b.SimdFn(am.SimdOpShuffleXor, false, x.arg(in, 0), x.arg(in, 1)))
	case ir.VWaveReadFirstLane:
		x.def(in, b.SimdFn(am.SimdOpBroadcastFirst, false, x.arg(in, 0)))
	case ir.VWaveBallot:
		mask := b.ZExt(x.arg(in, 1), am.ULong)
		x.def(in, b.And(b.SimdBallot(x.arg(in, 0)), mask))
	case ir.VWaveAny, ir.VWaveAll:
		// Over the lanes in the mask: any is some lane in it where c
		// holds, all is no lane in it where c fails.
		mask := b.ZExt(x.arg(in, 1), am.ULong)
		zero := am.ConstUint(am.ULong, 0)
		if in.Op().Verb == ir.VWaveAny {
			x.def(in, b.ICmp(am.NE, b.And(b.SimdBallot(x.arg(in, 0)), mask), zero))
		} else {
			fails := b.Xor(x.arg(in, 0), am.ConstBool(true))
			x.def(in, b.ICmp(am.EQ, b.And(b.SimdBallot(fails), mask), zero))
		}
	default:
		return fmt.Errorf("not lowered for AIR yet")
	}
	return nil
}

// fence is a memory fence. Relaxed ones are nothing. Metal has
// atomic_thread_fence from MSL 3.2; before it, a stronger fence has no
// spelling, and is refused by name rather than dropped.
func (x *fn) fence(in *ir.Inst) error {
	if in.SingleThread() {
		return nil
	}
	o := in.Orderings()[0]
	if o == ir.Unordered || o == ir.Monotonic {
		return nil
	}
	if x.l.out.Language < am.MSL32 {
		return fmt.Errorf("a %s fence: atomic_thread_fence is MSL 3.2's, and the module is MSL %v", o, x.l.out.Language)
	}
	scope := am.ScopeDevice
	flags := am.MemDevice | am.MemThreadgroup
	if in.Scope() == ir.Workgroup {
		scope = am.ScopeThreadgroup
	}
	x.cur.Fence(flags, am.SeqCst, scope)
	return nil
}

// atomic is §H over air's atomics. Metal's atomics are relaxed, and are
// on 32-bit integers and floats in device and threadgroup memory; an
// ordering stronger than relaxed is a fence before and after it, from MSL
// 3.2, and refused before. The space is the pointer's (space.go).
func (x *fn) atomic(in *ir.Inst) error {
	op := in.Op()
	t := op.Type
	switch t {
	case ir.TypeI32, ir.TypeF32:
	default:
		return fmt.Errorf("a %s atomic: Metal's atomics are on 32-bit integers and floats", t)
	}
	strong := false
	for _, o := range in.Orderings() {
		if o != ir.Unordered && o != ir.Monotonic {
			strong = true
		}
	}
	if strong {
		if err := x.fence(in); err != nil {
			return err
		}
	}
	b := x.cur
	vt, _ := x.regType(t, 0)
	switch op.Verb {
	case ir.VAtomicLoad:
		x.def(in, b.AtomicLoad(x.typed(x.arg(in, 0), vt)))
	case ir.VAtomicStore:
		b.AtomicStore(x.arg(in, 0), x.typed(x.arg(in, 1), vt))
	case ir.VAtomicRmwXchg:
		x.def(in, b.AtomicExchange(x.typed(x.arg(in, 1), vt), x.arg(in, 0)))
	case ir.VAtomicCas:
		// §H's compare-and-swap answers the old value; air's takes the
		// expected value in thread memory and answers the old value too.
		if t != ir.TypeI32 {
			return fmt.Errorf("a float compare-and-swap: Metal compares integers")
		}
		e := x.casSlot[in]
		b.Store(x.arg(in, 0), e)
		x.def(in, b.AtomicCmpXchg(x.typed(x.arg(in, 2), vt), e, x.arg(in, 1)))
	default:
		rmw := map[ir.Verb]struct {
			op     am.AtomicOp
			signed bool
		}{
			ir.VAtomicRmwAdd: {am.AtomicAdd, false}, ir.VAtomicRmwSub: {am.AtomicSub, false},
			ir.VAtomicRmwAnd: {am.AtomicAnd, false}, ir.VAtomicRmwOr: {am.AtomicOr, false},
			ir.VAtomicRmwXor:  {am.AtomicXor, false},
			ir.VAtomicRmwSMin: {am.AtomicMin, true}, ir.VAtomicRmwSMax: {am.AtomicMax, true},
			ir.VAtomicRmwUMin: {am.AtomicMin, false}, ir.VAtomicRmwUMax: {am.AtomicMax, false},
		}[op.Verb]
		if t == ir.TypeF32 && rmw.op != am.AtomicAdd && rmw.op != am.AtomicSub {
			return fmt.Errorf("a float atomic %s: Metal's float atomics add and subtract", rmw.op)
		}
		x.def(in, b.AtomicRMWSigned(rmw.op, x.typed(x.arg(in, 1), vt), x.arg(in, 0), rmw.signed))
	}
	if strong {
		return x.fence(in)
	}
	return nil
}
