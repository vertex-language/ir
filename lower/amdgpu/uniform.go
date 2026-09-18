package amdgpu

// Uniformity: which values are the same in every lane of a wave.
//
// A branch on a uniform condition is a scalar branch; a branch on a
// divergent one must run both arms under an execution mask, which is the
// exec-mask lowering this package does not have yet. So the analysis
// decides what can be lowered today, and errs in the safe direction: a
// value it cannot prove uniform is divergent, which costs nothing but a
// refusal it would not otherwise have made.
//
// The sound small version. Uniform: a constant; a kernel parameter; a
// workgroup id, a workgroup size, a workgroup count, the wave size; the
// address of a global; a load from a read-only global through a uniform
// pointer; and any pure instruction whose every operand is uniform.
// Divergent: workitem_id, lane_id, every other load, every atomic, every
// call result, every wave verb, and a block parameter with a divergent
// argument on any edge — computed to a fixed point, since parameters and
// edges are cyclic.

import "github.com/vertex-language/ir"

type uniformity struct {
	uniform map[*ir.Def]bool
	model   memModel // which atomics have instructions, for needsStructure
}

func analyzeUniformity(f *ir.Func, model memModel) *uniformity {
	u := &uniformity{uniform: map[*ir.Def]bool{}, model: model}
	for _, p := range f.Params() {
		u.uniform[p] = true
	}
	// Parameters start uniform and lose it when an edge brings a
	// divergent argument; instructions are recomputed until nothing
	// changes.
	for _, blk := range f.Blocks() {
		for _, p := range blk.Params() {
			u.uniform[p] = !blk.IsPad()
		}
	}
	for changed := true; changed; {
		changed = false
		for _, blk := range f.Blocks() {
			for _, in := range blk.All() {
				for _, d := range in.Results() {
					was := u.uniform[d]
					now := u.instUniform(in)
					if was != now {
						u.uniform[d] = now
						changed = true
					}
				}
				for _, t := range in.Targets() {
					params := t.Block().Params()
					for i, a := range t.Args() {
						if i < len(params) && !u.uniform[a] && u.uniform[params[i]] {
							u.uniform[params[i]] = false
							changed = true
						}
					}
				}
			}
		}
	}
	return u
}

func (u *uniformity) instUniform(in *ir.Inst) bool {
	op := in.Op()
	switch op.Verb {
	case ir.VConst, ir.VGetAddr, ir.VWorkgroupID, ir.VWorkgroupSize, ir.VNumWorkgroups, ir.VWaveSize:
		return true
	case ir.VWorkitemID, ir.VLaneID,
		ir.VWaveShflIdx, ir.VWaveShflUp, ir.VWaveShflDown, ir.VWaveShflXor,
		ir.VWaveBallot, ir.VWaveAny, ir.VWaveAll, ir.VWaveReadFirstLane,
		ir.VCall, ir.VCallInd, ir.VAlloc, ir.VAlloca,
		ir.VAtomicLoad, ir.VAtomicCas, ir.VAtomicRmwAdd, ir.VAtomicRmwSub,
		ir.VAtomicRmwAnd, ir.VAtomicRmwOr, ir.VAtomicRmwXor, ir.VAtomicRmwXchg,
		ir.VAtomicRmwSMin, ir.VAtomicRmwSMax, ir.VAtomicRmwUMin, ir.VAtomicRmwUMax,
		ir.VMemCmp, ir.VAsm:
		return false
	case ir.VLoad, ir.VSLoad8, ir.VSLoad16, ir.VSLoad32, ir.VULoad8, ir.VULoad16, ir.VULoad32:
		// A read-only global read through a uniform address is one
		// value; anything writable might have been written by a lane.
		p := in.Arg(0)
		if !u.uniform[p] {
			return false
		}
		return u.readOnly(p)
	}
	for _, a := range in.Args() {
		if !u.uniform[a] {
			return false
		}
	}
	return true
}

// readOnly follows a pointer back to a ptr.getaddr of a ro global.
func (u *uniformity) readOnly(p *ir.Def) bool {
	for depth := 0; depth < 8; depth++ {
		in := p.Inst()
		if in == nil {
			return false
		}
		switch in.Op().Verb {
		case ir.VGetAddr:
			g, ok := in.Symbol().(*ir.Global)
			return ok && g.Domain() == ir.RO
		case ir.VAdd, ir.VSub:
			if in.Op().Type != ir.TypePtr {
				return false
			}
			p = in.Arg(0)
		default:
			return false
		}
	}
	return false
}

// isUniform reports whether d holds one value across the wave.
func (u *uniformity) isUniform(d *ir.Def) bool { return u.uniform[d] }

// needsStructure reports whether f branches on a value that differs
// across the wave: a brif or br_table on one, or a bulk-memory row
// whose loop would, since isel writes those as branches of its own.
func (u *uniformity) needsStructure(f *ir.Func) bool {
	divergent := false
	f.WalkInsts(func(in *ir.Inst) bool {
		switch in.Op().Verb {
		case ir.VBrIf, ir.VBrTable, ir.VCallInd:
			if !u.isUniform(in.Arg(0)) {
				divergent = true
			}
		case ir.VMemCpy, ir.VMemSet:
			if !u.isUniform(in.Arg(2)) {
				divergent = true
			}
		case ir.VMemMove, ir.VMemCmp:
			for _, a := range in.Args() {
				if !u.isUniform(a) {
					divergent = true
				}
			}
		case ir.VAtomicCas8, ir.VAtomicCas16,
			ir.VAtomicRmwAdd8, ir.VAtomicRmwSub8, ir.VAtomicRmwAnd8, ir.VAtomicRmwOr8, ir.VAtomicRmwXor8, ir.VAtomicRmwXchg8,
			ir.VAtomicRmwAdd16, ir.VAtomicRmwSub16, ir.VAtomicRmwAnd16, ir.VAtomicRmwOr16, ir.VAtomicRmwXor16, ir.VAtomicRmwXchg16:
			// A compare-and-swap loop leaves lane by lane.
			divergent = true
		case ir.VAtomicRmwAdd:
			// A float add is one too, on a generation with no instruction.
			if in.Op().Type.IsFloat() && !u.floatAtomicAddNative(in) {
				divergent = true
			}
		}
		return !divergent
	})
	return divergent
}

// floatAtomicAddNative reports whether the target has an instruction
// for this float atomic add: always on workgroup storage, and on a
// flat pointer where the generation has one.
func (u *uniformity) floatAtomicAddNative(in *ir.Inst) bool {
	if sharedPtr(in.Arg(1)) {
		return true
	}
	if in.Op().Type == ir.TypeF64 {
		return u.model != modelGFX9
	}
	return u.model == modelGFX940
}
