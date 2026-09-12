package ir

// §W. Work-items.
//
// A kernel runs once per work-item, over a grid the host chose, and the
// first six verbs here are how a work-item learns where in that grid it
// is. They are i32 verbs with a literal axis rather than six special
// registers because the IR has no register that is not a value, and they
// are spelled in HSA's vocabulary — workitem, workgroup, wave — rather
// than CUDA's, because the IR names hardware and both vendors' hardware
// has these. threadIdx.x is i32.workitem_id x; blockDim.x is
// i32.workgroup_size x; warpSize is i32.wave_size.
//
// The wave verbs exchange values between the work-items of one wave
// without going through memory. Their mask operand is CUDA's member mask,
// which PTX has needed since independent thread scheduling; a target
// whose waves cannot diverge within an instruction has no use for it and
// reads the execution mask instead.
//
// barrier is bare: every work-item of the workgroup arrives before any
// leaves, and every access before it is visible to every work-item after
// it. A work-item that neither arrives nor exits keeps the others waiting,
// which is a hang and not a wrong answer — the same class of defined
// behaviour an infinite loop is.
//
// A CPU has none of this and refuses each verb by name.

// An Axis is §W's literal axis operand.
type Axis uint8

const (
	X Axis = iota
	Y
	Z
)

func (a Axis) String() string {
	switch a {
	case X:
		return "x"
	case Y:
		return "y"
	case Z:
		return "z"
	}
	return "<invalid axis>"
}

func (b *Builder) axis(v Verb, a Axis) *Def {
	op := Op{TypeI32, v}
	if a > Z {
		b.fail(op, ErrPlacement, "axis %d; the axes are x, y, z", a)
		return nil
	}
	return b.def1i(op, TypeI32, nil, &imm{axis: a, hasAxis: true})
}

// WorkitemID is the work-item's index within its workgroup along a.
func (n I32NS) WorkitemID(a Axis) I32 { return I32{n.b.axis(VWorkitemID, a)} }

// WorkgroupID is the workgroup's index within the grid along a.
func (n I32NS) WorkgroupID(a Axis) I32 { return I32{n.b.axis(VWorkgroupID, a)} }

// WorkgroupSize is the number of work-items in a workgroup along a.
func (n I32NS) WorkgroupSize(a Axis) I32 { return I32{n.b.axis(VWorkgroupSize, a)} }

// NumWorkgroups is the number of workgroups in the grid along a.
func (n I32NS) NumWorkgroups(a Axis) I32 { return I32{n.b.axis(VNumWorkgroups, a)} }

// LaneID is the work-item's index within its wave.
func (n I32NS) LaneID() I32 { return I32{n.b.def1(Op{TypeI32, VLaneID}, TypeI32)} }

// WaveSize is the number of work-items in a wave: 32 on one target, 32 or
// 64 on the other, and the module does not know which — the same reason
// sizeof is a value.
func (n I32NS) WaveSize() I32 { return I32{n.b.def1(Op{TypeI32, VWaveSize}, TypeI32)} }

// Barrier is a workgroup-wide execution and memory barrier.
func (b *Builder) Barrier() { b.void(Op{TypeNone, VBarrier}) }

// —— wave ——

func (n I32NS) shfl(v Verb, val, lane, mask I32) I32 {
	return I32{n.b.def1(Op{TypeI32, v}, TypeI32, val.d, lane.d, mask.d)}
}

// WaveShflIdx reads val from the work-item whose lane index is lane.
func (n I32NS) WaveShflIdx(val, lane, mask I32) I32 { return n.shfl(VWaveShflIdx, val, lane, mask) }

// WaveShflUp reads val from the work-item delta lanes below; a lane with
// none reads its own.
func (n I32NS) WaveShflUp(val, delta, mask I32) I32 { return n.shfl(VWaveShflUp, val, delta, mask) }

// WaveShflDown reads val from the work-item delta lanes above; a lane with
// none reads its own.
func (n I32NS) WaveShflDown(val, delta, mask I32) I32 {
	return n.shfl(VWaveShflDown, val, delta, mask)
}

// WaveShflXor reads val from the work-item whose lane index is this one's
// XOR lanemask.
func (n I32NS) WaveShflXor(val, lanemask, mask I32) I32 {
	return n.shfl(VWaveShflXor, val, lanemask, mask)
}

// WaveReadFirstLane is val as the lowest active lane holds it: a uniform
// value out of a divergent one.
func (n I32NS) WaveReadFirstLane(val I32) I32 {
	return I32{n.b.def1(Op{TypeI32, VWaveReadFirstLane}, TypeI32, val.d)}
}

// WaveBallot is one bit per lane, lane 0 in bit 0, set where c holds. It is
// i64 because Wave64 exists; a 32-wide target zero-extends.
func (n I64NS) WaveBallot(c I1, mask I32) I64 {
	return I64{n.b.def1(Op{TypeI64, VWaveBallot}, TypeI64, c.d, mask.d)}
}

// WaveAny reports whether c holds on any lane in mask.
func (n I1NS) WaveAny(c I1, mask I32) I1 {
	return I1{n.b.def1(Op{TypeI1, VWaveAny}, TypeI1, c.d, mask.d)}
}

// WaveAll reports whether c holds on every lane in mask.
func (n I1NS) WaveAll(c I1, mask I32) I1 {
	return I1{n.b.def1(Op{TypeI1, VWaveAll}, TypeI1, c.d, mask.d)}
}
