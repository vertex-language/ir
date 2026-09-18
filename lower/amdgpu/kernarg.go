package amdgpu

// The kernel's entry: how its arguments and its position in the grid
// reach registers.
//
// The dispatcher hands a wave a few SGPRs and VGPRs and nothing else. The
// user SGPRs come first, in the fixed order the descriptor enables them
// — here only the kernarg segment pointer, in s[0:1] — then the system
// SGPRs: the workgroup id along x, and along y and z where the kernel
// asked. v0 holds the work-item id: on gfx90a and later all three axes
// packed ten bits each, before that x alone with y and z in v1 and v2
// where asked.
//
// Arguments are s_load'ed from the kernarg segment at their natural
// offsets and copied into VGPRs, since every value lives in one. The
// workgroup size and count are not in any register: they are hidden
// arguments the runtime appends after the explicit ones, at offsets the
// metadata note publishes, which is how HIP's gridDim and blockDim reach
// a kernel on code object v5.

import (
	"fmt"

	amdgpuasm "github.com/vertex-language/amdgpu"
	"github.com/vertex-language/amdgpu/feature"
	amdgpuobj "github.com/vertex-language/amdgpu/obj"

	"github.com/vertex-language/ir"
	"github.com/vertex-language/ir/lower/mir"
)

// kernelPlan is what the descriptor and the metadata need to know, decided
// before selection from the signature and the §W verbs the body uses.
type kernelPlan struct {
	args        []amdgpuobj.KernelArg
	argOffset   []uint32 // by parameter index
	kernargSize uint32

	// Hidden arguments, by axis: the workgroup count and size, or -1.
	blockCount [3]int64
	groupSize  [3]int64

	wgIDs [3]bool // workgroup id y and z system SGPRs
	wiIDs int     // work-item id VGPRs: 1, 2 or 3
	lds   uint32  // group segment
	byval []int   // the parameters that are aggregates in the buffer
}

// byValOf is the aggregate a byval parameter carries, if it is one.
func byValOf(p ir.Param) (*ir.Type, bool) {
	for _, a := range p.Attrs {
		if a.IsByVal() {
			return a.Type(), true
		}
	}
	return nil, false
}

// newKernel plans the kernel argument buffer and the ids the body reads.
func (l *lowerer) newKernel(fn *ir.Func) (*kernelPlan, error) {
	k := &kernelPlan{blockCount: [3]int64{-1, -1, -1}, groupSize: [3]int64{-1, -1, -1}, wiIDs: 1}
	var off uint32
	for i, p := range fn.Signature().Params() {
		size, align, kind := uint32(4), uint32(4), amdgpuobj.ByValue
		if bv, ok := byValOf(p); ok {
			// The aggregate itself, in the buffer: the parameter is its
			// address there. The buffer is the dispatcher's and read-only;
			// a body that stores through the parameter is not caught yet.
			s, a, err := sizeAlign(bv.FType())
			if err != nil {
				return nil, fmt.Errorf("lower: @%s: parameter %d: %w", fn.Name(), i, err)
			}
			size, align = uint32(s), uint32(a)
			if align < 4 {
				align = 4
			}
			k.byval = append(k.byval, i)
		} else {
			switch p.Type {
			case ir.TypeI64, ir.TypeF64:
				size, align = 8, 8
			case ir.TypePtr:
				size, align, kind = 8, 8, amdgpuobj.GlobalBuffer
			case ir.TypeI1, ir.TypeI32, ir.TypeF32:
			default:
				return nil, fmt.Errorf("lower: @%s: parameter %d is %s, which no kernel argument holds", fn.Name(), i, p.Type)
			}
		}
		off = alignUp32(off, align)
		k.args = append(k.args, amdgpuobj.KernelArg{Name: p.Name, Size: size, Offset: off, Kind: kind})
		k.argOffset = append(k.argOffset, off)
		off += size
	}
	// The hidden arguments the body asks for, after the explicit ones at
	// an 8-byte boundary. Each is its own entry; the runtime fills them
	// by kind.
	hidden := func(kind amdgpuobj.ArgKind, size uint32) int64 {
		off = alignUp32(off, size)
		k.args = append(k.args, amdgpuobj.KernelArg{Size: size, Offset: off, Kind: kind})
		o := int64(off)
		off += size
		return o
	}
	first := true
	if len(callees(fn)) > 0 || hasCallInd(fn) {
		// A callee may read any of the ids: ask for all of them.
		k.wgIDs[1], k.wgIDs[2], k.wiIDs = true, true, 3
		off = alignUp32(off, 8)
		first = false
		for axis := 0; axis < 3; axis++ {
			k.blockCount[axis] = hidden([...]amdgpuobj.ArgKind{amdgpuobj.HiddenBlockCountX, amdgpuobj.HiddenBlockCountY, amdgpuobj.HiddenBlockCountZ}[axis], 4)
		}
		for axis := 0; axis < 3; axis++ {
			k.groupSize[axis] = hidden([...]amdgpuobj.ArgKind{amdgpuobj.HiddenGroupSizeX, amdgpuobj.HiddenGroupSizeY, amdgpuobj.HiddenGroupSizeZ}[axis], 2)
		}
	}
	fn.WalkInsts(func(in *ir.Inst) bool {
		axis, _ := in.Axis()
		switch in.Op().Verb {
		case ir.VWorkgroupID:
			if axis > 0 {
				k.wgIDs[axis] = true
			}
		case ir.VWorkitemID:
			if int(axis)+1 > k.wiIDs {
				k.wiIDs = int(axis) + 1
			}
		case ir.VNumWorkgroups:
			if k.blockCount[axis] < 0 {
				if first {
					off = alignUp32(off, 8)
					first = false
				}
				k.blockCount[axis] = hidden([...]amdgpuobj.ArgKind{amdgpuobj.HiddenBlockCountX, amdgpuobj.HiddenBlockCountY, amdgpuobj.HiddenBlockCountZ}[axis], 4)
			}
		case ir.VWorkgroupSize:
			if k.groupSize[axis] < 0 {
				if first {
					off = alignUp32(off, 8)
					first = false
				}
				k.groupSize[axis] = hidden([...]amdgpuobj.ArgKind{amdgpuobj.HiddenGroupSizeX, amdgpuobj.HiddenGroupSizeY, amdgpuobj.HiddenGroupSizeZ}[axis], 2)
			}
		}
		return true
	})
	k.kernargSize = off
	return k, nil
}

// packedIDs reports whether the ASIC packs the three work-item ids into v0.
func (l *lowerer) packedIDs() bool {
	switch l.opts.ASIC {
	case feature.GFX90A, feature.GFX940, feature.GFX942:
		return true
	}
	return false
}

// kernelOptions is the descriptor as planned, with the register counts
// the allocation settled.
func (x *fnState) kernelOptions(vgprs, sgprs int) []amdgpuasm.KernelOption {
	k := x.k
	opts := []amdgpuasm.KernelOption{
		amdgpuasm.WithVGPRs(vgprs),
		amdgpuasm.WithSGPRs(sgprs),
		amdgpuasm.WithFloatMode(true, true, true),
		amdgpuasm.WithArgs(8, k.args...),
		amdgpuasm.WithKernargs(int(k.kernargSize)),
		amdgpuasm.WithWorkgroupIDs(k.wgIDs[1], k.wgIDs[2]),
		amdgpuasm.WithWorkitemIDs(k.wiIDs),
	}
	// Every kernel of the module shares the module's LDS layout.
	if x.l.ldsSize > 0 {
		opts = append(opts, amdgpuasm.WithLDS(int(x.l.ldsSize)))
	}
	if x.scratch && !x.l.architectedScratch() {
		opts = append(opts, amdgpuasm.WithFlatScratchInit())
	}
	return opts
}

// fnState is one function's lowering state.
type fnState struct {
	extraSpill int // VGPRs past the reserved one holding spilled scalars

	l   *lowerer
	fn  *ir.Func
	mf  *mir.Func
	vr  *vregs
	uni *uniformity
	k   *kernelPlan

	// divergent says some brif's condition differs across the wave: the
	// function is structurized after selection, and every i1 copy is
	// masked from the start.
	divergent bool

	// scratch says the function has a private segment: it allocs, or it
	// spills. frame is its layout; fsInit and waveOff the SGPRs the
	// prologue builds FLAT_SCRATCH from, scratchZero the base the
	// scratch instructions take before gfx940.
	scratch     bool
	frame       *frame
	device      bool // a device function: parameters in VGPRs, a frame on the stack, s_setpc to return
	calls       bool // the function calls: the argument registers are kept clear and SP is set up
	fsInit      mir.VReg
	waveOff     mir.VReg
	scratchZero mir.VReg

	// The incoming registers, pinned: the kernarg pointer, the workgroup
	// ids, the work-item id VGPRs. A device function has the implicit
	// registers instead: tid[0] packed, and the sizes and counts.
	kernarg  mir.VReg
	wgID     [3]mir.VReg
	tid      [3]mir.VReg // v0, v1, v2 — or v0 alone when packed
	wgSize   [3]mir.VReg
	wgCount  [3]mir.VReg
	implicit bool
}

// entry fills the entry block's head: the pinned inputs, the argument
// loads, the copies into VGPRs.
func (x *fnState) entry(mb *mir.Block) error {
	c := x.cursor(mb)
	vr := x.vr

	// User SGPRs: the kernarg pointer in s[0:1], then flat scratch init
	// where a prologue needs it. System SGPRs follow: the workgroup ids,
	// then the wave's scratch offset.
	x.kernarg = vr.fresh(s64)
	vr.pin(x.kernarg, physOf(s64, 0))
	next := 2
	initScratch := x.scratch && !x.l.architectedScratch()
	if initScratch {
		x.fsInit = vr.fresh(s64)
		vr.pin(x.fsInit, physOf(s64, next))
		next += 2
	}
	for axis := 0; axis < 3; axis++ {
		if axis > 0 && !x.k.wgIDs[axis] {
			continue
		}
		x.wgID[axis] = vr.fresh(s32)
		vr.pin(x.wgID[axis], physOf(s32, next))
		next++
	}
	if initScratch {
		x.waveOff = vr.fresh(s32)
		vr.pin(x.waveOff, physOf(s32, next))
		next++
		// The zero base the scratch instructions take, pinned so that
		// the spiller — which names it in every spill — never spills it.
		// A pinned register inside the allocatable file is one the
		// allocator keeps clear for as long as the value lives.
		x.scratchZero = vr.fresh(s32)
		vr.pin(x.scratchZero, physOf(s32, next))
		next++
	}
	n := x.k.wiIDs
	if x.l.packedIDs() {
		n = 1
	}
	for i := 0; i < n; i++ {
		x.tid[i] = vr.fresh(v32)
		vr.pin(x.tid[i], physOf(v32, i))
	}
	if x.calls {
		// A call passes arguments in v0 upward: the ids leave their
		// pinned registers now, so no argument's pin collides with
		// theirs; the implicit registers are filled for the callees;
		// and SP starts past the kernel's own frame.
		for i := 0; i < n; i++ {
			t := vr.fresh(v32)
			emitCopy(c, t, x.tid[i], v32)
			x.tid[i] = t
		}
		x.fillImplicit(c)
		c.Emit(mir.Instr{Op: spInitOp{}})
	}

	if initScratch {
		x.scratchPrologue(c)
	}

	// The arguments: scalar loads at their offsets, then copies out. A
	// byval aggregate is where it lies: the parameter is the kernarg
	// pointer plus its offset.
	byval := map[int]bool{}
	for _, i := range x.k.byval {
		byval[i] = true
	}
	for i, d := range x.fn.Params() {
		w, _ := widthOf(d.Type())
		off := int64(x.k.argOffset[i])
		if byval[i] {
			s := vr.fresh(s64)
			c.Emit(mir.Instr{Op: amdOp{mn: "s_add_u32", ops: []opnd{{kind: oDefLo}, {kind: oUseLo, i: 0}, {kind: oImm, imm: off}}}, Defs: []mir.VReg{s}, Uses: []mir.VReg{x.kernarg}})
			c.Emit(mir.Instr{Op: amdOp{mn: "s_addc_u32", ops: []opnd{{kind: oDefHi}, {kind: oUseHi, i: 0}, {kind: oImm, imm: 0}}}, Defs: []mir.VReg{s}, Uses: []mir.VReg{x.kernarg}})
			v, err := vr.define(d)
			if err != nil {
				return err
			}
			x.movPair(c, v, s)
			continue
		}
		switch w {
		case v32, s64:
			s := vr.fresh(s32)
			c.Emit(mir.Instr{Op: amdOp{mn: "s_load_dword", wait: true, ops: []opnd{{kind: oDef}, {kind: oSMEM, i: 0, imm: off}}},
				Defs: []mir.VReg{s}, Uses: []mir.VReg{x.kernarg}})
			if w == s64 {
				// An i1 argument arrives as a dword; the mask is the
				// lanes where it is non-zero — every lane, or none.
				m, err := vr.define(d)
				if err != nil {
					return err
				}
				c.Emit(mir.Instr{Op: amdOp{mn: "v_cmp_ne_u32", ops: []opnd{{kind: oDef}, {kind: oImm, imm: 0}, {kind: oUse, i: 0}}},
					Defs: []mir.VReg{m}, Uses: []mir.VReg{s}})
				continue
			}
			v, err := vr.define(d)
			if err != nil {
				return err
			}
			c.Emit(mir.Instr{Op: amdOp{mn: "v_mov_b32", ops: []opnd{{kind: oDef}, {kind: oUse, i: 0}}}, Defs: []mir.VReg{v}, Uses: []mir.VReg{s}})
		case v64:
			s := vr.fresh(s64)
			c.Emit(mir.Instr{Op: amdOp{mn: "s_load_dwordx2", wait: true, ops: []opnd{{kind: oDef}, {kind: oSMEM, i: 0, imm: off}}},
				Defs: []mir.VReg{s}, Uses: []mir.VReg{x.kernarg}})
			v, err := vr.define(d)
			if err != nil {
				return err
			}
			x.movPair(c, v, s)
		}
	}
	return nil
}

// movPair copies a 64-bit SGPR pair into a VGPR pair, half by half.
func (x *fnState) movPair(c *cursor, dst, src mir.VReg) {
	c.Emit(mir.Instr{Op: amdOp{mn: "v_mov_b32", ops: []opnd{{kind: oDefLo}, {kind: oUseLo, i: 0}}}, Defs: []mir.VReg{dst}, Uses: []mir.VReg{src}})
	c.Emit(mir.Instr{Op: amdOp{mn: "v_mov_b32", ops: []opnd{{kind: oDefHi}, {kind: oUseHi, i: 0}}}, Defs: []mir.VReg{dst}, Uses: []mir.VReg{src}})
}

// workitem selects a §W identity verb.
func (x *fnState) workitem(c *cursor, in *ir.Inst) error {
	d, err := x.vr.define(in.Result(0))
	if err != nil {
		return err
	}
	axis, _ := in.Axis()
	switch in.Op().Verb {
	case ir.VWorkitemID:
		if x.l.packedIDs() || x.implicit {
			switch axis {
			case ir.X:
				c.Emit(mir.Instr{Op: amdOp{mn: "v_and_b32", ops: []opnd{{kind: oDef}, {kind: oImm, imm: 0x3ff}, {kind: oUse, i: 0}}}, Defs: []mir.VReg{d}, Uses: []mir.VReg{x.tid[0]}})
			case ir.Y:
				c.Emit(mir.Instr{Op: amdOp{mn: "v_bfe_u32", ops: []opnd{{kind: oDef}, {kind: oUse, i: 0}, {kind: oImm, imm: 10}, {kind: oImm, imm: 10}}}, Defs: []mir.VReg{d}, Uses: []mir.VReg{x.tid[0]}})
			case ir.Z:
				c.Emit(mir.Instr{Op: amdOp{mn: "v_lshrrev_b32", ops: []opnd{{kind: oDef}, {kind: oImm, imm: 20}, {kind: oUse, i: 0}}}, Defs: []mir.VReg{d}, Uses: []mir.VReg{x.tid[0]}})
			}
			return nil
		}
		emitCopy(c, d, x.tid[axis], v32)
	case ir.VWorkgroupID:
		c.Emit(mir.Instr{Op: amdOp{mn: "v_mov_b32", ops: []opnd{{kind: oDef}, {kind: oUse, i: 0}}}, Defs: []mir.VReg{d}, Uses: []mir.VReg{x.wgID[axis]}})
	case ir.VWorkgroupSize:
		if x.implicit {
			c.Emit(mir.Instr{Op: amdOp{mn: "v_mov_b32", ops: []opnd{def(0), use(0)}}, Defs: rs(d), Uses: rs(x.wgSize[axis])})
			return nil
		}
		// A halfword hidden argument: a dword load and a mask.
		off := x.k.groupSize[axis]
		s := x.vr.fresh(s32)
		c.Emit(mir.Instr{Op: amdOp{mn: "s_load_dword", wait: true, ops: []opnd{{kind: oDef}, {kind: oSMEM, i: 0, imm: off &^ 3}}}, Defs: []mir.VReg{s}, Uses: []mir.VReg{x.kernarg}})
		// VOP2's second source is a VGPR, so the dword moves first.
		c.Emit(mir.Instr{Op: amdOp{mn: "v_mov_b32", ops: []opnd{{kind: oDef}, {kind: oUse, i: 0}}}, Defs: []mir.VReg{d}, Uses: []mir.VReg{s}})
		if off%4 == 2 {
			c.Emit(mir.Instr{Op: amdOp{mn: "v_lshrrev_b32", ops: []opnd{{kind: oDef}, {kind: oImm, imm: 16}, {kind: oUse, i: 0}}}, Defs: []mir.VReg{d}, Uses: []mir.VReg{d}})
		} else {
			c.Emit(mir.Instr{Op: amdOp{mn: "v_and_b32", ops: []opnd{{kind: oDef}, {kind: oImm, imm: 0xffff}, {kind: oUse, i: 0}}}, Defs: []mir.VReg{d}, Uses: []mir.VReg{d}})
		}
	case ir.VNumWorkgroups:
		if x.implicit {
			c.Emit(mir.Instr{Op: amdOp{mn: "v_mov_b32", ops: []opnd{def(0), use(0)}}, Defs: rs(d), Uses: rs(x.wgCount[axis])})
			return nil
		}
		s := x.vr.fresh(s32)
		c.Emit(mir.Instr{Op: amdOp{mn: "s_load_dword", wait: true, ops: []opnd{{kind: oDef}, {kind: oSMEM, i: 0, imm: x.k.blockCount[axis]}}}, Defs: []mir.VReg{s}, Uses: []mir.VReg{x.kernarg}})
		c.Emit(mir.Instr{Op: amdOp{mn: "v_mov_b32", ops: []opnd{{kind: oDef}, {kind: oUse, i: 0}}}, Defs: []mir.VReg{d}, Uses: []mir.VReg{s}})
	case ir.VLaneID:
		// The count of active lanes below this one: mbcnt over an
		// all-ones mask, low half then high.
		t := x.vr.temp(v32)
		c.Emit(mir.Instr{Op: amdOp{mn: "v_mbcnt_lo_u32_b32", ops: []opnd{{kind: oDef}, {kind: oImm, imm: -1}, {kind: oImm, imm: 0}}}, Defs: []mir.VReg{t}})
		c.Emit(mir.Instr{Op: amdOp{mn: "v_mbcnt_hi_u32_b32", ops: []opnd{{kind: oDef}, {kind: oImm, imm: -1}, {kind: oUse, i: 0}}}, Defs: []mir.VReg{d}, Uses: []mir.VReg{t}})
	case ir.VWaveSize:
		n := int64(64)
		if !x.l.wave64() {
			n = 32
		}
		c.Emit(mir.Instr{Op: amdOp{mn: "v_mov_b32", ops: []opnd{{kind: oDef}, {kind: oImm, imm: n}}}, Defs: []mir.VReg{d}})
	}
	return nil
}

func alignUp32(n, a uint32) uint32 { return (n + a - 1) &^ (a - 1) }
func alignUp(n, a uint64) uint64   { return (n + a - 1) &^ (a - 1) }
