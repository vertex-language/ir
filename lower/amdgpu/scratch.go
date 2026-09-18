package amdgpu

// Private memory: ptr.alloc and register spills.
//
// A work-item's private segment is scratch, addressed per lane by the
// scratch_* instructions and reachable through a flat pointer in the
// private aperture — which is what ptr.alloc hands back, so that every
// later load and store through it is the flat instruction the rest of
// the backend already emits. The descriptor asks for the segment; on
// GFX9 and gfx90a a prologue also builds FLAT_SCRATCH from the
// flat-scratch-init user SGPRs and the wave's byte offset, which gfx940
// does in hardware.
//
// The frame is the allocs at their alignments, then a slot of eight
// bytes per spilled VGPR value. A spilled SGPR value goes to a lane of
// one reserved VGPR instead, through v_writelane and v_readlane, which
// ignore exec: a scalar spilled under one execution mask has to come
// back under another.
//
// Scratch is enabled for a function that allocs, and for one the
// allocator could not colour — which is known only after selection,
// with the entry already laid out without the scratch SGPRs. That
// function is selected again from the start with scratch on.

import (
	"errors"
	"fmt"

	"github.com/vertex-language/amdgpu/feature"

	"github.com/vertex-language/ir"
	"github.com/vertex-language/ir/lower/mir"
	"github.com/vertex-language/ir/lower/regalloc"
)

// sgprSpillVGPR is the VGPR whose lanes hold spilled scalars.
const sgprSpillVGPR = 59

// The first offset a 13-bit scratch offset field does not reach; past
// it the offset travels in offsetVGPR. The frame itself is bounded by
// the private segment, which the descriptor states in 32 bits.
const (
	maxScratchOffset = 4096
	offsetVGPR       = 57
)

// errNeedScratch says a function has to be lowered again with scratch.
var errNeedScratch = errors.New("the function spills")

// errNeedSpillVGPRs says a function has to be lowered again with n more
// VGPRs holding spilled scalars: its scalar spills outran the lanes.
type errNeedSpillVGPRs struct{ n int }

func (e errNeedSpillVGPRs) Error() string {
	return fmt.Sprintf("the function's scalar spills need %d more VGPRs", e.n)
}

// maxExtraSpillVGPRs bounds the VGPRs taken from the top of the singles
// for spilled scalars, past the reserved one: v49..v56 at most.
const maxExtraSpillVGPRs = 8

// spillLaneVGPR is the VGPR a spilled scalar's lane lies in: the
// reserved one for the first 64 lanes, then the extra ones downward
// from the top of the singles.
func spillLaneVGPR(lane int64) (vgpr int, laneIn int64) {
	if lane < 64 {
		return sgprSpillVGPR, lane
	}
	return vgprSinglesTo - int(lane/64-1), lane % 64
}

// frame is one function's private segment.
type frame struct {
	allocs   map[*ir.Inst]uint32 // each ptr.alloc's offset
	allocEnd uint32              // the allocs' extent, rounded to eight
	slots    int                 // spill slots of eight bytes, after them
}

func (fr *frame) size() uint32 { return fr.allocEnd + uint32(fr.slots)*8 }

// layoutFrame gives every ptr.alloc in the entry block its offset.
func layoutFrame(fn *ir.Func) (*frame, error) {
	fr := &frame{allocs: map[*ir.Inst]uint32{}}
	var at uint32
	for _, in := range fn.Entry().Insts() {
		if in.Op().Verb != ir.VAlloc {
			continue
		}
		size, align, err := allocShape(in)
		if err != nil {
			return nil, err
		}
		if align > 16 {
			return nil, fmt.Errorf("%s: an alignment of %d exceeds the private segment's 16", in.Op(), align)
		}
		at = alignUp32(at, uint32(align))
		fr.allocs[in] = at
		at += uint32(size)
	}
	fr.allocEnd = alignUp32(at, 8)
	return fr, nil
}

func allocShape(in *ir.Inst) (size, align uint64, err error) {
	if t := in.NamedType(); t != nil {
		size, align, err = sizeAlign(t.FType())
		if err != nil {
			return 0, 0, fmt.Errorf("%s: %w", in.Op(), err)
		}
	} else {
		size = in.Size()
		align, _ = in.Align()
		if align == 0 {
			align = 1
		}
	}
	if size == 0 {
		size = 1
	}
	return size, align, nil
}

// architectedScratch reports whether FLAT_SCRATCH is set up by the
// hardware, as on gfx940, or by a prologue.
func (l *lowerer) architectedScratch() bool {
	return l.opts.ASIC == feature.GFX940 || l.opts.ASIC == feature.GFX942
}

// scratchPrologue builds FLAT_SCRATCH where the hardware does not: the
// flat-scratch-init pair plus the wave's byte offset into the segment.
func (x *fnState) scratchPrologue(c *cursor) {
	if x.l.architectedScratch() {
		return
	}
	c.Emit(mir.Instr{Op: amdOp{mn: "s_add_u32", ops: []opnd{{kind: oFlatScratchLo}, useLo(0), use(1)}},
		Uses: rs(x.fsInit, x.waveOff)})
	c.Emit(mir.Instr{Op: amdOp{mn: "s_addc_u32", ops: []opnd{{kind: oFlatScratchHi}, useHi(0), imm(0)}},
		Uses: rs(x.fsInit)})
	// The scratch instructions want a scalar base; before gfx940 "off"
	// is not one, so zero it is.
	c.Emit(mir.Instr{Op: amdOp{mn: "s_mov_b32", ops: []opnd{def(0), imm(0)}}, Defs: rs(x.scratchZero)})
}

// apertureSGPR is the SGPR pair a function with scratch keeps the
// private aperture in, loaded once at entry: the first pair of the
// pool's range, which the pool then starts past. Reading src_private_base
// takes a scalar move, and an alloc's address is remade at every use
// when it spills (see allocAddrOp), which a fixed pair makes a plain
// VGPR move.
func apertureSGPR(calling bool) int {
	if calling {
		return callPairsFrom
	}
	return sgprPairsFrom
}

// allocAddrOp is a ptr.alloc's address: the private aperture's high
// half over the frame offset, or over the frame pointer plus it in a
// device function. One instruction with no operands, so that the
// allocator rematerializes it rather than keeping a hundred of them in
// VGPR pairs across a body that stores every local to its slot.
type allocAddrOp struct{ off int64 }

func (allocAddrOp) String() string { return "alloc_addr" }

// alloc is ptr.alloc: the private aperture's high half over the offset.
func (x *fnState) alloc(c *cursor, in *ir.Inst) error {
	off, ok := x.frame.allocs[in]
	if !ok {
		return fmt.Errorf("ptr.alloc outside the entry block")
	}
	d, err := x.result(in)
	if err != nil {
		return err
	}
	c.Emit(mir.Instr{Op: allocAddrOp{off: int64(off)}, Defs: rs(d)})
	if in.Zeroed() {
		size, _, _ := allocShape(in)
		zero := x.constV32(c, 0)
		var at uint64
		for ; at+4 <= size; at += 4 {
			x.emitMem(c, "flat_store_dword", nil, rs(d, zero), opnd{kind: oFlat, i: 0, imm: int64(at)}, use(1))
		}
		for ; at < size; at++ {
			x.emitMem(c, "flat_store_byte", nil, rs(d, zero), opnd{kind: oFlat, i: 0, imm: int64(at)}, use(1))
		}
	}
	return nil
}

// —— the spiller ——

// The spill ops, expanded by emit: a VGPR value to and from its scratch
// slot, an SGPR value to and from lanes of the reserved VGPR.
type (
	spillStoreOp struct {
		off int64
		w   width
	}
	spillLoadOp struct {
		off int64
		w   width
	}
)

type spiller struct{ x *fnState }

// Slot is the next eight bytes of the frame. Both kinds of spill take
// one: a scalar's lanes are numbered by it.
func (s *spiller) Slot() int {
	fr := s.x.frame
	slot := fr.slots
	fr.slots++
	return slot
}

// Rematerializable is a constant's materialization: a move of an
// immediate into a register, which is cheaper to repeat at each use
// than to keep in a lane or a slot.
func (s *spiller) Rematerializable(in mir.Instr) bool {
	if _, isAlloc := in.Op.(allocAddrOp); isAlloc {
		return true
	}
	op, ok := in.Op.(amdOp)
	if !ok || op.wait {
		return false
	}
	switch op.mn {
	case "s_mov_b32", "s_mov_b64", "v_mov_b32":
	default:
		return false
	}
	for _, o := range op.ops {
		switch o.kind {
		case oDef, oDefLo, oDefHi, oImm, oFImm:
		default:
			return false
		}
	}
	return true
}

// Store and Load reference the zero base on the generations whose
// scratch instructions take no "off" for a scalar base, so the
// allocator keeps it.
func (s *spiller) Store(slot int, v mir.VReg, c regalloc.Class) mir.Instr {
	w := widthOfClass(c)
	in := mir.Instr{Op: spillStoreOp{off: s.offset(slot, w), w: w}, Uses: rs(v)}
	if (w == v32 || w == v64) && !s.x.l.architectedScratch() {
		in.Uses = append(in.Uses, s.x.scratchZero)
	}
	return in
}

func (s *spiller) Load(slot int, v mir.VReg, c regalloc.Class) mir.Instr {
	w := widthOfClass(c)
	in := mir.Instr{Op: spillLoadOp{off: s.offset(slot, w), w: w}, Defs: rs(v)}
	if (w == v32 || w == v64) && !s.x.l.architectedScratch() {
		in.Uses = rs(s.x.scratchZero)
	}
	return in
}

// offset is a slot's byte offset for a vector value, or its first lane
// for a scalar one.
func (s *spiller) offset(slot int, w width) int64 {
	if w == s32 || w == s64 {
		return int64(slot * 2)
	}
	return int64(s.x.frame.allocEnd) + int64(slot)*8
}

func widthOfClass(c regalloc.Class) width {
	switch c {
	case classV64:
		return v64
	case classS32:
		return s32
	case classS64:
		return s64
	}
	return v32
}
