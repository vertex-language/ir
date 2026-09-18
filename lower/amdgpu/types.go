package amdgpu

// This backend's MIR: one opcode type, amdOp, carrying the mnemonic and an
// operand template. The template says where each assembler operand comes
// from — a def, a use, the low or high half of a 64-bit use, an
// immediate, a memory reference — and emit fills it in from the register
// assignment and hands the result to the assembler's Emit, which resolves
// the form by mnemonic and operand classes. That is the same resolution
// the differential test against llvm-mc exercises for every form, so a
// mnemonic spelled here is a mnemonic whose bytes are known to be right.
//
// # Registers
//
// Four register classes, disjoint by construction: 32-bit and 64-bit
// SGPRs, 32-bit and 64-bit VGPRs. A 64-bit value occupies an even-aligned
// pair, which gfx90a and later require and lower/regalloc cannot say — it
// hands out single identities — so each file is split statically: the low
// part of the file is singles, the high part is pairs, and a pair is one
// PhysReg whose halves emit addresses as base and base+1. Half of each
// file is wasted at O0, which is the honest cost of "only O0 exists".
//
// Every value lives in a VGPR except an i1, which is a lane mask in an
// SGPR pair, and the kernel's incoming arguments, which s_load into SGPRs
// at entry and are copied out. A branch whose condition is uniform tests
// the mask against zero and takes s_cbranch; one whose condition is not
// is refused by name until the exec-mask lowering lands.

import (
	"fmt"

	"github.com/vertex-language/amdgpu/reg"
	"github.com/vertex-language/ir"
	"github.com/vertex-language/ir/lower/regalloc"
)

// width is the register a value occupies.
type width uint8

const (
	v32 width = iota // a VGPR
	v64              // an aligned VGPR pair
	s32              // an SGPR
	s64              // an aligned SGPR pair; every i1 is one, as a lane mask
)

func (w width) class() regalloc.Class {
	switch w {
	case v64:
		return classV64
	case s32:
		return classS32
	case s64:
		return classS64
	}
	return regalloc.DefaultClass
}

func (w width) is64() bool { return w == v64 || w == s64 }

const (
	classV32 regalloc.Class = regalloc.DefaultClass
	classV64 regalloc.Class = 1
	classS32 regalloc.Class = 2
	classS64 regalloc.Class = 3
)

// widthOf is the VGPR width a VIR register type occupies. An i1 is a mask.
func widthOf(t ir.RegType) (width, bool) {
	switch t {
	case ir.TypeI1:
		return s64, true
	case ir.TypeI32, ir.TypeF32:
		return v32, true
	case ir.TypeI64, ir.TypePtr, ir.TypeF64:
		return v64, true
	}
	return 0, false
}

// The physical register numbering. A PhysReg is one identity across all
// four classes, so each file and width has its own range.
const (
	physS32 regalloc.PhysReg = 0    // + sgpr number
	physV32 regalloc.PhysReg = 1000 // + vgpr number
	physS64 regalloc.PhysReg = 2000 // + even sgpr number, the pair's base
	physV64 regalloc.PhysReg = 3000 // + even vgpr number, the pair's base
)

// The static split. gfx9 has 102 allocatable SGPRs (s0..s101) and 256
// VGPRs. Singles take the low part of each file and pairs the part above
// them, so that a small kernel's register count — which decides how many
// waves fit on a SIMD — stays small; the kernel's incoming SGPRs are
// pinned below the singles, and v0 holds the work-item id at entry.
const (
	sgprSinglesFrom, sgprSinglesTo = 8, 31   // s8..s31
	sgprPairsFrom, sgprPairsTo     = 32, 100 // s[32:33]..s[100:101]
	vgprSinglesFrom, vgprSinglesTo = 1, 56   // v1..v56; v57 a large scratch offset, v58 a callee's saved SGPRs, v59 spilled ones, v[60:63] casScratch
	vgprPairsFrom, vgprPairsTo     = 64, 254 // v[64:65]..v[254:255]
)

func physOf(w width, n int) regalloc.PhysReg {
	switch w {
	case v32:
		return physV32 + regalloc.PhysReg(n)
	case v64:
		return physV64 + regalloc.PhysReg(n)
	case s32:
		return physS32 + regalloc.PhysReg(n)
	}
	return physS64 + regalloc.PhysReg(n)
}

// regOf is the assembler's register for a PhysReg.
func regOf(p regalloc.PhysReg) reg.Reg {
	switch {
	case p >= physV64:
		return reg.V2(reg.VGPR(p - physV64))
	case p >= physS64:
		return reg.S2(reg.SGPR(p - physS64))
	case p >= physV32:
		return reg.VGPR(p - physV32)
	}
	return reg.SGPR(p)
}

// halfOf is one half of a pair register: the low VGPR or SGPR, or the high.
func halfOf(p regalloc.PhysReg, hi bool) reg.Reg {
	off := regalloc.PhysReg(0)
	if hi {
		off = 1
	}
	switch {
	case p >= physV64:
		return reg.VGPR(p - physV64 + off)
	case p >= physS64:
		return reg.SGPR(p - physS64 + off)
	}
	panic(fmt.Sprintf("amdgpu: %d is not a pair", p))
}

// pool is the four-class register pool. A function that calls, or is
// called, leaves the argument VGPRs and the convention's SGPRs alone;
// see call.go. extraSpill VGPRs at the top of the singles are left for
// spilled scalars past the reserved VGPR's lanes.
func pool(calling bool, extraSpill int, scratch bool) *regalloc.Pool {
	var v32s, v64s, s32s, s64s []regalloc.PhysReg
	vFrom, sTo, pFrom := vgprSinglesFrom, sgprSinglesTo, sgprPairsFrom
	if calling {
		vFrom, sTo, pFrom = argVGPRsFirst, retAddrSGPR-1, callPairsFrom
	}
	if scratch {
		// The first pair holds the private aperture.
		pFrom += 2
	}
	for n := vFrom; n <= vgprSinglesTo-extraSpill; n++ {
		v32s = append(v32s, physOf(v32, n))
	}
	for n := vgprPairsFrom; n <= vgprPairsTo; n += 2 {
		v64s = append(v64s, physOf(v64, n))
	}
	for n := sgprSinglesFrom; n <= sTo; n++ {
		s32s = append(s32s, physOf(s32, n))
	}
	for n := pFrom; n <= sgprPairsTo; n += 2 {
		s64s = append(s64s, physOf(s64, n))
	}
	p := regalloc.NewPool(v32s)
	p.AddClass(classV64, v64s)
	p.AddClass(classS32, s32s)
	p.AddClass(classS64, s64s)
	return p
}

// An opnd is one assembler operand of an amdOp, by where its value comes
// from.
type opnd struct {
	kind opndKind
	i    int   // Defs or Uses index
	imm  int64 // immediate, offset, or wait count
	sym  string

	// VOP3 source modifiers on a use.
	neg, abs bool

	// Cache bits on a memory operand or a cache-control instruction:
	// glc (sc0 on gfx940) and sc1.
	glc, sc1 bool
}

type opndKind uint8

const (
	oDef   opndKind = iota // Defs[i], whole register
	oUse                   // Uses[i], whole register
	oUseLo                 // the low half of the pair Uses[i]
	oUseHi                 // the high half
	oDefLo                 // the low half of the pair Defs[i]
	oDefHi
	oImm           // an inline constant or literal
	oFImm          // a float constant: imm holds the f32 bits
	oVCC           // the VCC register
	oExec          // the EXEC register
	oM0            // M0
	oFlat          // a flat address: [Uses[i]] + imm offset
	oGlobal        // a global address: [Uses[i] (VGPR pair)], off + imm offset
	oGlobalS       // a global address: [Uses[i] (VGPR) + Uses[i+1] (SGPR pair)] + imm
	oSMEM          // a scalar address: Uses[i] (SGPR pair) + imm offset
	oDS            // an LDS address: the low dword of the pair Uses[i], + imm offset
	oDS32          // an LDS address: the dword Uses[i] + imm offset
	oLabel         // a branch target, sym
	oWait          // s_waitcnt's counts
	oSymLo         // sym@rel32@lo+imm, a literal the linker fills
	oSymHi         // sym@rel32@hi+imm
	oSharedBase    // the LDS aperture, src_shared_base
	oPrivateBase   // the scratch aperture, src_private_base
	oFlatScratchLo // flat_scratch_lo
	oFlatScratchHi // flat_scratch_hi
	oFP            // s33, the frame pointer
	oCache         // a cache-control instruction's scope bits
	oFixedV        // a fixed VGPR, number imm; a tuple of i registers when i > 1
)

// amdOp is one machine instruction: a mnemonic and how to spell it.
type amdOp struct {
	mn   string
	ops  []opnd
	wait bool // a memory instruction: emit is followed by s_waitcnt
}

func (o amdOp) String() string { return o.mn }

// The few opcodes that are not one instruction: control flow that the
// cursor reads, and copies the allocator coalesces.
type (
	// movOp copies Uses[0] to Defs[0], a mir copy at width w. A masked
	// one is an s64 copy under the execution mask — read-modify-write,
	// Uses[1] the destination — which a structurized function needs for
	// every lane mask that crosses a region; see structurize.go.
	movOp struct {
		w      width
		masked bool
	}

	// branchOp is s_branch to a label.
	branchOp struct{ target string }

	// cbranchOp tests the lane mask Uses[0] against zero — s_and_b64
	// with exec into a scratch pair, s_cmp_lg_u64 — and takes then on
	// non-zero. The condition is uniform, so the mask is all or nothing.
	cbranchOp struct{ then, els string }

	// endpgmOp ends the wave.
	endpgmOp struct{}

	// trapOp is s_trap 2.
	trapOp struct{}

	// trapIfOp traps the wave when any active lane's bit in the mask
	// Uses[0] is set: s_and_b64 with exec, s_cbranch_scc1 to the one
	// s_trap at the end of the function.
	trapIfOp struct{}
)
