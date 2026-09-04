package amd64

import (
	"fmt"

	"github.com/vertex-language/amd64/reg"

	"github.com/vertex-language/ir"
	"github.com/vertex-language/ir/lower/mir"
	"github.com/vertex-language/ir/lower/regalloc"
)

// sysvIntArgs is the SysV AMD64 integer/pointer argument registers, first
// six, in order. A seventh argument goes on the stack.
var sysvIntArgs = []reg.R64{reg.RDI, reg.RSI, reg.RDX, reg.RCX, reg.R8Q, reg.R9Q}

// sysvFloatArgs is the SysV AMD64 SSE argument registers, first eight, in
// order. A float parameter past the eighth goes on the stack.
var sysvFloatArgs = []reg.Xmm{
	reg.XMM0, reg.XMM1, reg.XMM2, reg.XMM3,
	reg.XMM4, reg.XMM5, reg.XMM6, reg.XMM7,
}

// sysvIntRets is the SysV AMD64 integer return registers, in order. Note
// that RDX is also the third argument register and RAX is the variadic
// vector count: a call can name one register for two reasons, which is
// what callSite exists to keep straight.
var sysvIntRets = []reg.R64{reg.RAX, reg.RDX}

// sysvFloatRets is the SysV AMD64 SSE return registers, in order. Both of
// them are argument registers too.
var sysvFloatRets = []reg.Xmm{reg.XMM0, reg.XMM1}

// calleeSaved is every SysV callee-saved general-purpose register this
// package will allocate.
var calleeSaved = []reg.R64{reg.RBX, reg.R12Q, reg.R13Q, reg.R14Q, reg.R15Q}

// callerSaved is every SysV caller-saved general-purpose register, in
// the order scratchPool hands them out.
var callerSaved = []reg.R64{
	reg.RAX, reg.RCX, reg.RDX, reg.RSI, reg.RDI,
	reg.R8Q, reg.R9Q, reg.R10Q, reg.R11Q,
}

// allocatableXmm is every vector register this package will hand out.
//
// Under SysV that is all sixteen, none of them callee-saved. Under the
// Microsoft ABI the upper ten are callee-saved and this package has no
// machinery to save a vector register — usedCalleeSaved walks the
// general-purpose file only — so it hands out the six that need none.
func allocatableXmm(abi string) []regalloc.PhysReg {
	a := regsFor(abi)
	out := make([]regalloc.PhysReg, 0, len(a.xmm))
	for _, r := range a.xmm {
		out = append(out, regalloc.PhysReg(r))
	}
	return out
}

// allocatable is every register this package will hand out, in the order
// it prefers them: the caller-saved ones first, because a function that
// never calls anything then never touches a register it has to put back.
func allocatable(abi string) []regalloc.PhysReg {
	a := regsFor(abi)
	out := make([]regalloc.PhysReg, 0, len(a.callerSaved)+len(a.calleeSaved))
	for _, r := range a.callerSaved {
		out = append(out, regalloc.PhysReg(r))
	}
	for _, r := range a.calleeSaved {
		out = append(out, regalloc.PhysReg(r))
	}
	return out
}

// vregs is one function's value environment: which vreg holds each ir.Def
// isel has seen, and how wide that vreg is.
type vregs struct {
	mf   *mir.Func
	pool *regalloc.Pool

	of    map[*ir.Def]mir.VReg
	width map[mir.VReg]width

	// asmIDs numbers the inline asm statements in this function.
	asmIDs int
}

func newVRegs(mf *mir.Func, pool *regalloc.Pool, n int) *vregs {
	return &vregs{
		mf:    mf,
		pool:  pool,
		of:    make(map[*ir.Def]mir.VReg, n),
		width: make(map[mir.VReg]width, n),
	}
}

// define allocates a fresh vreg for d at the width d's type occupies.
func (v *vregs) define(d *ir.Def) (mir.VReg, error) {
	w, ok := widthOf(d.Type())
	if !ok {
		return 0, fmt.Errorf("%s is not a value this package holds in a register; only i1, i32, i64, ptr, f32, and f64 are supported", d.Type())
	}
	r := v.fresh(w)
	v.of[d] = r
	return r, nil
}

// fresh makes a vreg of the given width and tells the pool which register file that width lives in.
func (v *vregs) fresh(w width) mir.VReg {
	r := v.mf.NewVReg()
	v.width[r] = w
	v.pool.Classify(r, w.class())
	return r
}

// temp allocates a fresh vreg belonging to no ir.Def.
func (v *vregs) temp(w width) mir.VReg { return v.fresh(w) }

// physical allocates a fresh vreg that must live in p.
// This is how an instruction names a fixed register in a MIR.
func (v *vregs) physical(p reg.R64, w width) mir.VReg {
	r := v.fresh(w)
	v.pool.Pin(r, regalloc.PhysReg(p))
	return r
}

// physicalXmm is physical for the other register file.
func (v *vregs) physicalXmm(p reg.Xmm, w width) mir.VReg {
	r := v.fresh(w)
	v.pool.Pin(r, regalloc.PhysReg(p))
	return r
}

// lookup is the vreg holding d, and whether isel has defined one.
func (v *vregs) lookup(d *ir.Def) (mir.VReg, bool) {
	r, ok := v.of[d]
	return r, ok
}

// widthOfVReg is r's width.
func (v *vregs) widthOfVReg(r mir.VReg) width { return v.width[r] }

// nextAsmID hands out a number unique within this function, so two expansions
// of one inline asm template generate different local labels.
func (v *vregs) nextAsmID() int {
	v.asmIDs++
	return v.asmIDs - 1
}

// spiller is what regalloc calls when it runs out of registers: the two
// instructions it cannot write for itself, and the frame storage they name.
type spiller struct{ fr *frame }

// Slot reserves storage wide enough for the widest register this target
// allocates, which is regalloc.Spiller's contract — that package tracks
// no widths, so neither side can ask the other. Eight bytes until an
// f128 is in the function and then sixteen, aligned for MOVAPS: paying
// everywhere for a type most functions lack would be the wrong way
// round.
func (s *spiller) Slot() int {
	if !s.fr.wideSpills {
		return int(s.fr.reserve())
	}
	off, err := s.fr.reserveBytes(16, 16)
	if err != nil {
		// reserveBytes fails only past 2GB of frame, which planFrame
		// has already refused; returning the unaligned slot here would
		// be worse than the offset this cannot reach.
		return int(s.fr.reserve())
	}
	return int(off)
}

// Store carries the class because a store out of an integer register and
// one out of a vector register are different instructions.
func (s *spiller) Store(slot int, v mir.VReg, c regalloc.Class) mir.Instr {
	return mir.Instr{Op: spillOp{off: int32(slot), w: s.spillWidth(c)}, Uses: []mir.VReg{v}}
}

// Load is Store's other half.
func (s *spiller) Load(slot int, v mir.VReg, c regalloc.Class) mir.Instr {
	return mir.Instr{Op: reloadOp{off: int32(slot), w: s.spillWidth(c)}, Defs: []mir.VReg{v}}
}

// spillWidth is how much of a register one spill moves. The class is all
// regalloc knows, so an f64 in a function that also holds an f128 moves
// sixteen bytes too — eight of them unread, which is cheaper than
// tracking a width through a package whose point is not to.
func (s *spiller) spillWidth(c regalloc.Class) width {
	switch {
	case c != xmmClass:
		return w64
	case s.fr.wideSpills:
		return wv128
	}
	return wf64
}
