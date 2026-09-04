package arm64

import (
	"fmt"

	"github.com/vertex-language/arm64/reg"

	"github.com/vertex-language/ir"
	"github.com/vertex-language/ir/lower/mir"
	"github.com/vertex-language/ir/lower/regalloc"
)

// AAPCS64's argument registers: the first eight of each file, and the same
// eight are where results come back.
var (
	aapcsIntArgs   = []reg.X{reg.X0, reg.X1, reg.X2, reg.X3, reg.X4, reg.X5, reg.X6, reg.X7}
	aapcsFloatArgs = []reg.V{reg.V0, reg.V1, reg.V2, reg.V3, reg.V4, reg.V5, reg.V6, reg.V7}
)

// callerSaved is every general-purpose register a call may destroy.
//
// X8 is the indirect result register and X16 through X18 are reserved — IP0
// and IP1 belong to the linker's veneers and X18 is the platform register,
// which some platforms use and none of them promise not to. A register this
// package must not allocate is simply absent from the pool rather than being
// listed and filtered.
var callerSaved = []reg.X{
	reg.X0, reg.X1, reg.X2, reg.X3, reg.X4, reg.X5, reg.X6, reg.X7,
	reg.X9, reg.X10, reg.X11, reg.X12, reg.X13, reg.X14, reg.X15,
}

// calleeSaved is the general-purpose registers a call preserves. X29 and X30
// are the frame pointer and the link register: the prologue saves both, and
// neither is allocatable.
var calleeSaved = []reg.X{
	reg.X19, reg.X20, reg.X21, reg.X22, reg.X23,
	reg.X24, reg.X25, reg.X26, reg.X27, reg.X28,
}

// allocatable is every general-purpose register this package hands out, in the
// order it prefers them.
func allocatable() []regalloc.PhysReg {
	out := make([]regalloc.PhysReg, 0, len(callerSaved)+len(calleeSaved))
	for _, r := range callerSaved {
		out = append(out, regalloc.PhysReg(r))
	}
	for _, r := range calleeSaved {
		out = append(out, regalloc.PhysReg(r))
	}
	return out
}

// allocatableVec is every vector register this package hands out.
//
// V8 through V15 are callee-saved in their low 64 bits only, which is a
// promise this package cannot keep for an f64 and does not try to: they are
// allocated like any other and the prologue saves the ones used, at the width
// the ABI preserves.
func allocatableVec() []regalloc.PhysReg {
	out := make([]regalloc.PhysReg, 0, 32)
	for r := reg.V0; r <= reg.V31; r++ {
		out = append(out, regalloc.PhysReg(r))
	}
	return out
}

// calleeSavedVec is the vector registers a call preserves, low half only.
var calleeSavedVec = []reg.V{
	reg.V8, reg.V9, reg.V10, reg.V11, reg.V12, reg.V13, reg.V14, reg.V15,
}

// callerSavedVec is every vector register a call may destroy: the other
// twenty-four.
var callerSavedVec = func() []reg.V {
	var out []reg.V
	for r := reg.V0; r <= reg.V31; r++ {
		if r < reg.V8 || r > reg.V15 {
			out = append(out, r)
		}
	}
	return out
}()

// vregs is one function's value environment: which vreg holds each ir.Def isel
// has seen, and how wide that vreg is.
type vregs struct {
	mf   *mir.Func
	pool *regalloc.Pool

	of    map[*ir.Def]mir.VReg
	width map[mir.VReg]width

	// asmIDs numbers the inline asm statements in this function, so two
	// expansions of one template generate different local labels.
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

func (v *vregs) fresh(w width) mir.VReg {
	r := v.mf.NewVReg()
	v.width[r] = w
	v.pool.Classify(r, w.class())
	return r
}

// temp allocates a fresh vreg belonging to no ir.Def.
func (v *vregs) temp(w width) mir.VReg { return v.fresh(w) }

// physical allocates a fresh vreg that must live in p, which is how an
// instruction names a fixed register in a MIR.
func (v *vregs) physical(p reg.X, w width) mir.VReg {
	r := v.fresh(w)
	v.pool.Pin(r, regalloc.PhysReg(p))
	return r
}

// physicalVec is physical for the other register file.
func (v *vregs) physicalVec(p reg.V, w width) mir.VReg {
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

// nextAsmID hands out a number unique within this function, so that two
// expansions of one inline asm template generate different local labels.
func (v *vregs) nextAsmID() int {
	v.asmIDs++
	return v.asmIDs - 1
}

// spiller is what regalloc calls when it runs out of registers.
type spiller struct{ fr *frame }

func (s *spiller) Slot() int { return int(s.fr.reserve()) }

func (s *spiller) Store(slot int, v mir.VReg, c regalloc.Class) mir.Instr {
	return mir.Instr{Op: spillOp{off: int64(slot), float: c == vecClass}, Uses: []mir.VReg{v}}
}

func (s *spiller) Load(slot int, v mir.VReg, c regalloc.Class) mir.Instr {
	return mir.Instr{Op: reloadOp{off: int64(slot), float: c == vecClass}, Defs: []mir.VReg{v}}
}
