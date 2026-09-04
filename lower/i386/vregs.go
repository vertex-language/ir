package i386

import (
	"fmt"

	"github.com/vertex-language/i386/reg"

	"github.com/vertex-language/ir"
	"github.com/vertex-language/ir/lower/mir"
	"github.com/vertex-language/ir/lower/regalloc"
)

// callerSaved is every register a call may destroy, and calleeSaved the rest.
//
// EBP is the frame pointer and ESP the stack pointer; neither is allocatable,
// which is what leaves six.
var (
	callerSaved = []reg.R32{reg.EAX, reg.ECX, reg.EDX}
	calleeSaved = []reg.R32{reg.EBX, reg.ESI, reg.EDI}
)

// allocatable is every register this package hands out, in the order it
// prefers them.
func allocatable() []regalloc.PhysReg {
	out := make([]regalloc.PhysReg, 0, 6)
	for _, r := range callerSaved {
		out = append(out, regalloc.PhysReg(r))
	}
	for _, r := range calleeSaved {
		out = append(out, regalloc.PhysReg(r))
	}
	return out
}

// allocatableVec is the XMM registers, all eight of which a call destroys.
//
// The psABI has no callee-saved vector register, which is one thing this
// target makes simpler than the other two: there is no save area to plan and
// no half-preserved register to reason about.
func allocatableVec() []regalloc.PhysReg {
	out := make([]regalloc.PhysReg, 0, 8)
	for r := reg.XMM0; r <= reg.XMM7; r++ {
		out = append(out, regalloc.PhysReg(r))
	}
	return out
}

// A value is where one ir.Def lives.
//
// One register, or two for an i64: lo is the low thirty-two bits and hi the
// high thirty-two. This is the shape the whole package is arranged around,
// and the reason isel here asks for halves where the other backends ask for
// operands.
type value struct {
	lo, hi mir.VReg
	w      width
}

// vregs is one function's value environment.
type vregs struct {
	mf   *mir.Func
	pool *regalloc.Pool

	of map[*ir.Def]value

	// asmIDs numbers the inline asm statements in this function, so two
	// expansions of one template generate different local labels.
	asmIDs int
}

// nextAsmID hands out a number unique within this function.
func (v *vregs) nextAsmID() int {
	v.asmIDs++
	return v.asmIDs - 1
}

func newVRegs(mf *mir.Func, pool *regalloc.Pool, n int) *vregs {
	return &vregs{mf: mf, pool: pool, of: make(map[*ir.Def]value, n)}
}

// define allocates the registers d's type occupies.
func (v *vregs) define(d *ir.Def) (value, error) {
	w, ok := widthOf(d.Type())
	if !ok {
		return value{}, fmt.Errorf("%s is not a value this package holds in registers; only i1, i32, i64, ptr, f32 and f64 are, f80 and f128 being types this target does not have", d.Type())
	}
	val := v.fresh(w)
	v.of[d] = val
	return val, nil
}

func (v *vregs) fresh(w width) value {
	val := value{lo: v.mf.NewVReg(), w: w}
	v.pool.Classify(val.lo, w.class())
	if w.pairs() {
		val.hi = v.mf.NewVReg()
		v.pool.Classify(val.hi, regalloc.DefaultClass)
	}
	return val
}

// vec allocates one bare XMM vreg.
func (v *vregs) vec() mir.VReg {
	r := v.mf.NewVReg()
	v.pool.Classify(r, vecClass)
	return r
}

// physicalVec allocates a fresh vreg pinned to an XMM register.
func (v *vregs) physicalVec(p reg.Xmm) mir.VReg {
	r := v.vec()
	v.pool.Pin(r, regalloc.PhysReg(p))
	return r
}

// temp allocates a fresh value belonging to no ir.Def.
func (v *vregs) temp(w width) value { return v.fresh(w) }

// reg32 allocates one bare register, for a scratch that is never half of a pair.
func (v *vregs) reg32() mir.VReg {
	r := v.mf.NewVReg()
	v.pool.Classify(r, regalloc.DefaultClass)
	return r
}

// physical allocates a fresh vreg pinned to p.
func (v *vregs) physical(p reg.R32) mir.VReg {
	r := v.reg32()
	v.pool.Pin(r, regalloc.PhysReg(p))
	return r
}

// lookup is the value holding d, and whether isel has defined one.
func (v *vregs) lookup(d *ir.Def) (value, bool) {
	val, ok := v.of[d]
	return val, ok
}

// spiller is what regalloc calls when it runs out of registers.
//
// It runs out often here: six registers, and an i64 takes two of them.
type spiller struct{ fr *frame }

// Slot takes eight bytes rather than four. A general register needs four and
// a float needs eight, and Slot is not told which is asking — so it hands out
// the larger, which costs a word of frame per spill and cannot be too small.
func (s *spiller) Slot() int { return int(s.fr.reserve8()) }

func (s *spiller) Store(slot int, v mir.VReg, c regalloc.Class) mir.Instr {
	if c == vecClass {
		// A spilled float goes out at its full eight bytes whatever it
		// is: the slot is the allocator's and nothing reads it but the
		// matching reload, so the wide store costs a byte of encoding
		// and saves knowing the width here.
		return mir.Instr{Op: fspillOp{off: int32(slot)}, Uses: []mir.VReg{v}}
	}
	return mir.Instr{Op: spillOp{off: int32(slot)}, Uses: []mir.VReg{v}}
}

func (s *spiller) Load(slot int, v mir.VReg, c regalloc.Class) mir.Instr {
	if c == vecClass {
		return mir.Instr{Op: freloadOp{off: int32(slot)}, Defs: []mir.VReg{v}}
	}
	return mir.Instr{Op: reloadOp{off: int32(slot)}, Defs: []mir.VReg{v}}
}

// usedCalleeSaved is the callee-saved registers this allocation named.
//
// The vector file is not consulted: the psABI preserves no XMM register, so
// there is nothing there to give back.
func usedCalleeSaved(pool *regalloc.Pool, assigned map[mir.VReg]regalloc.PhysReg) []reg.R32 {
	used := map[reg.R32]bool{}
	for v, p := range assigned {
		if pool.ClassOf(v) != regalloc.DefaultClass {
			continue
		}
		used[reg.R32(p)] = true
	}
	var out []reg.R32
	for _, r := range calleeSaved {
		if used[r] {
			out = append(out, r)
		}
	}
	return out
}
