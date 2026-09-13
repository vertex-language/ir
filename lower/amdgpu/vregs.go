package amdgpu

import (
	"fmt"

	"github.com/vertex-language/ir"
	"github.com/vertex-language/ir/lower/mir"
	"github.com/vertex-language/ir/lower/regalloc"
)

// vregs is one function's value environment: which vreg holds each ir.Def
// isel has seen, and how wide that vreg is.
type vregs struct {
	mf   *mir.Func
	pool *regalloc.Pool

	of    map[*ir.Def]mir.VReg
	width map[mir.VReg]width
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
		return 0, fmt.Errorf("%s is not a value this package holds in a register", d.Type())
	}
	r := v.fresh(w)
	v.of[d] = r
	return r, nil
}

// fresh is a new vreg of width w, classified for the allocator.
func (v *vregs) fresh(w width) mir.VReg {
	r := v.mf.NewVReg()
	v.width[r] = w
	if c := w.class(); c != regalloc.DefaultClass {
		v.pool.Classify(r, c)
	}
	return r
}

// temp is fresh, by another name: a value isel made up.
func (v *vregs) temp(w width) mir.VReg { return v.fresh(w) }

// lookup is the vreg holding d.
func (v *vregs) lookup(d *ir.Def) (mir.VReg, bool) {
	r, ok := v.of[d]
	return r, ok
}

// use is lookup as isel wants it: a vreg or an error naming the operand.
func (v *vregs) use(d *ir.Def) (mir.VReg, error) {
	if r, ok := v.of[d]; ok {
		return r, nil
	}
	return 0, fmt.Errorf("%%%s has no register; defined after its use or outside the function", d)
}

func (v *vregs) widthOfVReg(r mir.VReg) width { return v.width[r] }

// pin binds a vreg to a physical register: an incoming argument.
func (v *vregs) pin(r mir.VReg, p regalloc.PhysReg) { v.pool.Pin(r, p) }
