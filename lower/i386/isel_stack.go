package i386

import (
	"fmt"

	"github.com/vertex-language/ir"
	"github.com/vertex-language/ir/lower/mir"
)

// iselAlloca lowers §D3's dynamic allocation.
//
// The size is rounded to the stack's alignment before ESP moves: the psABI
// wants sixteen bytes at every call, and an odd allocation would break the
// outgoing arguments rather than only itself. The address is above the
// outgoing area, because a call writes its stack arguments from ESP and an
// allocation beginning there would be underneath them.
func iselAlloca(c *cursor, vr *vregs, fr *frame, in *ir.Inst) error {
	op := in.Op()
	if align, stated := in.Align(); stated && align > maxAlign {
		return fmt.Errorf("%s: wants %d-byte alignment; the stack guarantees %d", op, align, maxAlign)
	}
	n, ok := vr.lookup(in.Arg(0))
	if !ok {
		return fmt.Errorf("%s: size defined outside the function", op)
	}
	dst, err := vr.define(in.Result(0))
	if err != nil {
		return fmt.Errorf("%s: %w", op, err)
	}

	// The size arrives as an i64 and only its low half can be a size here:
	// a 32-bit stack pointer cannot move by more than a 32-bit amount.
	c.Emit(mir.Instr{
		Op:   allocaOp{outArgs: fr.outgoing()},
		Defs: []mir.VReg{dst.lo, vr.reg32()},
		Uses: []mir.VReg{n.lo},
	})

	if !in.Zeroed() {
		return nil
	}
	// The same memset a zeroed ptr.alloc gets, over a size that is a value
	// rather than a constant — and over the requested count rather than
	// the rounded one, which keeps the unrounded size live across the
	// subtraction that moved ESP.
	//
	// The call is safe after the allocation for the reason the allocation
	// was placed where it was: the outgoing area stays at the bottom of
	// the frame and ESP still names it.
	return emitMemsetZero(c, vr, fr, dst, value{lo: n.lo, w: w32})
}

// iselStackSave lowers §D3's stacksave: the opaque token that names where the
// stack was, which here is ESP itself.
func iselStackSave(c *cursor, vr *vregs, in *ir.Inst) error {
	dst, err := vr.define(in.Result(0))
	if err != nil {
		return fmt.Errorf("%s: %w", in.Op(), err)
	}
	c.Emit(mir.Instr{Op: stackSaveOp{}, Defs: []mir.VReg{dst.lo}})
	return nil
}

func iselStackRestore(c *cursor, vr *vregs, in *ir.Inst) error {
	src, ok := vr.lookup(in.Arg(0))
	if !ok {
		return fmt.Errorf("%s: token defined outside the function", in.Op())
	}
	c.Emit(mir.Instr{Op: stackRestoreOp{}, Uses: []mir.VReg{src.lo}})
	return nil
}

// iselFrameAddr lowers §D3's frameaddr and returnaddr, which are level zero
// only: this frame's own base, and the return address the call pushed just
// below it.
func iselFrameAddr(c *cursor, vr *vregs, in *ir.Inst, ret bool) error {
	dst, err := vr.define(in.Result(0))
	if err != nil {
		return fmt.Errorf("%s: %w", in.Op(), err)
	}
	if ret {
		c.Emit(mir.Instr{Op: frameLoadOp{off: 4}, Defs: []mir.VReg{dst.lo}})
		return nil
	}
	c.Emit(mir.Instr{Op: frameOp{off: 0}, Defs: []mir.VReg{dst.lo}})
	return nil
}

// iselBlockAddr lowers §D3's blockaddr.
//
// One instruction, and no relocation subtlety: a 32-bit address fits an
// immediate, and this object is not position-independent, so the label's
// address is an absolute reference the linker fills in.
func iselBlockAddr(c *cursor, vr *vregs, in *ir.Inst) error {
	labels := in.Labels()
	if len(labels) != 1 {
		return fmt.Errorf("%s: %d blocks named, want exactly one", in.Op(), len(labels))
	}
	dst, err := vr.define(in.Result(0))
	if err != nil {
		return fmt.Errorf("%s: %w", in.Op(), err)
	}
	blk := labels[0]
	c.Emit(mir.Instr{
		Op:   blockAddrOp{label: blockLabel(blk.Func(), blk)},
		Defs: []mir.VReg{dst.lo},
	})
	return nil
}
