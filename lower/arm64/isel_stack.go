package arm64

import (
	"fmt"

	"github.com/vertex-language/ir"
	"github.com/vertex-language/ir/lower/mir"
)

// iselAlloca lowers §D3's dynamic allocation.
//
// The size is rounded up to the stack's alignment and subtracted from SP. It
// has to be rounded: AAPCS64 requires SP to hold a sixteen-byte alignment at
// every instruction that uses it as a base, which is every access to a local
// and every outgoing argument, so an odd allocation would break the rest of
// the frame rather than only itself.
//
// The address is above the outgoing argument area rather than at SP. A call
// writes its stack arguments from SP, so an allocation that began there would
// be underneath the next call's arguments.
func iselAlloca(c *cursor, vr *vregs, fr *frame, in *ir.Inst, opts Options) error {
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

	c.Emit(mir.Instr{
		Op:   allocaOp{outArgs: fr.outgoing()},
		Defs: []mir.VReg{dst, vr.temp(w64)},
		Uses: []mir.VReg{n},
	})

	if !in.Zeroed() {
		return nil
	}
	// A memset over the size that was asked for, not the rounded one:
	// what §D3 guarantees reads as zero is the allocation, and the bytes
	// the rounding took are padding outside it.
	zero := vr.temp(w32)
	c.Emit(mir.Instr{Op: constOp{imm: 0, w: w32}, Defs: []mir.VReg{zero}})
	return emitLibcall(c, vr, memsetSym, opts, []mir.VReg{dst, zero, n})
}

// iselStackSave lowers §D3's stacksave: the opaque token that names where the
// stack was, which here is SP itself.
func iselStackSave(c *cursor, vr *vregs, in *ir.Inst) error {
	dst, err := vr.define(in.Result(0))
	if err != nil {
		return fmt.Errorf("%s: %w", in.Op(), err)
	}
	c.Emit(mir.Instr{Op: stackSaveOp{}, Defs: []mir.VReg{dst}})
	return nil
}

// iselStackRestore puts it back.
func iselStackRestore(c *cursor, vr *vregs, in *ir.Inst) error {
	src, ok := vr.lookup(in.Arg(0))
	if !ok {
		return fmt.Errorf("%s: token defined outside the function", in.Op())
	}
	c.Emit(mir.Instr{Op: stackRestoreOp{}, Uses: []mir.VReg{src}})
	return nil
}
