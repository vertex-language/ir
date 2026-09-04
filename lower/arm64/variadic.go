package arm64

// §I, under Apple's variadic variant.
//
// Every variadic argument occupies one eight-byte stack slot, in order, and
// va_list is a plain pointer at the next one. That is the whole convention:
// there is no register save area, no two-region walk, and no counting — which
// is the difference from the base standard, and from the other architecture.
//
// An eight-byte slot whatever the type. A four-byte value sits in the low half
// of its slot and a little-endian load of four bytes finds it there, which is
// why va_arg reads at the result's own width and not at the slot's.

import (
	"fmt"

	"github.com/vertex-language/ir"
	"github.com/vertex-language/ir/lower/mir"
)

// vaSlot is the stack a single variadic argument takes.
const vaSlot = 8

// checkVariadicABI refuses the convention this package has not written.
func checkVariadicABI(op ir.Op, opts Options) error {
	if opts.Variadic != VariadicDarwin {
		return fmt.Errorf("%s: Options.Variadic names the base standard's convention, whose register save area and two-region walk are not written", op)
	}
	return nil
}

// iselVaStart lowers §I's va_start: the address of the first variadic slot,
// stored into the list.
func iselVaStart(c *cursor, vr *vregs, fr *frame, in *ir.Inst, opts Options) error {
	op := in.Op()
	if err := checkVariadicABI(op, opts); err != nil {
		return err
	}
	if !fr.variadic {
		return fmt.Errorf("%s: this function is not variadic, so it has no list to start", op)
	}
	ap, ok := vr.lookup(in.Arg(0))
	if !ok {
		return fmt.Errorf("%s: the list is defined outside the function", op)
	}

	// Where the tail begins: after whatever this function's own named
	// parameters left on the caller's outgoing area.
	first := vr.temp(w64)
	c.Emit(mir.Instr{
		Op:   frameOp{off: stackParamOff(fr.vaOffset)},
		Defs: []mir.VReg{first},
	})
	c.Emit(mir.Instr{Op: storeOp{w: w64}, Uses: []mir.VReg{first, ap}})
	return nil
}

// iselVaEnd lowers §I's va_end, which under this convention has nothing to
// undo: the list is a pointer and nothing was allocated to make it.
func iselVaEnd(in *ir.Inst, opts Options) error {
	return checkVariadicABI(in.Op(), opts)
}

// iselVaCopy lowers §I's va_copy: one pointer copied through memory.
func iselVaCopy(c *cursor, vr *vregs, in *ir.Inst, opts Options) error {
	op := in.Op()
	if err := checkVariadicABI(op, opts); err != nil {
		return err
	}
	ops, err := operands(vr, in, 2)
	if err != nil {
		return err
	}
	v := vr.temp(w64)
	c.Emit(mir.Instr{Op: loadOp{w: w64}, Defs: []mir.VReg{v}, Uses: []mir.VReg{ops[1]}})
	c.Emit(mir.Instr{Op: storeOp{w: w64}, Uses: []mir.VReg{v, ops[0]}})
	return nil
}

// iselVaArg lowers §I's va_arg: the value at the list's cursor, and the
// cursor moved past its slot.
func iselVaArg(c *cursor, vr *vregs, in *ir.Inst, opts Options) error {
	op := in.Op()
	if err := checkVariadicABI(op, opts); err != nil {
		return err
	}
	ap, ok := vr.lookup(in.Arg(0))
	if !ok {
		return fmt.Errorf("%s: the list is defined outside the function", op)
	}
	dst, err := vr.define(in.Result(0))
	if err != nil {
		return fmt.Errorf("%s: %w", op, err)
	}
	cur := vaCursor(c, vr, ap)
	c.Emit(mir.Instr{
		Op:   loadOp{w: vr.widthOfVReg(dst)},
		Defs: []mir.VReg{dst}, Uses: []mir.VReg{cur},
	})
	vaAdvance(c, vr, ap, cur, vaSlot)
	return nil
}

// iselVaArgRef lowers §I's va_arg_ref: the address of the next argument,
// which is how va_arg of an aggregate is written.
//
// Where that address is depends on how the aggregate was passed. AAPCS64 puts
// an aggregate of sixteen bytes or less on the stack by value, so its address
// is the slot itself and the cursor moves past as many slots as it filled; a
// larger one is passed as a pointer in one slot, so the address is what that
// slot holds and the cursor moves past the single pointer.
func iselVaArgRef(c *cursor, vr *vregs, in *ir.Inst, opts Options) error {
	op := in.Op()
	if err := checkVariadicABI(op, opts); err != nil {
		return err
	}
	t := in.NamedType()
	if t == nil {
		return fmt.Errorf("%s: no type named", op)
	}
	size, _, err := sizeAlign(t.FType())
	if err != nil {
		return fmt.Errorf("%s: %w", op, err)
	}
	ap, ok := vr.lookup(in.Arg(0))
	if !ok {
		return fmt.Errorf("%s: the list is defined outside the function", op)
	}
	dst, err := vr.define(in.Result(0))
	if err != nil {
		return fmt.Errorf("%s: %w", op, err)
	}

	cur := vaCursor(c, vr, ap)
	if size > 16 {
		// Indirect: the slot holds the address.
		c.Emit(mir.Instr{Op: loadOp{w: w64}, Defs: []mir.VReg{dst}, Uses: []mir.VReg{cur}})
		vaAdvance(c, vr, ap, cur, vaSlot)
		return nil
	}
	emitCopy(c, dst, cur, w64)
	vaAdvance(c, vr, ap, cur, int64(alignUp(size, vaSlot)))
	return nil
}

// vaCursor loads the list's current position.
func vaCursor(c *cursor, vr *vregs, ap mir.VReg) mir.VReg {
	cur := vr.temp(w64)
	c.Emit(mir.Instr{Op: loadOp{w: w64}, Defs: []mir.VReg{cur}, Uses: []mir.VReg{ap}})
	return cur
}

// vaAdvance moves it on by n bytes and writes it back.
func vaAdvance(c *cursor, vr *vregs, ap, cur mir.VReg, n int64) {
	next := vr.temp(w64)
	c.Emit(mir.Instr{Op: addImmOp{imm: n}, Defs: []mir.VReg{next}, Uses: []mir.VReg{cur}})
	c.Emit(mir.Instr{Op: storeOp{w: w64}, Uses: []mir.VReg{next, ap}})
}
