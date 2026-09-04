package i386

// §I, which on this target is nearly nothing.
//
// The Intel386 psABI passes every argument on the stack, named or variadic
// alike, so a va_list is a pointer at the next one and va_arg is a load and
// an increment. There is no register save area to write, no two-region walk,
// and no second convention to choose between — the three things that make §I
// substantial on the other two targets are all consequences of having
// register arguments, which this architecture does not.

import (
	"fmt"

	"github.com/vertex-language/ir"
	"github.com/vertex-language/ir/lower/mir"
)

// iselVaStart lowers §I's va_start: the address of the first variadic slot,
// stored into the list.
func iselVaStart(c *cursor, vr *vregs, fr *frame, in *ir.Inst) error {
	op := in.Op()
	if !fr.variadic {
		return fmt.Errorf("%s: this function is not variadic, so it has no list to start", op)
	}
	ap, ok := vr.lookup(in.Arg(0))
	if !ok {
		return fmt.Errorf("%s: the list is defined outside the function", op)
	}
	first := vr.reg32()
	c.Emit(mir.Instr{
		Op:   frameOp{off: stackParamOff(fr.vaOffset)},
		Defs: []mir.VReg{first},
	})
	c.Emit(mir.Instr{Op: storeOp{}, Uses: []mir.VReg{first, ap.lo}})
	return nil
}

// iselVaEnd lowers §I's va_end, which has nothing to undo: the list is a
// pointer and nothing was allocated to make it.
func iselVaEnd(*ir.Inst) error { return nil }

// iselVaCopy lowers §I's va_copy: one pointer copied through memory.
func iselVaCopy(c *cursor, vr *vregs, in *ir.Inst) error {
	ops, err := operands(vr, in, 2)
	if err != nil {
		return err
	}
	v := vr.reg32()
	c.Emit(mir.Instr{Op: loadOp{}, Defs: []mir.VReg{v}, Uses: []mir.VReg{ops[1].lo}})
	c.Emit(mir.Instr{Op: storeOp{}, Uses: []mir.VReg{v, ops[0].lo}})
	return nil
}

// iselVaArg lowers §I's va_arg: the value at the list's cursor, and the
// cursor moved past it.
//
// Past its own width, not a fixed slot: an i64 argument occupies two slots
// here where every other type occupies one, which is the same rule the call
// side places arguments by.
func iselVaArg(c *cursor, vr *vregs, in *ir.Inst) error {
	op := in.Op()
	ap, ok := vr.lookup(in.Arg(0))
	if !ok {
		return fmt.Errorf("%s: the list is defined outside the function", op)
	}
	dst, err := vr.define(in.Result(0))
	if err != nil {
		return fmt.Errorf("%s: %w", op, err)
	}

	cur := vr.reg32()
	c.Emit(mir.Instr{Op: loadOp{}, Defs: []mir.VReg{cur}, Uses: []mir.VReg{ap.lo}})
	c.Emit(mir.Instr{Op: loadOp{}, Defs: []mir.VReg{dst.lo}, Uses: []mir.VReg{cur}})
	step := int64(4)
	if dst.w.pairs() {
		c.Emit(mir.Instr{Op: loadOp{off: 4}, Defs: []mir.VReg{dst.hi}, Uses: []mir.VReg{cur}})
		step = 8
	}
	return vaAdvance(c, vr, ap.lo, cur, step)
}

// iselVaArgRef lowers §I's va_arg_ref: the address of the next argument.
//
// An aggregate is passed by value on this psABI whatever its size — there is
// no by-reference rule to tell apart, which is the other thing having no
// register arguments simplifies — so the address is the slot itself and the
// cursor moves past as many slots as the type fills.
func iselVaArgRef(c *cursor, vr *vregs, in *ir.Inst) error {
	op := in.Op()
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

	cur := vr.reg32()
	c.Emit(mir.Instr{Op: loadOp{}, Defs: []mir.VReg{cur}, Uses: []mir.VReg{ap.lo}})
	emitCopy(c, dst.lo, cur)
	return vaAdvance(c, vr, ap.lo, cur, int64(alignUp(size, 4)))
}

// vaAdvance moves the list on by n bytes and writes it back.
func vaAdvance(c *cursor, vr *vregs, ap, cur mir.VReg, n int64) error {
	next := vr.reg32()
	emitCopy(c, next, cur)
	c.Emit(mir.Instr{Op: addImmOp{imm: n}, Defs: []mir.VReg{next}, Uses: []mir.VReg{next}})
	c.Emit(mir.Instr{Op: storeOp{}, Uses: []mir.VReg{next, ap}})
	return nil
}
