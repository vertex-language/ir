package i386

import (
	"fmt"

	"github.com/vertex-language/i386/reg"

	"github.com/vertex-language/ir"
	"github.com/vertex-language/ir/lower/mir"
)

// iselDivide lowers §A's four division verbs.
//
// The dividend is EDX:EAX and the quotient and remainder come back in EAX and
// EDX, which is why both are pinned. §A says a zero divisor traps and that a
// signed division traps where its quotient does not fit; x86 raises #DE for
// both, so there is no check to emit here — the trap is the instruction's.
// That is the reverse of AArch64, where the same two cases are silent and the
// backend has to guard them.
//
// At sixty-four bits there is no instruction at all: the division becomes a
// call to the same helper a C compiler would call.
func iselDivide(c *cursor, vr *vregs, fr *frame, in *ir.Inst, signed, wantRem bool) error {
	op := in.Op()
	ops, err := operands(vr, in, 2)
	if err != nil {
		return err
	}
	dst, err := vr.define(in.Result(0))
	if err != nil {
		return fmt.Errorf("%s: %w", op, err)
	}

	if dst.w.pairs() {
		return emitLibcall(c, vr, fr, divHelper(signed, wantRem), []value{ops[0], ops[1]}, dst, true)
	}

	// The divisor is copied rather than read where it lies: it may already
	// be in EAX or EDX, which are the two registers this instruction
	// overwrites. The copy gives the allocator a vreg it is free to place
	// elsewhere, and the interference graph keeps it out of both since it
	// is an operand of an instruction that defines them.
	div := vr.reg32()
	emitCopy(c, div, ops[1].lo)

	eax := vr.physical(reg.EAX)
	edx := vr.physical(reg.EDX)
	emitCopy(c, eax, ops[0].lo)
	c.Emit(mir.Instr{Op: widenOp{signed: signed}, Defs: []mir.VReg{edx}, Uses: []mir.VReg{eax}})
	c.Emit(mir.Instr{
		Op:   divOp{signed: signed},
		Defs: []mir.VReg{eax, edx},
		Uses: []mir.VReg{eax, edx, div},
	})

	src := eax
	if wantRem {
		src = edx
	}
	emitCopy(c, dst.lo, src)
	return nil
}

// divHelper is the C library's name for a 64-bit division. These are
// compiler-rt's and libgcc's alike, which is what makes them the names to
// call rather than something this package invents.
func divHelper(signed, rem bool) string {
	switch {
	case signed && rem:
		return "__moddi3"
	case signed:
		return "__divdi3"
	case rem:
		return "__umoddi3"
	}
	return "__udivdi3"
}

// iselMulHi lowers §A's smulhi and umulhi: the half of the product the
// ordinary multiply discards, which the one-operand form leaves in EDX.
func iselMulHi(c *cursor, vr *vregs, in *ir.Inst, signed bool) error {
	ops, err := operands(vr, in, 2)
	if err != nil {
		return err
	}
	dst, err := vr.define(in.Result(0))
	if err != nil {
		return fmt.Errorf("%s: %w", in.Op(), err)
	}
	if dst.w.pairs() {
		return iselMulHi64(c, vr, dst, ops[0], ops[1], signed)
	}

	mul := vr.reg32()
	emitCopy(c, mul, ops[1].lo)
	eax := vr.physical(reg.EAX)
	edx := vr.physical(reg.EDX)
	emitCopy(c, eax, ops[0].lo)
	c.Emit(mir.Instr{
		Op:   wideMulOp{signed: signed},
		Defs: []mir.VReg{eax, edx},
		Uses: []mir.VReg{eax, mul},
	})
	emitCopy(c, dst.lo, edx)
	return nil
}

// iselMul64 lowers a 64-bit multiply.
//
// The schoolbook expansion of (ah·2³² + al)(bh·2³² + bl) with the term that
// overflows the width dropped:
//
//	lo      = low(al·bl)
//	hi      = high(al·bl) + al·bh + ah·bl
//
// One full 32×32 product for the low half, which the one-operand MUL puts in
// EDX:EAX, and two ordinary 32-bit products for the crossing terms — their
// own high halves are past bit 63 and do not matter.
func iselMul64(c *cursor, vr *vregs, dst, a, b value) error {
	mul := vr.reg32()
	emitCopy(c, mul, b.lo)
	eax := vr.physical(reg.EAX)
	edx := vr.physical(reg.EDX)
	emitCopy(c, eax, a.lo)
	c.Emit(mir.Instr{
		Op:   wideMulOp{},
		Defs: []mir.VReg{eax, edx},
		Uses: []mir.VReg{eax, mul},
	})

	lo := vr.reg32()
	hi := vr.reg32()
	emitCopy(c, lo, eax)
	emitCopy(c, hi, edx)

	cross := vr.reg32()
	emitCopy(c, cross, a.lo)
	c.Emit(mir.Instr{Op: aluOp{verb: ir.VMul}, Defs: []mir.VReg{cross}, Uses: []mir.VReg{cross, b.hi}})
	c.Emit(mir.Instr{Op: aluOp{verb: ir.VAdd}, Defs: []mir.VReg{hi}, Uses: []mir.VReg{hi, cross}})

	cross2 := vr.reg32()
	emitCopy(c, cross2, a.hi)
	c.Emit(mir.Instr{Op: aluOp{verb: ir.VMul}, Defs: []mir.VReg{cross2}, Uses: []mir.VReg{cross2, b.lo}})
	c.Emit(mir.Instr{Op: aluOp{verb: ir.VAdd}, Defs: []mir.VReg{hi}, Uses: []mir.VReg{hi, cross2}})

	emitCopy(c, dst.lo, lo)
	emitCopy(c, dst.hi, hi)
	return nil
}

// iselOverflow lowers §A2's five predicates.
//
// Each is the arithmetic instruction and the flag it already sets: ADD and
// SUB set OF on signed overflow and CF on unsigned carry, and the two-operand
// IMUL sets both when the product does not fit. The value the arithmetic
// produces is discarded — §A2 pairs each predicate with the wrapping verb in
// §A rather than producing both.
func iselOverflow(c *cursor, vr *vregs, in *ir.Inst, verb ir.Verb) error {
	ops, err := operands(vr, in, 2)
	if err != nil {
		return err
	}
	dst, err := vr.define(in.Result(0))
	if err != nil {
		return fmt.Errorf("%s: %w", in.Op(), err)
	}
	w := ops[0].w

	cond := condO
	if verb == ir.VUAddO || verb == ir.VUMulO {
		cond = condB // CF, which SETC and SETB are the same instruction for
	}

	switch verb {
	case ir.VSAddO, ir.VUAddO, ir.VSSubO:
		alu, carry := ir.VAdd, false
		if verb == ir.VSSubO {
			alu, carry = ir.VSub, true
		}
		sum := vr.reg32()
		emitCopy(c, sum, ops[0].lo)
		c.Emit(mir.Instr{Op: aluOp{verb: alu}, Defs: []mir.VReg{sum}, Uses: []mir.VReg{sum, ops[1].lo}})
		if w.pairs() {
			high := vr.reg32()
			emitCopy(c, high, ops[0].hi)
			c.Emit(mir.Instr{
				Op:   carryOp{sub: carry || verb == ir.VSSubO},
				Defs: []mir.VReg{high}, Uses: []mir.VReg{high, ops[1].hi},
			})
		}
		c.Emit(mir.Instr{Op: setccOp{cond: cond}, Defs: []mir.VReg{dst.lo}})
		return nil
	}

	if w.pairs() {
		return iselMulOverflow64(c, vr, dst, ops[0], ops[1], verb == ir.VSMulO)
	}
	if verb == ir.VSMulO {
		prod := vr.reg32()
		emitCopy(c, prod, ops[0].lo)
		c.Emit(mir.Instr{Op: aluOp{verb: ir.VMul}, Defs: []mir.VReg{prod}, Uses: []mir.VReg{prod, ops[1].lo}})
		c.Emit(mir.Instr{Op: setccOp{cond: condO}, Defs: []mir.VReg{dst.lo}})
		return nil
	}

	// Unsigned: the one-operand MUL sets CF when the high half is not zero,
	// which is the carry out of thirty-two bits.
	mul := vr.reg32()
	emitCopy(c, mul, ops[1].lo)
	eax := vr.physical(reg.EAX)
	edx := vr.physical(reg.EDX)
	emitCopy(c, eax, ops[0].lo)
	c.Emit(mir.Instr{
		Op:   wideMulOp{},
		Defs: []mir.VReg{eax, edx},
		Uses: []mir.VReg{eax, mul},
	})
	c.Emit(mir.Instr{Op: setccOp{cond: condB}, Defs: []mir.VReg{dst.lo}})
	return nil
}
