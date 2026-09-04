package arm64

import (
	"fmt"

	"github.com/vertex-language/ir"
	"github.com/vertex-language/ir/lower/mir"
)

// iselRotL lowers §A5's rotl, which A64 does not have: there is a rotate
// right and no rotate left.
//
// Rotating left by n is rotating right by W−n, and RORV reads only the low
// five or six bits of its amount — so the subtraction is a negation and needs
// no mask and no correction at n = 0, where the negation is zero and the
// rotate is the identity.
func iselRotL(c *cursor, vr *vregs, in *ir.Inst) error {
	ops, err := operands(vr, in, 2)
	if err != nil {
		return err
	}
	dst, err := vr.define(in.Result(0))
	if err != nil {
		return fmt.Errorf("%s: %w", in.Op(), err)
	}
	w := vr.widthOfVReg(dst)

	back := vr.temp(w)
	c.Emit(mir.Instr{Op: unOp{verb: ir.VNeg, w: w}, Defs: []mir.VReg{back}, Uses: []mir.VReg{ops[1]}})
	c.Emit(mir.Instr{
		Op:   aluOp{verb: ir.VRotR, w: w},
		Defs: []mir.VReg{dst},
		Uses: []mir.VReg{ops[0], back},
	})
	return nil
}

// iselCtz lowers §A6's ctz, which A64 also does not have: RBIT reverses the
// bit order and turns the trailing zeros into leading ones, which CLZ counts.
func iselCtz(c *cursor, vr *vregs, in *ir.Inst) error {
	ops, err := operands(vr, in, 1)
	if err != nil {
		return err
	}
	dst, err := vr.define(in.Result(0))
	if err != nil {
		return fmt.Errorf("%s: %w", in.Op(), err)
	}
	w := vr.widthOfVReg(dst)

	rev := vr.temp(w)
	c.Emit(mir.Instr{Op: rbitOp{w: w}, Defs: []mir.VReg{rev}, Uses: ops})
	c.Emit(mir.Instr{Op: unOp{verb: ir.VClz, w: w}, Defs: []mir.VReg{dst}, Uses: []mir.VReg{rev}})
	return nil
}

// iselPopcnt lowers §A6's popcnt.
//
// The instruction that counts bits on this architecture is CNT, which is a
// SIMD instruction over the lanes of a vector register: a scalar popcount is
// an FMOV into the vector file, a CNT, an ADDV to sum the lanes, and an FMOV
// back. The assembler this package emits through has no vector-lane forms, so
// that route is not available and this is the well-known SWAR sequence
// instead — pairs, then nibbles, then bytes, then one multiply that sums the
// bytes into the top one.
//
// Fourteen instructions where the vector route is four. Written this way
// because it is exact and reaches no instruction the assembler does not have;
// it is the obvious thing to replace when CNT arrives.
func iselPopcnt(c *cursor, vr *vregs, in *ir.Inst) error {
	ops, err := operands(vr, in, 1)
	if err != nil {
		return err
	}
	dst, err := vr.define(in.Result(0))
	if err != nil {
		return fmt.Errorf("%s: %w", in.Op(), err)
	}
	w := vr.widthOfVReg(dst)
	if w.isFloat() {
		return fmt.Errorf("%s: not an integer operation", in.Op())
	}

	// The masks and the byte-summing multiplier, at whichever width.
	m1, m2, m4, ones, top := int64(0x55555555), int64(0x33333333), int64(0x0f0f0f0f), int64(0x01010101), int64(24)
	if w == w64 {
		m1, m2, m4 = 0x5555555555555555, 0x3333333333333333, 0x0f0f0f0f0f0f0f0f
		ones, top = 0x0101010101010101, 56
	}

	konst := func(v int64) mir.VReg {
		r := vr.temp(w)
		c.Emit(mir.Instr{Op: constOp{imm: v, w: w}, Defs: []mir.VReg{r}})
		return r
	}
	alu := func(verb ir.Verb, a, b mir.VReg) mir.VReg {
		r := vr.temp(w)
		c.Emit(mir.Instr{Op: aluOp{verb: verb, w: w}, Defs: []mir.VReg{r}, Uses: []mir.VReg{a, b}})
		return r
	}

	// x -= (x >> 1) & 0x5555...  — every adjacent pair now holds its own
	// population, in place.
	x := alu(ir.VSub, ops[0], alu(ir.VAnd, alu(ir.VUShr, ops[0], konst(1)), konst(m1)))
	// x = (x & 0x3333...) + ((x >> 2) & 0x3333...) — pairs into nibbles.
	x = alu(ir.VAdd, alu(ir.VAnd, x, konst(m2)), alu(ir.VAnd, alu(ir.VUShr, x, konst(2)), konst(m2)))
	// x = (x + (x >> 4)) & 0x0f0f... — nibbles into bytes. The mask comes
	// after the add because no byte's count can reach 16 and carry.
	x = alu(ir.VAnd, alu(ir.VAdd, x, alu(ir.VUShr, x, konst(4))), konst(m4))
	// The multiply sums every byte into the top one, which the shift reads.
	c.Emit(mir.Instr{
		Op:   aluOp{verb: ir.VUShr, w: w},
		Defs: []mir.VReg{dst},
		Uses: []mir.VReg{alu(ir.VMul, x, konst(ones)), konst(top)},
	})
	return nil
}

// iselOverflow lowers §A2's five predicates.
//
// The three additive ones are the arithmetic instruction's flag-setting form
// and a CSET: ADDS and SUBS set V on signed overflow and C on unsigned carry,
// which is what saddo, ssubo and uaddo each ask for, and the sum itself is
// discarded. §A2 pairs each predicate with the wrapping verb in §A rather
// than producing both, so the value this computes is only the flag.
//
// The multiplicative two have no flag. A 64-bit product overflows when its
// high half is not the sign extension of its low half — or, unsigned, when
// the high half is not zero — and SMULH and UMULH are the instructions that
// name that half. At 32 bits the whole product fits in an X register, so the
// same question is asked of one 64-bit multiply.
func iselOverflow(c *cursor, vr *vregs, in *ir.Inst, verb ir.Verb) error {
	ops, err := operands(vr, in, 2)
	if err != nil {
		return err
	}
	a, b := ops[0], ops[1]
	dst, err := vr.define(in.Result(0))
	if err != nil {
		return fmt.Errorf("%s: %w", in.Op(), err)
	}
	w := vr.widthOfVReg(a)
	if w.isFloat() {
		return fmt.Errorf("%s: not an integer operation", in.Op())
	}

	switch verb {
	case ir.VSAddO, ir.VUAddO, ir.VSSubO:
		sum := vr.temp(w)
		cond := condVS
		op := ir.VAdd
		switch verb {
		case ir.VUAddO:
			cond = condHS
		case ir.VSSubO:
			op = ir.VSub
		}
		c.Emit(mir.Instr{
			Op:   flagAluOp{verb: op, w: w},
			Defs: []mir.VReg{sum},
			Uses: []mir.VReg{a, b},
		})
		c.Emit(mir.Instr{Op: csetOp{cond: cond}, Defs: []mir.VReg{dst}})
		return nil
	}

	signed := verb == ir.VSMulO

	if w == w32 {
		// Both operands widened, one 64-bit multiply, and the question of
		// whether the answer still fits in 32 bits.
		wa, wb := vr.temp(w64), vr.temp(w64)
		c.Emit(mir.Instr{Op: extOp{from: a32, signed: signed}, Defs: []mir.VReg{wa}, Uses: []mir.VReg{a}})
		c.Emit(mir.Instr{Op: extOp{from: a32, signed: signed}, Defs: []mir.VReg{wb}, Uses: []mir.VReg{b}})
		prod := vr.temp(w64)
		c.Emit(mir.Instr{Op: aluOp{verb: ir.VMul, w: w64}, Defs: []mir.VReg{prod}, Uses: []mir.VReg{wa, wb}})

		if !signed {
			// Anything above bit 31 is a carry out.
			sh := vr.temp(w64)
			hi := vr.temp(w64)
			c.Emit(mir.Instr{Op: constOp{imm: 32, w: w64}, Defs: []mir.VReg{sh}})
			c.Emit(mir.Instr{Op: aluOp{verb: ir.VUShr, w: w64}, Defs: []mir.VReg{hi}, Uses: []mir.VReg{prod, sh}})
			c.Emit(mir.Instr{Op: cmpImmOp{imm: 0, w: w64}, Uses: []mir.VReg{hi}})
			c.Emit(mir.Instr{Op: csetOp{cond: condNE}, Defs: []mir.VReg{dst}})
			return nil
		}
		// Signed: the product fits iff it is its own low half sign-extended.
		back := vr.temp(w64)
		c.Emit(mir.Instr{Op: extOp{from: a32, signed: true}, Defs: []mir.VReg{back}, Uses: []mir.VReg{prod}})
		c.Emit(mir.Instr{Op: cmpOp{w: w64}, Uses: []mir.VReg{prod, back}})
		c.Emit(mir.Instr{Op: csetOp{cond: condNE}, Defs: []mir.VReg{dst}})
		return nil
	}

	hi := vr.temp(w64)
	c.Emit(mir.Instr{Op: mulhOp{signed: signed}, Defs: []mir.VReg{hi}, Uses: []mir.VReg{a, b}})

	if !signed {
		c.Emit(mir.Instr{Op: cmpImmOp{imm: 0, w: w64}, Uses: []mir.VReg{hi}})
		c.Emit(mir.Instr{Op: csetOp{cond: condNE}, Defs: []mir.VReg{dst}})
		return nil
	}

	// Signed: the high half has to be the low half's sign, which is the
	// low half arithmetic-shifted down by 63.
	lo := vr.temp(w64)
	c.Emit(mir.Instr{Op: aluOp{verb: ir.VMul, w: w64}, Defs: []mir.VReg{lo}, Uses: []mir.VReg{a, b}})
	sh := vr.temp(w64)
	sign := vr.temp(w64)
	c.Emit(mir.Instr{Op: constOp{imm: 63, w: w64}, Defs: []mir.VReg{sh}})
	c.Emit(mir.Instr{Op: aluOp{verb: ir.VSShr, w: w64}, Defs: []mir.VReg{sign}, Uses: []mir.VReg{lo, sh}})
	c.Emit(mir.Instr{Op: cmpOp{w: w64}, Uses: []mir.VReg{hi, sign}})
	c.Emit(mir.Instr{Op: csetOp{cond: condNE}, Defs: []mir.VReg{dst}})
	return nil
}

// iselPtrInt lowers §C4, which on a target whose pointers are already sixty-
// four bits is a move: there is nothing to truncate and nothing to extend.
func iselPtrInt(c *cursor, vr *vregs, in *ir.Inst) error {
	ops, err := operands(vr, in, 1)
	if err != nil {
		return err
	}
	dst, err := vr.define(in.Result(0))
	if err != nil {
		return fmt.Errorf("%s: %w", in.Op(), err)
	}
	emitCopy(c, dst, ops[0], w64)
	return nil
}

// iselDiff lowers §D3's ptr.diff: a subtraction, sign-extended from ptrbits,
// which at sixty-four bits is the subtraction alone.
func iselDiff(c *cursor, vr *vregs, in *ir.Inst) error {
	ops, err := operands(vr, in, 2)
	if err != nil {
		return err
	}
	dst, err := vr.define(in.Result(0))
	if err != nil {
		return fmt.Errorf("%s: %w", in.Op(), err)
	}
	c.Emit(mir.Instr{
		Op:   aluOp{verb: ir.VSub, w: w64},
		Defs: []mir.VReg{dst},
		Uses: ops,
	})
	return nil
}
