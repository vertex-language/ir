package arm64

import (
	"fmt"
	"math"

	"github.com/vertex-language/ir"
	"github.com/vertex-language/ir/lower/mir"
)

// iselDivide lowers §A's four division verbs.
//
// SDIV and UDIV do not trap. A zero divisor gives zero and INT_MIN/−1 gives
// INT_MIN, both quietly, and §A says both trap — so the trap is a guard this
// package emits in front of the divide rather than something the instruction
// does. Two guards for the signed verbs and one for the unsigned, since
// there is no unsigned pair whose quotient does not fit.
//
// The remainder verbs are the same guards and one more instruction: A64 has
// no remainder, so it is the quotient multiplied back out and subtracted,
// which MSUB does in one.
func iselDivide(c *cursor, vr *vregs, in *ir.Inst, signed, rem bool) error {
	op := in.Op()
	ops, err := operands(vr, in, 2)
	if err != nil {
		return err
	}
	a, b := ops[0], ops[1]
	dst, err := vr.define(in.Result(0))
	if err != nil {
		return fmt.Errorf("%s: %w", op, err)
	}
	w := vr.widthOfVReg(dst)

	trap := c.open("trap")

	// A zero divisor, which is every verb's trap.
	nonZero := c.open("nonzero")
	c.Emit(mir.Instr{Op: cmpImmOp{imm: 0, w: w}, Uses: []mir.VReg{b}})
	c.branch(condEQ, trap, nonZero)

	// INT_MIN/−1, which is the signed verbs' second trap: the quotient is
	// one past the top of the range and there is nowhere to put it. Two
	// compares rather than one, because the pair of operands has to match
	// and neither half alone is wrong.
	if signed {
		minusOne := vr.temp(w)
		divisible := c.open("divisible")
		notMinusOne := c.open("notminusone")
		c.Emit(mir.Instr{Op: constOp{imm: -1, w: w}, Defs: []mir.VReg{minusOne}})
		c.Emit(mir.Instr{Op: cmpOp{w: w}, Uses: []mir.VReg{b, minusOne}})
		c.branch(condNE, notMinusOne, divisible)

		// The divisor is −1, so the dividend decides.
		c.resume(divisible)
		min := int64(math.MinInt32)
		if w == w64 {
			min = math.MinInt64
		}
		intMin := vr.temp(w)
		c.Emit(mir.Instr{Op: constOp{imm: min, w: w}, Defs: []mir.VReg{intMin}})
		c.Emit(mir.Instr{Op: cmpOp{w: w}, Uses: []mir.VReg{a, intMin}})
		c.branch(condEQ, trap, notMinusOne)

		c.resume(notMinusOne)
	}

	// BRK, and nothing after it: a trap is a terminator with no successors.
	trap.Emit(mir.Instr{Op: trapOp{}})

	if !rem {
		c.Emit(mir.Instr{
			Op:   divOp{signed: signed, w: w},
			Defs: []mir.VReg{dst},
			Uses: []mir.VReg{a, b},
		})
		return nil
	}

	q := vr.temp(w)
	c.Emit(mir.Instr{
		Op:   divOp{signed: signed, w: w},
		Defs: []mir.VReg{q},
		Uses: []mir.VReg{a, b},
	})
	c.Emit(mir.Instr{
		Op:   msubOp{w: w},
		Defs: []mir.VReg{dst},
		Uses: []mir.VReg{q, b, a},
	})
	return nil
}
