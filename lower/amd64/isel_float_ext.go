package amd64

import (
	"fmt"
	"github.com/vertex-language/ir"
	"github.com/vertex-language/ir/lower/mir"
)

func iselFma(c *cursor, vr *vregs, in *ir.Inst) error {
	a, ok := vr.lookup(in.Arg(0))
	if !ok {
		return fmt.Errorf("vfma operand 0 missing")
	}
	b, ok := vr.lookup(in.Arg(1))
	if !ok {
		return fmt.Errorf("vfma operand 1 missing")
	}
	d, ok := vr.lookup(in.Arg(2))
	if !ok {
		return fmt.Errorf("vfma operand 2 missing")
	}

	dst, err := vr.define(in.Result(0))
	if err != nil {
		return err
	}

	w := vr.widthOfVReg(dst)

	// VFMA a, b, d -> dst = a * b + d
	// fmaOp expects dst to be populated with d first if they are different!
	// emit.go handles it by emitting Movaps if dst != c_val.
	// Wait, emit.go: dst, a, b, c_val := in.Defs[0], in.Uses[0], in.Uses[1], in.Uses[2]
	c.Emit(mir.Instr{
		Op:   fmaOp{w: w},
		Defs: []mir.VReg{dst},
		Uses: []mir.VReg{a, b, d},
	})
	return nil
}

func iselFloatRound(c *cursor, vr *vregs, in *ir.Inst) error {
	src, ok := vr.lookup(in.Arg(0))
	if !ok {
		return fmt.Errorf("round operand missing")
	}
	dst, err := vr.define(in.Result(0))
	if err != nil {
		return err
	}
	w := vr.widthOfVReg(dst)

	var mode int64
	switch in.Op().Verb {
	case ir.VNearest:
		mode = 0
	case ir.VFloor:
		mode = 1
	case ir.VCeil:
		mode = 2
	case ir.VTrunc:
		mode = 3
	}

	c.Emit(mir.Instr{
		Op:   fRoundOp{w: w, mode: mode},
		Defs: []mir.VReg{dst},
		Uses: []mir.VReg{src},
	})
	return nil
}

func iselFloatMinMax(c *cursor, vr *vregs, in *ir.Inst) error {
	a, ok := vr.lookup(in.Arg(0))
	if !ok {
		return fmt.Errorf("missing a")
	}
	b, ok := vr.lookup(in.Arg(1))
	if !ok {
		return fmt.Errorf("missing b")
	}
	dst, err := vr.define(in.Result(0))
	if err != nil {
		return err
	}
	w := vr.widthOfVReg(dst)

	verb := in.Op().Verb
	isMax := verb == ir.VMaximum || verb == ir.VMaxNum
	isNum := verb == ir.VMinNum || verb == ir.VMaxNum

	if isNum {
		b_nan := c.open("b_nan")
		next := c.open("next")
		c.Emit(mir.Instr{Op: cmpOp{w: w}, Uses: []mir.VReg{b, b}})
		c.branch(condP, b_nan, next)

		equal := c.open("equal")
		mins := c.open("mins")
		c.Emit(mir.Instr{Op: cmpOp{w: w}, Uses: []mir.VReg{a, b}})
		c.branch(condE, equal, mins)

		done := c.open("done")

		c.resume(mins)
		emitCopy(c, dst, a, w)
		c.Emit(mir.Instr{Op: hwMinMaxOp{isMax: isMax, w: w}, Defs: []mir.VReg{dst}, Uses: []mir.VReg{dst, b}})
		c.to(done)

		c.resume(equal)
		emitCopy(c, dst, a, w)
		op := fOr
		if isMax {
			op = fAnd
		}
		c.Emit(mir.Instr{Op: fLogicOp{op: op, w: w}, Defs: []mir.VReg{dst}, Uses: []mir.VReg{dst, b}})
		c.to(done)

		c.resume(b_nan)
		emitCopy(c, dst, a, w)
		c.to(done)

		c.resume(done)
	} else {
		a_nan := c.open("a_nan")
		next := c.open("next")
		c.Emit(mir.Instr{Op: cmpOp{w: w}, Uses: []mir.VReg{a, a}})
		c.branch(condP, a_nan, next)

		equal := c.open("equal")
		mins := c.open("mins")
		c.Emit(mir.Instr{Op: cmpOp{w: w}, Uses: []mir.VReg{a, b}})
		c.branch(condE, equal, mins)

		done := c.open("done")

		c.resume(mins)
		emitCopy(c, dst, a, w)
		c.Emit(mir.Instr{Op: hwMinMaxOp{isMax: isMax, w: w}, Defs: []mir.VReg{dst}, Uses: []mir.VReg{dst, b}})
		c.to(done)

		c.resume(equal)
		emitCopy(c, dst, a, w)
		op := fOr
		if isMax {
			op = fAnd
		}
		c.Emit(mir.Instr{Op: fLogicOp{op: op, w: w}, Defs: []mir.VReg{dst}, Uses: []mir.VReg{dst, b}})
		c.to(done)

		c.resume(a_nan)
		emitCopy(c, dst, a, w)
		c.to(done)

		c.resume(done)
	}
	return nil
}

func iselFloatToIntExt(c *cursor, vr *vregs, in *ir.Inst) error {
	op := in.Op()
	src, ok := vr.lookup(in.Arg(0))
	if !ok {
		return fmt.Errorf("missing src")
	}
	dst, err := vr.define(in.Result(0))
	if err != nil {
		return err
	}

	from := vr.widthOfVReg(src)
	to := vr.widthOfVReg(dst)

	verb := op.Verb

	if verb == ir.VUCvtF32 || verb == ir.VUCvtF64 {
		if to == w32 {
			// Convert to signed 64-bit first, which covers the entire uint32 range.
			// Range for uint32 is [0, 2^32). But ucvt truncates towards zero. So it's valid for (-1, 2^32).
			// If it's out of range, trap.
			trap := c.open("trap")

			lo := vr.temp(from)
			if err := iselFloatConst(c, vr, lo, from, -1.0); err != nil {
				return err
			}
			inRange1 := c.open("inRange1")
			c.Emit(mir.Instr{Op: cmpOp{w: from}, Uses: []mir.VReg{src, lo}})
			c.branch(condBE, trap, inRange1) // if src <= -1.0 or unordered, trap

			c.resume(inRange1)
			hi := vr.temp(from)
			if err := iselFloatConst(c, vr, hi, from, 4294967296.0); err != nil {
				return err
			}
			convert := c.open("convert")
			c.Emit(mir.Instr{Op: cmpOp{w: from}, Uses: []mir.VReg{hi, src}})
			c.branch(condBE, trap, convert) // if hi <= src or unordered, trap

			trap.Emit(mir.Instr{Op: trapOp{}})

			c.resume(convert)
			dst64 := vr.temp(w64)
			c.Emit(mir.Instr{
				Op:   cvtFloatToIntOp{from: from, to: w64},
				Defs: []mir.VReg{dst64}, Uses: []mir.VReg{src},
			})
			emitCopy(c, dst, dst64, w32) // implicit truncation
			return nil
		}
		if to == w64 {
			return iselUCvtToI64(c, vr, src, dst, from)
		}
	}

	if verb == ir.VSCvtSatF32 || verb == ir.VSCvtSatF64 {
		r, ok := f2iRange[[2]width{from, to}]
		if !ok {
			return fmt.Errorf("no range for scvtsat")
		}

		nan := c.open("nan")
		notNan := c.open("notNan")
		c.Emit(mir.Instr{Op: cmpOp{w: from}, Uses: []mir.VReg{src, src}})
		c.branch(condP, nan, notNan)

		c.resume(nan)
		zero := vr.temp(to)
		c.Emit(mir.Instr{Op: constOp{imm: 0, w: to}, Defs: []mir.VReg{zero}})
		emitCopy(c, dst, zero, to)

		done := c.open("done")
		c.to(done)

		c.resume(notNan)
		lo := vr.temp(from)
		iselFloatConst(c, vr, lo, from, r.lo)
		below := c.open("below")
		checkHi := c.open("checkHi")

		belowCond := condB
		if r.loStrict {
			belowCond = condBE
		}
		c.Emit(mir.Instr{Op: cmpOp{w: from}, Uses: []mir.VReg{src, lo}})
		c.branch(belowCond, below, checkHi)

		c.resume(below)
		minInt := vr.temp(to)
		var minVal int64 = -2147483648
		if to == w64 {
			minVal = -9223372036854775808
		}
		c.Emit(mir.Instr{Op: constOp{imm: minVal, w: to}, Defs: []mir.VReg{minInt}})
		emitCopy(c, dst, minInt, to)
		c.to(done)

		c.resume(checkHi)
		hi := vr.temp(from)
		iselFloatConst(c, vr, hi, from, r.hi)
		above := c.open("above")
		convert := c.open("convert")
		c.Emit(mir.Instr{Op: cmpOp{w: from}, Uses: []mir.VReg{hi, src}})
		c.branch(condBE, above, convert)

		c.resume(above)
		maxInt := vr.temp(to)
		var maxVal int64 = 2147483647
		if to == w64 {
			maxVal = 9223372036854775807
		}
		c.Emit(mir.Instr{Op: constOp{imm: maxVal, w: to}, Defs: []mir.VReg{maxInt}})
		emitCopy(c, dst, maxInt, to)
		c.to(done)

		c.resume(convert)
		c.Emit(mir.Instr{
			Op:   cvtFloatToIntOp{from: from, to: to},
			Defs: []mir.VReg{dst}, Uses: []mir.VReg{src},
		})
		c.to(done)

		c.resume(done)
		return nil
	}

	return fmt.Errorf("iselFloatToIntExt not fully implemented for %s", op)
}

// iselUCvtToI64 lowers a float to a 64-bit unsigned integer — §C2's other
// row with no instruction behind it, and the mirror of isel.go's
// iselUCvtI64.
//
// CVTTSD2SI writes its result as signed, so it can only produce values below
// 2^63: everything above that saturates to the indefinite value rather than
// wrapping into the upper half. Below 2^63 it is already the answer, and the
// branch is between those two cases.
//
// At or above it, 2^63 is subtracted first, which is exact — the constant is
// a power of two and the source has at most fifty-three significant bits, so
// nothing is lost — and the sign bit is put back with an OR. The subtraction
// brings the value into the signed range that the instruction can express,
// and the bit pattern that comes out is the unsigned one.
//
// Out of range is undefined in C (§6.3.1.4p1), and the w32 path above traps
// rather than answering something; this does the same, with the range
// (-1, 2^64). An unordered comparison takes the trap branch too, which is
// what makes a NaN trap rather than convert.
func iselUCvtToI64(c *cursor, vr *vregs, src, dst mir.VReg, from width) error {
	trap := c.open("ucvttrap")
	inRange := c.open("ucvtinrange")

	lo := vr.temp(from)
	if err := iselFloatConst(c, vr, lo, from, -1.0); err != nil {
		return err
	}
	c.Emit(mir.Instr{Op: cmpOp{w: from}, Uses: []mir.VReg{src, lo}})
	c.branch(condBE, trap, inRange) // src <= -1.0, or unordered

	c.resume(inRange)
	limit := vr.temp(from)
	if err := iselFloatConst(c, vr, limit, from, 18446744073709551616.0); err != nil { // 2^64
		return err
	}
	small := c.open("ucvtsmall")
	c.Emit(mir.Instr{Op: cmpOp{w: from}, Uses: []mir.VReg{limit, src}})
	c.branch(condBE, trap, small) // 2^64 <= src, or unordered

	trap.Emit(mir.Instr{Op: trapOp{}})

	// Below 2^63 the signed instruction is the answer.
	c.resume(small)
	half := vr.temp(from)
	if err := iselFloatConst(c, vr, half, from, 9223372036854775808.0); err != nil { // 2^63
		return err
	}
	big := c.open("ucvtbig")
	direct := c.open("ucvtdirect")
	done := c.open("ucvtdone")
	c.Emit(mir.Instr{Op: cmpOp{w: from}, Uses: []mir.VReg{src, half}})
	c.branch(condB, direct, big)

	c.resume(direct)
	c.Emit(mir.Instr{
		Op:   cvtFloatToIntOp{from: from, to: w64},
		Defs: []mir.VReg{dst}, Uses: []mir.VReg{src},
	})
	c.to(done)

	// At or above it: bias down by 2^63, convert, and put the sign bit back.
	c.resume(big)
	biased := vr.temp(from)
	emitCopy(c, biased, src, from)
	c.Emit(mir.Instr{
		Op:   fAluOp{verb: ir.VSub, w: from},
		Defs: []mir.VReg{biased}, Uses: []mir.VReg{biased, half},
	})
	c.Emit(mir.Instr{
		Op:   cvtFloatToIntOp{from: from, to: w64},
		Defs: []mir.VReg{dst}, Uses: []mir.VReg{biased},
	})
	sign := vr.temp(w64)
	c.Emit(mir.Instr{Op: constOp{imm: -9223372036854775808, w: w64}, Defs: []mir.VReg{sign}})
	c.Emit(mir.Instr{
		Op:   aluOp{verb: ir.VOr, w: w64},
		Defs: []mir.VReg{dst}, Uses: []mir.VReg{dst, sign},
	})
	c.to(done)

	c.resume(done)
	return nil
}
