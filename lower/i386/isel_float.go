package i386

// §A3, §B's float half, and §C's conversions, in the vector unit.
//
// SSE2 rather than x87, which is a choice with consequences. x87 is the
// psABI's own float unit and its registers are a stack, which nothing in this
// tree's register allocator can model; SSE2's are eight flat registers that
// behave like any other file. The price is that the baseline stops being a
// 386 — SSE2 arrived with the Pentium 4 — and that the return convention
// still has to go through x87, because the psABI says a float comes back in
// ST(0) and no amount of SSE changes that.

import (
	"fmt"
	"math"

	"github.com/vertex-language/ir"
	"github.com/vertex-language/ir/lower/mir"
)

// floatBinOps is every §A3 verb that is one instruction.
var floatBinOps = map[ir.Verb]bool{
	ir.VAdd: true, ir.VSub: true, ir.VMul: true, ir.VDiv: true,
}

// floatConds maps §B's float verbs onto what answers them after UCOMIS.
//
// UCOMIS sets ZF, PF and CF as if the two values had been compared unsigned,
// with PF standing for unordered. That makes A and AE the ordered
// greater-than and greater-or-equal — and nothing directly the ordered
// less-than, which is why §B's lt and le are lowered with their operands
// swapped rather than with a different condition.
//
// eq and ne need more than a condition each: an unordered compare sets ZF as
// well as PF, so EQ alone would call a NaN equal to anything. See
// iselFloatCompare.
var floatConds = map[ir.Verb]struct {
	cond condCode
	swap bool
}{
	ir.VLt: {cond: condA, swap: true},
	ir.VLe: {cond: condAE, swap: true},
}

// iselFloatInst lowers the verbs that work in the vector file. It reports
// whether the verb was one of them.
func iselFloatInst(c *cursor, vr *vregs, fr *frame, in *ir.Inst, opts Options) (bool, error) {
	op := in.Op()
	verb := op.Verb

	if op.Type.IsFloat() {
		switch {
		case floatBinOps[verb]:
			return true, iselFloatBinary(c, vr, in)
		case verb == ir.VSqrt:
			return true, iselFloatSqrt(c, vr, in)
		case verb == ir.VNeg, verb == ir.VAbs:
			return true, iselFloatSign(c, vr, fr, in)
		case verb == ir.VCopySign:
			return true, iselCopySign(c, vr, fr, in)
		case verb == ir.VMinimum, verb == ir.VMaximum,
			verb == ir.VMinNum, verb == ir.VMaxNum:
			return true, iselFloatMinMax(c, vr, in)
		case verb == ir.VCeil, verb == ir.VFloor,
			verb == ir.VTrunc, verb == ir.VNearest, verb == ir.VFMA:
			return true, iselFloatLibcall(c, vr, fr, in)
		case verb == ir.VEq, verb == ir.VNe, verb == ir.VUno:
			return true, iselFloatEquality(c, vr, in, verb)
		}
		if fc, ok := floatConds[verb]; ok {
			return true, iselFloatCompare(c, vr, in, fc.cond, fc.swap)
		}
	}

	switch verb {
	case ir.VSCvtI32, ir.VUCvtI32, ir.VSCvtI64, ir.VUCvtI64:
		return true, iselIntToFloat(c, vr, fr, in, opts)
	case ir.VSCvtF32, ir.VSCvtF64, ir.VUCvtF32, ir.VUCvtF64,
		ir.VSCvtSatF32, ir.VSCvtSatF64, ir.VUCvtSatF32, ir.VUCvtSatF64:
		return true, iselFloatToInt(c, vr, fr, in, opts)
	case ir.VFCvtF32, ir.VFCvtF64:
		return true, iselFloatWidth(c, vr, in)
	case ir.VBitcastF32, ir.VBitcastF64, ir.VBitcastI32, ir.VBitcastI64:
		return true, iselBitcast(c, vr, fr, in)
	}
	return false, nil
}

// ftwo puts a's value into dst, so that a two-address instruction can write
// dst without destroying a. The vector forms are two-address like every
// other x86 instruction.
func ftwo(c *cursor, dst, a mir.VReg) {
	c.Emit(mir.Instr{Op: fmovOp{}, Defs: []mir.VReg{dst}, Uses: []mir.VReg{a}, Copy: true})
}

func iselFloatBinary(c *cursor, vr *vregs, in *ir.Inst) error {
	ops, err := operands(vr, in, 2)
	if err != nil {
		return err
	}
	dst, err := vr.define(in.Result(0))
	if err != nil {
		return fmt.Errorf("%s: %w", in.Op(), err)
	}
	ftwo(c, dst.lo, ops[0].lo)
	c.Emit(mir.Instr{
		Op:   fbinOp{verb: in.Op().Verb, w: dst.w},
		Defs: []mir.VReg{dst.lo},
		Uses: []mir.VReg{dst.lo, ops[1].lo},
	})
	return nil
}

func iselFloatSqrt(c *cursor, vr *vregs, in *ir.Inst) error {
	ops, err := operands(vr, in, 1)
	if err != nil {
		return err
	}
	dst, err := vr.define(in.Result(0))
	if err != nil {
		return fmt.Errorf("%s: %w", in.Op(), err)
	}
	c.Emit(mir.Instr{
		Op:   fsqrtOp{w: dst.w},
		Defs: []mir.VReg{dst.lo},
		Uses: []mir.VReg{ops[0].lo},
	})
	return nil
}

// iselFloatSign lowers §A3's neg and abs, which are one mask each.
//
// Both preserve a NaN's payload, which §A3 requires and which is exactly what
// a bit operation does and an arithmetic one does not: negating by
// subtracting from zero would turn a NaN into the processor's own.
func iselFloatSign(c *cursor, vr *vregs, fr *frame, in *ir.Inst) error {
	ops, err := operands(vr, in, 1)
	if err != nil {
		return err
	}
	dst, err := vr.define(in.Result(0))
	if err != nil {
		return fmt.Errorf("%s: %w", in.Op(), err)
	}

	sign := floatConst(c, vr, fr, dst.w, signBit(dst.w))
	if in.Op().Verb == ir.VAbs {
		// ANDNPS computes ~dst & src, not dst & ~src, so the mask is
		// what goes into the destination and the value is the operand.
		ftwo(c, dst.lo, sign)
		c.Emit(mir.Instr{
			Op:   fbitOp{op: maskAndn, w: dst.w},
			Defs: []mir.VReg{dst.lo},
			Uses: []mir.VReg{dst.lo, ops[0].lo},
		})
		return nil
	}
	ftwo(c, dst.lo, ops[0].lo)
	c.Emit(mir.Instr{
		Op:   fbitOp{op: maskXor, w: dst.w}, // neg flips the sign bit
		Defs: []mir.VReg{dst.lo},
		Uses: []mir.VReg{dst.lo, sign},
	})
	return nil
}

// iselCopySign lowers §A3's copysign: the magnitude of one and the sign of
// the other, which is two masks and a merge.
func iselCopySign(c *cursor, vr *vregs, fr *frame, in *ir.Inst) error {
	ops, err := operands(vr, in, 2)
	if err != nil {
		return err
	}
	dst, err := vr.define(in.Result(0))
	if err != nil {
		return fmt.Errorf("%s: %w", in.Op(), err)
	}

	sign := floatConst(c, vr, fr, dst.w, signBit(dst.w))
	theirs := vr.vec()
	c.Emit(mir.Instr{Op: fmovOp{}, Defs: []mir.VReg{theirs}, Uses: []mir.VReg{ops[1].lo}})
	c.Emit(mir.Instr{
		Op:   fbitOp{op: maskAnd, w: dst.w},
		Defs: []mir.VReg{theirs}, Uses: []mir.VReg{theirs, sign},
	})

	// The magnitude, by the same ~dst & src reading of ANDNPS: the mask
	// is the destination and the value the operand.
	ftwo(c, dst.lo, sign)
	c.Emit(mir.Instr{
		Op:   fbitOp{op: maskAndn, w: dst.w},
		Defs: []mir.VReg{dst.lo}, Uses: []mir.VReg{dst.lo, ops[0].lo},
	})
	c.Emit(mir.Instr{
		Op:   fbitOp{op: maskOr, w: dst.w},
		Defs: []mir.VReg{dst.lo}, Uses: []mir.VReg{dst.lo, theirs},
	})
	return nil
}

// iselFloatMinMax lowers §A3's four min and max verbs.
//
// MINSS is not IEEE minimum and MAXSS is not maximum. Both return their
// *second* operand whenever the pair is unordered or the two compare equal,
// which is wrong for a NaN under one reading and for a signed zero under
// both — and is not even commutative. So each verb is the instruction for the
// ordinary case and an explicit answer for the two that are not.
//
// UCOMIS makes the split cheap: it sets ZF for an equal pair *and* for an
// unordered one, and PF only for the unordered one. So a single JE separates
// "distinct ordered values", where the instruction is right, from everything
// else.
//
// What the remaining cases want:
//
//   - Both zeroes, differing in sign. §A3 pins these: minimum and minnum give
//     −0, maximum and maxnum give +0. OR'ing the two operands together gives
//     −0 and AND'ing gives +0, which is exactly the pair.
//
//   - Either a NaN. minimum and maximum propagate it, which OR does — the
//     exponent's ones and the mantissa's survive it — and AND does not, so a
//     maximum separates the unordered pair from the equal one rather than
//     merging both the same way. minnum and maxnum discard the NaN, which
//     needs the other operand, so those test for it first and hand back the
//     one that is not.
func iselFloatMinMax(c *cursor, vr *vregs, in *ir.Inst) error {
	verb := in.Op().Verb
	ops, err := operands(vr, in, 2)
	if err != nil {
		return err
	}
	dst, err := vr.define(in.Result(0))
	if err != nil {
		return fmt.Errorf("%s: %w", in.Op(), err)
	}
	a, b := ops[0].lo, ops[1].lo
	max := verb == ir.VMaximum || verb == ir.VMaxNum
	keepNaN := verb == ir.VMinimum || verb == ir.VMaximum

	// The merge for a pair that is equal or unordered: OR for a minimum,
	// AND for a maximum, which is the signed-zero rule.
	merge := maskOr
	if max {
		merge = maskAnd
	}

	done := c.open("done")

	if !keepNaN {
		// minnum and maxnum: whichever operand is not a NaN is the
		// answer. A value is a NaN exactly when it is unordered with
		// itself.
		takeA := c.open("bnan")
		next := c.open("bok")
		c.Emit(mir.Instr{Op: fcmpOp{w: ops[0].w}, Uses: []mir.VReg{b, b}})
		c.branch(condP, takeA, next)

		takeB := c.open("anan")
		both := c.open("bothnum")
		c.Emit(mir.Instr{Op: fcmpOp{w: ops[0].w}, Uses: []mir.VReg{a, a}})
		c.branch(condP, takeB, both)

		takeA.Emit(mir.Instr{Op: fmovOp{}, Defs: []mir.VReg{dst.lo}, Uses: []mir.VReg{a}})
		takeA.Emit(mir.Instr{Op: jmpOp{target: done.Label}})
		c.mf.Succ(takeA, done.Label)

		takeB.Emit(mir.Instr{Op: fmovOp{}, Defs: []mir.VReg{dst.lo}, Uses: []mir.VReg{b}})
		takeB.Emit(mir.Instr{Op: jmpOp{target: done.Label}})
		c.mf.Succ(takeB, done.Label)

		c.resume(both)
	}

	// Now: for minimum and maximum, an unordered pair still reaches here
	// and takes the merge, which is what propagates the NaN. For minnum
	// and maxnum both operands are numbers by construction.
	equal := c.open("equal")
	distinct := c.open("distinct")
	c.Emit(mir.Instr{Op: fcmpOp{w: ops[0].w}, Uses: []mir.VReg{a, b}})
	c.branch(condE, equal, distinct)

	// Distinct ordered values, where the instruction is right.
	c.Emit(mir.Instr{Op: fmovOp{}, Defs: []mir.VReg{dst.lo}, Uses: []mir.VReg{a}})
	c.Emit(mir.Instr{
		Op:   fminmaxOp{max: max, w: dst.w},
		Defs: []mir.VReg{dst.lo}, Uses: []mir.VReg{dst.lo, b},
	})
	c.to(done)

	c.resume(equal)
	if keepNaN && merge != maskOr {
		// A maximum's merge is AND, and AND does not propagate a NaN:
		// the exponent's ones survive it but the mantissa's bits need
		// not, and 0x7ff8... AND 0x3ff0... is 1.0. So the unordered
		// pair, which arrives here alongside the equal one, takes the
		// OR that does propagate, and the AND is left to the two
		// zeroes it was for.
		unord := c.open("unord")
		zeroes := c.open("zeroes")
		c.Emit(mir.Instr{Op: fcmpOp{w: ops[0].w}, Uses: []mir.VReg{a, b}})
		c.branch(condP, unord, zeroes)

		unord.Emit(mir.Instr{Op: fmovOp{}, Defs: []mir.VReg{dst.lo}, Uses: []mir.VReg{a}})
		unord.Emit(mir.Instr{
			Op:   fbitOp{op: maskOr, w: dst.w},
			Defs: []mir.VReg{dst.lo}, Uses: []mir.VReg{dst.lo, b},
		})
		unord.Emit(mir.Instr{Op: jmpOp{target: done.Label}})
		c.mf.Succ(unord, done.Label)
	}
	c.Emit(mir.Instr{Op: fmovOp{}, Defs: []mir.VReg{dst.lo}, Uses: []mir.VReg{a}})
	c.Emit(mir.Instr{
		Op:   fbitOp{op: merge, w: dst.w},
		Defs: []mir.VReg{dst.lo}, Uses: []mir.VReg{dst.lo, b},
	})
	c.to(done)
	return nil
}

// iselFloatCompare lowers §B's ordered comparisons.
//
// UCOMIS answers greater-than and greater-or-equal directly — A and AE are
// false for an unordered pair, which is what "ordered" means — and answers
// nothing directly for less-than. So lt and le swap their operands and ask
// the greater-than question instead.
func iselFloatCompare(c *cursor, vr *vregs, in *ir.Inst, cond condCode, swap bool) error {
	ops, err := operands(vr, in, 2)
	if err != nil {
		return err
	}
	dst, err := vr.define(in.Result(0))
	if err != nil {
		return fmt.Errorf("%s: %w", in.Op(), err)
	}
	a, b := ops[0].lo, ops[1].lo
	if swap {
		a, b = b, a
	}
	c.Emit(mir.Instr{Op: fcmpOp{w: ops[0].w}, Uses: []mir.VReg{a, b}})
	c.Emit(mir.Instr{Op: setccOp{cond: cond}, Defs: []mir.VReg{dst.lo}})
	return nil
}

// iselFloatEquality lowers §B's eq, ne and uno.
//
// uno is PF alone. eq and ne are not one condition each: an unordered compare
// sets ZF as well as PF, so E would call a NaN equal to anything. §B's eq is
// ordered and its ne is the exact negation, so eq is E and not-PF, and ne is
// NE or PF.
func iselFloatEquality(c *cursor, vr *vregs, in *ir.Inst, verb ir.Verb) error {
	ops, err := operands(vr, in, 2)
	if err != nil {
		return err
	}
	dst, err := vr.define(in.Result(0))
	if err != nil {
		return fmt.Errorf("%s: %w", in.Op(), err)
	}

	c.Emit(mir.Instr{Op: fcmpOp{w: ops[0].w}, Uses: []mir.VReg{ops[0].lo, ops[1].lo}})
	if verb == ir.VUno {
		c.Emit(mir.Instr{Op: setccOp{cond: condP}, Defs: []mir.VReg{dst.lo}})
		return nil
	}

	ordered := vr.reg32()
	c.Emit(mir.Instr{Op: setccOp{cond: condNP}, Defs: []mir.VReg{ordered}})
	if verb == ir.VEq {
		c.Emit(mir.Instr{Op: setccOp{cond: condE}, Defs: []mir.VReg{dst.lo}})
		c.Emit(mir.Instr{
			Op:   aluOp{verb: ir.VAnd},
			Defs: []mir.VReg{dst.lo}, Uses: []mir.VReg{dst.lo, ordered},
		})
		return nil
	}
	// ne: not equal, or unordered.
	c.Emit(mir.Instr{Op: setccOp{cond: condNE}, Defs: []mir.VReg{dst.lo}})
	unordered := vr.reg32()
	c.Emit(mir.Instr{Op: constOp{imm: 1}, Defs: []mir.VReg{unordered}})
	c.Emit(mir.Instr{
		Op:   aluOp{verb: ir.VXor},
		Defs: []mir.VReg{unordered}, Uses: []mir.VReg{unordered, ordered},
	})
	c.Emit(mir.Instr{
		Op:   aluOp{verb: ir.VOr},
		Defs: []mir.VReg{dst.lo}, Uses: []mir.VReg{dst.lo, unordered},
	})
	return nil
}

// signBit is the mask holding one float's sign and nothing else.
func signBit(w width) uint64 {
	if w == wf32 {
		return 0x80000000
	}
	return 0x8000000000000000
}

// floatConst materializes a literal into a vector register, through the
// integer file: SSE2 has no instruction taking a float immediate, and a
// constant pool would be a relocation for a value that fits in a register.
func floatConst(c *cursor, vr *vregs, fr *frame, w width, bits uint64) mir.VReg {
	out := vr.vec()
	if w == wf32 {
		tmp := vr.reg32()
		c.Emit(mir.Instr{Op: constOp{imm: int64(int32(uint32(bits)))}, Defs: []mir.VReg{tmp}})
		c.Emit(mir.Instr{Op: movdToXmmOp{}, Defs: []mir.VReg{out}, Uses: []mir.VReg{tmp}})
		return out
	}
	// Eight bytes cross the files four at a time, so a double goes out
	// through a frame slot: two integer stores and one vector load. The
	// slot is one per function and reused, since nothing reads it across
	// an instruction boundary.
	lo := vr.reg32()
	hi := vr.reg32()
	c.Emit(mir.Instr{Op: constOp{imm: int64(int32(uint32(bits)))}, Defs: []mir.VReg{lo}})
	c.Emit(mir.Instr{Op: constOp{imm: int64(int32(uint32(bits >> 32)))}, Defs: []mir.VReg{hi}})
	c.Emit(mir.Instr{
		Op:   pairToFloatOp{off: fr.scratch()},
		Defs: []mir.VReg{out}, Uses: []mir.VReg{lo, hi},
	})
	return out
}

// iselFloatConst materializes §A7's float literal.
func iselFloatConst(c *cursor, vr *vregs, fr *frame, dst value, v float64) {
	bits := math.Float64bits(v)
	if dst.w == wf32 {
		bits = uint64(math.Float32bits(float32(v)))
	}
	src := floatConst(c, vr, fr, dst.w, bits)
	c.Emit(mir.Instr{Op: fmovOp{}, Defs: []mir.VReg{dst.lo}, Uses: []mir.VReg{src}, Copy: true})
}
