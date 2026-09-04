package arm64

import (
	"fmt"
	"math"

	"github.com/vertex-language/ir"
	"github.com/vertex-language/ir/lower/mir"
)

// floatBinOps is every §A3 verb that is one three-address instruction.
//
// Including all four of §A3's min and max, which is the part worth naming.
// FMIN and FMAX are IEEE-754-2019 minimum and maximum — a NaN operand
// propagates, and −0 compares below +0 — and FMINNM and FMAXNM are
// 754-2008 minNum and maxNum, which discard a NaN in favour of the other
// operand. That is §A3's two pairs exactly, in that order, with the signed
// zero already the way §A3 specifies. The fixup the spec warns a target may
// have to pay is not owed here.
var floatBinOps = map[ir.Verb]bool{
	ir.VAdd: true, ir.VSub: true, ir.VMul: true, ir.VDiv: true,
	ir.VMinimum: true, ir.VMaximum: true,
	ir.VMinNum: true, ir.VMaxNum: true,
}

// floatUnOps is every §A3 verb that is one one-source instruction: the sign
// verbs, the square root, and the four roundings, which are the four FRINTs
// with an explicit rounding mode rather than the ambient one.
var floatUnOps = map[ir.Verb]bool{
	ir.VNeg: true, ir.VAbs: true, ir.VSqrt: true,
	ir.VCeil: true, ir.VFloor: true, ir.VTrunc: true, ir.VNearest: true,
}

// floatConds maps §B's float verbs onto the condition each answers after an
// FCMP.
//
// Not the integer conditions. FCMP sets C and V on an unordered compare, so
// LT and LE are both true for a NaN and neither is the ordered comparison §B
// asks for. MI is N set, which only an ordered less-than produces; LS is C
// clear or Z set, which only an ordered less-or-equal produces. VS is the
// unordered case itself, which is what §B's uno is.
//
// eq and ne need no such care: §B's float ne is the exact negation of eq —
// true when either operand is NaN — and EQ and NE are exact negations of each
// other, so the pair lands right without a second condition.
var floatConds = map[ir.Verb]condCode{
	ir.VEq:  condEQ,
	ir.VNe:  condNE,
	ir.VLt:  condMI,
	ir.VLe:  condLS,
	ir.VUno: condVS,
}

// iselFloatInst lowers the §A3, §B and §C verbs that work in the vector file.
// It reports whether the verb was one of them.
func iselFloatInst(c *cursor, vr *vregs, in *ir.Inst) (bool, error) {
	op := in.Op()
	verb := op.Verb

	if op.Type.IsFloat() {
		switch {
		case floatBinOps[verb]:
			return true, iselFloatBinary(c, vr, in)
		case floatUnOps[verb]:
			return true, iselFloatUnary(c, vr, in)
		case verb == ir.VFMA:
			return true, iselFMA(c, vr, in)
		case verb == ir.VCopySign:
			return true, iselCopySign(c, vr, in)
		}
		if cond, ok := floatConds[verb]; ok {
			return true, iselFloatCompare(c, vr, in, cond)
		}
	}

	switch verb {
	case ir.VSCvtI32, ir.VSCvtI64:
		return true, iselIntToFloat(c, vr, in, true)
	case ir.VUCvtI32, ir.VUCvtI64:
		return true, iselIntToFloat(c, vr, in, false)
	case ir.VSCvtF32, ir.VSCvtF64:
		return true, iselFloatToInt(c, vr, in, true)
	case ir.VUCvtF32, ir.VUCvtF64:
		return true, iselFloatToInt(c, vr, in, false)
	case ir.VSCvtSatF32, ir.VSCvtSatF64:
		return true, iselFloatToIntSat(c, vr, in, true)
	case ir.VUCvtSatF32, ir.VUCvtSatF64:
		return true, iselFloatToIntSat(c, vr, in, false)
	case ir.VFCvtF32, ir.VFCvtF64:
		return true, iselFloatWidth(c, vr, in)
	case ir.VBitcastF32, ir.VBitcastF64:
		return true, iselBitcast(c, vr, in, true)
	case ir.VBitcastI32, ir.VBitcastI64:
		return true, iselBitcast(c, vr, in, false)
	}
	return false, nil
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
	c.Emit(mir.Instr{
		Op:   fbinOp{verb: in.Op().Verb, w: vr.widthOfVReg(dst)},
		Defs: []mir.VReg{dst},
		Uses: ops,
	})
	return nil
}

func iselFloatUnary(c *cursor, vr *vregs, in *ir.Inst) error {
	ops, err := operands(vr, in, 1)
	if err != nil {
		return err
	}
	dst, err := vr.define(in.Result(0))
	if err != nil {
		return fmt.Errorf("%s: %w", in.Op(), err)
	}
	c.Emit(mir.Instr{
		Op:   funOp{verb: in.Op().Verb, w: vr.widthOfVReg(dst)},
		Defs: []mir.VReg{dst},
		Uses: ops,
	})
	return nil
}

// iselFMA lowers §A3's fma, which rounds once. FMADD is that instruction:
// the product is not rounded before the addend joins it.
func iselFMA(c *cursor, vr *vregs, in *ir.Inst) error {
	ops, err := operands(vr, in, 3)
	if err != nil {
		return err
	}
	dst, err := vr.define(in.Result(0))
	if err != nil {
		return fmt.Errorf("%s: %w", in.Op(), err)
	}
	c.Emit(mir.Instr{
		Op:   fmaOp{w: vr.widthOfVReg(dst)},
		Defs: []mir.VReg{dst},
		Uses: ops,
	})
	return nil
}

// iselCopySign lowers §A3's copysign through the integer file.
//
// A64 has no scalar copysign. The vector unit has BSL, which would do it in
// one instruction given a mask register to hold; through the integer file it
// is two FMOVs out, three data-processing instructions and one FMOV back,
// with no vector-lane operand for this package to grow a notion of.
func iselCopySign(c *cursor, vr *vregs, in *ir.Inst) error {
	ops, err := operands(vr, in, 2)
	if err != nil {
		return err
	}
	dst, err := vr.define(in.Result(0))
	if err != nil {
		return fmt.Errorf("%s: %w", in.Op(), err)
	}
	w := vr.widthOfVReg(dst)

	iw := w32
	var rest, sign int64 = 0x7fffffff, -0x80000000
	if w == wf64 {
		iw = w64
		rest, sign = 0x7fffffffffffffff, math.MinInt64
	}

	magBits := vr.temp(iw)
	signBits := vr.temp(iw)
	c.Emit(mir.Instr{Op: floatToBitsOp{w: w}, Defs: []mir.VReg{magBits}, Uses: []mir.VReg{ops[0]}})
	c.Emit(mir.Instr{Op: floatToBitsOp{w: w}, Defs: []mir.VReg{signBits}, Uses: []mir.VReg{ops[1]}})

	restMask := vr.temp(iw)
	signMask := vr.temp(iw)
	c.Emit(mir.Instr{Op: constOp{imm: rest, w: iw}, Defs: []mir.VReg{restMask}})
	c.Emit(mir.Instr{Op: constOp{imm: sign, w: iw}, Defs: []mir.VReg{signMask}})

	mag := vr.temp(iw)
	sgn := vr.temp(iw)
	both := vr.temp(iw)
	c.Emit(mir.Instr{Op: aluOp{verb: ir.VAnd, w: iw}, Defs: []mir.VReg{mag}, Uses: []mir.VReg{magBits, restMask}})
	c.Emit(mir.Instr{Op: aluOp{verb: ir.VAnd, w: iw}, Defs: []mir.VReg{sgn}, Uses: []mir.VReg{signBits, signMask}})
	c.Emit(mir.Instr{Op: aluOp{verb: ir.VOr, w: iw}, Defs: []mir.VReg{both}, Uses: []mir.VReg{mag, sgn}})

	c.Emit(mir.Instr{Op: bitsToFloatOp{w: w}, Defs: []mir.VReg{dst}, Uses: []mir.VReg{both}})
	return nil
}

// iselFloatCompare lowers §B's float comparisons: FCMP sets the flags and
// CSET reads them into the i1 the comparison is.
func iselFloatCompare(c *cursor, vr *vregs, in *ir.Inst, cond condCode) error {
	ops, err := operands(vr, in, 2)
	if err != nil {
		return err
	}
	dst, err := vr.define(in.Result(0))
	if err != nil {
		return fmt.Errorf("%s: %w", in.Op(), err)
	}
	c.Emit(mir.Instr{Op: fcmpOp{w: vr.widthOfVReg(ops[0])}, Uses: ops})
	c.Emit(mir.Instr{Op: csetOp{cond: cond}, Defs: []mir.VReg{dst}})
	return nil
}

// iselFloatConst materializes a float literal into dst.
//
// A64 has FMOV with an immediate, but it reaches only the 256 values an
// eight-bit exponent-and-mantissa encoding names, and a literal that is not
// one of them would need a constant pool and the relocation to reach it.
// Every literal goes the same way instead: the bit pattern into an integer
// register the way any wide constant goes, then one FMOV across the files.
func iselFloatConst(c *cursor, vr *vregs, dst mir.VReg, w width, v float64) {
	iw, bits := w64, int64(math.Float64bits(v))
	if w == wf32 {
		iw, bits = w32, int64(math.Float32bits(float32(v)))
	}
	tmp := vr.temp(iw)
	c.Emit(mir.Instr{Op: constOp{imm: bits, w: iw}, Defs: []mir.VReg{tmp}})
	c.Emit(mir.Instr{Op: bitsToFloatOp{w: w}, Defs: []mir.VReg{dst}, Uses: []mir.VReg{tmp}})
}

// iselIntToFloat lowers §C2's int-to-float conversions: SCVTF and UCVTF,
// which are exact where the destination can hold the value and
// round-to-nearest-even where it cannot, which is what §C2 asks for.
func iselIntToFloat(c *cursor, vr *vregs, in *ir.Inst, signed bool) error {
	ops, err := operands(vr, in, 1)
	if err != nil {
		return err
	}
	dst, err := vr.define(in.Result(0))
	if err != nil {
		return fmt.Errorf("%s: %w", in.Op(), err)
	}
	w := vr.widthOfVReg(dst)
	if !w.isFloat() {
		return fmt.Errorf("%s: the destination is not a float", in.Op())
	}
	c.Emit(mir.Instr{
		Op:   cvtIntToFloatOp{signed: signed, from: vr.widthOfVReg(ops[0]), w: w},
		Defs: []mir.VReg{dst},
		Uses: ops,
	})
	return nil
}

// iselFloatToIntSat lowers §C2's saturating float-to-int conversions.
//
// One instruction. FCVTZS and FCVTZU round toward zero, clamp an
// out-of-range value to the endpoint it passed, and give zero for a NaN —
// which is `_sat_`'s specification word for word. The trapping forms are the
// ones that cost something here, which is the reverse of the other
// architecture.
func iselFloatToIntSat(c *cursor, vr *vregs, in *ir.Inst, signed bool) error {
	ops, err := operands(vr, in, 1)
	if err != nil {
		return err
	}
	dst, err := vr.define(in.Result(0))
	if err != nil {
		return fmt.Errorf("%s: %w", in.Op(), err)
	}
	c.Emit(mir.Instr{
		Op:   cvtFloatToIntOp{signed: signed, from: vr.widthOfVReg(ops[0]), w: vr.widthOfVReg(dst)},
		Defs: []mir.VReg{dst},
		Uses: ops,
	})
	return nil
}

// f2iKey is a source float width, a destination integer width, and whether
// the destination is signed.
type f2iKey struct {
	from   width
	to     width
	signed bool
}

// f2iRange is the interval of source values one trapping §C2 conversion
// admits: valid is `lo <= x < hi`, or `lo < x < hi` where loStrict.
//
// Every bound is a power of two or one less than one, and so exactly
// representable in both source formats — which is what makes the comparison
// against it exact and the interval the true one rather than a conservative
// one. The strict bounds are where the exact endpoint is a value that would
// itself convert in range: −2147483649 is out of an i32 by a whole unit, so
// admitting everything strictly above it admits −2147483648.5, which
// truncates to −2147483648 and belongs.
var f2iRange = map[f2iKey]struct {
	lo, hi   float64
	loStrict bool
}{
	{wf32, w32, true}: {lo: -2147483648, hi: 2147483648},
	{wf64, w32, true}: {lo: -2147483649, hi: 2147483648, loStrict: true},
	{wf32, w64, true}: {lo: -9223372036854775808, hi: 9223372036854775808},
	{wf64, w64, true}: {lo: -9223372036854775808, hi: 9223372036854775808},

	{wf32, w32, false}: {lo: -1, hi: 4294967296, loStrict: true},
	{wf64, w32, false}: {lo: -1, hi: 4294967296, loStrict: true},
	{wf32, w64, false}: {lo: -1, hi: 18446744073709551616, loStrict: true},
	{wf64, w64, false}: {lo: -1, hi: 18446744073709551616, loStrict: true},
}

// iselFloatToInt lowers §C2's trapping float-to-integer conversions.
//
// FCVTZS does not trap — it saturates, which is the other verb's answer — so
// the trap §C2 asks for is a range check this package emits in front of it.
// Two compares against exact bounds, each branching to a BRK.
//
// The conditions are chosen so that the unordered case satisfies both, which
// is what makes a NaN trap without a third comparison: LT and LE are true for
// a NaN because it leaves N clear and V set, and PL is true for a NaN because
// it leaves N clear. Neither bound is compared the other way round, which is
// the difference from the amd64 backend — there the unordered case has to be
// steered into the condition by swapping the operands.
func iselFloatToInt(c *cursor, vr *vregs, in *ir.Inst, signed bool) error {
	op := in.Op()
	ops, err := operands(vr, in, 1)
	if err != nil {
		return err
	}
	src := ops[0]
	dst, err := vr.define(in.Result(0))
	if err != nil {
		return fmt.Errorf("%s: %w", op, err)
	}
	from, to := vr.widthOfVReg(src), vr.widthOfVReg(dst)

	r, ok := f2iRange[f2iKey{from: from, to: to, signed: signed}]
	if !ok {
		return fmt.Errorf("%s: no range check for this pair of widths", op)
	}

	trap := c.open("trap")

	// Below the interval, or unordered.
	lo := vr.temp(from)
	iselFloatConst(c, vr, lo, from, r.lo)
	below := condLT
	if r.loStrict {
		below = condLE
	}
	inRange := c.open("inrange")
	c.Emit(mir.Instr{Op: fcmpOp{w: from}, Uses: []mir.VReg{src, lo}})
	c.branch(below, trap, inRange)

	// At or above it, or unordered.
	hi := vr.temp(from)
	iselFloatConst(c, vr, hi, from, r.hi)
	convert := c.open("cvt")
	c.Emit(mir.Instr{Op: fcmpOp{w: from}, Uses: []mir.VReg{src, hi}})
	c.branch(condPL, trap, convert)

	// BRK, and nothing after it. A trap is a terminator with no
	// successors: control does not leave the instruction, so there is no
	// frame to tear down and nothing to fall through to.
	trap.Emit(mir.Instr{Op: trapOp{}})

	c.Emit(mir.Instr{
		Op:   cvtFloatToIntOp{signed: signed, from: from, w: to},
		Defs: []mir.VReg{dst},
		Uses: []mir.VReg{src},
	})
	return nil
}

// iselFloatWidth lowers §C3's fcvt between the two float widths.
func iselFloatWidth(c *cursor, vr *vregs, in *ir.Inst) error {
	ops, err := operands(vr, in, 1)
	if err != nil {
		return err
	}
	dst, err := vr.define(in.Result(0))
	if err != nil {
		return fmt.Errorf("%s: %w", in.Op(), err)
	}
	from, to := vr.widthOfVReg(ops[0]), vr.widthOfVReg(dst)
	if !from.isFloat() || !to.isFloat() {
		return fmt.Errorf("%s: fcvt is between two floats", in.Op())
	}
	c.Emit(mir.Instr{
		Op:   cvtFloatOp{from: from, w: to},
		Defs: []mir.VReg{dst},
		Uses: ops,
	})
	return nil
}

// iselBitcast lowers §C3's bitcast, which is FMOV across the two register
// files: the same bits, read as the other kind of thing, with no conversion
// and no rounding.
func iselBitcast(c *cursor, vr *vregs, in *ir.Inst, toInt bool) error {
	ops, err := operands(vr, in, 1)
	if err != nil {
		return err
	}
	dst, err := vr.define(in.Result(0))
	if err != nil {
		return fmt.Errorf("%s: %w", in.Op(), err)
	}
	if toInt {
		c.Emit(mir.Instr{
			Op:   floatToBitsOp{w: vr.widthOfVReg(ops[0])},
			Defs: []mir.VReg{dst}, Uses: ops,
		})
		return nil
	}
	c.Emit(mir.Instr{
		Op:   bitsToFloatOp{w: vr.widthOfVReg(dst)},
		Defs: []mir.VReg{dst}, Uses: ops,
	})
	return nil
}
