package ptx

// Instruction selection: one VIR instruction to one or several PTX
// instructions, in a block whose registers are already the values' own.
//
// The rows that are more than one instruction are the ones where PTX and
// §0 disagree about what happens at the edge: division by zero, a
// conversion out of range, a shift count past the width, a rotate wider
// than shf reaches. Each is spelled out here rather than left to the
// hardware's answer, because the hardware's answer is not the one the IR
// promised.

import (
	"fmt"
	"math"

	"github.com/vertex-language/ir"
	"github.com/vertex-language/ir/lower/globals"
	"github.com/vertex-language/ptx"
)

func alignUp(n, a uint64) uint64 {
	if a == 0 {
		return n
	}
	return (n + a - 1) &^ (a - 1)
}

// block emits one VIR block: its label, its instructions, its terminator.
func (x *fn) block(blk *ir.Block) error {
	x.b.Bind(x.labels[blk])
	for _, in := range blk.Insts() {
		if err := x.inst(in); err != nil {
			return fmt.Errorf("@%s: %s: %w", blk.Label(), in.Op(), err)
		}
	}
	if err := x.term(blk.Term()); err != nil {
		return fmt.Errorf("@%s: %s: %w", blk.Label(), blk.Term().Op(), err)
	}
	return nil
}

func (x *fn) inst(in *ir.Inst) error {
	op := in.Op()
	if op.IsBare() {
		return x.bare(in)
	}
	t := op.Type
	switch op.Verb {
	// —— §A7 ——
	case ir.VConst:
		return x.constant(in)

	// —— §A, §A3 ——
	case ir.VAdd, ir.VSub, ir.VMul:
		return x.arith(in)
	case ir.VSMulHi:
		x.b.Mul(signedT(t), x.res(in), x.arg(in, 0), x.arg(in, 1), ptx.MulHi)
	case ir.VUMulHi:
		x.b.Mul(unsignedT(t), x.res(in), x.arg(in, 0), x.arg(in, 1), ptx.MulHi)
	case ir.VSDiv, ir.VUDiv, ir.VSRem, ir.VURem:
		return x.divide(in)
	case ir.VNeg:
		if t.IsFloat() {
			x.b.Neg(floatT(t), x.res(in), x.arg(in, 0))
		} else {
			x.b.Neg(signedT(t), x.res(in), x.arg(in, 0))
		}
	case ir.VDiv:
		x.b.Div(floatT(t), x.res(in), x.arg(in, 0), x.arg(in, 1), ptx.RN)
	case ir.VFMA:
		x.b.Fma(floatT(t), x.res(in), x.arg(in, 0), x.arg(in, 1), x.arg(in, 2), ptx.RN)
	case ir.VAbs:
		x.b.Abs(floatT(t), x.res(in), x.arg(in, 0))
	case ir.VSqrt:
		x.b.Sqrt(floatT(t), x.res(in), x.arg(in, 0), ptx.RN)
	case ir.VMinNum:
		x.b.Min(floatT(t), x.res(in), x.arg(in, 0), x.arg(in, 1))
	case ir.VMaxNum:
		x.b.Max(floatT(t), x.res(in), x.arg(in, 0), x.arg(in, 1))
	case ir.VMinimum, ir.VMaximum:
		return x.minimum(in)
	case ir.VCopySign:
		// PTX's copysign takes the sign first and the magnitude second;
		// C's, and VIR's, take them the other way round.
		x.b.Copysign(floatT(t), x.res(in), x.arg(in, 1), x.arg(in, 0))
	case ir.VCeil:
		x.b.Cvt(floatT(t), floatT(t), x.res(in), x.arg(in, 0), ptx.RPI)
	case ir.VFloor:
		x.b.Cvt(floatT(t), floatT(t), x.res(in), x.arg(in, 0), ptx.RMI)
	case ir.VTrunc:
		x.b.Cvt(floatT(t), floatT(t), x.res(in), x.arg(in, 0), ptx.RZI)
	case ir.VNearest:
		x.b.Cvt(floatT(t), floatT(t), x.res(in), x.arg(in, 0), ptx.RNI)
	case ir.VRcpApprox:
		x.b.Rcp(ptx.F32, x.res(in), x.arg(in, 0), ptx.Approx)
	case ir.VRsqrtApprox:
		x.b.Rsqrt(ptx.F32, x.res(in), x.arg(in, 0), ptx.Approx)
	case ir.VExp2Approx:
		x.b.Ex2(ptx.F32, x.res(in), x.arg(in, 0), ptx.Approx)
	case ir.VLog2Approx:
		x.b.Lg2(ptx.F32, x.res(in), x.arg(in, 0), ptx.Approx)
	case ir.VSinApprox:
		x.b.Sin(ptx.F32, x.res(in), x.arg(in, 0), ptx.Approx)
	case ir.VCosApprox:
		x.b.Cos(ptx.F32, x.res(in), x.arg(in, 0), ptx.Approx)

	// —— §A2 ——
	case ir.VSAddO, ir.VUAddO, ir.VSSubO, ir.VSMulO, ir.VUMulO:
		return x.overflow(in)

	// —— §A4 ——
	case ir.VNot:
		x.b.Not(bitsT(t), x.res(in), x.arg(in, 0))
	case ir.VAnd:
		x.b.And(bitsT(t), x.res(in), x.arg(in, 0), x.arg(in, 1))
	case ir.VOr:
		x.b.Or(bitsT(t), x.res(in), x.arg(in, 0), x.arg(in, 1))
	case ir.VXor:
		x.b.Xor(bitsT(t), x.res(in), x.arg(in, 0), x.arg(in, 1))

	// —— §A5 ——
	case ir.VShl, ir.VSShr, ir.VUShr:
		return x.shift(in)
	case ir.VRotL, ir.VRotR:
		return x.rotate(in)

	// —— §A6 ——
	case ir.VClz, ir.VCtz, ir.VPopcnt, ir.VBswap:
		return x.bits(in)

	// —— §B ——
	case ir.VEq, ir.VNe, ir.VSLt, ir.VULt, ir.VSLe, ir.VULe, ir.VLt, ir.VLe, ir.VUno:
		return x.compare(in)

	// —— §C–§C4 ——
	case ir.VWrapI64:
		x.b.Cvt(ptx.U32, ptx.U64, x.res(in), x.arg(in, 0))
	case ir.VSExtI32:
		x.b.Cvt(ptx.S64, ptx.S32, x.res(in), x.arg(in, 0))
	case ir.VZExtI32:
		x.b.Cvt(ptx.U64, ptx.U32, x.res(in), x.arg(in, 0))
	case ir.VZExtI1:
		x.b.Selp(unsignedT(t), x.res(in), ptx.Imm(1), ptx.Imm(0), x.arg(in, 0))
	case ir.VSCvtI32:
		x.b.Cvt(floatT(t), ptx.S32, x.res(in), x.arg(in, 0), ptx.RN)
	case ir.VSCvtI64:
		x.b.Cvt(floatT(t), ptx.S64, x.res(in), x.arg(in, 0), ptx.RN)
	case ir.VUCvtI32:
		x.b.Cvt(floatT(t), ptx.U32, x.res(in), x.arg(in, 0), ptx.RN)
	case ir.VUCvtI64:
		x.b.Cvt(floatT(t), ptx.U64, x.res(in), x.arg(in, 0), ptx.RN)
	case ir.VSCvtF32, ir.VSCvtF64, ir.VUCvtF32, ir.VUCvtF64:
		return x.floatToInt(in, true)
	case ir.VSCvtSatF32, ir.VSCvtSatF64, ir.VUCvtSatF32, ir.VUCvtSatF64:
		return x.floatToInt(in, false)
	case ir.VFCvtF32:
		x.b.Cvt(ptx.F64, ptx.F32, x.res(in), x.arg(in, 0))
	case ir.VFCvtF64:
		x.b.Cvt(ptx.F32, ptx.F64, x.res(in), x.arg(in, 0), ptx.RN)
	case ir.VBitcastF32, ir.VBitcastI32:
		x.b.Mov(ptx.B32, x.res(in), x.arg(in, 0))
	case ir.VBitcastF64, ir.VBitcastI64:
		x.b.Mov(ptx.B64, x.res(in), x.arg(in, 0))
	case ir.VFromI64, ir.VFromPtr:
		x.b.Mov(ptx.B64, x.res(in), x.arg(in, 0))

	// —— §D, §D2 ——
	case ir.VLoad, ir.VSLoad8, ir.VSLoad16, ir.VSLoad32, ir.VULoad8, ir.VULoad16, ir.VULoad32:
		return x.load(in)
	case ir.VStore, ir.VStore8, ir.VStore16, ir.VStore32:
		return x.store(in)

	// —— §D3 ——
	case ir.VAlloc:
		off, ok := x.allocAt[in]
		if !ok {
			return fmt.Errorf("ptr.alloc outside the entry block")
		}
		x.b.Add(ptx.U64, x.res(in), x.depot, ptx.Imm(off))
		if in.Zeroed() {
			size, _, _ := allocShape(in)
			x.zero(x.res(in), size)
		}
	case ir.VAlloca:
		if err := x.l.needSM(op, 52, "alloca"); err != nil {
			return err
		}
		align, _ := in.Align()
		if align == 0 {
			align = 1
		}
		x.b.Alloca(ptx.U64, x.res(in), x.arg(in, 0), ptx.Imm(int64(align)))
		if in.Zeroed() {
			return fmt.Errorf("a zeroed ptr.alloca is not lowered yet; the size is a value")
		}
	case ir.VStackSave:
		if err := x.l.needSM(op, 52, "stacksave"); err != nil {
			return err
		}
		x.b.StackSave(ptx.U64, x.res(in))
	case ir.VStackRestore:
		if err := x.l.needSM(op, 52, "stackrestore"); err != nil {
			return err
		}
		x.b.StackRestore(ptx.U64, x.arg(in, 0))
	case ir.VGetAddr:
		return x.getaddr(in)
	case ir.VDiff:
		x.b.Sub(ptx.S64, x.res(in), x.arg(in, 0), x.arg(in, 1))
	case ir.VTLSAddr, ir.VBlockAddr, ir.VFrameAddr, ir.VReturnAddr:
		return fmt.Errorf("no PTX equivalent")

	// —— §F ——
	case ir.VSelect:
		return x.selectV(in)

	// —— §H ——
	case ir.VAtomicLoad, ir.VAtomicStore, ir.VAtomicCas,
		ir.VAtomicRmwAdd, ir.VAtomicRmwSub, ir.VAtomicRmwAnd, ir.VAtomicRmwOr,
		ir.VAtomicRmwXor, ir.VAtomicRmwXchg,
		ir.VAtomicRmwSMin, ir.VAtomicRmwSMax, ir.VAtomicRmwUMin, ir.VAtomicRmwUMax:
		return x.atomic(in)
	case ir.VAtomicULoad8, ir.VAtomicULoad16, ir.VAtomicStore8, ir.VAtomicStore16,
		ir.VAtomicCas8, ir.VAtomicCas16,
		ir.VAtomicRmwAdd8, ir.VAtomicRmwSub8, ir.VAtomicRmwAnd8, ir.VAtomicRmwOr8,
		ir.VAtomicRmwXor8, ir.VAtomicRmwXchg8,
		ir.VAtomicRmwAdd16, ir.VAtomicRmwSub16, ir.VAtomicRmwAnd16, ir.VAtomicRmwOr16,
		ir.VAtomicRmwXor16, ir.VAtomicRmwXchg16:
		return fmt.Errorf("narrow atomics are a compare-and-swap loop on the containing word, which is not lowered yet")

	// —— §I ——
	case ir.VVaArg, ir.VVaArgRef:
		return fmt.Errorf("PTX has no va_list")

	// —— §W ——
	case ir.VWorkitemID, ir.VWorkgroupID, ir.VWorkgroupSize, ir.VNumWorkgroups,
		ir.VLaneID, ir.VWaveSize:
		return x.workitem(in)
	case ir.VWaveShflIdx, ir.VWaveShflUp, ir.VWaveShflDown, ir.VWaveShflXor,
		ir.VWaveReadFirstLane, ir.VWaveBallot, ir.VWaveAny, ir.VWaveAll:
		return x.wave(in)

	default:
		return fmt.Errorf("not lowered")
	}
	return nil
}

// bare is §17's bare set, less the terminators.
func (x *fn) bare(in *ir.Inst) error {
	switch in.Op().Verb {
	case ir.VCall, ir.VCallInd:
		return x.call(in)
	case ir.VFence:
		return x.fence(in)
	case ir.VBarrier:
		x.b.BarSync(ptx.Imm(0))
	case ir.VMemCpy, ir.VMemMove, ir.VMemSet, ir.VMemCmp:
		return x.bulk(in)
	case ir.VAsm:
		return x.asm(in)
	case ir.VVaStart, ir.VVaEnd, ir.VVaCopy:
		return fmt.Errorf("PTX has no va_list")
	default:
		return fmt.Errorf("not lowered")
	}
	return nil
}

// —— constants ——

func (x *fn) constant(in *ir.Inst) error {
	t := in.Op().Type
	lit, _ := in.Lit()
	d := x.res(in)
	switch t {
	case ir.TypeF32:
		if lit.Kind() != ir.ConstFloat {
			return fmt.Errorf("a float constant needs a float literal")
		}
		x.b.Mov(ptx.F32, d, ptx.F32Imm(float32(lit.Float())))
		return nil
	case ir.TypeF64:
		if lit.Kind() != ir.ConstFloat {
			return fmt.Errorf("a float constant needs a float literal")
		}
		x.b.Mov(ptx.F64, d, ptx.F64Imm(lit.Float()))
		return nil
	}
	v, err := globals.ConstInt(layout{}, lit)
	if err != nil {
		return err
	}
	switch t {
	case ir.TypeI1:
		// There is no predicate immediate. A register holding the bit,
		// compared against zero, is the shortest spelling ptxas takes.
		r := x.temp(ptx.B32)
		x.b.Mov(ptx.B32, r, ptx.Imm(v&1))
		x.b.Setp(ptx.B32, ptx.Ne, d, r, ptx.Imm(0))
	case ir.TypeI32:
		x.b.Mov(ptx.B32, d, ptx.Imm(int64(int32(v))))
	default:
		x.b.Mov(ptx.B64, d, ptx.Imm(v))
	}
	return nil
}

// imm32 is a 32-bit immediate operand.
func imm32(v int32) ptx.Operand { return ptx.Imm(int64(v)) }

// —— §A ——

func (x *fn) arith(in *ir.Inst) error {
	t := in.Op().Type
	d, a, c := x.res(in), x.arg(in, 0), x.arg(in, 1)
	if t.IsFloat() {
		switch in.Op().Verb {
		case ir.VAdd:
			x.b.Add(floatT(t), d, a, c, ptx.RN)
		case ir.VSub:
			x.b.Sub(floatT(t), d, a, c, ptx.RN)
		case ir.VMul:
			x.b.Mul(floatT(t), d, a, c, ptx.RN)
		}
		return nil
	}
	switch in.Op().Verb {
	case ir.VAdd:
		x.b.Add(signedT(t), d, a, c)
	case ir.VSub:
		x.b.Sub(signedT(t), d, a, c)
	case ir.VMul:
		x.b.Mul(signedT(t), d, a, c, ptx.MulLo)
	}
	return nil
}

// divide is §A's four division rows with §0's traps ahead of them: a GPU
// does not fault on a zero divisor, and PTX's div gives an unspecified
// answer for one.
func (x *fn) divide(in *ir.Inst) error {
	t := in.Op().Type
	verb := in.Op().Verb
	d, a, c := x.res(in), x.arg(in, 0), x.arg(in, 1)
	signed := verb == ir.VSDiv || verb == ir.VSRem

	zero := x.temp(ptx.Pred)
	x.b.Setp(bitsT(t), ptx.Eq, zero, c, ptx.Imm(0))
	x.b.Trap().If(zero)

	if signed {
		// INT_MIN / -1 overflows, and PTX's answer to it is whatever the
		// hardware does, which on some generations is a hang.
		minv := int64(math.MinInt32)
		if t == ir.TypeI64 {
			minv = math.MinInt64
		}
		pa, pb := x.temp(ptx.Pred), x.temp(ptx.Pred)
		x.b.Setp(signedT(t), ptx.Eq, pa, a, ptx.Imm(minv))
		x.b.Setp(signedT(t), ptx.Eq, pb, c, ptx.Imm(-1))
		x.b.And(ptx.Pred, pa, pa, pb)
		x.b.Trap().If(pa)
	}

	ty := unsignedT(t)
	if signed {
		ty = signedT(t)
	}
	if verb == ir.VSDiv || verb == ir.VUDiv {
		x.b.Div(ty, d, a, c)
	} else {
		x.b.Rem(ty, d, a, c)
	}
	return nil
}

// minimum is IEEE-754-2019 minimum/maximum: a NaN operand yields NaN.
// sm_80 has it as a qualifier; below that it is min.f32, which discards
// the NaN, corrected by a select.
func (x *fn) minimum(in *ir.Inst) error {
	t := in.Op().Type
	d, a, c := x.res(in), x.arg(in, 0), x.arg(in, 1)
	ft := floatT(t)
	if x.l.opts.sm().SM >= 80 {
		if in.Op().Verb == ir.VMinimum {
			x.b.Min(ft, d, a, c, ptx.NaN)
		} else {
			x.b.Max(ft, d, a, c, ptx.NaN)
		}
		return nil
	}
	r := x.temp(ft)
	if in.Op().Verb == ir.VMinimum {
		x.b.Min(ft, r, a, c)
	} else {
		x.b.Max(ft, r, a, c)
	}
	nan := x.temp(ptx.Pred)
	x.b.Setp(ft, ptx.Nan, nan, a, c)
	q := x.temp(ft)
	if t == ir.TypeF32 {
		x.b.Mov(ptx.F32, q, ptx.F32Imm(float32(math.NaN())))
	} else {
		x.b.Mov(ptx.F64, q, ptx.F64Imm(math.NaN()))
	}
	x.b.Selp(ft, d, q, r, nan)
	return nil
}

// —— §A2 ——

func (x *fn) overflow(in *ir.Inst) error {
	t := in.Op().Type
	p, a, c := x.res(in), x.arg(in, 0), x.arg(in, 1)
	bt, st := bitsT(t), signedT(t)
	switch in.Op().Verb {
	case ir.VSAddO:
		// Signed overflow of a+b is when a and b agree in sign and the
		// sum does not: (a^r)&(b^r) has its top bit set.
		r, t1, t2 := x.temp(bt), x.temp(bt), x.temp(bt)
		x.b.Add(st, r, a, c)
		x.b.Xor(bt, t1, a, r)
		x.b.Xor(bt, t2, c, r)
		x.b.And(bt, t1, t1, t2)
		x.b.Setp(st, ptx.Lt, p, t1, ptx.Imm(0))
	case ir.VSSubO:
		// Signed overflow of a-b is when a and b differ in sign and the
		// difference disagrees with a: (a^b)&(a^r).
		r, t1, t2 := x.temp(bt), x.temp(bt), x.temp(bt)
		x.b.Sub(st, r, a, c)
		x.b.Xor(bt, t1, a, c)
		x.b.Xor(bt, t2, a, r)
		x.b.And(bt, t1, t1, t2)
		x.b.Setp(st, ptx.Lt, p, t1, ptx.Imm(0))
	case ir.VUAddO:
		// Carry out is r < a, which needs only the inputs and the sum.
		r := x.temp(bt)
		x.b.Add(unsignedT(t), r, a, c)
		x.b.Setp(unsignedT(t), ptx.Lo, p, r, a)
	case ir.VUMulO:
		h := x.temp(bt)
		x.b.Mul(unsignedT(t), h, a, c, ptx.MulHi)
		x.b.Setp(unsignedT(t), ptx.Ne, p, h, ptx.Imm(0))
	case ir.VSMulO:
		// The high half of the signed product is the sign extension of
		// the low half exactly when nothing overflowed.
		h, lo, s := x.temp(bt), x.temp(bt), x.temp(bt)
		x.b.Mul(st, h, a, c, ptx.MulHi)
		x.b.Mul(st, lo, a, c, ptx.MulLo)
		x.b.Shr(st, s, lo, ptx.Imm(int64(width(t)-1)))
		x.b.Setp(st, ptx.Ne, p, h, s)
	}
	return nil
}

// —— §A5 ——

// shift masks the count to the width, as §0 requires and PTX does not:
// PTX clamps a count past the width, which gives zero where VIR gives a
// shift by the count modulo the width. The count operand of every PTX
// shift is 32 bits, so an i64 count is masked and narrowed.
func (x *fn) shift(in *ir.Inst) error {
	t := in.Op().Type
	d, a := x.res(in), x.arg(in, 0)
	m := x.shiftCount(t, x.arg(in, 1))
	switch in.Op().Verb {
	case ir.VShl:
		x.b.Shl(bitsT(t), d, a, m)
	case ir.VSShr:
		x.b.Shr(signedT(t), d, a, m)
	case ir.VUShr:
		x.b.Shr(unsignedT(t), d, a, m)
	}
	return nil
}

// shiftCount is a shift count masked to the width and narrowed to the
// 32 bits every PTX shift takes its count as.
func (x *fn) shiftCount(t ir.RegType, n ptx.Reg) ptx.Reg {
	if t == ir.TypeI32 {
		m := x.temp(ptx.B32)
		x.b.And(ptx.B32, m, n, ptx.Imm(31))
		return m
	}
	m64, m := x.temp(ptx.B64), x.temp(ptx.B32)
	x.b.And(ptx.B64, m64, n, ptx.Imm(63))
	x.b.Cvt(ptx.U32, ptx.U64, m, m64)
	return m
}

// rotate is shf's funnel shift at 32 bits, which wraps the count itself,
// and three shifts at 64, where a count of zero shifts the other half by
// 64 and PTX's clamp makes that zero — which is the right answer.
func (x *fn) rotate(in *ir.Inst) error {
	t := in.Op().Type
	d, a, n := x.res(in), x.arg(in, 0), x.arg(in, 1)
	left := in.Op().Verb == ir.VRotL
	if t == ir.TypeI32 {
		if left {
			x.b.Shf(ptx.B32, d, a, a, n, ptx.DirL, ptx.Wrap)
		} else {
			x.b.Shf(ptx.B32, d, a, a, n, ptx.DirR, ptx.Wrap)
		}
		return nil
	}
	m := x.shiftCount(t, n)
	inv := x.temp(ptx.B32)
	x.b.Sub(ptx.U32, inv, ptx.Imm(64), m)
	lo, hi := x.temp(ptx.B64), x.temp(ptx.B64)
	if left {
		x.b.Shl(ptx.B64, lo, a, m)
		x.b.Shr(ptx.U64, hi, a, inv)
	} else {
		x.b.Shr(ptx.U64, lo, a, m)
		x.b.Shl(ptx.B64, hi, a, inv)
	}
	x.b.Or(ptx.B64, d, lo, hi)
	return nil
}

// —— §A6 ——

func (x *fn) bits(in *ir.Inst) error {
	t := in.Op().Type
	d, a := x.res(in), x.arg(in, 0)
	bt := bitsT(t)
	// clz and popc answer in 32 bits whatever the width they count.
	widen := func(r32 ptx.Reg) {
		if t == ir.TypeI64 {
			x.b.Cvt(ptx.U64, ptx.U32, d, r32)
		}
	}
	r := d
	if t == ir.TypeI64 {
		r = x.temp(ptx.B32)
	}
	switch in.Op().Verb {
	case ir.VClz:
		x.b.Clz(bt, r, a)
		widen(r)
	case ir.VCtz:
		rev := x.temp(bt)
		x.b.Brev(bt, rev, a)
		x.b.Clz(bt, r, rev)
		widen(r)
	case ir.VPopcnt:
		x.b.Popc(bt, r, a)
		widen(r)
	case ir.VBswap:
		if t == ir.TypeI32 {
			x.b.Prmt(ptx.B32, d, a, ptx.Imm(0), ptx.Imm(0x0123))
			return nil
		}
		lo, hi := x.temp(ptx.B32), x.temp(ptx.B32)
		x.b.Emit("mov.b64", []ptx.Operand{ptx.Vec(lo, hi)}, []ptx.Operand{a})
		lo2, hi2 := x.temp(ptx.B32), x.temp(ptx.B32)
		x.b.Prmt(ptx.B32, lo2, lo, ptx.Imm(0), ptx.Imm(0x0123))
		x.b.Prmt(ptx.B32, hi2, hi, ptx.Imm(0), ptx.Imm(0x0123))
		x.b.Emit("mov.b64", []ptx.Operand{d}, []ptx.Operand{ptx.Vec(hi2, lo2)})
	}
	return nil
}

// —— §B ——

func (x *fn) compare(in *ir.Inst) error {
	t := in.Op().Type
	p, a, c := x.res(in), x.arg(in, 0), x.arg(in, 1)
	var ty ptx.Type
	var cmp ptx.Cmp
	switch in.Op().Verb {
	case ir.VEq:
		ty, cmp = bitsT(t), ptx.Eq
		if t.IsFloat() {
			ty = floatT(t)
		}
	case ir.VNe:
		ty, cmp = bitsT(t), ptx.Ne
		if t.IsFloat() {
			ty, cmp = floatT(t), ptx.Neu
		}
	case ir.VSLt:
		ty, cmp = signedT(t), ptx.Lt
	case ir.VSLe:
		ty, cmp = signedT(t), ptx.Le
	case ir.VULt:
		ty, cmp = unsignedT(t), ptx.Lo
	case ir.VULe:
		ty, cmp = unsignedT(t), ptx.Ls
	case ir.VLt:
		ty, cmp = floatT(t), ptx.Lt
		if t == ir.TypePtr {
			ty, cmp = ptx.U64, ptx.Lo
		}
	case ir.VLe:
		ty, cmp = floatT(t), ptx.Le
		if t == ir.TypePtr {
			ty, cmp = ptx.U64, ptx.Ls
		}
	case ir.VUno:
		ty, cmp = floatT(t), ptx.Nan
	}
	x.b.Setp(ty, cmp, p, a, c)
	return nil
}

// —— §C2 ——

// floatToInt is cvt.rzi, which saturates and sends NaN to zero — what the
// _sat_ rows ask for, and what the trapping rows must test for first.
func (x *fn) floatToInt(in *ir.Inst, trapping bool) error {
	t := in.Op().Type
	d, a := x.res(in), x.arg(in, 0)
	src := in.Arg(0).Type()
	verb := in.Op().Verb
	signed := verb == ir.VSCvtF32 || verb == ir.VSCvtF64 || verb == ir.VSCvtSatF32 || verb == ir.VSCvtSatF64

	if trapping {
		// trunc(a) is in range exactly when a is in (lo-1, hi+1), and
		// the bounds below are the ones representable in the source
		// type that enclose that interval and nothing outside it.
		var lo, hi float64
		switch {
		case signed && t == ir.TypeI32:
			lo, hi = -2147483648, 2147483648 // lo inclusive (f32 has no -2^31-1), hi exclusive
		case signed:
			lo, hi = -9223372036854775808, 9223372036854775808
		case t == ir.TypeI32:
			lo, hi = -1, 4294967296 // lo exclusive
		default:
			lo, hi = -1, 18446744073709551616
		}
		loCmp := ptx.Ge
		if !signed || (src == ir.TypeF64 && t == ir.TypeI32) {
			// An f64 can hold -2147483649 exactly, so the signed i32
			// bound is exclusive there; the unsigned bound is -1
			// exclusive everywhere.
			loCmp = ptx.Gt
			if signed {
				lo = -2147483649
			}
		}
		ft := floatT(src)
		p1, p2 := x.temp(ptx.Pred), x.temp(ptx.Pred)
		x.b.Setp(ft, loCmp, p1, a, fimm(src, lo))
		x.b.Setp(ft, ptx.Lt, p2, a, fimm(src, hi))
		x.b.And(ptx.Pred, p1, p1, p2)
		x.b.Trap().IfNot(p1)
	}
	dt := unsignedT(t)
	if signed {
		dt = signedT(t)
	}
	x.b.Cvt(dt, floatT(src), d, a, ptx.RZI)
	return nil
}

// fimm is a float immediate in the source's own width.
func fimm(t ir.RegType, v float64) ptx.Operand {
	if t == ir.TypeF32 {
		return ptx.F32Imm(float32(v))
	}
	return ptx.F64Imm(v)
}

// —— §D ——

// loadShape is a §D or §D2 load's instruction type: the memory width and
// the extension it names.
func loadShape(t ir.RegType, v ir.Verb) ptx.Type {
	switch v {
	case ir.VSLoad8:
		return ptx.S8
	case ir.VSLoad16:
		return ptx.S16
	case ir.VSLoad32:
		return ptx.S32
	case ir.VULoad8:
		return ptx.U8
	case ir.VULoad16:
		return ptx.U16
	case ir.VULoad32:
		return ptx.U32
	}
	return memT(t)
}

func storeShape(t ir.RegType, v ir.Verb) ptx.Type {
	switch v {
	case ir.VStore8:
		return ptx.U8
	case ir.VStore16:
		return ptx.U16
	case ir.VStore32:
		return ptx.U32
	}
	return memT(t)
}

// checkAlign is §D4's align attribute. One below the natural alignment
// is refused: PTX has no unaligned access, and a byte-wise assembly of
// one is a sequence this package does not emit yet.
func checkAlign(in *ir.Inst, natural uint64) error {
	if a, ok := in.Align(); ok && a < natural {
		return fmt.Errorf("align %d below the natural %d; PTX has no unaligned access", a, natural)
	}
	return nil
}

func (x *fn) load(in *ir.Inst) error {
	t := in.Op().Type
	ty := loadShape(t, in.Op().Verb)
	if err := checkAlign(in, uint64(ty.Bits()/8)); err != nil {
		return err
	}
	d := x.res(in)
	if t == ir.TypeI1 {
		return fmt.Errorf("i1 has no storage width")
	}
	// A shared or const global whose address is visible right here is
	// loaded from its own space, which is faster and what every
	// tile[ty][tx] is. Anything else is a generic load.
	space, base := x.provenance(in.Arg(0))
	if in.Volatile() {
		x.b.Emit("ld.volatile"+space.String()+ty.String(), []ptx.Operand{d}, []ptx.Operand{base})
		return nil
	}
	x.b.Ld(ty, d, base, space)
	return nil
}

func (x *fn) store(in *ir.Inst) error {
	t := in.Op().Type
	ty := storeShape(t, in.Op().Verb)
	if err := checkAlign(in, uint64(ty.Bits()/8)); err != nil {
		return err
	}
	if t == ir.TypeI1 {
		return fmt.Errorf("i1 has no storage width")
	}
	v := x.arg(in, 0)
	space, base := x.provenance(in.Arg(1))
	if in.Volatile() {
		x.b.Emit("st.volatile"+space.String()+ty.String(), nil, []ptx.Operand{base, v})
		return nil
	}
	x.b.St(ty, base, v, space)
	return nil
}

// provenance is the state space an address is known to be in, and the
// operand that addresses it there. A generic pointer answers NoSpace and
// itself. The address of a global answers the global's space and the
// symbol — plus a constant where one was added, or through cvta.to where
// a value was, which is the fold that turns a tile[ty][tx] into
// ld.shared. A shared or global access through its own space skips the
// aperture check every generic access pays.
func (x *fn) provenance(p *ir.Def) (ptx.Space, ptx.Mem) {
	v, off, ok := x.symbolic(p, 0)
	switch {
	case !ok:
		return ptx.NoSpace, ptx.At(x.v(p))
	case v != nil:
		return v.Space, ptx.At(v, off)
	}
	// A symbol's space with a variable offset: the generic pointer,
	// converted back into the space it came from.
	sp := x.spaceOf(p)
	t := x.temp(ptx.B64)
	x.b.Cvta(ptx.U64, t, x.v(p), sp, ptx.To)
	return sp, ptx.At(t)
}

// symbolic follows ptr.add chains back to a ptr.getaddr. It answers the
// global and the constant offset where the whole chain is constant, a nil
// global with ok where some step was a value, and !ok for anything else.
func (x *fn) symbolic(p *ir.Def, depth int) (v *ptx.Var, off int64, ok bool) {
	in := p.Inst()
	if in == nil || depth > 8 {
		return nil, 0, false
	}
	switch in.Op().Verb {
	case ir.VGetAddr:
		v, ok := x.l.vars[in.Symbol()]
		if !ok || v.Linkage == ptx.Extern {
			return nil, 0, false
		}
		return v, 0, true
	case ir.VAdd, ir.VSub:
		if in.Op().Type != ir.TypePtr {
			return nil, 0, false
		}
		v, off, ok := x.symbolic(in.Arg(0), depth+1)
		if !ok {
			return nil, 0, false
		}
		c, isConst := constOf(in.Arg(1))
		if !isConst || v == nil {
			return nil, 0, true
		}
		if in.Op().Verb == ir.VSub {
			c = -c
		}
		return v, off + c, true
	}
	return nil, 0, false
}

// spaceOf is the space symbolic found, for the variable-offset case.
func (x *fn) spaceOf(p *ir.Def) ptx.Space {
	for in := p.Inst(); in != nil; in = in.Arg(0).Inst() {
		if in.Op().Verb == ir.VGetAddr {
			return x.l.vars[in.Symbol()].Space
		}
	}
	return ptx.NoSpace
}

// constOf is a value's integer literal, where it is an i64.const.
func constOf(d *ir.Def) (int64, bool) {
	in := d.Inst()
	if in == nil || in.Op().Verb != ir.VConst {
		return 0, false
	}
	lit, _ := in.Lit()
	if lit.Kind() != ir.ConstInt {
		return 0, false
	}
	return lit.Int(), true
}

// zero writes n zero bytes at p: the zeroed form of ptr.alloc.
func (x *fn) zero(p ptx.Reg, n uint64) {
	z64 := x.temp(ptx.B64)
	x.b.Mov(ptx.B64, z64, ptx.Imm(0))
	var off uint64
	for ; off+8 <= n; off += 8 {
		x.b.St(ptx.U64, ptx.At(p, int64(off)), z64)
	}
	if off < n {
		z32 := x.temp(ptx.B32)
		x.b.Mov(ptx.B32, z32, ptx.Imm(0))
		for ; off+4 <= n; off += 4 {
			x.b.St(ptx.U32, ptx.At(p, int64(off)), z32)
		}
		for ; off < n; off++ {
			x.b.St(ptx.U8, ptx.At(p, int64(off)), z32)
		}
	}
}

// getaddr is a symbol's address as a generic pointer. mov gives the
// address within the symbol's own state space; cvta makes it generic.
func (x *fn) getaddr(in *ir.Inst) error {
	d := x.res(in)
	sym := in.Symbol()
	if pf, ok := x.l.funcs[sym]; ok {
		x.b.Mov(ptx.U64, d, pf)
		return nil
	}
	if f, ok := sym.(*ir.Func); ok {
		if _, k := x.l.kernels[f]; k {
			return fmt.Errorf("@%s is a kernel; its address is the host's to take", sym.Name())
		}
	}
	v, ok := x.l.vars[sym]
	if !ok {
		return fmt.Errorf("@%s is not a symbol this module declared", sym.Name())
	}
	raw := x.temp(ptx.B64)
	x.b.Mov(ptx.U64, raw, v)
	x.b.Cvta(ptx.U64, d, raw, v.Space)
	return nil
}

// —— §F ——

func (x *fn) selectV(in *ir.Inst) error {
	t := in.Op().Type
	d, c, a, b := x.res(in), x.arg(in, 0), x.arg(in, 1), x.arg(in, 2)
	if t == ir.TypeI1 {
		// No selp.pred: (c & a) | (!c & b).
		t1, t2, nc := x.temp(ptx.Pred), x.temp(ptx.Pred), x.temp(ptx.Pred)
		x.b.And(ptx.Pred, t1, c, a)
		x.b.Not(ptx.Pred, nc, c)
		x.b.And(ptx.Pred, t2, nc, b)
		x.b.Or(ptx.Pred, d, t1, t2)
		return nil
	}
	ty := bitsT(t)
	if t.IsFloat() {
		ty = floatT(t)
	}
	x.b.Selp(ty, d, a, b, c)
	return nil
}

// —— §W ——

func (x *fn) workitem(in *ir.Inst) error {
	d := x.res(in)
	axis, _ := in.Axis()
	var s ptx.SReg
	switch in.Op().Verb {
	case ir.VWorkitemID:
		s = [3]ptx.SReg{ptx.TidX, ptx.TidY, ptx.TidZ}[axis]
	case ir.VWorkgroupID:
		s = [3]ptx.SReg{ptx.CtaIdX, ptx.CtaIdY, ptx.CtaIdZ}[axis]
	case ir.VWorkgroupSize:
		s = [3]ptx.SReg{ptx.NTidX, ptx.NTidY, ptx.NTidZ}[axis]
	case ir.VNumWorkgroups:
		s = [3]ptx.SReg{ptx.NCtaIdX, ptx.NCtaIdY, ptx.NCtaIdZ}[axis]
	case ir.VLaneID:
		s = ptx.LaneID
	case ir.VWaveSize:
		s = ptx.WarpSz
	}
	x.b.MovSReg(d, s)
	return nil
}

func (x *fn) wave(in *ir.Inst) error {
	op := in.Op()
	if err := x.l.needSM(op, 30, "the wave verbs"); err != nil {
		return err
	}
	d := x.res(in)
	switch op.Verb {
	case ir.VWaveShflIdx, ir.VWaveShflUp, ir.VWaveShflDown, ir.VWaveShflXor:
		val, lane, mask := x.arg(in, 0), x.arg(in, 1), x.arg(in, 2)
		var mode ptx.ShflMode
		clamp := ptx.Imm(0x1f)
		switch op.Verb {
		case ir.VWaveShflIdx:
			mode = ptx.ShflIdx
		case ir.VWaveShflUp:
			mode, clamp = ptx.ShflUp, ptx.Imm(0)
		case ir.VWaveShflDown:
			mode = ptx.ShflDown
		case ir.VWaveShflXor:
			mode = ptx.ShflBfly
		}
		x.b.ShflSync(ptx.B32, d, val, lane, clamp, mask, mode)
	case ir.VWaveReadFirstLane:
		// The lowest active lane: the count of trailing zeros of the
		// active mask, read through shfl.idx under that same mask.
		m, rev, first := x.temp(ptx.B32), x.temp(ptx.B32), x.temp(ptx.B32)
		x.b.Activemask(ptx.B32, m)
		x.b.Brev(ptx.B32, rev, m)
		x.b.Clz(ptx.B32, first, rev)
		x.b.ShflSync(ptx.B32, d, x.arg(in, 0), first, ptx.Imm(0x1f), m, ptx.ShflIdx)
	case ir.VWaveBallot:
		r := x.temp(ptx.B32)
		x.b.VoteSync(ptx.B32, r, x.arg(in, 0), x.arg(in, 1), ptx.VoteBallot)
		x.b.Cvt(ptx.U64, ptx.U32, d, r)
	case ir.VWaveAny:
		x.b.VoteSync(ptx.Pred, d, x.arg(in, 0), x.arg(in, 1), ptx.VoteAny)
	case ir.VWaveAll:
		x.b.VoteSync(ptx.Pred, d, x.arg(in, 0), x.arg(in, 1), ptx.VoteAll)
	}
	return nil
}
