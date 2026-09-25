package air

// Instruction selection: one VIR instruction to AIR instructions in the
// current block. AIR is LLVM IR, so most rows are one instruction; the
// rows that are more are where AIR and §0 disagree, and each says so.

import (
	"fmt"
	"math"

	am "github.com/vertex-language/air"
	"github.com/vertex-language/ir"
	"github.com/vertex-language/ir/lower/globals"
)

func (x *fn) inst(in *ir.Inst) error {
	op := in.Op()
	if op.IsBare() {
		return x.bare(in)
	}
	t := op.Type
	b := x.cur
	switch op.Verb {
	// —— §A7 ——
	case ir.VConst:
		return x.constant(in)

	// —— §A, §A3 ——
	case ir.VAdd, ir.VSub, ir.VMul:
		return x.arith(in)
	case ir.VSMulHi:
		x.def(in, b.IntFn(am.IntMulHi, true, x.arg(in, 0), x.arg(in, 1)))
	case ir.VUMulHi:
		x.def(in, b.IntFn(am.IntMulHi, false, x.arg(in, 0), x.arg(in, 1)))
	case ir.VSDiv, ir.VUDiv, ir.VSRem, ir.VURem:
		return x.divide(in)
	case ir.VNeg:
		if t.IsFloat() {
			x.def(in, b.FNeg(x.arg(in, 0)))
		} else {
			x.def(in, b.Sub(x.zero(t), x.arg(in, 0)))
		}
	case ir.VDiv:
		x.def(in, b.FDiv(x.arg(in, 0), x.arg(in, 1)))
	case ir.VFMA:
		x.def(in, b.MathForm(am.Fma, false, x.arg(in, 0), x.arg(in, 1), x.arg(in, 2)))
	case ir.VAbs:
		x.def(in, b.MathForm(am.Fabs, false, x.arg(in, 0)))
	case ir.VSqrt:
		x.def(in, b.MathForm(am.Sqrt, false, x.arg(in, 0)))
	case ir.VMinNum:
		x.def(in, b.MathForm(am.FMin, false, x.arg(in, 0), x.arg(in, 1)))
	case ir.VMaxNum:
		x.def(in, b.MathForm(am.FMax, false, x.arg(in, 0), x.arg(in, 1)))
	case ir.VMinimum, ir.VMaximum:
		x.minimum(in)
	case ir.VCopySign:
		x.copySign(in)
	case ir.VCeil:
		x.def(in, b.MathForm(am.Ceil, false, x.arg(in, 0)))
	case ir.VFloor:
		x.def(in, b.MathForm(am.Floor, false, x.arg(in, 0)))
	case ir.VTrunc:
		x.def(in, b.MathForm(am.Trunc, false, x.arg(in, 0)))
	case ir.VNearest:
		x.def(in, b.MathForm(am.Rint, false, x.arg(in, 0)))
	case ir.VRcpApprox:
		r := b.FDiv(am.ConstFloat(am.Float, 1), x.arg(in, 0))
		r.Fast = true
		x.def(in, r)
	case ir.VRsqrtApprox:
		x.def(in, b.MathForm(am.Rsqrt, true, x.arg(in, 0)))
	case ir.VExp2Approx:
		x.def(in, b.MathForm(am.Exp2, true, x.arg(in, 0)))
	case ir.VLog2Approx:
		x.def(in, b.MathForm(am.Log2, true, x.arg(in, 0)))
	case ir.VSinApprox:
		x.def(in, b.MathForm(am.Sin, true, x.arg(in, 0)))
	case ir.VCosApprox:
		x.def(in, b.MathForm(am.Cos, true, x.arg(in, 0)))

	// —— §A2 ——
	case ir.VSAddO, ir.VUAddO, ir.VSSubO, ir.VSMulO, ir.VUMulO:
		x.overflow(in)

	// —— §A4 ——
	case ir.VNot:
		x.def(in, b.Xor(x.arg(in, 0), x.ones(t)))
	case ir.VAnd:
		x.def(in, b.And(x.arg(in, 0), x.arg(in, 1)))
	case ir.VOr:
		x.def(in, b.Or(x.arg(in, 0), x.arg(in, 1)))
	case ir.VXor:
		x.def(in, b.Xor(x.arg(in, 0), x.arg(in, 1)))

	// —— §A5 ——
	case ir.VShl, ir.VSShr, ir.VUShr:
		x.shift(in)
	case ir.VRotL, ir.VRotR:
		x.rotate(in)

	// —— §A6 ——
	case ir.VClz:
		x.def(in, b.Clz(x.arg(in, 0)))
	case ir.VCtz:
		x.def(in, b.Ctz(x.arg(in, 0)))
	case ir.VPopcnt:
		x.def(in, b.Popcount(x.arg(in, 0)))
	case ir.VBswap:
		x.def(in, x.bswap(x.arg(in, 0), t))

	// —— §B ——
	case ir.VEq, ir.VNe, ir.VSLt, ir.VULt, ir.VSLe, ir.VULe, ir.VLt, ir.VLe, ir.VUno:
		return x.compare(in)

	// —— §C–§C4 ——
	case ir.VWrapI64:
		x.def(in, b.Trunc(x.arg(in, 0), am.UInt))
	case ir.VSExtI32:
		x.def(in, b.SExt(x.arg(in, 0), am.ULong))
	case ir.VZExtI32:
		x.def(in, b.ZExt(x.arg(in, 0), am.ULong))
	case ir.VZExtI1:
		rt, _ := x.regType(t, 0)
		x.def(in, b.ZExt(x.arg(in, 0), rt))
	case ir.VSCvtI32, ir.VSCvtI64:
		if t != ir.TypeF32 {
			return fmt.Errorf("to %s: Apple GPUs have no floating point wider than 32 bits", t)
		}
		x.def(in, b.SIToFP(x.arg(in, 0), am.Float))
	case ir.VUCvtI32, ir.VUCvtI64:
		if t != ir.TypeF32 {
			return fmt.Errorf("to %s: Apple GPUs have no floating point wider than 32 bits", t)
		}
		x.def(in, b.UIToFP(x.arg(in, 0), am.Float))
	case ir.VSCvtF32, ir.VSCvtSatF32:
		x.floatToInt(in, true)
	case ir.VUCvtF32, ir.VUCvtSatF32:
		x.floatToInt(in, false)
	case ir.VSCvtF64, ir.VUCvtF64, ir.VSCvtSatF64, ir.VUCvtSatF64, ir.VFCvtF32, ir.VFCvtF64:
		return fmt.Errorf("Apple GPUs have no f64")
	case ir.VBitcastF32, ir.VBitcastI32, ir.VBitcastI64:
		rt, err := x.regType(t, 0)
		if err != nil {
			return err
		}
		x.def(in, b.Bitcast(x.arg(in, 0), rt))
	case ir.VBitcastF64:
		return fmt.Errorf("Apple GPUs have no f64")
	case ir.VFromI64:
		x.def(in, b.IntToPtr(x.arg(in, 0), am.Ptr(x.space[in.Result(0)], am.Char)))
	case ir.VFromPtr:
		x.def(in, b.PtrToInt(x.arg(in, 0), am.ULong))

	// —— §D, §D2 ——
	case ir.VLoad, ir.VSLoad8, ir.VSLoad16, ir.VSLoad32, ir.VULoad8, ir.VULoad16, ir.VULoad32:
		return x.load(in)
	case ir.VStore, ir.VStore8, ir.VStore16, ir.VStore32:
		return x.store(in)

	// —— §D3 ——
	case ir.VAlloc:
		return x.alloc(in)
	case ir.VAlloca:
		return fmt.Errorf("a dynamically sized ptr.alloca: an Apple GPU's thread memory is fixed at compile time")
	case ir.VGetAddr:
		return x.getaddr(in)
	case ir.VDiff:
		d := b.Sub(b.PtrToInt(x.arg(in, 0), am.ULong), b.PtrToInt(x.arg(in, 1), am.ULong))
		x.def(in, d)

	// —— §F ——
	case ir.VSelect:
		x.def(in, b.Select(x.arg(in, 0), x.arg(in, 1), x.arg(in, 2)))

	// —— §H ——
	case ir.VAtomicLoad, ir.VAtomicStore, ir.VAtomicCas,
		ir.VAtomicRmwAdd, ir.VAtomicRmwSub, ir.VAtomicRmwAnd, ir.VAtomicRmwOr,
		ir.VAtomicRmwXor, ir.VAtomicRmwXchg,
		ir.VAtomicRmwSMin, ir.VAtomicRmwSMax, ir.VAtomicRmwUMin, ir.VAtomicRmwUMax:
		return x.atomic(in)

	// —— §W ——
	case ir.VWorkitemID, ir.VWorkgroupID, ir.VWorkgroupSize, ir.VNumWorkgroups, ir.VLaneID, ir.VWaveSize:
		x.workitem(in)
	case ir.VWaveShflIdx, ir.VWaveShflUp, ir.VWaveShflDown, ir.VWaveShflXor,
		ir.VWaveReadFirstLane, ir.VWaveBallot, ir.VWaveAny, ir.VWaveAll:
		return x.wave(in)

	case ir.VVaArg, ir.VVaArgRef:
		return fmt.Errorf("there is no va_list on the GPU")
	case ir.VTLSAddr, ir.VBlockAddr, ir.VFrameAddr, ir.VReturnAddr, ir.VStackSave, ir.VStackRestore:
		return fmt.Errorf("AIR has no equivalent")
	default:
		return fmt.Errorf("not lowered for AIR yet")
	}
	return nil
}

// bare is §17's bare set, less the terminators.
func (x *fn) bare(in *ir.Inst) error {
	switch in.Op().Verb {
	case ir.VBarrier:
		x.cur.Barrier(am.MemDevice | am.MemThreadgroup)
	case ir.VFence:
		return x.fence(in)
	case ir.VCall, ir.VCallInd:
		if callee, ok := in.Callee().(*ir.Func); ok {
			return fmt.Errorf("the call to @%s is still here after inlining: AIR code has every device function inlined into its kernel", callee.Name())
		}
		return fmt.Errorf("an indirect call: AIR has no function pointers")
	case ir.VMemCpy:
		x.cur.Memcpy(x.arg(in, 0), x.arg(in, 1), x.arg(in, 2))
	case ir.VMemMove:
		x.cur.Memmove(x.arg(in, 0), x.arg(in, 1), x.arg(in, 2))
	case ir.VMemSet:
		// The low byte of the value, as memset takes it.
		x.cur.Memset(x.arg(in, 0), x.cur.Trunc(x.arg(in, 1), am.Char), x.arg(in, 2))
	case ir.VMemCmp:
		return fmt.Errorf("memcmp: Metal has no library to call, and AIR no compare of memory")
	case ir.VAsm:
		return fmt.Errorf("inline assembly: an Apple GPU's instruction set is not published")
	case ir.VVaStart, ir.VVaEnd, ir.VVaCopy:
		return fmt.Errorf("there is no va_list on the GPU")
	default:
		return fmt.Errorf("not lowered for AIR yet")
	}
	return nil
}

// —— constants ————————————————————————————————————————————————————

func (x *fn) intType(t ir.RegType) *am.IntType {
	switch t {
	case ir.TypeI1:
		return am.I1
	case ir.TypeI64:
		return am.ULong
	}
	return am.UInt
}

func (x *fn) zero(t ir.RegType) am.Value { return am.ConstUint(x.intType(t), 0) }
func (x *fn) ones(t ir.RegType) am.Value { return am.ConstInt(x.intType(t), -1) }

func width(t ir.RegType) int {
	switch t {
	case ir.TypeI1:
		return 1
	case ir.TypeI64:
		return 64
	}
	return 32
}

func (x *fn) constant(in *ir.Inst) error {
	t := in.Op().Type
	lit, _ := in.Lit()
	switch t {
	case ir.TypeF32:
		if lit.Kind() != ir.ConstFloat {
			return fmt.Errorf("a float constant needs a float literal")
		}
		x.vals[in.Result(0)] = am.ConstFloat(am.Float, lit.Float())
		return nil
	case ir.TypeF64, ir.TypeF80, ir.TypeF128:
		return fmt.Errorf("Apple GPUs have no floating point wider than 32 bits")
	}
	v, err := globals.ConstInt(globalTarget{l: x.l}, lit)
	if err != nil {
		return err
	}
	switch t {
	case ir.TypePtr:
		pt := am.Ptr(x.space[in.Result(0)], am.Char)
		if v == 0 {
			x.vals[in.Result(0)] = am.Zero(pt)
		} else {
			x.def(in, x.cur.IntToPtr(am.ConstInt(am.ULong, v), pt))
		}
	default:
		x.vals[in.Result(0)] = am.ConstInt(x.intType(t), v)
	}
	return nil
}

// —— §A ———————————————————————————————————————————————————————————

func (x *fn) arith(in *ir.Inst) error {
	t := in.Op().Type
	b := x.cur
	a, c := x.arg(in, 0), x.arg(in, 1)
	switch {
	case t == ir.TypePtr:
		if in.Op().Verb != ir.VAdd {
			return fmt.Errorf("ptr.%s: a pointer is moved by ptr.add", in.Op().Verb)
		}
		g := b.GEP(a, c)
		g.InBounds = false // ptr.add promises nothing about bounds
		x.def(in, g)
	case t.IsFloat():
		switch in.Op().Verb {
		case ir.VAdd:
			x.def(in, b.FAdd(a, c))
		case ir.VSub:
			x.def(in, b.FSub(a, c))
		case ir.VMul:
			x.def(in, b.FMul(a, c))
		}
	default:
		switch in.Op().Verb {
		case ir.VAdd:
			x.def(in, b.Add(a, c))
		case ir.VSub:
			x.def(in, b.Sub(a, c))
		case ir.VMul:
			// A multiply stays one, a power of two's too: Apple's compiler
			// folds an index times an element's size into the address of a
			// load, which a shift it does not.
			x.def(in, b.Mul(a, c))
		}
	}
	return nil
}

// divide guards its divisor. §0 traps on a zero one, and on the signed
// quotient that overflows (MIN / -1); an Apple GPU cannot trap, and LLVM
// leaves both undefined, which the compiler below would be free to use.
// So the divisor is made 1 in either case: the quotient is the dividend,
// the remainder zero -- defined, and a divergence from §0 this backend
// documents rather than hides.
func (x *fn) divide(in *ir.Inst) error {
	t := in.Op().Type
	b := x.cur
	a, d := x.arg(in, 0), x.arg(in, 1)
	it := x.intType(t)
	// A divisor that is a constant, not zero and not -1, needs no guard:
	// the division keeps its constant, which the guard's select would hide.
	// (Rewriting a power of two as shifts here was measured to slow the
	// quantized matrix-vector kernels, whose register use it changed; a
	// kernel wanting shifts writes them.)
	if c, ok := d.(*am.Const); ok && c.Kind == am.ConstIntKind && c.Bits != 0 {
		w := width(t)
		minusOne := c.Bits == mask64(w)
		if !minusOne || in.Op().Verb == ir.VUDiv || in.Op().Verb == ir.VURem {
			switch in.Op().Verb {
			case ir.VSDiv:
				x.def(in, b.SDiv(a, d))
			case ir.VUDiv:
				x.def(in, b.UDiv(a, d))
			case ir.VSRem:
				x.def(in, b.SRem(a, d))
			case ir.VURem:
				x.def(in, b.URem(a, d))
			}
			return nil
		}
	}
	bad := am.Value(b.ICmp(am.EQ, d, am.ConstUint(it, 0)))
	signed := in.Op().Verb == ir.VSDiv || in.Op().Verb == ir.VSRem
	if signed {
		min := am.ConstUint(it, 1<<uint(width(t)-1))
		ovf := b.And(b.ICmp(am.EQ, a, min), b.ICmp(am.EQ, d, am.ConstInt(it, -1)))
		bad = b.Or(bad, ovf)
	}
	safe := b.Select(bad, am.ConstUint(it, 1), d)
	switch in.Op().Verb {
	case ir.VSDiv:
		x.def(in, b.SDiv(a, safe))
	case ir.VUDiv:
		x.def(in, b.UDiv(a, safe))
	case ir.VSRem:
		x.def(in, b.SRem(a, safe))
	case ir.VURem:
		x.def(in, b.URem(a, safe))
	}
	return nil
}

// minimum is IEEE-754-2019's minimum and maximum: a NaN in either operand
// is the answer, and -0 is less than +0. fmin and fmax are minNum and
// maxNum, which prefer the number and may answer either zero.
func (x *fn) minimum(in *ir.Inst) {
	b := x.cur
	a, c := x.arg(in, 0), x.arg(in, 1)
	isMin := in.Op().Verb == ir.VMinimum
	var r am.Value
	if isMin {
		r = b.MathForm(am.FMin, false, a, c)
	} else {
		r = b.MathForm(am.FMax, false, a, c)
	}
	// Equal operands are either the same number or two zeros: then the
	// sign bits decide, OR for minimum, AND for maximum.
	ab, cb := b.Bitcast(a, am.UInt), b.Bitcast(c, am.UInt)
	var signs am.Value
	if isMin {
		signs = b.Or(ab, cb)
	} else {
		signs = b.And(ab, cb)
	}
	r = b.Select(b.FCmp(am.FOEQ, a, c), b.Bitcast(signs, am.Float), r)
	nan := b.FCmp(am.FUNO, a, c)
	x.def(in, b.Select(nan, b.FAdd(a, c), r))
}

// copySign is the magnitude of the first operand with the sign of the
// second, in bits: xcrun expands copysign the same way.
func (x *fn) copySign(in *ir.Inst) {
	b := x.cur
	mag := b.And(b.Bitcast(x.arg(in, 0), am.UInt), am.ConstUint(am.UInt, 0x7fffffff))
	sign := b.And(b.Bitcast(x.arg(in, 1), am.UInt), am.ConstUint(am.UInt, 0x80000000))
	x.def(in, b.Bitcast(b.Or(mag, sign), am.Float))
}

func (x *fn) overflow(in *ir.Inst) {
	t := in.Op().Type
	b := x.cur
	a, c := x.arg(in, 0), x.arg(in, 1)
	zero := x.zero(t)
	switch in.Op().Verb {
	case ir.VSAddO:
		r := b.Add(a, c)
		x.def(in, b.ICmp(am.SLT, b.And(b.Xor(a, r), b.Xor(c, r)), zero))
	case ir.VSSubO:
		r := b.Sub(a, c)
		x.def(in, b.ICmp(am.SLT, b.And(b.Xor(a, c), b.Xor(a, r)), zero))
	case ir.VUAddO:
		x.def(in, b.ICmp(am.ULT, b.Add(a, c), a))
	case ir.VUMulO:
		x.def(in, b.ICmp(am.NE, b.IntFn(am.IntMulHi, false, a, c), zero))
	case ir.VSMulO:
		hi := b.IntFn(am.IntMulHi, true, a, c)
		sign := b.AShr(b.Mul(a, c), am.ConstUint(x.intType(t), uint64(width(t)-1)))
		x.def(in, b.ICmp(am.NE, hi, sign))
	}
}

// —— §A5 ——————————————————————————————————————————————————————————

// shift masks the count to the width, as §0 requires; LLVM leaves a count
// past the width undefined.
func (x *fn) shift(in *ir.Inst) {
	t := in.Op().Type
	b := x.cur
	n := x.shiftCount(x.arg(in, 1), t)
	switch in.Op().Verb {
	case ir.VShl:
		x.def(in, b.Shl(x.arg(in, 0), n))
	case ir.VSShr:
		x.def(in, b.AShr(x.arg(in, 0), n))
	case ir.VUShr:
		x.def(in, b.LShr(x.arg(in, 0), n))
	}
}

// shiftCount masks a shift's count to the width. A constant count is
// masked here, so the shift takes a literal count as xcrun's does: Apple's
// GPU compiler has been seen to drop a shift by 'and 16, 31' outright when
// the kernel also compares floats.
func (x *fn) shiftCount(n am.Value, t ir.RegType) am.Value {
	m := uint64(width(t) - 1)
	if c, ok := n.(*am.Const); ok && c.Kind == am.ConstIntKind {
		return am.ConstUint(x.intType(t), c.Bits&m)
	}
	return x.cur.And(n, am.ConstUint(x.intType(t), m))
}

// rotate is two shifts and an or, the second count masked too, so that a
// count of zero shifts by zero both ways.
func (x *fn) rotate(in *ir.Inst) {
	t := in.Op().Type
	b := x.cur
	it := x.intType(t)
	mask := am.ConstUint(it, uint64(width(t)-1))
	a := x.arg(in, 0)
	n := b.And(x.arg(in, 1), mask)
	m := b.And(b.Sub(am.ConstUint(it, uint64(width(t))), n), mask)
	if in.Op().Verb == ir.VRotL {
		x.def(in, b.Or(b.Shl(a, n), b.LShr(a, m)))
	} else {
		x.def(in, b.Or(b.LShr(a, n), b.Shl(a, m)))
	}
}

// bswap reverses a value's bytes with shifts and masks.
func (x *fn) bswap(a am.Value, t ir.RegType) am.Value {
	b := x.cur
	if t == ir.TypeI64 {
		lo := x.bswap(b.Trunc(a, am.UInt), ir.TypeI32)
		hi := x.bswap(b.Trunc(b.LShr(a, am.ConstUint(am.ULong, 32)), am.UInt), ir.TypeI32)
		return b.Or(b.Shl(b.ZExt(lo, am.ULong), am.ConstUint(am.ULong, 32)), b.ZExt(hi, am.ULong))
	}
	c := func(v uint64) am.Value { return am.ConstUint(am.UInt, v) }
	r := b.Shl(a, c(24))
	r = b.Or(r, b.And(b.Shl(a, c(8)), c(0xff0000)))
	r = b.Or(r, b.And(b.LShr(a, c(8)), c(0xff00)))
	return b.Or(r, b.LShr(a, c(24)))
}

// —— §B ———————————————————————————————————————————————————————————

func (x *fn) compare(in *ir.Inst) error {
	t := in.Op().Type
	b := x.cur
	a, c := x.arg(in, 0), x.arg(in, 1)
	if t == ir.TypeF32 && in.Arg(0) == in.Arg(1) {
		return x.selfCompare(in)
	}
	if t.IsFloat() {
		pred := map[ir.Verb]am.Pred{ir.VEq: am.FOEQ, ir.VNe: am.FUNE, ir.VLt: am.FOLT, ir.VLe: am.FOLE, ir.VUno: am.FUNO}
		p, ok := pred[in.Op().Verb]
		if !ok {
			return fmt.Errorf("not a float comparison")
		}
		x.def(in, b.FCmp(p, a, c))
		return nil
	}
	pred := map[ir.Verb]am.Pred{ir.VEq: am.EQ, ir.VNe: am.NE, ir.VSLt: am.SLT, ir.VULt: am.ULT, ir.VSLe: am.SLE, ir.VULe: am.ULE}
	if t == ir.TypePtr {
		// §B's ordering on pointers is the one verb, and it is unsigned:
		// an address is a number and there is no signed reading of it.
		pred[ir.VLt], pred[ir.VLe] = am.ULT, am.ULE
	}
	p, ok := pred[in.Op().Verb]
	if !ok {
		return fmt.Errorf("not an integer comparison")
	}
	if t == ir.TypePtr && !am.Identical(a.Type(), c.Type()) {
		return fmt.Errorf("compares a pointer into %v with one into %v", x.space[in.Arg(0)], x.space[in.Arg(1)])
	}
	x.def(in, b.ICmp(p, a, c))
	return nil
}

// —— §C ———————————————————————————————————————————————————————————

// floatToInt saturates. §C2's _sat_ rows ask for exactly this -- the
// clamp to the range, NaN to zero -- and the trapping rows ask for a trap
// out of range, which an Apple GPU cannot take: they saturate too, a
// divergence documented rather than an undefined conversion.
func (x *fn) floatToInt(in *ir.Inst, signed bool) {
	t := in.Op().Type
	b := x.cur
	it := x.intType(t)
	w := width(t)
	v := x.arg(in, 0)
	f := func(v float64) am.Value { return am.ConstFloat(am.Float, v) }
	var lo, hi float64
	var loV, hiV am.Value
	if signed {
		lo, hi = -math.Ldexp(1, w-1), math.Ldexp(1, w-1)
		loV, hiV = am.ConstUint(it, 1<<uint(w-1)), am.ConstUint(it, 1<<uint(w-1)-1)
	} else {
		lo, hi = -1, math.Ldexp(1, w)
		loV, hiV = am.ConstUint(it, 0), am.ConstInt(it, -1)
	}
	big := b.FCmp(am.FOGE, v, f(hi))
	var small am.Value
	if signed {
		small = b.FCmp(am.FOLT, v, f(lo))
	} else {
		small = b.FCmp(am.FOLE, v, f(lo))
	}
	nan := b.FCmp(am.FUNO, v, v)
	out := b.Or(b.Or(big, small), nan)
	safe := b.Select(out, f(0), v)
	var r am.Value
	if signed {
		r = b.FPToSI(safe, it)
	} else {
		r = b.FPToUI(safe, it)
	}
	r = b.Select(big, hiV, r)
	r = b.Select(small, loV, r)
	x.def(in, b.Select(nan, am.ConstUint(it, 0), r))
}

// —— §D ———————————————————————————————————————————————————————————

// memType is the type a load or store of a verb moves in memory.
func (x *fn) memType(in *ir.Inst) (am.Type, error) {
	switch in.Op().Verb {
	case ir.VSLoad8, ir.VULoad8, ir.VStore8:
		return am.UChar, nil
	case ir.VSLoad16, ir.VULoad16, ir.VStore16:
		return am.UShort, nil
	case ir.VSLoad32, ir.VULoad32, ir.VStore32:
		return am.UInt, nil
	}
	t := in.Op().Type
	switch t {
	case ir.TypeI1:
		return nil, fmt.Errorf("i1 has no storage width")
	case ir.TypePtr:
		// A pointer in memory points into device memory (space.go).
		return am.Ptr(am.Device, am.Char), nil
	}
	return x.regType(t, 0)
}

// typed is a pointer cast to point to t, in its own space.
func (x *fn) typed(p am.Value, t am.Type) am.Value {
	pt := p.Type().(*am.PointerType)
	if am.Identical(pt.Elem, t) {
		return p
	}
	return x.cur.Bitcast(p, am.Ptr(pt.Space, t))
}

func setAlign(i *am.Inst, in *ir.Inst) {
	if a, ok := in.Align(); ok && a > 0 {
		i.Align = int(a)
	}
}

func (x *fn) load(in *ir.Inst) error {
	mt, err := x.memType(in)
	if err != nil {
		return err
	}
	b := x.cur
	ld := b.Load(x.typed(x.arg(in, 0), mt))
	setAlign(ld, in)
	switch in.Op().Verb {
	case ir.VSLoad8, ir.VSLoad16, ir.VSLoad32:
		x.def(in, b.SExt(ld, x.intType(in.Op().Type)))
	case ir.VULoad8, ir.VULoad16, ir.VULoad32:
		x.def(in, b.ZExt(ld, x.intType(in.Op().Type)))
	default:
		x.def(in, ld)
	}
	return nil
}

func (x *fn) store(in *ir.Inst) error {
	mt, err := x.memType(in)
	if err != nil {
		return err
	}
	b := x.cur
	v := x.arg(in, 0)
	switch in.Op().Verb {
	case ir.VStore8, ir.VStore16, ir.VStore32:
		v = b.Trunc(v, mt)
	}
	st := b.Store(v, x.typed(x.arg(in, 1), mt))
	setAlign(st, in)
	return nil
}

// alloc is thread memory in the kernel's frame: an alloca of its bytes.
func (x *fn) alloc(in *ir.Inst) error {
	b := x.cur
	size := in.Size()
	if size == 0 {
		size = 1
	}
	t := am.Array(am.Char, int(size))
	a := b.Alloca(t)
	if al, ok := in.Align(); ok && al > 0 {
		a.Align = int(al)
	}
	if in.Zeroed() {
		b.Store(am.Zero(t), a)
	}
	x.def(in, b.Bitcast(a, am.Ptr(am.Thread, am.Char)))
	return nil
}

func (x *fn) getaddr(in *ir.Inst) error {
	switch s := in.Symbol().(type) {
	case *ir.Global:
		g, ok := x.l.globals[s.Name()]
		if !ok {
			return fmt.Errorf("@%s was not laid out", s.Name())
		}
		x.def(in, x.cur.Bitcast(g, am.Ptr(g.Space, am.Char)))
		return nil
	case *ir.GlobalImport:
		if p, ok := x.shared[s]; ok {
			x.vals[in.Result(0)] = p
			return nil
		}
		return fmt.Errorf("@%s is imported: an AIR library has no other module to link against", s.Name())
	}
	return fmt.Errorf("the address of a function: AIR has no function pointers")
}

// selfCompare lowers a float compared with itself, which asks only whether
// it is a NaN, as MSL's isnan does: on the bits, as integers. Apple's GPU
// compiler miscompiles 'fcmp une %f, %f' and its kin: a kernel holding one
// drops the shift from 'lshr (bitcast %g), 16', for any float %g. xcrun
// -fno-fast-math's own output for 'f != f' does the same.
func (x *fn) selfCompare(in *ir.Inst) error {
	b := x.cur
	bits := b.And(b.Bitcast(x.arg(in, 0), am.UInt), am.ConstUint(am.UInt, 0x7fffffff))
	nan := b.ICmp(am.UGT, bits, am.ConstUint(am.UInt, 0x7f800000))
	switch in.Op().Verb {
	case ir.VNe, ir.VUno:
		x.def(in, nan)
	case ir.VEq, ir.VLe:
		x.def(in, b.Xor(nan, am.ConstBool(true)))
	case ir.VLt:
		x.def(in, am.ConstBool(false))
	default:
		return fmt.Errorf("not a float comparison")
	}
	return nil
}

// mask64 is the w low bits set.
func mask64(w int) uint64 {
	if w >= 64 {
		return ^uint64(0)
	}
	return 1<<uint(w) - 1
}
