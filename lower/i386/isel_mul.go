package i386

// The full 128-bit product of two 64-bit values, which four verbs need and
// this machine has no instruction for.
//
// §A's smulhi and umulhi want the half an ordinary multiply throws away, and
// §A2's smulo and umulo want to know whether that half is anything other than
// what the low half's sign implies. Both questions are the same computation,
// and at sixty-four bits on a thirty-two-bit machine that computation is the
// schoolbook expansion: four 32x32 products, and a carry chain across them.
//
// The one-operand MUL is what makes each of the four exact — it puts the
// whole product in EDX:EAX rather than discarding the top — and the carries
// are ADC against a zero, which is the idiom that turns a flag into a number
// without a branch.

import (
	"github.com/vertex-language/i386/reg"

	"github.com/vertex-language/ir"
	"github.com/vertex-language/ir/lower/mir"
)

// wide is a 128-bit value as its four words, least significant first.
type wide struct{ w0, w1, w2, w3 mir.VReg }

// low is the product's low sixty-four bits, and high the top sixty-four.
func (p wide) low() value  { return value{lo: p.w0, hi: p.w1, w: w64} }
func (p wide) high() value { return value{lo: p.w2, hi: p.w3, w: w64} }

// mul32 is one 32x32 product, both halves of it: the one-operand MUL, whose
// destination is EDX:EAX and whose second operand is a register.
func mul32(c *cursor, vr *vregs, a, b mir.VReg) (lo, hi mir.VReg) {
	other := vr.reg32()
	emitCopy(c, other, b)
	eax := vr.physical(reg.EAX)
	edx := vr.physical(reg.EDX)
	emitCopy(c, eax, a)
	c.Emit(mir.Instr{
		Op:   wideMulOp{},
		Defs: []mir.VReg{eax, edx},
		Uses: []mir.VReg{eax, other},
	})
	lo, hi = vr.reg32(), vr.reg32()
	emitCopy(c, lo, eax)
	emitCopy(c, hi, edx)
	return lo, hi
}

// zero32 is a register holding zero, which the carry chain adds into.
func zero32(c *cursor, vr *vregs) mir.VReg {
	z := vr.reg32()
	c.Emit(mir.Instr{Op: constOp{imm: 0}, Defs: []mir.VReg{z}})
	return z
}

// addCarry is dst += src, with the carry it produces accumulated into carry.
//
// The ADC has to follow the ADD with nothing between it that writes flags,
// which holds for the same reason every other carry chain in this package
// holds: the only instructions the allocator inserts are moves, and a move on
// x86 does not touch the flags. The zero is materialized before the pair
// rather than between them for that reason.
func addCarry(c *cursor, dst, src, carry, zero mir.VReg) {
	c.Emit(mir.Instr{Op: aluOp{verb: ir.VAdd}, Defs: []mir.VReg{dst}, Uses: []mir.VReg{dst, src}})
	c.Emit(mir.Instr{Op: carryOp{}, Defs: []mir.VReg{carry}, Uses: []mir.VReg{carry, zero}})
}

// emitWideProduct is the unsigned 128-bit product of a and b.
//
// Writing (a1:a0) and (b1:b0) for the halves, the product is
//
//	a0·b0 + (a0·b1 + a1·b0)·2³² + a1·b1·2⁶⁴
//
// which is four exact products and two columns of addition. The first column
// is a0·b0's high word plus the two crossing products' low words, and can
// carry twice — so the carry out of it is counted rather than tested, and
// added into the second column as a number.
func emitWideProduct(c *cursor, vr *vregs, a, b value) wide {
	p0l, p0h := mul32(c, vr, a.lo, b.lo)
	p1l, p1h := mul32(c, vr, a.lo, b.hi)
	p2l, p2h := mul32(c, vr, a.hi, b.lo)
	p3l, p3h := mul32(c, vr, a.hi, b.hi)

	// Bits 32 to 63, and how far the column overflowed.
	zero := zero32(c, vr)
	k1 := zero32(c, vr)
	w1 := vr.reg32()
	emitCopy(c, w1, p0h)
	addCarry(c, w1, p1l, k1, zero)
	addCarry(c, w1, p2l, k1, zero)

	// Bits 64 to 95, into which the column above carried.
	k2 := zero32(c, vr)
	w2 := vr.reg32()
	emitCopy(c, w2, p3l)
	addCarry(c, w2, p1h, k2, zero)
	addCarry(c, w2, p2h, k2, zero)
	addCarry(c, w2, k1, k2, zero)

	// And the top, which cannot carry: the product of two 64-bit values
	// occupies 128 bits and no more.
	w3 := vr.reg32()
	emitCopy(c, w3, p3h)
	c.Emit(mir.Instr{Op: aluOp{verb: ir.VAdd}, Defs: []mir.VReg{w3}, Uses: []mir.VReg{w3, k2}})

	return wide{w0: p0l, w1: w1, w2: w2, w3: w3}
}

// signFix subtracts from the unsigned high half what makes it the signed one.
//
// The two products differ by exactly the terms the unsigned reading adds for
// each negative operand: reading a as unsigned adds b·2⁶⁴ when a is negative,
// and likewise for b. So the signed high half is the unsigned one less b for
// a negative a and less a for a negative b — and "for a negative" is a mask,
// the arithmetic shift of the top word by thirty-one, rather than a branch.
func signFix(c *cursor, vr *vregs, high, a, b value) value {
	sub := func(dst value, mask mir.VReg, v value) {
		lo, hi := vr.reg32(), vr.reg32()
		emitCopy(c, lo, v.lo)
		c.Emit(mir.Instr{Op: aluOp{verb: ir.VAnd}, Defs: []mir.VReg{lo}, Uses: []mir.VReg{lo, mask}})
		emitCopy(c, hi, v.hi)
		c.Emit(mir.Instr{Op: aluOp{verb: ir.VAnd}, Defs: []mir.VReg{hi}, Uses: []mir.VReg{hi, mask}})
		c.Emit(mir.Instr{Op: aluOp{verb: ir.VSub}, Defs: []mir.VReg{dst.lo}, Uses: []mir.VReg{dst.lo, lo}})
		c.Emit(mir.Instr{Op: carryOp{sub: true}, Defs: []mir.VReg{dst.hi}, Uses: []mir.VReg{dst.hi, hi}})
	}

	out := value{lo: vr.reg32(), hi: vr.reg32(), w: w64}
	emitCopy(c, out.lo, high.lo)
	emitCopy(c, out.hi, high.hi)

	maskA := signMask(c, vr, a.hi)
	sub(out, maskA, b)
	maskB := signMask(c, vr, b.hi)
	sub(out, maskB, a)
	return out
}

// signMask is all ones when the word is negative and zero when it is not.
func signMask(c *cursor, vr *vregs, hi mir.VReg) mir.VReg {
	m := vr.reg32()
	emitCopy(c, m, hi)
	c.Emit(mir.Instr{Op: shiftImmOp{verb: ir.VSShr, n: 31}, Defs: []mir.VReg{m}, Uses: []mir.VReg{m}})
	return m
}

// iselMulHi64 lowers §A's smulhi and umulhi at sixty-four bits.
func iselMulHi64(c *cursor, vr *vregs, dst, a, b value, signed bool) error {
	p := emitWideProduct(c, vr, a, b)
	high := p.high()
	if signed {
		high = signFix(c, vr, high, a, b)
	}
	emitCopy(c, dst.lo, high.lo)
	emitCopy(c, dst.hi, high.hi)
	return nil
}

// iselMulOverflow64 lowers §A2's smulo and umulo at sixty-four bits.
//
// Unsigned overflows when the high half is anything at all. Signed overflows
// when the high half is not what the low half's sign already says it must be:
// a 128-bit product fits in sixty-four signed bits exactly when its top
// sixty-four are the sign extension of its low sixty-four, so the test is
// against that word rather than against zero.
func iselMulOverflow64(c *cursor, vr *vregs, dst, a, b value, signed bool) error {
	p := emitWideProduct(c, vr, a, b)
	high := p.high()
	want := zero32(c, vr)
	if signed {
		high = signFix(c, vr, high, a, b)
		want = signMask(c, vr, p.w1)
	}

	// The two words folded into one, so that a single flag answers both.
	acc := vr.reg32()
	emitCopy(c, acc, high.lo)
	c.Emit(mir.Instr{Op: aluOp{verb: ir.VXor}, Defs: []mir.VReg{acc}, Uses: []mir.VReg{acc, want}})
	top := vr.reg32()
	emitCopy(c, top, high.hi)
	c.Emit(mir.Instr{Op: aluOp{verb: ir.VXor}, Defs: []mir.VReg{top}, Uses: []mir.VReg{top, want}})
	c.Emit(mir.Instr{Op: aluOp{verb: ir.VOr}, Defs: []mir.VReg{acc}, Uses: []mir.VReg{acc, top}})

	c.Emit(mir.Instr{Op: testOp{}, Uses: []mir.VReg{acc}})
	c.Emit(mir.Instr{Op: setccOp{cond: condNE}, Defs: []mir.VReg{dst.lo}})
	return nil
}
