package amd64

// §V's v128 namespace, lowered onto SSE2.
//
// SSE2 is the floor and not a choice: it is part of what x86-64 means, so
// every row here is an instruction the target has by definition and nothing
// is gated on a feature bit. That is also why the IR's verb set is the shape
// it is — a verb exists where the baseline has an instruction, and the ones
// an orthogonal table would predict but the hardware lacks (a signed byte
// minimum, an arithmetic quadword shift) are absent from §V rather than
// present and expensive.
//
// Almost every one of these is two-address: the first operand is read and
// written. A result that is not in the first operand's register needs a copy
// first, which is emitCopy's job here as it is aluOp's. There is no
// three-operand form until AVX, where the same instruction gains one.

import (
	"fmt"

	amd64asm "github.com/vertex-language/amd64"
	"github.com/vertex-language/amd64/operand"
	"github.com/vertex-language/amd64/reg"
	"github.com/vertex-language/ir"
	"github.com/vertex-language/ir/lower/mir"
)

// The MIR ops. Each names the VIR verb it came from rather than the
// instruction it becomes, so that the mapping lives in one table — vecEmit
// below — and a reader comparing §V against SSE2 has one place to look.
type (
	// vecAluOp is a two-address lane operation: Defs[0] = Uses[0] op Uses[1].
	vecAluOp struct{ verb ir.Verb }

	// vecImmOp is a two-address lane operation with a literal second
	// operand: the whole-register byte shifts.
	vecImmOp struct {
		verb ir.Verb
		k    int64
	}

	// vecShuffleOp is the permutes, which read their source rather than
	// their destination and so are not two-address. That is what makes
	// them the cheap way to move a lane: no copy first.
	vecShuffleOp struct {
		verb ir.Verb
		k    int64
	}

	// vecShiftOp is a lane shift by a count in the low quadword of a
	// vector register. Uses[1] is that register, which isel fills from a
	// general one when the count is not a literal.
	vecShiftOp struct{ verb ir.Verb }

	// vecSplatOp fills every lane from one general register.
	vecSplatOp struct{ verb ir.Verb }

	// vecZExtOp puts a general register in the low lane and zeroes the
	// rest. Not a splat: it fills nothing else.
	vecZExtOp struct{ w width }

	// vecBitmaskOp gathers one bit per lane into a general register.
	vecBitmaskOp struct{}

	// vecExtractOp reads one lane into a general register.
	vecExtractOp struct{ lane int64 }

	// vecReplaceOp writes one lane from a general register. Two-address in
	// the vector operand: Defs[0] = Uses[0] with Uses[1] at lane.
	vecReplaceOp struct{ lane int64 }

	// vecOnesOp is the all-ones register, which PCMPEQD of anything with
	// itself produces and which no move can produce as cheaply.
	vecOnesOp struct{}
)

// vecAlu is the two-address lane operations: every §V verb whose shape is
// "two vector registers in, one out, destination is the first source".
//
// PANDN is the one row whose operands are the other way round. SSE2 computes
// NOT dst AND src, and §V's andnot is a AND NOT b, so isel hands the emitter
// its operands swapped and the table says so here rather than leaving it to
// be rediscovered at the call.
var vecAlu = map[ir.Verb]func(*amd64asm.Section, reg.Xmm, operand.RM128){
	ir.VAnd:       (*amd64asm.Section).PandXmmRM128,
	ir.VOr:        (*amd64asm.Section).PorXmmRM128,
	ir.VXor:       (*amd64asm.Section).PxorXmmRM128,
	ir.VVecAndNot: (*amd64asm.Section).PandnXmmRM128,

	ir.VI8x16Add: (*amd64asm.Section).PaddbXmmRM128,
	ir.VI16x8Add: (*amd64asm.Section).PaddwXmmRM128,
	ir.VI32x4Add: (*amd64asm.Section).PadddXmmRM128,
	ir.VI64x2Add: (*amd64asm.Section).PaddqXmmRM128,
	ir.VI8x16Sub: (*amd64asm.Section).PsubbXmmRM128,
	ir.VI16x8Sub: (*amd64asm.Section).PsubwXmmRM128,
	ir.VI32x4Sub: (*amd64asm.Section).PsubdXmmRM128,
	ir.VI64x2Sub: (*amd64asm.Section).PsubqXmmRM128,

	ir.VI8x16AddSatS: (*amd64asm.Section).PaddsbXmmRM128,
	ir.VI16x8AddSatS: (*amd64asm.Section).PaddswXmmRM128,
	ir.VI8x16AddSatU: (*amd64asm.Section).PaddusbXmmRM128,
	ir.VI16x8AddSatU: (*amd64asm.Section).PadduswXmmRM128,
	ir.VI8x16SubSatS: (*amd64asm.Section).PsubsbXmmRM128,
	ir.VI16x8SubSatS: (*amd64asm.Section).PsubswXmmRM128,
	ir.VI8x16SubSatU: (*amd64asm.Section).PsubusbXmmRM128,
	ir.VI16x8SubSatU: (*amd64asm.Section).PsubuswXmmRM128,

	ir.VI16x8Mul:      (*amd64asm.Section).PmullwXmmRM128,
	ir.VI16x8MulHiS:   (*amd64asm.Section).PmulhwXmmRM128,
	ir.VI16x8MulHiU:   (*amd64asm.Section).PmulhuwXmmRM128,
	ir.VI32x4MulEvenU: (*amd64asm.Section).PmuludqXmmRM128,
	ir.VI16x8MaddS:    (*amd64asm.Section).PmaddwdXmmRM128,
	ir.VI8x16SadU:     (*amd64asm.Section).PsadbwXmmRM128,

	ir.VI8x16MinU:  (*amd64asm.Section).PminubXmmRM128,
	ir.VI8x16MaxU:  (*amd64asm.Section).PmaxubXmmRM128,
	ir.VI16x8MinS:  (*amd64asm.Section).PminswXmmRM128,
	ir.VI16x8MaxS:  (*amd64asm.Section).PmaxswXmmRM128,
	ir.VI8x16AvgrU: (*amd64asm.Section).PavgbXmmRM128,
	ir.VI16x8AvgrU: (*amd64asm.Section).PavgwXmmRM128,

	ir.VI8x16Eq:  (*amd64asm.Section).PcmpeqbXmmRM128,
	ir.VI16x8Eq:  (*amd64asm.Section).PcmpeqwXmmRM128,
	ir.VI32x4Eq:  (*amd64asm.Section).PcmpeqdXmmRM128,
	ir.VI8x16GtS: (*amd64asm.Section).PcmpgtbXmmRM128,
	ir.VI16x8GtS: (*amd64asm.Section).PcmpgtwXmmRM128,
	ir.VI32x4GtS: (*amd64asm.Section).PcmpgtdXmmRM128,

	ir.VI8x16NarrowS: (*amd64asm.Section).PacksswbXmmRM128,
	ir.VI8x16NarrowU: (*amd64asm.Section).PackuswbXmmRM128,
	ir.VI16x8NarrowS: (*amd64asm.Section).PackssdwXmmRM128,

	ir.VI8x16UnpackLow: (*amd64asm.Section).PunpcklbwXmmRM128,
	ir.VI16x8UnpackLow: (*amd64asm.Section).PunpcklwdXmmRM128,
	ir.VI32x4UnpackLow: (*amd64asm.Section).PunpckldqXmmRM128,
	ir.VI64x2UnpackLow: (*amd64asm.Section).PunpcklqdqXmmRM128,
	ir.VI8x16UnpackHi:  (*amd64asm.Section).PunpckhbwXmmRM128,
	ir.VI16x8UnpackHi:  (*amd64asm.Section).PunpckhwdXmmRM128,
	ir.VI32x4UnpackHi:  (*amd64asm.Section).PunpckhdqXmmRM128,
	ir.VI64x2UnpackHi:  (*amd64asm.Section).PunpckhqdqXmmRM128,
}

// vecShiftReg and vecShiftImm are the same eight operations reached two
// ways. The hardware has both because a literal count is the common one and
// costs no register; isel picks between them by whether the count is a
// literal, which is the only thing that decides it.
var vecShiftReg = map[ir.Verb]func(*amd64asm.Section, reg.Xmm, operand.RM128){
	ir.VI16x8Shl:  (*amd64asm.Section).PsllwXmmRM128,
	ir.VI32x4Shl:  (*amd64asm.Section).PslldXmmRM128,
	ir.VI64x2Shl:  (*amd64asm.Section).PsllqXmmRM128,
	ir.VI16x8ShrU: (*amd64asm.Section).PsrlwXmmRM128,
	ir.VI32x4ShrU: (*amd64asm.Section).PsrldXmmRM128,
	ir.VI64x2ShrU: (*amd64asm.Section).PsrlqXmmRM128,
	ir.VI16x8ShrS: (*amd64asm.Section).PsrawXmmRM128,
	ir.VI32x4ShrS: (*amd64asm.Section).PsradXmmRM128,
}

var vecShiftImm = map[ir.Verb]func(*amd64asm.Section, reg.Xmm, int64){
	ir.VI16x8Shl:  (*amd64asm.Section).PsllwXmmImm8,
	ir.VI32x4Shl:  (*amd64asm.Section).PslldXmmImm8,
	ir.VI64x2Shl:  (*amd64asm.Section).PsllqXmmImm8,
	ir.VI16x8ShrU: (*amd64asm.Section).PsrlwXmmImm8,
	ir.VI32x4ShrU: (*amd64asm.Section).PsrldXmmImm8,
	ir.VI64x2ShrU: (*amd64asm.Section).PsrlqXmmImm8,
	ir.VI16x8ShrS: (*amd64asm.Section).PsrawXmmImm8,
	ir.VI32x4ShrS: (*amd64asm.Section).PsradXmmImm8,

	ir.VVecShlBytes: (*amd64asm.Section).PslldqXmmImm8,
	ir.VVecShrBytes: (*amd64asm.Section).PsrldqXmmImm8,
}

var vecShuffle = map[ir.Verb]func(*amd64asm.Section, reg.Xmm, operand.RM128, int64){
	ir.VI32x4Shuffle:    (*amd64asm.Section).PshufdXmmRM128Imm8,
	ir.VI16x8ShuffleLow: (*amd64asm.Section).PshuflwXmmRM128Imm8,
	ir.VI16x8ShuffleHi:  (*amd64asm.Section).PshufhwXmmRM128Imm8,
}

// ---- instruction selection ------------------------------------------------

// vecVerbs is the whole of §V, which is how the dispatch in isel.go knows a
// verb belongs here without listing a hundred cases.
func vecVerb(v ir.Verb) bool {
	if _, ok := vecAlu[v]; ok {
		return true
	}
	if _, ok := vecShiftReg[v]; ok {
		return true
	}
	if _, ok := vecShuffle[v]; ok {
		return true
	}
	switch v {
	case ir.VVecShlBytes, ir.VVecShrBytes,
		ir.VVecZExtI32, ir.VVecZExtI64,
		ir.VI8x16Splat, ir.VI16x8Splat, ir.VI32x4Splat, ir.VI64x2Splat,
		ir.VI8x16Bitmask, ir.VI16x8ExtractU, ir.VI16x8Replace,
		ir.VI32x4Extract, ir.VI64x2Extract,
		ir.VNot:
		return true
	}
	return false
}

// iselVec lowers one §V instruction.
func iselVec(c *cursor, vr *vregs, in *ir.Inst) error {
	op := in.Op()

	// Every verb here defines exactly one register, and all but bitmask
	// and extract define a vector one.
	arg := func(i int) (mir.VReg, error) {
		v, ok := vr.lookup(in.Arg(i))
		if !ok {
			return 0, fmt.Errorf("%s: operand defined outside the function", op)
		}
		return v, nil
	}
	def := func() (mir.VReg, error) {
		d, err := vr.define(in.Result(0))
		if err != nil {
			return 0, fmt.Errorf("%s: %w", op, err)
		}
		return d, nil
	}

	if _, ok := vecAlu[op.Verb]; ok {
		a, err := arg(0)
		if err != nil {
			return err
		}
		b, err := arg(1)
		if err != nil {
			return err
		}
		dst, err := def()
		if err != nil {
			return err
		}
		// PANDN reads its destination as the negated operand, so the
		// operand that is *not* negated has to be the one in memory or
		// in the second register. §V's andnot negates the second.
		if op.Verb == ir.VVecAndNot {
			a, b = b, a
		}
		c.Emit(mir.Instr{Op: vecAluOp{verb: op.Verb}, Defs: []mir.VReg{dst}, Uses: []mir.VReg{a, b}})
		return nil
	}

	if _, ok := vecShuffle[op.Verb]; ok {
		a, err := arg(0)
		if err != nil {
			return err
		}
		k, ok := in.Lit()
		if !ok {
			return fmt.Errorf("%s: the pattern is not a literal", op)
		}
		dst, err := def()
		if err != nil {
			return err
		}
		c.Emit(mir.Instr{Op: vecShuffleOp{verb: op.Verb, k: k.Int() & 0xff},
			Defs: []mir.VReg{dst}, Uses: []mir.VReg{a}})
		return nil
	}

	switch op.Verb {
	case ir.VVecShlBytes, ir.VVecShrBytes:
		a, err := arg(0)
		if err != nil {
			return err
		}
		k, ok := in.Lit()
		if !ok {
			return fmt.Errorf("%s: the byte count is not a literal", op)
		}
		dst, err := def()
		if err != nil {
			return err
		}
		c.Emit(mir.Instr{Op: vecImmOp{verb: op.Verb, k: k.Int() & 0xff},
			Defs: []mir.VReg{dst}, Uses: []mir.VReg{a}})
		return nil

	case ir.VNot:
		// No PNOT. All-ones from PCMPEQD of a register with itself, then
		// XOR — which is what every compiler emits, and cheaper than a
		// sixteen-byte load of the same constant.
		a, err := arg(0)
		if err != nil {
			return err
		}
		dst, err := def()
		if err != nil {
			return err
		}
		ones := vr.temp(wv128)
		c.Emit(mir.Instr{Op: vecOnesOp{}, Defs: []mir.VReg{ones}})
		c.Emit(mir.Instr{Op: vecAluOp{verb: ir.VXor}, Defs: []mir.VReg{dst}, Uses: []mir.VReg{a, ones}})
		return nil

	case ir.VVecZExtI32, ir.VVecZExtI64:
		a, err := arg(0)
		if err != nil {
			return err
		}
		dst, err := def()
		if err != nil {
			return err
		}
		w := w32
		if op.Verb == ir.VVecZExtI64 {
			w = w64
		}
		c.Emit(mir.Instr{Op: vecZExtOp{w: w}, Defs: []mir.VReg{dst}, Uses: []mir.VReg{a}})
		return nil

	case ir.VI8x16Splat, ir.VI16x8Splat, ir.VI32x4Splat, ir.VI64x2Splat:
		a, err := arg(0)
		if err != nil {
			return err
		}
		dst, err := def()
		if err != nil {
			return err
		}
		c.Emit(mir.Instr{Op: vecSplatOp{verb: op.Verb}, Defs: []mir.VReg{dst}, Uses: []mir.VReg{a}})
		return nil

	case ir.VI8x16Bitmask:
		a, err := arg(0)
		if err != nil {
			return err
		}
		dst, err := def()
		if err != nil {
			return err
		}
		c.Emit(mir.Instr{Op: vecBitmaskOp{}, Defs: []mir.VReg{dst}, Uses: []mir.VReg{a}})
		return nil

	case ir.VI32x4Extract, ir.VI64x2Extract:
		a, err := arg(0)
		if err != nil {
			return err
		}
		lane, ok := in.Lit()
		if !ok {
			return fmt.Errorf("%s: the lane index is not a literal", op)
		}
		dst, err := def()
		if err != nil {
			return err
		}
		// Lane zero is MOVD or MOVQ. A higher one is a permute first,
		// into a scratch register, because the extract itself only
		// reads the low one and PSHUFD writes what it is given.
		w, src := w32, a
		k := lane.Int() & 3
		if op.Verb == ir.VI64x2Extract {
			w, k = w64, lane.Int()&1
			k *= 2 // a quadword lane is two doubleword lanes along
		}
		if k != 0 {
			perm := vr.temp(wv128)
			pat := int64(0)
			for i := int64(0); i < 4; i++ {
				pat |= ((k + i) & 3) << (2 * i)
			}
			c.Emit(mir.Instr{Op: vecShuffleOp{verb: ir.VI32x4Shuffle, k: pat},
				Defs: []mir.VReg{perm}, Uses: []mir.VReg{a}})
			src = perm
		}
		c.Emit(mir.Instr{Op: vecLaneOutOp{w: w}, Defs: []mir.VReg{dst}, Uses: []mir.VReg{src}})
		return nil

	case ir.VI16x8ExtractU:
		a, err := arg(0)
		if err != nil {
			return err
		}
		lane, ok := in.Lit()
		if !ok {
			return fmt.Errorf("%s: the lane index is not a literal", op)
		}
		dst, err := def()
		if err != nil {
			return err
		}
		c.Emit(mir.Instr{Op: vecExtractOp{lane: lane.Int() & 7},
			Defs: []mir.VReg{dst}, Uses: []mir.VReg{a}})
		return nil

	case ir.VI16x8Replace:
		a, err := arg(0)
		if err != nil {
			return err
		}
		v, err := arg(1)
		if err != nil {
			return err
		}
		lane, ok := in.Lit()
		if !ok {
			return fmt.Errorf("%s: the lane index is not a literal", op)
		}
		dst, err := def()
		if err != nil {
			return err
		}
		c.Emit(mir.Instr{Op: vecReplaceOp{lane: lane.Int() & 7},
			Defs: []mir.VReg{dst}, Uses: []mir.VReg{a, v}})
		return nil
	}

	if _, ok := vecShiftReg[op.Verb]; ok {
		return iselVecShift(c, vr, in)
	}
	return fmt.Errorf("unsupported instruction %s", op)
}

// iselVecShift lowers a lane shift. A literal count becomes the immediate
// form; anything else moves into the low quadword of a vector register,
// which is where the by-register form reads it.
func iselVecShift(c *cursor, vr *vregs, in *ir.Inst) error {
	op := in.Op()
	a, ok := vr.lookup(in.Arg(0))
	if !ok {
		return fmt.Errorf("%s: operand defined outside the function", op)
	}
	amt, ok := vr.lookup(in.Arg(1))
	if !ok {
		return fmt.Errorf("%s: shift amount defined outside the function", op)
	}
	dst, err := vr.define(in.Result(0))
	if err != nil {
		return fmt.Errorf("%s: %w", op, err)
	}

	if k, ok := constArg(in.Arg(1)); ok {
		if k < 0 || k > 255 {
			k = 255 // any count past the lane width is the same answer: zero
		}
		c.Emit(mir.Instr{Op: vecImmOp{verb: op.Verb, k: k}, Defs: []mir.VReg{dst}, Uses: []mir.VReg{a}})
		return nil
	}

	count := vr.temp(wv128)
	c.Emit(mir.Instr{Op: vecZExtOp{w: w32}, Defs: []mir.VReg{count}, Uses: []mir.VReg{amt}})
	c.Emit(mir.Instr{Op: vecShiftOp{verb: op.Verb}, Defs: []mir.VReg{dst}, Uses: []mir.VReg{a, count}})
	return nil
}

// constArg reports the literal a definition holds, if it is an i32.const
// and nothing else. A shift by a literal is the common case and the one the
// immediate form exists for.
func constArg(d *ir.Def) (int64, bool) {
	if d == nil {
		return 0, false
	}
	in := d.Inst()
	if in == nil || in.Op().Verb != ir.VConst {
		return 0, false
	}
	lit, ok := in.Lit()
	if !ok || lit.Kind() != ir.ConstInt {
		return 0, false
	}
	return lit.Int(), true
}

// iselVecConst materializes a v128 literal: sixteen bytes, which is two
// quadwords in .rodata unless every one of them is zero, and then it is a
// PXOR of a register with itself.
func iselVecConst(c *cursor, vr *vregs, in *ir.Inst, b []byte) error {
	dst, err := vr.define(in.Result(0))
	if err != nil {
		return fmt.Errorf("%s: %w", in.Op(), err)
	}
	if len(b) != 16 {
		return fmt.Errorf("%s: a v128 literal is sixteen bytes, not %d", in.Op(), len(b))
	}

	var lo, hi uint64
	for i := 0; i < 8; i++ {
		lo |= uint64(b[i]) << (8 * i)
		hi |= uint64(b[8+i]) << (8 * i)
	}
	if lo == 0 && hi == 0 {
		c.Emit(mir.Instr{Op: vecZeroOp{}, Defs: []mir.VReg{dst}})
		return nil
	}
	c.Emit(mir.Instr{Op: wideConstOp{lo: lo, hi: hi}, Defs: []mir.VReg{dst}})
	return nil
}

// vecZeroOp is the all-zero register. PXOR of a register with itself is two
// bytes and reads nothing, which no load of sixteen zero bytes can match.
type vecZeroOp struct{}

// iselVecSelect lowers §F's select at v128. There is no conditional move in
// the vector file, so the condition becomes a mask — zero or all-ones across
// every lane — and the answer is a bitwise blend. Branch-free, which is what
// select is for.
func iselVecSelect(c *cursor, vr *vregs, cond, yes, no, dst mir.VReg) {
	// NEG turns 0 into 0 and 1 into -1, which is the mask, in a general
	// register; MOVD and a broadcast put it in every lane.
	wide := vr.temp(w32)
	emitCopy(c, wide, cond, w32)
	c.Emit(mir.Instr{Op: unOp{verb: ir.VNeg, w: w32}, Defs: []mir.VReg{wide}, Uses: []mir.VReg{wide}})

	mask := vr.temp(wv128)
	c.Emit(mir.Instr{Op: vecZExtOp{w: w32}, Defs: []mir.VReg{mask}, Uses: []mir.VReg{wide}})
	c.Emit(mir.Instr{Op: vecShuffleOp{verb: ir.VI32x4Shuffle, k: 0}, Defs: []mir.VReg{mask}, Uses: []mir.VReg{mask}})

	// (yes AND mask) OR (no AND NOT mask), the second half through PANDN,
	// whose destination is the operand it negates.
	taken := vr.temp(wv128)
	c.Emit(mir.Instr{Op: vecAluOp{verb: ir.VAnd}, Defs: []mir.VReg{taken}, Uses: []mir.VReg{yes, mask}})
	other := vr.temp(wv128)
	c.Emit(mir.Instr{Op: vecAluOp{verb: ir.VVecAndNot}, Defs: []mir.VReg{other}, Uses: []mir.VReg{mask, no}})
	c.Emit(mir.Instr{Op: vecAluOp{verb: ir.VOr}, Defs: []mir.VReg{dst}, Uses: []mir.VReg{taken, other}})
}

// ---- emission -------------------------------------------------------------

// emitVec writes one §V op. It reports whether it recognized the op, so the
// caller's switch can fall through to its own default.
func emitVec(text *amd64asm.Section, in mir.Instr, xmm func(mir.VReg) reg.Xmm,
	r32 func(mir.VReg) reg.R32, r64 func(mir.VReg) reg.R64) bool {

	// two prepares a two-address op: the destination holds the first
	// operand, copied there if it is not already, and the second operand
	// is what the instruction reads.
	two := func() (reg.Xmm, reg.Xmm) {
		dst, a, b := xmm(in.Defs[0]), xmm(in.Uses[0]), xmm(in.Uses[1])
		if dst != a {
			text.MovdqaXmmRM128(dst, a)
		}
		return dst, b
	}

	switch op := in.Op.(type) {
	case vecAluOp:
		dst, b := two()
		vecAlu[op.verb](text, dst, b)

	case vecShiftOp:
		dst, b := two()
		vecShiftReg[op.verb](text, dst, b)

	case vecImmOp:
		dst, a := xmm(in.Defs[0]), xmm(in.Uses[0])
		if dst != a {
			text.MovdqaXmmRM128(dst, a)
		}
		vecShiftImm[op.verb](text, dst, op.k)

	case vecShuffleOp:
		// Source, not destination: these read the operand register, so
		// there is nothing to copy first.
		vecShuffle[op.verb](text, xmm(in.Defs[0]), xmm(in.Uses[0]), op.k)

	case vecZeroOp:
		d := xmm(in.Defs[0])
		text.PxorXmmRM128(d, d)

	case vecOnesOp:
		d := xmm(in.Defs[0])
		text.PcmpeqdXmmRM128(d, d)

	case vecZExtOp:
		if op.w == w32 {
			text.MovdXmmRM32(xmm(in.Defs[0]), r32(in.Uses[0]))
			break
		}
		text.MovqXmmRM64(xmm(in.Defs[0]), r64(in.Uses[0]))

	case vecLaneOutOp:
		if op.w == w32 {
			text.MovdRM32Xmm(r32(in.Defs[0]), xmm(in.Uses[0]))
			break
		}
		text.MovqRM64Xmm(r64(in.Defs[0]), xmm(in.Uses[0]))

	case vecBitmaskOp:
		text.PmovmskbR32Xmm(r32(in.Defs[0]), xmm(in.Uses[0]))

	case vecExtractOp:
		text.PextrwR32XmmImm8(r32(in.Defs[0]), xmm(in.Uses[0]), op.lane)

	case vecReplaceOp:
		dst, a := xmm(in.Defs[0]), xmm(in.Uses[0])
		if dst != a {
			text.MovdqaXmmRM128(dst, a)
		}
		text.PinsrwXmmRM32Imm8(dst, r32(in.Uses[1]), op.lane)

	case vecSplatOp:
		emitSplat(text, op.verb, xmm(in.Defs[0]), in, r32, r64)

	default:
		return false
	}
	return true
}

// emitSplat writes one lane from a general register and then spreads it.
//
// Each width is the next one's problem solved once more: a byte becomes a
// word by interleaving the register with itself, that word becomes a
// doubleword the same way, and a doubleword becomes the whole register with
// one shuffle. Nothing here is a peephole — it is the sequence the hardware
// leaves, since SSE2 has no broadcast.
func emitSplat(text *amd64asm.Section, verb ir.Verb, dst reg.Xmm, in mir.Instr,
	r32 func(mir.VReg) reg.R32, r64 func(mir.VReg) reg.R64) {

	if verb == ir.VI64x2Splat {
		text.MovqXmmRM64(dst, r64(in.Uses[0]))
		text.PunpcklqdqXmmRM128(dst, dst)
		return
	}

	text.MovdXmmRM32(dst, r32(in.Uses[0]))
	switch verb {
	case ir.VI8x16Splat:
		text.PunpcklbwXmmRM128(dst, dst)
		text.PunpcklwdXmmRM128(dst, dst)
	case ir.VI16x8Splat:
		text.PunpcklwdXmmRM128(dst, dst)
	}
	text.PshufdXmmRM128Imm8(dst, dst, 0)
}

// statedUnaligned reports whether a §D4 align attribute takes an access
// below what the instruction that would otherwise carry it requires.
//
// Only the sixteen-byte vector moves care. Every narrower access on this
// target is indifferent to alignment — an unaligned MOV is a MOV — so a
// stated align on one of those is a fact for the optimizer and not for the
// encoder.
func statedUnaligned(in *ir.Inst, w width) bool {
	if w != wv128 {
		return false
	}
	align, stated := in.Align()
	return stated && align < 16
}

// vecLaneOutOp is the low lane into a general register: MOVD at w32 and MOVQ
// at w64, the inverse of vecZExtOp. Which lane it is was decided before this,
// by a permute or by there being nothing to do.
type vecLaneOutOp struct{ w width }
