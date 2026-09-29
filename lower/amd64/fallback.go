package amd64

// What a gated verb becomes on a processor without its instruction.
//
// popcnt, clz and ctz carry CPUID bits (POPCNT, LZCNT, BMI1), and so do the
// rounding verbs (SSE4.1's ROUNDSS/ROUNDSD) and fma (FMA3). The baseline
// processor -- x86-64 as AMD first shipped it, and what an unqualified
// x86_64 target means to every toolchain -- has none of them. A frontend
// writes these verbs whatever the processor, because the language defines
// them whatever the processor: Swift's leadingZeroBitCount and rounded(.towardZero)
// are there on every machine. So the lowering is the processor's question
// and not the frontend's, and the answer on a processor without the
// instruction is the one LLVM gives:
//
//   - clz and ctz are BSR and BSF, which every AMD64 processor has, with a
//     conditional move for zero -- the one input where BSR and BSF leave
//     their destination undefined and LZCNT and TZCNT answer the width.
//   - popcnt is the SWAR sum: bits counted in pairs, nibbles, bytes, and the
//     bytes summed by one multiply.
//   - The rounding verbs and fma are calls to the C library's ceil, floor,
//     trunc, nearbyint and fma, which is what they are defined to be.
//
// gatedVerbs says which feature makes the instruction available; nothing is
// refused for the lack of one.

import (
	"fmt"

	amd64asm "github.com/vertex-language/amd64"
	"github.com/vertex-language/amd64/feature"
	"github.com/vertex-language/amd64/reg"

	"github.com/vertex-language/ir"
	"github.com/vertex-language/ir/lower/mir"
)

// gatedVerbs is the verbs whose instruction carries a CPUID bit.
var gatedVerbs = map[ir.Verb]feature.Feature{
	ir.VPopcnt:  feature.POPCNT,
	ir.VClz:     feature.LZCNT,
	ir.VCtz:     feature.BMI1,
	ir.VFMA:     feature.FMA,
	ir.VCeil:    feature.SSE41,
	ir.VFloor:   feature.SSE41,
	ir.VTrunc:   feature.SSE41,
	ir.VNearest: feature.SSE41,
}

// native reports whether the processor has the instruction a verb lowers to
// directly. A verb that is not gated always has.
func native(set feature.Set, v ir.Verb) bool {
	f, gated := gatedVerbs[v]
	return !gated || set.Has(f)
}

// fallbackLibcalls is the C library function each float verb is, by width,
// for a processor without the instruction.
var fallbackLibcalls = map[ir.Verb][2]string{
	ir.VCeil:    {"ceilf", "ceil"},
	ir.VFloor:   {"floorf", "floor"},
	ir.VTrunc:   {"truncf", "trunc"},
	ir.VNearest: {"nearbyintf", "nearbyint"},
	ir.VFMA:     {"fmaf", "fma"},
}

// fallbackLibcall is the C function a float verb calls on this processor,
// or false when it has the instruction or the verb is not one of these.
// nearbyint rounds to nearest, ties to even, in the default rounding mode,
// which is ir's nearest; and it raises no inexact, as ROUNDSD with the
// exception bit set does not.
func fallbackLibcall(set feature.Set, v ir.Verb, t ir.RegType) (string, bool) {
	names, ok := fallbackLibcalls[v]
	if !ok || native(set, v) {
		return "", false
	}
	if t == ir.TypeF32 {
		return names[0], true
	}
	return names[1], true
}

// iselFloatLibcall lowers a rounding verb or fma as a call to the C library.
func iselFloatLibcall(c *cursor, vr *vregs, in *ir.Inst, sym string) error {
	dst, err := vr.define(in.Result(0))
	if err != nil {
		return err
	}
	w := vr.widthOfVReg(dst)
	args := make([]libArg, len(in.Args()))
	for i, a := range in.Args() {
		v, ok := vr.lookup(a)
		if !ok {
			return fmt.Errorf("%s: operand %d defined outside the function", in.Op(), i)
		}
		args[i] = libArg{v: v, w: w}
	}
	return emitLibcall(c, vr, sym, args, dst, true)
}

// bitCountFallbackOp is clz, ctz or popcnt on a processor without LZCNT,
// TZCNT or POPCNT: a sequence rather than an instruction, over scratch
// registers. Defs are the result and one scratch, and a second for a 64-bit
// popcnt, whose masks do not fit an immediate; Uses the operand. See
// emitBitCountFallback.
type bitCountFallbackOp struct {
	verb ir.Verb
	w    width
}

// iselBitCountFallback lowers one of them.
func iselBitCountFallback(c *cursor, vr *vregs, in *ir.Inst, src, dst mir.VReg, w width) {
	defs := []mir.VReg{dst, vr.temp(w)}
	if in.Op().Verb == ir.VPopcnt && w == w64 {
		defs = append(defs, vr.temp(w))
	}
	c.Emit(mir.Instr{
		Op:   bitCountFallbackOp{verb: in.Op().Verb, w: w},
		Defs: defs,
		Uses: []mir.VReg{src},
	})
}

// emitBitCountFallback writes one bitCountFallbackOp. The operand is read
// by the first instruction and never again, so the allocator may give any
// of the definitions its register. For a width of w bits:
//
//	clz     bsr   d, x      the highest set bit's index; ZF when x is 0
//	        mov   t, 2w-1
//	        cmovz d, t      2w-1 for zero, which the xor makes w
//	        xor   d, w-1    w-1-index, as the index is below w
//
//	ctz     bsf   d, x      the lowest set bit's index; ZF when x is 0
//	        mov   t, w
//	        cmovz d, t
//
//	popcnt  d = x - ((x >> 1) & 0x55..)              pairs
//	        d = (d & 0x33..) + ((d >> 2) & 0x33..)   nibbles
//	        d = (d + (d >> 4)) & 0x0f..              bytes
//	        d = (d * 0x01..) >> (w-8)                the bytes summed
//
// with u holding the masks a 64-bit form cannot take as an immediate.
func emitBitCountFallback(text *amd64asm.Section, op bitCountFallbackOp, d, t, u, x reg.R64) {
	if op.w == w32 {
		emitBitCountFallback32(text, op.verb, reg.R32(d), reg.R32(t), reg.R32(x))
		return
	}
	switch op.verb {
	case ir.VClz:
		text.BsrR64RM64(d, x)
		text.MovR32Imm32(reg.R32(t), 127)
		text.CmovzR64RM64(d, t)
		text.XorRM64Imm8(d, 63)
	case ir.VCtz:
		text.BsfR64RM64(d, x)
		text.MovR32Imm32(reg.R32(t), 64)
		text.CmovzR64RM64(d, t)
	case ir.VPopcnt:
		if d != x {
			text.MovR64RM64(d, x)
		}
		text.MovR64RM64(t, d)
		text.ShrRM64Imm8(t, 1)
		text.MovR64Imm64(u, 0x5555555555555555)
		text.AndR64RM64(t, u)
		text.SubR64RM64(d, t)
		text.MovR64RM64(t, d)
		text.ShrRM64Imm8(t, 2)
		text.MovR64Imm64(u, 0x3333333333333333)
		text.AndR64RM64(t, u)
		text.AndR64RM64(d, u)
		text.AddR64RM64(d, t)
		text.MovR64RM64(t, d)
		text.ShrRM64Imm8(t, 4)
		text.AddR64RM64(d, t)
		text.MovR64Imm64(u, 0x0f0f0f0f0f0f0f0f)
		text.AndR64RM64(d, u)
		text.MovR64Imm64(u, 0x0101010101010101)
		text.ImulR64RM64(d, u)
		text.ShrRM64Imm8(d, 56)
	}
}

// emitBitCountFallback32 is the 32-bit forms, whose masks fit an immediate
// and so need one scratch register rather than two.
func emitBitCountFallback32(text *amd64asm.Section, verb ir.Verb, d, t, x reg.R32) {
	switch verb {
	case ir.VClz:
		text.BsrR32RM32(d, x)
		text.MovR32Imm32(t, 63)
		text.CmovzR32RM32(d, t)
		text.XorRM32Imm8(d, 31)
	case ir.VCtz:
		text.BsfR32RM32(d, x)
		text.MovR32Imm32(t, 32)
		text.CmovzR32RM32(d, t)
	case ir.VPopcnt:
		if d != x {
			text.MovR32RM32(d, x)
		}
		text.MovR32RM32(t, d)
		text.ShrRM32Imm8(t, 1)
		text.AndRM32Imm32(t, 0x55555555)
		text.SubR32RM32(d, t)
		text.MovR32RM32(t, d)
		text.ShrRM32Imm8(t, 2)
		text.AndRM32Imm32(t, 0x33333333)
		text.AndRM32Imm32(d, 0x33333333)
		text.AddR32RM32(d, t)
		text.MovR32RM32(t, d)
		text.ShrRM32Imm8(t, 4)
		text.AddR32RM32(d, t)
		text.AndRM32Imm32(d, 0x0f0f0f0f)
		text.ImulR32RM32Imm32(d, d, 0x01010101)
		text.ShrRM32Imm8(d, 24)
	}
}
