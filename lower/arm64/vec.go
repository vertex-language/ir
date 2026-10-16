package arm64

// §V's v128 namespace and §V5's and §A8's cryptographic verbs, lowered onto
// Advanced SIMD and the cryptographic extensions.
//
// Advanced SIMD is the floor: it is part of what AArch64 means, so §V's rows
// need no feature. §V's verbs are SSE2's shapes, and most are one NEON
// instruction; where NEON cuts an operation differently -- the high half of
// a product, a sum of absolute differences, a byte mask -- the sequence is
// written here, once, so that a reader comparing §V with the architecture
// has one place to look. The cryptographic verbs reach this file only where
// the function may execute them (LegalizeCrypto made the rest calls), and
// each is the instruction its capability names.
//
// The MIR op is one shape for all of them: an emitter that writes the
// instruction given the registers allocation chose, Defs first then Uses.
// An instruction that reads its destination -- AESE, BSL, the SHA rounds,
// INS -- is given its first operand copied into the result first, and reads
// and writes that vreg, which is how allocation keeps the other operands
// out of its register.

import (
	"fmt"

	arm64asm "github.com/vertex-language/arm64"
	"github.com/vertex-language/arm64/feature"
	"github.com/vertex-language/arm64/reg"
	"github.com/vertex-language/ir"
	"github.com/vertex-language/ir/lower/mir"
	"github.com/vertex-language/ir/lower/regalloc"
)

// vecOp is one vector instruction, or a short fixed sequence of them.
type vecOp struct {
	name string
	emit func(t *arm64asm.Section, r []regalloc.PhysReg)
}

func (o vecOp) String() string { return o.name }

// emitVec writes a vecOp with its registers, Defs then Uses.
func emitVec(text *arm64asm.Section, op vecOp, in mir.Instr, assigned map[mir.VReg]regalloc.PhysReg) {
	r := make([]regalloc.PhysReg, 0, len(in.Defs)+len(in.Uses))
	for _, d := range in.Defs {
		r = append(r, assigned[d])
	}
	for _, u := range in.Uses {
		r = append(r, assigned[u])
	}
	op.emit(text, r)
}

func v(r regalloc.PhysReg) reg.V { return reg.V(r) }

// vsel is the selection state of one function's vector verbs.
type vsel struct {
	c    *cursor
	vr   *vregs
	in   *ir.Inst
	feat arm64asm.FeatureSet
}

func (s *vsel) fail(why string) error { return fmt.Errorf("%s: %s", s.in.Op(), why) }

func (s *vsel) arg(i int) (mir.VReg, error) {
	r, ok := s.vr.lookup(s.in.Arg(i))
	if !ok {
		return 0, s.fail(fmt.Sprintf("operand %d defined outside the function", i))
	}
	return r, nil
}

func (s *vsel) args() ([]mir.VReg, error) {
	out := make([]mir.VReg, s.in.NumArgs())
	for i := range out {
		r, err := s.arg(i)
		if err != nil {
			return nil, err
		}
		out[i] = r
	}
	return out, nil
}

func (s *vsel) lit() (int64, error) {
	k, ok := s.in.Lit()
	if !ok || k.Kind() != ir.ConstInt {
		return 0, s.fail("needs its literal")
	}
	return k.Int(), nil
}

func (s *vsel) result() (mir.VReg, error) {
	r, err := s.vr.define(s.in.Result(0))
	if err != nil {
		return 0, s.fail(err.Error())
	}
	return r, nil
}

// op emits name with dst defined and uses read.
func (s *vsel) op(name string, dst mir.VReg, uses []mir.VReg, f func(t *arm64asm.Section, r []regalloc.PhysReg)) {
	var defs []mir.VReg
	if dst != noVReg {
		defs = []mir.VReg{dst}
	}
	s.c.Emit(mir.Instr{Op: vecOp{name: name, emit: f}, Defs: defs, Uses: uses})
}

// tied emits an instruction that reads its destination: the first operand
// is copied into dst, and dst is both read and written.
func (s *vsel) tied(name string, dst mir.VReg, first mir.VReg, rest []mir.VReg, f func(t *arm64asm.Section, r []regalloc.PhysReg)) {
	emitCopy(s.c, dst, first, wv128)
	s.op(name, dst, append([]mir.VReg{dst}, rest...), f)
}

// temp is a fresh vector vreg.
func (s *vsel) temp() mir.VReg { return s.vr.temp(wv128) }

// zero is a fresh vreg holding sixteen zero bytes.
func (s *vsel) zero() mir.VReg {
	z := s.temp()
	s.op("movi #0", z, nil, func(t *arm64asm.Section, r []regalloc.PhysReg) { t.MoviV2D(v(r[0]), 0) })
	return z
}

// bytes is a fresh vreg holding a constant: MOVI where every byte is 0x00
// or 0xff, and otherwise the two halves through general registers.
func (s *vsel) bytes(b [16]byte) mir.VReg {
	lo := uint64(0)
	hi := uint64(0)
	for i := 7; i >= 0; i-- {
		lo = lo<<8 | uint64(b[i])
		hi = hi<<8 | uint64(b[i+8])
	}
	mask := true
	for _, x := range b {
		if x != 0 && x != 0xff {
			mask = false
		}
	}
	d := s.temp()
	if mask && lo == hi {
		s.op("movi", d, nil, func(t *arm64asm.Section, r []regalloc.PhysReg) { t.MoviV2D(v(r[0]), lo) })
		return d
	}
	gl := s.vr.temp(w64)
	s.c.Emit(mir.Instr{Op: constOp{imm: int64(lo), w: w64}, Defs: []mir.VReg{gl}})
	s.op("fmov d, x", d, []mir.VReg{gl}, func(t *arm64asm.Section, r []regalloc.PhysReg) {
		t.FmovXToD(reg.D(r[0]), reg.X(r[1]))
	})
	if hi != 0 {
		gh := s.vr.temp(w64)
		s.c.Emit(mir.Instr{Op: constOp{imm: int64(hi), w: w64}, Defs: []mir.VReg{gh}})
		s.op("ins d[1]", d, []mir.VReg{d, gh}, func(t *arm64asm.Section, r []regalloc.PhysReg) {
			t.InsDGen(v(r[0]), 1, reg.X(r[2]))
		})
	}
	return d
}

// noVReg is no destination: an instruction defining nothing.
const noVReg mir.VReg = 1<<31 - 1

// has reports whether the target has f, for the rows that are one
// instruction with it and two or three without.
func (s *vsel) has(f feature.Feature) bool { return s.feat.Has(f) }

type emitFn = func(t *arm64asm.Section, r []regalloc.PhysReg)

// bin2 is a three-register instruction on r[0] = r[1] op r[2].
func bin2(f func(t *arm64asm.Section, d, n, m reg.V)) emitFn {
	return func(t *arm64asm.Section, r []regalloc.PhysReg) { f(t, v(r[0]), v(r[1]), v(r[2])) }
}

func un1(f func(t *arm64asm.Section, d, n reg.V)) emitFn {
	return func(t *arm64asm.Section, r []regalloc.PhysReg) { f(t, v(r[0]), v(r[1])) }
}

// vecBin is every §V verb that is one three-register instruction, operands
// in the IR's order.
var vecBin = map[ir.Verb]func(t *arm64asm.Section, d, n, m reg.V){
	ir.VAnd: (*arm64asm.Section).AndV16B, ir.VOr: (*arm64asm.Section).OrrV16B,
	ir.VXor: (*arm64asm.Section).EorV16B, ir.VVecAndNot: (*arm64asm.Section).BicV16B,

	ir.VI8x16Add: (*arm64asm.Section).AddV16B, ir.VI16x8Add: (*arm64asm.Section).AddV8H,
	ir.VI32x4Add: (*arm64asm.Section).AddV4S, ir.VI64x2Add: (*arm64asm.Section).AddV2D,
	ir.VI8x16Sub: (*arm64asm.Section).SubV16B, ir.VI16x8Sub: (*arm64asm.Section).SubV8H,
	ir.VI32x4Sub: (*arm64asm.Section).SubV4S, ir.VI64x2Sub: (*arm64asm.Section).SubV2D,
	ir.VI16x8Mul: (*arm64asm.Section).MulV8H,

	ir.VI8x16AddSatS: (*arm64asm.Section).SqaddV16B, ir.VI8x16AddSatU: (*arm64asm.Section).UqaddV16B,
	ir.VI16x8AddSatS: (*arm64asm.Section).SqaddV8H, ir.VI16x8AddSatU: (*arm64asm.Section).UqaddV8H,
	ir.VI8x16SubSatS: (*arm64asm.Section).SqsubV16B, ir.VI8x16SubSatU: (*arm64asm.Section).UqsubV16B,
	ir.VI16x8SubSatS: (*arm64asm.Section).SqsubV8H, ir.VI16x8SubSatU: (*arm64asm.Section).UqsubV8H,

	ir.VI8x16MinU: (*arm64asm.Section).UminV16B, ir.VI8x16MaxU: (*arm64asm.Section).UmaxV16B,
	ir.VI16x8MinS: (*arm64asm.Section).SminV8H, ir.VI16x8MaxS: (*arm64asm.Section).SmaxV8H,
	ir.VI8x16AvgrU: (*arm64asm.Section).UrhaddV16B, ir.VI16x8AvgrU: (*arm64asm.Section).UrhaddV8H,

	ir.VI8x16Eq: (*arm64asm.Section).CmeqV16B, ir.VI16x8Eq: (*arm64asm.Section).CmeqV8H,
	ir.VI32x4Eq: (*arm64asm.Section).CmeqV4S,
	ir.VI8x16GtS: (*arm64asm.Section).CmgtV16B, ir.VI16x8GtS: (*arm64asm.Section).CmgtV8H,
	ir.VI32x4GtS: (*arm64asm.Section).CmgtV4S,

	ir.VI8x16UnpackLow: (*arm64asm.Section).Zip1V16B, ir.VI8x16UnpackHi: (*arm64asm.Section).Zip2V16B,
	ir.VI16x8UnpackLow: (*arm64asm.Section).Zip1V8H, ir.VI16x8UnpackHi: (*arm64asm.Section).Zip2V8H,
	ir.VI32x4UnpackLow: (*arm64asm.Section).Zip1V4S, ir.VI32x4UnpackHi: (*arm64asm.Section).Zip2V4S,
	ir.VI64x2UnpackLow: (*arm64asm.Section).Zip1V2D, ir.VI64x2UnpackHi: (*arm64asm.Section).Zip2V2D,

	// TBL's out-of-range rule is §V4's: an index of 16 or more is zero.
	ir.VI8x16Swizzle: func(t *arm64asm.Section, d, n, m reg.V) { t.TblV16B(d, n, m) },
}

// vecUn is every §V verb that is one two-register instruction.
var vecUn = map[ir.Verb]func(t *arm64asm.Section, d, n reg.V){
	ir.VNot:         (*arm64asm.Section).NotV16B,
	ir.VI16x8Bswap:  (*arm64asm.Section).Rev16V16B,
	ir.VI32x4Bswap:  (*arm64asm.Section).Rev32V16B,
	ir.VI64x2Bswap:  (*arm64asm.Section).Rev64V16B,
}

// iselVec lowers one v128 instruction, a §A8 checksum, or i1.cpu_has, and
// reports whether it was one.
func iselVec(c *cursor, vr *vregs, in *ir.Inst, feat arm64asm.FeatureSet) (bool, error) {
	op := in.Op()
	if _, crypto := op.Verb.Capability(); !crypto && op.Type != ir.TypeV128 {
		return false, nil
	}
	if op.Verb == ir.VLoad || op.Verb == ir.VStore {
		return false, nil // §D's rows, which the general load and store handle at wv128
	}
	s := &vsel{c: c, vr: vr, in: in, feat: feat}
	return true, s.sel()
}

func (s *vsel) sel() error {
	verb := s.in.Op().Verb
	if f, ok := vecBin[verb]; ok {
		a, err := s.args()
		if err != nil {
			return err
		}
		d, err := s.result()
		if err != nil {
			return err
		}
		s.op(string(verb), d, a, bin2(f))
		return nil
	}
	if f, ok := vecUn[verb]; ok {
		a, err := s.args()
		if err != nil {
			return err
		}
		d, err := s.result()
		if err != nil {
			return err
		}
		s.op(string(verb), d, a, un1(f))
		return nil
	}
	if s.in.Op().Type == ir.TypeI32 {
		return s.crc()
	}
	switch verb {
	case ir.VConst:
		return s.constant()
	case ir.VSelect:
		return s.selectV()
	case ir.VVecShlBytes, ir.VVecShrBytes:
		return s.byteShift(verb == ir.VVecShlBytes)
	case ir.VVecZExtI32, ir.VVecZExtI64:
		return s.zext(verb == ir.VVecZExtI64)
	case ir.VI8x16Splat, ir.VI16x8Splat, ir.VI32x4Splat, ir.VI64x2Splat:
		return s.splat(verb)
	case ir.VI16x8ExtractU, ir.VI32x4Extract, ir.VI64x2Extract:
		return s.extract(verb)
	case ir.VI16x8Replace:
		return s.replace()
	case ir.VI16x8MulHiS, ir.VI16x8MulHiU:
		return s.mulHi(verb == ir.VI16x8MulHiS)
	case ir.VI32x4MulEvenU:
		return s.mulEven()
	case ir.VI8x16SadU:
		return s.sad()
	case ir.VI16x8MaddS:
		return s.madd()
	case ir.VI8x16NarrowS, ir.VI8x16NarrowU, ir.VI16x8NarrowS:
		return s.narrow(verb)
	case ir.VI8x16Bitmask:
		return s.bitmask()
	case ir.VI16x8Shl, ir.VI32x4Shl, ir.VI64x2Shl, ir.VI16x8ShrU, ir.VI32x4ShrU, ir.VI64x2ShrU,
		ir.VI16x8ShrS, ir.VI32x4ShrS:
		return s.shift(verb)
	case ir.VI32x4Shuffle, ir.VI16x8ShuffleLow, ir.VI16x8ShuffleHi:
		return s.shuffle(verb)
	case ir.VVecXor3, ir.VVecXorAndNot:
		return s.xor3(verb)
	case ir.VI32x4Rotl, ir.VI64x2Rotl:
		return s.rotl(verb == ir.VI64x2Rotl)
	}
	return s.crypto()
}

// ---- §V1 ----------------------------------------------------------------

func (s *vsel) constant() error {
	k, ok := s.in.Lit()
	if !ok || k.Kind() != ir.ConstBytes || len(k.Bytes()) != 16 {
		return s.fail("needs sixteen bytes")
	}
	d, err := s.result()
	if err != nil {
		return err
	}
	var b [16]byte
	copy(b[:], k.Bytes())
	emitCopy(s.c, d, s.bytes(b), wv128)
	return nil
}

// selectV is one condition for every lane: a mask of all ones or none from
// the i1, and BSL by it.
func (s *vsel) selectV() error {
	a, err := s.args()
	if err != nil {
		return err
	}
	d, err := s.result()
	if err != nil {
		return err
	}
	m := s.vr.temp(w64)
	s.c.Emit(mir.Instr{Op: vecGPROp{name: "sbfx", emit: func(t *arm64asm.Section, r []regalloc.PhysReg) {
		t.Sbfx64(reg.X(r[0]), reg.X(r[1]), 0, 1)
	}}, Defs: []mir.VReg{m}, Uses: []mir.VReg{a[0]}})
	s.op("dup mask", d, []mir.VReg{m}, func(t *arm64asm.Section, r []regalloc.PhysReg) { t.DupV2DGen(v(r[0]), reg.X(r[1])) })
	s.op("bsl", d, []mir.VReg{d, a[1], a[2]}, bin3tied((*arm64asm.Section).BslV16B))
	return nil
}

// bin3tied is a tied instruction r[0] = f(r[0] (read), r[2], r[3]): Uses are
// the destination, then the two sources.
func bin3tied(f func(t *arm64asm.Section, d, n, m reg.V)) emitFn {
	return func(t *arm64asm.Section, r []regalloc.PhysReg) { f(t, v(r[0]), v(r[2]), v(r[3])) }
}

// byteShift is EXT against zero: left by k is the top 16-k bytes of
// zero:a, right by k the bottom 16-k of a:zero.
func (s *vsel) byteShift(left bool) error {
	a, err := s.arg(0)
	if err != nil {
		return err
	}
	k, err := s.lit()
	if err != nil {
		return err
	}
	d, err := s.result()
	if err != nil {
		return err
	}
	switch {
	case k <= 0:
		emitCopy(s.c, d, a, wv128)
	case k >= 16:
		emitCopy(s.c, d, s.zero(), wv128)
	case left:
		s.op("ext", d, []mir.VReg{s.zero(), a}, func(t *arm64asm.Section, r []regalloc.PhysReg) {
			t.ExtV16B(v(r[0]), v(r[1]), v(r[2]), uint8(16-k))
		})
	default:
		s.op("ext", d, []mir.VReg{a, s.zero()}, func(t *arm64asm.Section, r []regalloc.PhysReg) {
			t.ExtV16B(v(r[0]), v(r[1]), v(r[2]), uint8(k))
		})
	}
	return nil
}

// zext is FMOV into the low lane, which zeroes the rest.
func (s *vsel) zext(wide bool) error {
	a, err := s.arg(0)
	if err != nil {
		return err
	}
	d, err := s.result()
	if err != nil {
		return err
	}
	s.op("fmov", d, []mir.VReg{a}, func(t *arm64asm.Section, r []regalloc.PhysReg) {
		if wide {
			t.FmovXToD(reg.D(r[0]), reg.X(r[1]))
		} else {
			t.FmovWToS(reg.S(r[0]), reg.W(r[1]))
		}
	})
	return nil
}

// ---- §V2 ----------------------------------------------------------------

// mulHi is the high halves of eight products: the two widening multiplies
// and the odd halves of their lanes.
func (s *vsel) mulHi(signed bool) error {
	a, err := s.args()
	if err != nil {
		return err
	}
	d, err := s.result()
	if err != nil {
		return err
	}
	lo, hi := s.temp(), s.temp()
	mul, mul2 := (*arm64asm.Section).UmullV4S, (*arm64asm.Section).Umull2V4S
	if signed {
		mul, mul2 = (*arm64asm.Section).SmullV4S, (*arm64asm.Section).Smull2V4S
	}
	s.op("mull", lo, a, bin2(mul))
	s.op("mull2", hi, a, bin2(mul2))
	s.op("uzp2", d, []mir.VReg{lo, hi}, bin2((*arm64asm.Section).Uzp2V8H))
	return nil
}

// mulEven is lanes 0 and 2 of each, multiplied into two i64: the even lanes
// gathered to the bottom, and one UMULL.
func (s *vsel) mulEven() error {
	a, err := s.args()
	if err != nil {
		return err
	}
	d, err := s.result()
	if err != nil {
		return err
	}
	x, y := s.temp(), s.temp()
	s.op("uzp1", x, []mir.VReg{a[0], a[0]}, bin2((*arm64asm.Section).Uzp1V4S))
	s.op("uzp1", y, []mir.VReg{a[1], a[1]}, bin2((*arm64asm.Section).Uzp1V4S))
	s.op("umull", d, []mir.VReg{x, y}, bin2((*arm64asm.Section).UmullV2D))
	return nil
}

// sad is each eight bytes' absolute differences summed into the low word
// of their quadword: UABD, then three pairwise widening adds.
func (s *vsel) sad() error {
	a, err := s.args()
	if err != nil {
		return err
	}
	d, err := s.result()
	if err != nil {
		return err
	}
	t := s.temp()
	s.op("uabd", t, a, bin2((*arm64asm.Section).UabdV16B))
	t8, t4 := s.temp(), s.temp()
	s.op("uaddlp", t8, []mir.VReg{t}, un1((*arm64asm.Section).UaddlpV8H))
	s.op("uaddlp", t4, []mir.VReg{t8}, un1((*arm64asm.Section).UaddlpV4S))
	s.op("uaddlp", d, []mir.VReg{t4}, un1((*arm64asm.Section).UaddlpV2D))
	return nil
}

// madd is PMADDWD: each pair of adjacent signed products summed.
func (s *vsel) madd() error {
	a, err := s.args()
	if err != nil {
		return err
	}
	d, err := s.result()
	if err != nil {
		return err
	}
	lo, hi := s.temp(), s.temp()
	s.op("smull", lo, a, bin2((*arm64asm.Section).SmullV4S))
	s.op("smull2", hi, a, bin2((*arm64asm.Section).Smull2V4S))
	s.op("addp", d, []mir.VReg{lo, hi}, bin2((*arm64asm.Section).AddpV4S))
	return nil
}

// narrow is the saturating narrowing of two registers into one, the first
// operand's lanes low.
func (s *vsel) narrow(verb ir.Verb) error {
	a, err := s.args()
	if err != nil {
		return err
	}
	d, err := s.result()
	if err != nil {
		return err
	}
	var lo func(t *arm64asm.Section, d, n reg.V)
	var hi func(t *arm64asm.Section, d, n reg.V)
	switch verb {
	case ir.VI8x16NarrowS:
		lo, hi = (*arm64asm.Section).SqxtnV8B, (*arm64asm.Section).Sqxtn2V16B
	case ir.VI8x16NarrowU:
		lo, hi = (*arm64asm.Section).SqxtunV8B, (*arm64asm.Section).Sqxtun2V16B
	default:
		lo, hi = (*arm64asm.Section).SqxtnV4H, (*arm64asm.Section).Sqxtn2V8H
	}
	s.op("narrow", d, []mir.VReg{a[0]}, un1(lo))
	s.op("narrow2", d, []mir.VReg{d, a[1]}, func(t *arm64asm.Section, r []regalloc.PhysReg) { hi(t, v(r[0]), v(r[2])) })
	return nil
}

// bitmask is the top bit of each byte, lane 0 in bit 0: each byte made 0 or
// 1, shifted to its place within its half, and each half's bytes summed.
func (s *vsel) bitmask() error {
	a, err := s.arg(0)
	if err != nil {
		return err
	}
	d, err := s.result()
	if err != nil {
		return err
	}
	var place [16]byte
	for i := range place {
		place[i] = byte(i % 8)
	}
	weights := s.bytes(place)
	bit, at := s.temp(), s.temp()
	s.op("ushr #7", bit, []mir.VReg{a}, func(t *arm64asm.Section, r []regalloc.PhysReg) { t.UshrV16B(v(r[0]), v(r[1]), 7) })
	s.op("ushl", at, []mir.VReg{bit, weights}, bin2((*arm64asm.Section).UshlV16B))
	hiHalf := s.temp()
	s.op("ext #8", hiHalf, []mir.VReg{at, at}, func(t *arm64asm.Section, r []regalloc.PhysReg) {
		t.ExtV16B(v(r[0]), v(r[1]), v(r[2]), 8)
	})
	lo, hi := s.temp(), s.temp()
	s.op("addv", lo, []mir.VReg{at}, func(t *arm64asm.Section, r []regalloc.PhysReg) { t.AddvV8B(reg.B(r[0]), v(r[1])) })
	s.op("addv", hi, []mir.VReg{hiHalf}, func(t *arm64asm.Section, r []regalloc.PhysReg) { t.AddvV8B(reg.B(r[0]), v(r[1])) })
	gl, gh := s.vr.temp(w32), s.vr.temp(w32)
	s.c.Emit(mir.Instr{Op: vecGPROp{name: "umov", emit: func(t *arm64asm.Section, r []regalloc.PhysReg) {
		t.UmovB(reg.W(r[0]), v(r[1]), 0)
	}}, Defs: []mir.VReg{gl}, Uses: []mir.VReg{lo}})
	s.c.Emit(mir.Instr{Op: vecGPROp{name: "umov", emit: func(t *arm64asm.Section, r []regalloc.PhysReg) {
		t.UmovB(reg.W(r[0]), v(r[1]), 0)
	}}, Defs: []mir.VReg{gh}, Uses: []mir.VReg{hi}})
	s.c.Emit(mir.Instr{Op: vecGPROp{name: "orr lsl 8", emit: func(t *arm64asm.Section, r []regalloc.PhysReg) {
		t.OrrShifted32(reg.W(r[0]), reg.W(r[1]), reg.W(r[2]), arm64asm.Shifted(arm64asm.LSL, 8))
	}}, Defs: []mir.VReg{d}, Uses: []mir.VReg{gl, gh}})
	return nil
}

// vecGPROp is a vecOp whose registers include general ones: the emitter
// reads each by what it is, as vecOp's do.
type vecGPROp = vecOp

// ---- §V3 ----------------------------------------------------------------

// shiftEmitters are a lane shift's two forms: by a literal, and by a count
// in every lane of a register (USHL and SSHL, negative for a right shift).
type shiftForm struct {
	bits  int64
	imm   func(t *arm64asm.Section, d, n reg.V, k uint8)
	byReg func(t *arm64asm.Section, d, n, m reg.V)
	dup   func(t *arm64asm.Section, d reg.V, n reg.W)
	left  bool
	sign  bool
}

var shiftForms = map[ir.Verb]shiftForm{
	ir.VI16x8Shl:  {16, (*arm64asm.Section).ShlV8H, (*arm64asm.Section).UshlV8H, (*arm64asm.Section).DupV8HGen, true, false},
	ir.VI32x4Shl:  {32, (*arm64asm.Section).ShlV4S, (*arm64asm.Section).UshlV4S, (*arm64asm.Section).DupV4SGen, true, false},
	ir.VI64x2Shl:  {64, (*arm64asm.Section).ShlV2D, (*arm64asm.Section).UshlV2D, nil, true, false},
	ir.VI16x8ShrU: {16, (*arm64asm.Section).UshrV8H, (*arm64asm.Section).UshlV8H, (*arm64asm.Section).DupV8HGen, false, false},
	ir.VI32x4ShrU: {32, (*arm64asm.Section).UshrV4S, (*arm64asm.Section).UshlV4S, (*arm64asm.Section).DupV4SGen, false, false},
	ir.VI64x2ShrU: {64, (*arm64asm.Section).UshrV2D, (*arm64asm.Section).UshlV2D, nil, false, false},
	ir.VI16x8ShrS: {16, (*arm64asm.Section).SshrV8H, (*arm64asm.Section).SshlV8H, (*arm64asm.Section).DupV8HGen, false, true},
	ir.VI32x4ShrS: {32, (*arm64asm.Section).SshrV4S, (*arm64asm.Section).SshlV4S, (*arm64asm.Section).DupV4SGen, false, true},
}

// shift is §V3's saturating lane shift. By a literal it is one instruction,
// a count at or past the lane width zero (or sign fill); by a register the
// count is clamped to the width -- USHL reads the low byte of each lane as
// signed, so 200 would be a right shift -- and broadcast.
func (s *vsel) shift(verb ir.Verb) error {
	f := shiftForms[verb]
	a, err := s.arg(0)
	if err != nil {
		return err
	}
	d, err := s.result()
	if err != nil {
		return err
	}
	if k, ok := literalOf(s.in.Arg(1)); ok && k >= 0 {
		switch {
		case k == 0:
			emitCopy(s.c, d, a, wv128)
		case k >= f.bits && f.sign:
			s.op("sshr", d, []mir.VReg{a}, func(t *arm64asm.Section, r []regalloc.PhysReg) { f.imm(t, v(r[0]), v(r[1]), uint8(f.bits)) })
		case k >= f.bits:
			emitCopy(s.c, d, s.zero(), wv128)
		default:
			s.op("shift", d, []mir.VReg{a}, func(t *arm64asm.Section, r []regalloc.PhysReg) { f.imm(t, v(r[0]), v(r[1]), uint8(k)) })
		}
		return nil
	}
	n, err := s.arg(1)
	if err != nil {
		return err
	}
	// count = min(n, width), unsigned, negated for a right shift.
	limit := s.vr.temp(w32)
	s.c.Emit(mir.Instr{Op: constOp{imm: f.bits, w: w32}, Defs: []mir.VReg{limit}})
	s.c.Emit(mir.Instr{Op: cmpOp{w: w32}, Uses: []mir.VReg{n, limit}})
	count := s.vr.temp(w32)
	s.c.Emit(mir.Instr{Op: cselOp{cond: condLO, w: w32}, Defs: []mir.VReg{count}, Uses: []mir.VReg{n, limit}})
	if !f.left {
		neg := s.vr.temp(w32)
		s.c.Emit(mir.Instr{Op: unOp{verb: ir.VNeg, w: w32}, Defs: []mir.VReg{neg}, Uses: []mir.VReg{count}})
		count = neg
	}
	by := s.temp()
	if f.dup != nil {
		s.op("dup", by, []mir.VReg{count}, func(t *arm64asm.Section, r []regalloc.PhysReg) { f.dup(t, v(r[0]), reg.W(r[1])) })
	} else {
		wide := s.vr.temp(w64)
		s.c.Emit(mir.Instr{Op: extOp{from: a32, signed: true}, Defs: []mir.VReg{wide}, Uses: []mir.VReg{count}})
		s.op("dup", by, []mir.VReg{wide}, func(t *arm64asm.Section, r []regalloc.PhysReg) { t.DupV2DGen(v(r[0]), reg.X(r[1])) })
	}
	s.op("shl by reg", d, []mir.VReg{a, by}, bin2(f.byReg))
	return nil
}

// literalOf is the value of an i32 or i64 constant operand, if it is one.
func literalOf(d *ir.Def) (int64, bool) {
	if d == nil || d.Inst() == nil || d.Inst().Op().Verb != ir.VConst {
		return 0, false
	}
	k, ok := d.Inst().Lit()
	if !ok || k.Kind() != ir.ConstInt {
		return 0, false
	}
	return k.Int(), true
}

// ---- §V4 ----------------------------------------------------------------

func (s *vsel) splat(verb ir.Verb) error {
	a, err := s.arg(0)
	if err != nil {
		return err
	}
	d, err := s.result()
	if err != nil {
		return err
	}
	s.op("dup", d, []mir.VReg{a}, func(t *arm64asm.Section, r []regalloc.PhysReg) {
		switch verb {
		case ir.VI8x16Splat:
			t.DupV16BGen(v(r[0]), reg.W(r[1]))
		case ir.VI16x8Splat:
			t.DupV8HGen(v(r[0]), reg.W(r[1]))
		case ir.VI32x4Splat:
			t.DupV4SGen(v(r[0]), reg.W(r[1]))
		default:
			t.DupV2DGen(v(r[0]), reg.X(r[1]))
		}
	})
	return nil
}

func (s *vsel) extract(verb ir.Verb) error {
	a, err := s.arg(0)
	if err != nil {
		return err
	}
	k, err := s.lit()
	if err != nil {
		return err
	}
	d, err := s.result()
	if err != nil {
		return err
	}
	s.c.Emit(mir.Instr{Op: vecGPROp{name: "umov", emit: func(t *arm64asm.Section, r []regalloc.PhysReg) {
		switch verb {
		case ir.VI16x8ExtractU:
			t.UmovH(reg.W(r[0]), v(r[1]), uint8(k&7))
		case ir.VI32x4Extract:
			t.UmovS(reg.W(r[0]), v(r[1]), uint8(k&3))
		default:
			t.UmovD(reg.X(r[0]), v(r[1]), uint8(k&1))
		}
	}}, Defs: []mir.VReg{d}, Uses: []mir.VReg{a}})
	return nil
}

func (s *vsel) replace() error {
	a, err := s.args()
	if err != nil {
		return err
	}
	k, err := s.lit()
	if err != nil {
		return err
	}
	d, err := s.result()
	if err != nil {
		return err
	}
	emitCopy(s.c, d, a[0], wv128)
	s.op("ins", d, []mir.VReg{d, a[1]}, func(t *arm64asm.Section, r []regalloc.PhysReg) {
		t.InsHGen(v(r[0]), uint8(k&7), reg.W(r[2]))
	})
	return nil
}

// shuffle is TBL by the pattern's byte indices, a constant: one instruction
// and the constant, whatever the pattern.
func (s *vsel) shuffle(verb ir.Verb) error {
	a, err := s.arg(0)
	if err != nil {
		return err
	}
	p, err := s.lit()
	if err != nil {
		return err
	}
	d, err := s.result()
	if err != nil {
		return err
	}
	var idx [16]byte
	for i := range idx {
		idx[i] = byte(i)
	}
	switch verb {
	case ir.VI32x4Shuffle:
		for lane := 0; lane < 4; lane++ {
			src := int(p>>(2*lane)) & 3
			for b := 0; b < 4; b++ {
				idx[4*lane+b] = byte(4*src + b)
			}
		}
	default:
		base := 0
		if verb == ir.VI16x8ShuffleHi {
			base = 4
		}
		for w := 0; w < 4; w++ {
			src := base + int(p>>(2*w))&3
			idx[2*(base+w)] = byte(2 * src)
			idx[2*(base+w)+1] = byte(2*src + 1)
		}
	}
	s.op("tbl", d, []mir.VReg{a, s.bytes(idx)}, bin2(func(t *arm64asm.Section, d, n, m reg.V) { t.TblV16B(d, n, m) }))
	return nil
}

// xor3 is EOR3 and BCAX with FEAT_SHA3, and two instructions without.
func (s *vsel) xor3(verb ir.Verb) error {
	a, err := s.args()
	if err != nil {
		return err
	}
	d, err := s.result()
	if err != nil {
		return err
	}
	if s.has(feature.SHA3) {
		f := (*arm64asm.Section).Eor3
		if verb == ir.VVecXorAndNot {
			f = (*arm64asm.Section).Bcax
		}
		s.op(string(verb), d, a, func(t *arm64asm.Section, r []regalloc.PhysReg) { f(t, v(r[0]), v(r[1]), v(r[2]), v(r[3])) })
		return nil
	}
	t := s.temp()
	if verb == ir.VVecXorAndNot {
		s.op("bic", t, []mir.VReg{a[1], a[2]}, bin2((*arm64asm.Section).BicV16B))
	} else {
		s.op("eor", t, []mir.VReg{a[1], a[2]}, bin2((*arm64asm.Section).EorV16B))
	}
	s.op("eor", d, []mir.VReg{a[0], t}, bin2((*arm64asm.Section).EorV16B))
	return nil
}

// rotl is a lane rotate by a literal: REV32 of halfwords for an i32 by 16,
// XAR against zero for an i64 with FEAT_SHA3, and SHL then USRA otherwise --
// the two halves have no bit in common, so their sum is their or.
func (s *vsel) rotl(wide bool) error {
	a, err := s.arg(0)
	if err != nil {
		return err
	}
	k, err := s.lit()
	if err != nil {
		return err
	}
	d, err := s.result()
	if err != nil {
		return err
	}
	bits := int64(32)
	if wide {
		bits = 64
	}
	k %= bits
	switch {
	case k == 0:
		emitCopy(s.c, d, a, wv128)
	case !wide && k == 16:
		s.op("rev32 .8h", d, []mir.VReg{a}, un1((*arm64asm.Section).Rev32V8H))
	case wide && s.has(feature.SHA3):
		s.op("xar", d, []mir.VReg{a, s.zero()}, func(t *arm64asm.Section, r []regalloc.PhysReg) {
			t.Xar(v(r[0]), v(r[1]), v(r[2]), uint8(64-k))
		})
	default:
		shl, usra := (*arm64asm.Section).ShlV4S, (*arm64asm.Section).UsraV4S
		if wide {
			shl, usra = (*arm64asm.Section).ShlV2D, (*arm64asm.Section).UsraV2D
		}
		s.op("shl", d, []mir.VReg{a}, func(t *arm64asm.Section, r []regalloc.PhysReg) { shl(t, v(r[0]), v(r[1]), uint8(k)) })
		s.op("usra", d, []mir.VReg{d, a}, func(t *arm64asm.Section, r []regalloc.PhysReg) {
			usra(t, v(r[0]), v(r[2]), uint8(bits-k))
		})
	}
	return nil
}

// ---- §A8 ----------------------------------------------------------------

func (s *vsel) crc() error {
	a, err := s.args()
	if err != nil {
		return err
	}
	d, err := s.result()
	if err != nil {
		return err
	}
	f := map[ir.Verb]func(t *arm64asm.Section, d, n, m reg.W){
		ir.VCrc32cU8: (*arm64asm.Section).Crc32cb, ir.VCrc32cU16: (*arm64asm.Section).Crc32ch,
		ir.VCrc32cU32: (*arm64asm.Section).Crc32cw, ir.VCrc32U8: (*arm64asm.Section).Crc32b,
		ir.VCrc32U16: (*arm64asm.Section).Crc32h, ir.VCrc32U32: (*arm64asm.Section).Crc32w,
	}
	verb := s.in.Op().Verb
	s.c.Emit(mir.Instr{Op: vecGPROp{name: string(verb), emit: func(t *arm64asm.Section, r []regalloc.PhysReg) {
		switch verb {
		case ir.VCrc32cU64:
			t.Crc32cx(reg.W(r[0]), reg.W(r[1]), reg.X(r[2]))
		case ir.VCrc32U64:
			t.Crc32x(reg.W(r[0]), reg.W(r[1]), reg.X(r[2]))
		default:
			f[verb](t, reg.W(r[0]), reg.W(r[1]), reg.W(r[2]))
		}
	}}, Defs: []mir.VReg{d}, Uses: a})
	return nil
}

// ---- §V5 ----------------------------------------------------------------

// aesKeygenIndex is what TBL picks from SR(SB(x)) -- AESE against zero --
// to answer aeskeygenassist's SubWord(x1), RotWord(SubWord(x1)),
// SubWord(x3), RotWord(SubWord(x3)): ShiftRows moved byte 4c+r of x to
// 4((c-r) mod 4)+r, and RotWord is the word's bytes 1, 2, 3, 0.
var aesKeygenIndex = func() [16]byte {
	var idx [16]byte
	at := func(j int) byte { c, r := j/4, j%4; return byte(4*((c-r+4)%4) + r) }
	for half, col := range []int{1, 3} {
		for b := 0; b < 4; b++ {
			idx[8*half+b] = at(4*col + b)
			idx[8*half+4+b] = at(4*col + (b+1)%4)
		}
	}
	return idx
}()

func (s *vsel) crypto() error {
	verb := s.in.Op().Verb
	a, err := s.args()
	if err != nil {
		return err
	}
	d, err := s.result()
	if err != nil {
		return err
	}
	q := func(r regalloc.PhysReg) reg.Q { return reg.Q(r) }
	switch verb {
	case ir.VAesEnc, ir.VAesEncLast, ir.VAesDec, ir.VAesDecLast:
		// AESE and AESD XOR their key before SubBytes, x86's rounds after
		// MixColumns: a round here is AESE against zero, AESMC, and the
		// key EOR'd at the end, which the next round's AESE takes back.
		t := s.temp()
		dec := verb == ir.VAesDec || verb == ir.VAesDecLast
		s.tied("aese", t, a[0], []mir.VReg{s.zero()}, func(tx *arm64asm.Section, r []regalloc.PhysReg) {
			if dec {
				tx.Aesd(v(r[0]), v(r[2]))
			} else {
				tx.Aese(v(r[0]), v(r[2]))
			}
		})
		if verb == ir.VAesEnc || verb == ir.VAesDec {
			m := s.temp()
			s.op("aesmc", m, []mir.VReg{t}, un1(func(tx *arm64asm.Section, d, n reg.V) {
				if dec {
					tx.Aesimc(d, n)
				} else {
					tx.Aesmc(d, n)
				}
			}))
			t = m
		}
		s.op("eor", d, []mir.VReg{t, a[1]}, bin2((*arm64asm.Section).EorV16B))
	case ir.VAesInvMixColumns:
		s.op("aesimc", d, a, un1((*arm64asm.Section).Aesimc))
	case ir.VAesKeygenAssist:
		rcon, err := s.lit()
		if err != nil {
			return err
		}
		t := s.temp()
		s.tied("aese", t, a[0], []mir.VReg{s.zero()}, func(tx *arm64asm.Section, r []regalloc.PhysReg) { tx.Aese(v(r[0]), v(r[2])) })
		picked := s.temp()
		s.op("tbl", picked, []mir.VReg{t, s.bytes(aesKeygenIndex)}, bin2(func(tx *arm64asm.Section, d, n, m reg.V) { tx.TblV16B(d, n, m) }))
		if rcon&0xff == 0 {
			emitCopy(s.c, d, picked, wv128)
			break
		}
		var rc [16]byte
		rc[4], rc[12] = byte(rcon), byte(rcon)
		s.op("eor rcon", d, []mir.VReg{picked, s.bytes(rc)}, bin2((*arm64asm.Section).EorV16B))
	case ir.VI64x2Clmul:
		sel, err := s.lit()
		if err != nil {
			return err
		}
		x, y := a[0], a[1]
		switch sel & 3 {
		case 0:
			s.op("pmull", d, []mir.VReg{x, y}, bin2((*arm64asm.Section).PmullV1Q))
		case 3:
			s.op("pmull2", d, []mir.VReg{x, y}, bin2((*arm64asm.Section).Pmull2V1Q))
		default:
			// One high lane and one low: the high one moved down first.
			hi := x
			if sel&3 == 2 {
				hi = y
			}
			moved := s.temp()
			s.op("ext #8", moved, []mir.VReg{hi, hi}, func(tx *arm64asm.Section, r []regalloc.PhysReg) {
				tx.ExtV16B(v(r[0]), v(r[1]), v(r[2]), 8)
			})
			if sel&3 == 1 {
				x = moved
			} else {
				y = moved
			}
			s.op("pmull", d, []mir.VReg{x, y}, bin2((*arm64asm.Section).PmullV1Q))
		}
	case ir.VSha1RoundsCh, ir.VSha1RoundsParity, ir.VSha1RoundsMaj:
		e := s.vr.temp(wf32)
		s.op("fmov s, w", e, []mir.VReg{a[1]}, func(tx *arm64asm.Section, r []regalloc.PhysReg) { tx.FmovWToS(reg.S(r[0]), reg.W(r[1])) })
		f := map[ir.Verb]func(*arm64asm.Section, reg.Q, reg.S, reg.V){
			ir.VSha1RoundsCh: (*arm64asm.Section).Sha1c, ir.VSha1RoundsParity: (*arm64asm.Section).Sha1p,
			ir.VSha1RoundsMaj: (*arm64asm.Section).Sha1m,
		}[verb]
		s.tied(string(verb), d, a[0], []mir.VReg{e, a[2]}, func(tx *arm64asm.Section, r []regalloc.PhysReg) {
			f(tx, q(r[0]), reg.S(r[2]), v(r[3]))
		})
	case ir.VSha1Msg1:
		s.tied("sha1su0", d, a[0], a[1:], func(tx *arm64asm.Section, r []regalloc.PhysReg) { tx.Sha1su0(v(r[0]), v(r[2]), v(r[3])) })
	case ir.VSha1Msg2:
		s.tied("sha1su1", d, a[0], a[1:], func(tx *arm64asm.Section, r []regalloc.PhysReg) { tx.Sha1su1(v(r[0]), v(r[2])) })
	case ir.VSha256RoundsAbcd, ir.VSha256RoundsEfgh:
		f := (*arm64asm.Section).Sha256h
		if verb == ir.VSha256RoundsEfgh {
			f = (*arm64asm.Section).Sha256h2
		}
		s.tied(string(verb), d, a[0], a[1:], func(tx *arm64asm.Section, r []regalloc.PhysReg) { f(tx, q(r[0]), q(r[2]), v(r[3])) })
	case ir.VSha256Msg1:
		s.tied("sha256su0", d, a[0], a[1:], func(tx *arm64asm.Section, r []regalloc.PhysReg) { tx.Sha256su0(v(r[0]), v(r[2])) })
	case ir.VSha256Msg2:
		s.tied("sha256su1", d, a[0], a[1:], func(tx *arm64asm.Section, r []regalloc.PhysReg) { tx.Sha256su1(v(r[0]), v(r[2]), v(r[3])) })
	case ir.VSha512Rounds1, ir.VSha512Rounds2:
		f := (*arm64asm.Section).Sha512h
		if verb == ir.VSha512Rounds2 {
			f = (*arm64asm.Section).Sha512h2
		}
		s.tied(string(verb), d, a[0], a[1:], func(tx *arm64asm.Section, r []regalloc.PhysReg) { f(tx, q(r[0]), q(r[2]), v(r[3])) })
	case ir.VSha512Msg1:
		s.tied("sha512su0", d, a[0], a[1:], func(tx *arm64asm.Section, r []regalloc.PhysReg) { tx.Sha512su0(v(r[0]), v(r[2])) })
	case ir.VSha512Msg2:
		s.tied("sha512su1", d, a[0], a[1:], func(tx *arm64asm.Section, r []regalloc.PhysReg) { tx.Sha512su1(v(r[0]), v(r[2]), v(r[3])) })
	case ir.VSm3Ss1:
		s.op("sm3ss1", d, a, func(tx *arm64asm.Section, r []regalloc.PhysReg) { tx.Sm3ss1(v(r[0]), v(r[1]), v(r[2]), v(r[3])) })
	case ir.VSm3Tt1a, ir.VSm3Tt1b, ir.VSm3Tt2a, ir.VSm3Tt2b:
		lane, err := s.lit()
		if err != nil {
			return err
		}
		f := map[ir.Verb]func(*arm64asm.Section, reg.V, reg.V, reg.V, uint8){
			ir.VSm3Tt1a: (*arm64asm.Section).Sm3tt1a, ir.VSm3Tt1b: (*arm64asm.Section).Sm3tt1b,
			ir.VSm3Tt2a: (*arm64asm.Section).Sm3tt2a, ir.VSm3Tt2b: (*arm64asm.Section).Sm3tt2b,
		}[verb]
		s.tied(string(verb), d, a[0], a[1:], func(tx *arm64asm.Section, r []regalloc.PhysReg) {
			f(tx, v(r[0]), v(r[2]), v(r[3]), uint8(lane&3))
		})
	case ir.VSm3Partw1, ir.VSm3Partw2:
		f := (*arm64asm.Section).Sm3partw1
		if verb == ir.VSm3Partw2 {
			f = (*arm64asm.Section).Sm3partw2
		}
		s.tied(string(verb), d, a[0], a[1:], func(tx *arm64asm.Section, r []regalloc.PhysReg) { f(tx, v(r[0]), v(r[2]), v(r[3])) })
	case ir.VSm4Rounds:
		s.tied("sm4e", d, a[0], a[1:], func(tx *arm64asm.Section, r []regalloc.PhysReg) { tx.Sm4e(v(r[0]), v(r[2])) })
	case ir.VSm4Key:
		s.op("sm4ekey", d, a, bin2((*arm64asm.Section).Sm4ekey))
	default:
		return s.fail("is not lowered for AArch64")
	}
	return nil
}

// capabilitiesOf is the IR capabilities a feature set gives: what
// LegalizeCrypto may leave as instructions.
func capabilitiesOf(f arm64asm.FeatureSet) ir.Capabilities {
	var c ir.Capabilities
	if f.Has(feature.AES) {
		c = c.With(ir.CapAES, ir.CapClmul)
	}
	if f.Has(feature.SHA2) {
		c = c.With(ir.CapSHA1, ir.CapSHA256)
	}
	if f.Has(feature.SHA3) {
		c = c.With(ir.CapSHA512)
	}
	if f.Has(feature.SM4) {
		c = c.With(ir.CapSM3, ir.CapSM4)
	}
	if f.Has(feature.CRC) {
		c = c.With(ir.CapCRC32, ir.CapCRC32C)
	}
	return c
}

// featuresOf is the feature set a function that requires cs adds.
func featuresOf(cs ir.Capabilities) []feature.Feature {
	var out []feature.Feature
	for _, c := range cs.List() {
		switch c {
		case ir.CapAES, ir.CapClmul:
			out = append(out, feature.AES)
		case ir.CapSHA1, ir.CapSHA256:
			out = append(out, feature.SHA2)
		case ir.CapSHA512:
			out = append(out, feature.SHA3)
		case ir.CapSM3, ir.CapSM4:
			out = append(out, feature.SM4)
		case ir.CapCRC32, ir.CapCRC32C:
			out = append(out, feature.CRC)
		}
	}
	return out
}
