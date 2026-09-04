package arm64

import (
	"fmt"

	"github.com/vertex-language/arm64/reg"

	"github.com/vertex-language/ir"
	"github.com/vertex-language/ir/lower/globals"
	"github.com/vertex-language/ir/lower/mir"
)

// binOps is every §A verb that is one three-address instruction here.
var binOps = map[ir.Verb]bool{
	ir.VAdd: true, ir.VSub: true, ir.VMul: true,
	ir.VAnd: true, ir.VOr: true, ir.VXor: true,
	ir.VShl: true, ir.VSShr: true, ir.VUShr: true, ir.VRotR: true,
}

// unOps is every §A verb that is one one-source instruction.
var unOps = map[ir.Verb]bool{
	ir.VNeg: true, ir.VNot: true,
	ir.VClz: true, ir.VBswap: true,
}

// compareConds maps §B's verbs onto the condition each answers.
var compareConds = map[ir.Verb]condCode{
	ir.VEq:  condEQ,
	ir.VNe:  condNE,
	ir.VSLt: condLT,
	ir.VSLe: condLE,
	ir.VULt: condLO,
	ir.VULe: condLS,
}

// iselInst lowers one non-terminator instruction.
func iselInst(c *cursor, vr *vregs, fr *frame, in *ir.Inst, opts Options) error {
	op := in.Op()
	verb := op.Verb

	if binOps[verb] && !op.Type.IsFloat() {
		return iselBinary(c, vr, in)
	}
	if unOps[verb] && !op.Type.IsFloat() {
		return iselUnary(c, vr, in)
	}
	if cond, ok := compareConds[verb]; ok && !op.Type.IsFloat() {
		return iselCompare(c, vr, in, cond)
	}
	if handled, err := iselFloatInst(c, vr, in); handled {
		return err
	}

	if narrow, ok := atomicLoads[verb]; ok {
		return iselAtomicLoad(c, vr, in, narrow)
	}
	if narrow, ok := atomicStores[verb]; ok {
		return iselAtomicStore(c, vr, in, narrow)
	}
	if narrow, ok := atomicCases[verb]; ok {
		return iselAtomicCas(c, vr, in, narrow)
	}
	if rmw, ok := atomicRmwAlus[verb]; ok {
		return iselAtomicRmw(c, vr, in, rmw.a, rmw.alu)
	}

	if verb == ir.VAsm {
		return iselAsm(c.fn, c, vr, in, vr.nextAsmID(), nil)
	}

	if _, ok := libcalls[verb]; ok {
		return iselLibcall(c, vr, in, opts)
	}

	switch verb {
	case ir.VFence:
		return iselFence(c, in)
	case ir.VConst:
		return iselConst(c, vr, in)
	case ir.VSMulHi:
		return iselMulHi(c, vr, in, true)
	case ir.VUMulHi:
		return iselMulHi(c, vr, in, false)
	case ir.VSDiv:
		return iselDivide(c, vr, in, true, false)
	case ir.VUDiv:
		return iselDivide(c, vr, in, false, false)
	case ir.VSRem:
		return iselDivide(c, vr, in, true, true)
	case ir.VURem:
		return iselDivide(c, vr, in, false, true)
	case ir.VRotL:
		return iselRotL(c, vr, in)
	case ir.VCtz:
		return iselCtz(c, vr, in)
	case ir.VPopcnt:
		return iselPopcnt(c, vr, in)
	case ir.VSAddO, ir.VUAddO, ir.VSSubO, ir.VSMulO, ir.VUMulO:
		return iselOverflow(c, vr, in, verb)
	case ir.VFromI64, ir.VFromPtr:
		return iselPtrInt(c, vr, in)
	case ir.VDiff:
		return iselDiff(c, vr, in)
	case ir.VSelect:
		return iselSelect(c, vr, in)
	case ir.VLoad:
		return iselLoad(c, vr, in)
	case ir.VStore:
		return iselStore(c, vr, in)
	case ir.VAlloc:
		return iselAlloc(c, vr, fr, in, opts)
	case ir.VAlloca:
		return iselAlloca(c, vr, fr, in, opts)
	case ir.VStackSave:
		return iselStackSave(c, vr, in)
	case ir.VStackRestore:
		return iselStackRestore(c, vr, in)
	case ir.VGetAddr:
		return iselGetAddr(c, vr, in)
	case ir.VBlockAddr:
		return iselBlockAddr(c, vr, in)
	case ir.VVaStart:
		return iselVaStart(c, vr, fr, in, opts)
	case ir.VVaEnd:
		return iselVaEnd(in, opts)
	case ir.VVaCopy:
		return iselVaCopy(c, vr, in, opts)
	case ir.VVaArg:
		return iselVaArg(c, vr, in, opts)
	case ir.VVaArgRef:
		return iselVaArgRef(c, vr, in, opts)
	case ir.VCall:
		return iselCall(c, vr, in, opts)
	case ir.VCallInd:
		return iselCallInd(c, vr, in, opts)
	case ir.VWrapI64:
		return iselWrap(c, vr, in)
	case ir.VSExtI32:
		return iselExtend(c, vr, in, a32, true)
	case ir.VZExtI32, ir.VZExtI1:
		return iselExtend(c, vr, in, a32, false)
	case ir.VSLoad8:
		return iselExtLoad(c, vr, in, a8, true)
	case ir.VSLoad16:
		return iselExtLoad(c, vr, in, a16, true)
	case ir.VSLoad32:
		return iselExtLoad(c, vr, in, a32, true)
	case ir.VULoad8:
		return iselExtLoad(c, vr, in, a8, false)
	case ir.VULoad16:
		return iselExtLoad(c, vr, in, a16, false)
	case ir.VULoad32:
		return iselExtLoad(c, vr, in, a32, false)
	case ir.VStore8:
		return iselSubStore(c, vr, in, a8)
	case ir.VStore16:
		return iselSubStore(c, vr, in, a16)
	case ir.VStore32:
		return iselSubStore(c, vr, in, a32)
	}
	return fmt.Errorf("unsupported instruction %s", op)
}

// operands is the vregs a two-operand instruction reads.
func operands(vr *vregs, in *ir.Inst, n int) ([]mir.VReg, error) {
	out := make([]mir.VReg, n)
	for i := 0; i < n; i++ {
		v, ok := vr.lookup(in.Arg(i))
		if !ok {
			return nil, fmt.Errorf("%s: operand %d defined outside the function", in.Op(), i)
		}
		out[i] = v
	}
	return out, nil
}

func iselBinary(c *cursor, vr *vregs, in *ir.Inst) error {
	ops, err := operands(vr, in, 2)
	if err != nil {
		return err
	}
	dst, err := vr.define(in.Result(0))
	if err != nil {
		return fmt.Errorf("%s: %w", in.Op(), err)
	}
	c.Emit(mir.Instr{
		Op:   aluOp{verb: in.Op().Verb, w: vr.widthOfVReg(dst)},
		Defs: []mir.VReg{dst},
		Uses: ops,
	})
	return nil
}

func iselUnary(c *cursor, vr *vregs, in *ir.Inst) error {
	ops, err := operands(vr, in, 1)
	if err != nil {
		return err
	}
	dst, err := vr.define(in.Result(0))
	if err != nil {
		return fmt.Errorf("%s: %w", in.Op(), err)
	}
	var op any = unOp{verb: in.Op().Verb, w: vr.widthOfVReg(dst)}
	if in.Op().Type == ir.TypeI1 && in.Op().Verb == ir.VNot {
		op = i1NotOp{}
	}
	c.Emit(mir.Instr{
		Op:   op,
		Defs: []mir.VReg{dst},
		Uses: ops,
	})
	return nil
}

// iselCompare lowers §B: a compare that sets the flags, then a CSET that reads
// them into a register.
//
// Always materialized, never fused into the branch that reads it. §B's result
// is an i1 value and a value has to exist; folding the pair away when its only
// reader is a brif is a peephole, and this package has nowhere to put one.
func iselCompare(c *cursor, vr *vregs, in *ir.Inst, cond condCode) error {
	ops, err := operands(vr, in, 2)
	if err != nil {
		return err
	}
	dst, err := vr.define(in.Result(0))
	if err != nil {
		return fmt.Errorf("%s: %w", in.Op(), err)
	}
	c.Emit(mir.Instr{Op: cmpOp{w: vr.widthOfVReg(ops[0])}, Uses: ops})
	c.Emit(mir.Instr{Op: csetOp{cond: cond}, Defs: []mir.VReg{dst}})
	return nil
}

func iselSelect(c *cursor, vr *vregs, in *ir.Inst) error {
	ops, err := operands(vr, in, 3)
	if err != nil {
		return err
	}
	dst, err := vr.define(in.Result(0))
	if err != nil {
		return fmt.Errorf("%s: %w", in.Op(), err)
	}
	// The condition against zero, then the choice: CSEL takes the first
	// source when the condition holds, so NE against zero picks the true arm.
	// The condition is an i1 in the integer file whatever the arms are, so
	// only the choice changes register file with them.
	w := vr.widthOfVReg(dst)
	c.Emit(mir.Instr{Op: cmpImmOp{imm: 0, w: w32}, Uses: []mir.VReg{ops[0]}})
	sel := mir.Instr{
		Op:   cselOp{cond: condNE, w: w},
		Defs: []mir.VReg{dst},
		Uses: []mir.VReg{ops[1], ops[2]},
	}
	if w.isFloat() {
		sel.Op = fcselOp{cond: condNE, w: w}
	}
	c.Emit(sel)
	return nil
}

func iselMulHi(c *cursor, vr *vregs, in *ir.Inst, signed bool) error {
	ops, err := operands(vr, in, 2)
	if err != nil {
		return err
	}
	dst, err := vr.define(in.Result(0))
	if err != nil {
		return fmt.Errorf("%s: %w", in.Op(), err)
	}
	if vr.widthOfVReg(dst) == w64 {
		c.Emit(mir.Instr{Op: mulhOp{signed: signed}, Defs: []mir.VReg{dst}, Uses: ops})
		return nil
	}

	// At thirty-two bits there is no SMULH to reach for, and none is
	// needed: the whole product of two 32-bit values fits in a register,
	// so both operands widen to sixty-four, multiply once, and the half
	// MUL would have discarded is what the top of that word already
	// holds. The extension is the operation's own signedness — a signed
	// high half is the high half of a signed product — and the final copy
	// is a write to the narrow view, which takes the low thirty-two bits
	// of the shifted result and nothing else.
	wide := make([]mir.VReg, 2)
	for i, o := range ops {
		wide[i] = vr.temp(w64)
		c.Emit(mir.Instr{
			Op:   extOp{from: a32, signed: signed},
			Defs: []mir.VReg{wide[i]}, Uses: []mir.VReg{o},
		})
	}
	prod := vr.temp(w64)
	c.Emit(mir.Instr{Op: aluOp{verb: ir.VMul, w: w64}, Defs: []mir.VReg{prod}, Uses: wide})

	shift := vr.temp(w64)
	c.Emit(mir.Instr{Op: constOp{imm: 32, w: w64}, Defs: []mir.VReg{shift}})
	high := vr.temp(w64)
	c.Emit(mir.Instr{
		Op:   aluOp{verb: ir.VUShr, w: w64},
		Defs: []mir.VReg{high}, Uses: []mir.VReg{prod, shift},
	})
	emitCopy(c, dst, high, w32)
	return nil
}

func iselConst(c *cursor, vr *vregs, in *ir.Inst) error {
	lit, ok := in.Lit()
	if !ok {
		return fmt.Errorf("%s: only a plain literal is supported", in.Op())
	}
	dst, err := vr.define(in.Result(0))
	if err != nil {
		return fmt.Errorf("%s: %w", in.Op(), err)
	}
	w := vr.widthOfVReg(dst)
	if w.isFloat() {
		if lit.Kind() != ir.ConstFloat {
			return fmt.Errorf("%s: only a plain float literal is supported", in.Op())
		}
		iselFloatConst(c, vr, dst, w, lit.Float())
		return nil
	}
	// An integer literal, or one of §2's three symbolic constants, which
	// are integers this target has to be asked for.
	v, err := globals.ConstInt(layout{}, lit)
	if err != nil {
		return fmt.Errorf("%s: %w", in.Op(), err)
	}
	c.Emit(mir.Instr{Op: constOp{imm: v, w: w}, Defs: []mir.VReg{dst}})
	return nil
}

func iselLoad(c *cursor, vr *vregs, in *ir.Inst) error {
	ops, err := operands(vr, in, 1)
	if err != nil {
		return err
	}
	dst, err := vr.define(in.Result(0))
	if err != nil {
		return fmt.Errorf("%s: %w", in.Op(), err)
	}
	c.Emit(mir.Instr{Op: loadOp{w: vr.widthOfVReg(dst)}, Defs: []mir.VReg{dst}, Uses: ops})
	return nil
}

func iselStore(c *cursor, vr *vregs, in *ir.Inst) error {
	ops, err := operands(vr, in, 2)
	if err != nil {
		return err
	}
	c.Emit(mir.Instr{Op: storeOp{w: vr.widthOfVReg(ops[0])}, Uses: ops})
	return nil
}

func iselExtLoad(c *cursor, vr *vregs, in *ir.Inst, from access, signed bool) error {
	ops, err := operands(vr, in, 1)
	if err != nil {
		return err
	}
	dst, err := vr.define(in.Result(0))
	if err != nil {
		return fmt.Errorf("%s: %w", in.Op(), err)
	}
	c.Emit(mir.Instr{
		Op:   extLoadOp{from: from, signed: signed, w: vr.widthOfVReg(dst)},
		Defs: []mir.VReg{dst},
		Uses: ops,
	})
	return nil
}

func iselSubStore(c *cursor, vr *vregs, in *ir.Inst, to access) error {
	ops, err := operands(vr, in, 2)
	if err != nil {
		return err
	}
	c.Emit(mir.Instr{Op: subStoreOp{to: to}, Uses: ops})
	return nil
}

// iselWrap lowers §C's wrap_i64, which is a move: a W register *is* the low
// half of its X, so reading the narrow view is the truncation.
func iselWrap(c *cursor, vr *vregs, in *ir.Inst) error {
	ops, err := operands(vr, in, 1)
	if err != nil {
		return err
	}
	dst, err := vr.define(in.Result(0))
	if err != nil {
		return fmt.Errorf("%s: %w", in.Op(), err)
	}
	emitCopy(c, dst, ops[0], w32)
	return nil
}

func iselExtend(c *cursor, vr *vregs, in *ir.Inst, from access, signed bool) error {
	ops, err := operands(vr, in, 1)
	if err != nil {
		return err
	}
	dst, err := vr.define(in.Result(0))
	if err != nil {
		return fmt.Errorf("%s: %w", in.Op(), err)
	}
	c.Emit(mir.Instr{
		Op:   extOp{from: from, signed: signed},
		Defs: []mir.VReg{dst},
		Uses: ops,
	})
	return nil
}

func iselAlloc(c *cursor, vr *vregs, fr *frame, in *ir.Inst, opts Options) error {
	off, ok := fr.slot[in]
	if !ok {
		return fmt.Errorf("%s: no frame slot was planned", in.Op())
	}
	dst, err := vr.define(in.Result(0))
	if err != nil {
		return fmt.Errorf("%s: %w", in.Op(), err)
	}
	c.Emit(mir.Instr{Op: frameOp{off: off}, Defs: []mir.VReg{dst}})

	if !in.Zeroed() {
		return nil
	}
	// §D3's zeroed, which guarantees the storage reads as all-zero on
	// entry to the allocation's live range. The same memset the dynamic
	// form gets, over a size that was known when the frame was planned.
	size, _, err := allocSizeAlign(in)
	if err != nil {
		return fmt.Errorf("%s: %w", in.Op(), err)
	}
	zero := vr.temp(w32)
	n := vr.temp(w64)
	c.Emit(mir.Instr{Op: constOp{imm: 0, w: w32}, Defs: []mir.VReg{zero}})
	c.Emit(mir.Instr{Op: constOp{imm: int64(size), w: w64}, Defs: []mir.VReg{n}})
	return emitLibcall(c, vr, memsetSym, opts, []mir.VReg{dst, zero, n})
}

func iselGetAddr(c *cursor, vr *vregs, in *ir.Inst) error {
	sym := in.Symbol()
	if sym == nil {
		return fmt.Errorf("%s: no symbol named", in.Op())
	}
	dst, err := vr.define(in.Result(0))
	if err != nil {
		return fmt.Errorf("%s: %w", in.Op(), err)
	}
	if isImportSym(sym) {
		// A symbol this module only declares may be defined in a shared
		// library, whose address is not a link-time constant. ADRP+ADD would
		// bake one in; the address has to be read from the GOT, which the
		// loader fills. This is also right for an import another object of
		// the same link defines — the linker relaxes that case if it wants
		// to — and it is what the platform's compilers emit.
		c.Emit(mir.Instr{Op: symGotAddrOp{sym: sym.Name()}, Defs: []mir.VReg{dst}})
		return nil
	}
	c.Emit(mir.Instr{Op: symAddrOp{sym: sym.Name()}, Defs: []mir.VReg{dst}})
	return nil
}

// isImportSym reports whether a symbol is declared here and defined
// elsewhere.
func isImportSym(s ir.Symbol) bool {
	switch s.(type) {
	case *ir.GlobalImport, *ir.FuncImport:
		return true
	}
	return false
}

// A callSite is the physical registers one call names, each mapped to the
// single vreg that stands for it.
//
// The collision the amd64 backend has to work around is milder here: AAPCS64
// returns in the same registers it passes in, so X0 is both the first argument
// and the first result. One vreg per register keeps that one fact.
type callSite struct {
	vr   *vregs
	ints map[reg.X]mir.VReg
	vecs map[reg.V]mir.VReg
}

func newCallSite(vr *vregs) *callSite {
	return &callSite{vr: vr, ints: map[reg.X]mir.VReg{}, vecs: map[reg.V]mir.VReg{}}
}

func (s *callSite) intReg(r reg.X, w width) mir.VReg {
	if v, ok := s.ints[r]; ok {
		return v
	}
	v := s.vr.physical(r, w)
	s.ints[r] = v
	return v
}

func (s *callSite) vecReg(r reg.V, w width) mir.VReg {
	if v, ok := s.vecs[r]; ok {
		return v
	}
	v := s.vr.physicalVec(r, w)
	s.vecs[r] = v
	return v
}

func (s *callSite) namedInt(r reg.X) bool { _, ok := s.ints[r]; return ok }
func (s *callSite) namedVec(r reg.V) bool { _, ok := s.vecs[r]; return ok }

// iselCall lowers §G's direct call.
func iselCall(c *cursor, vr *vregs, in *ir.Inst, opts Options) error {
	sym := in.Symbol()
	if sym == nil {
		return fmt.Errorf("call: no callee named")
	}
	what := "call @" + sym.Name()

	var sig *ir.Sig
	if callee := in.Callee(); callee != nil {
		sig = callee.Signature()
	}
	return iselCallSeq(c, vr, what, sig, in.Args(), in.Results(), nil, callOp{sym: sym.Name()}, opts)
}

// iselCallInd lowers §G's indirect call, whose convention comes from the
// named func type rather than from a declaration.
func iselCallInd(c *cursor, vr *vregs, in *ir.Inst, opts Options) error {
	addr, ok := vr.lookup(in.Arg(0))
	if !ok {
		return fmt.Errorf("callind: callee defined outside the function")
	}

	// The address into a vreg of this call's own rather than the one the
	// value already lives in. Every caller-saved register is a destination
	// of the call, so the address interferes with all of them and the
	// allocator can only place it somewhere the call does not write if it
	// is a value the call uses rather than one of the pinned vregs it also
	// defines.
	target := vr.temp(w64)
	emitCopy(c, target, addr, w64)

	var sig *ir.Sig
	if t := in.NamedType(); t != nil {
		sig = t.Sig()
	}
	return iselCallSeq(c, vr, "callind", sig, in.Args()[1:], in.Results(), []mir.VReg{target}, callIndOp{}, opts)
}

// iselCallSeq is the body every call form shares: the arguments into the
// places AAPCS64 names, the call, and the results back out.
func iselCallSeq(c *cursor, vr *vregs, what string, sig *ir.Sig,
	args []*ir.Def, results []*ir.Def, extraUses []mir.VReg, op any, opts Options) error {

	if len(results) > 2 {
		return fmt.Errorf("%s: %d results; more than two comes back through memory, which is sret and is not written yet", what, len(results))
	}

	named := len(args)
	variadic := false
	if sig != nil && sig.IsVariadic() {
		variadic, named = true, len(sig.Params())
	}
	sret := sretOf(sig)
	places, err := classifyCall(sigArgSpec(sig, args), named, variadic, opts.Variadic, sret)
	if err != nil {
		return fmt.Errorf("%s: %w", what, err)
	}

	// Whether this call's result comes back in registers rather than
	// through the storage the sret parameter names.
	var sretAgg *aggregate
	if agg, inRegs, err := sretInRegs(sret); err != nil {
		return fmt.Errorf("%s: %w", what, err)
	} else if inRegs {
		sretAgg = &agg
	}

	srcs := make([]mir.VReg, len(args))
	for i, a := range args {
		v, ok := vr.lookup(a)
		if !ok {
			return fmt.Errorf("%s: argument %d defined outside the function", what, i)
		}
		srcs[i] = v
	}

	dsts := make([]mir.VReg, len(results))
	for i, d := range results {
		v, err := vr.define(d)
		if err != nil {
			return fmt.Errorf("%s: %w", what, err)
		}
		dsts[i] = v
	}

	return emitCallSeq(c, vr, places, srcs, dsts, extraUses, op, opts, sretAgg)
}

// copyToOutgoing copies a byval aggregate into the outgoing argument area,
// for the case where the registers of its file ran out. The callee reads it
// there as one contiguous object, so what has to arrive is the bytes.
func copyToOutgoing(c *cursor, vr *vregs, pl place, src mir.VReg, opts Options) error {
	dst := vr.temp(w64)
	c.Emit(mir.Instr{Op: outArgAddrOp{off: pl.off}, Defs: []mir.VReg{dst}})

	n := vr.temp(w64)
	c.Emit(mir.Instr{Op: constOp{w: w64, imm: int64(pl.size)}, Defs: []mir.VReg{n}})

	return emitLibcall(c, vr, "memcpy", opts, []mir.VReg{dst, src, n})
}

// emitCallSeq is the part with the difficulty in it, over vregs that are
// already resolved: a call this package invents has no ir.Defs to look up.
//
// The stack arguments go first, into the outgoing area at the bottom of the
// frame, before any register copy: a stack argument may be living in a
// register a copy below is about to write, and a store that has already
// happened is a value that no longer cares where it was.
func emitCallSeq(c *cursor, vr *vregs, places []place,
	srcs, dsts, extraUses []mir.VReg, op any, opts Options, sretAgg *aggregate) error {

	for i, pl := range places {
		if pl.kind != placeStack {
			continue
		}
		if pl.isAggregate() {
			// The bytes, not the pointer. byval says the callee receives
			// what the pointer names, and an aggregate that ran out of
			// registers is received as a copy in the outgoing area — so
			// the caller is what makes that copy.
			if err := copyToOutgoing(c, vr, pl, srcs[i], opts); err != nil {
				return err
			}
			continue
		}
		c.Emit(mir.Instr{Op: argStoreOp{off: pl.off, w: pl.w}, Uses: []mir.VReg{srcs[i]}})
	}

	site := newCallSite(vr)
	inRegs := append([]mir.VReg(nil), extraUses...)
	for i, pl := range places {
		if pl.kind == placeStack {
			continue
		}
		if pl.isAggregate() {
			// An aggregate small enough for registers travels as its own
			// bytes, read out of the storage the pointer names: one load
			// per register, at the offset that register carries.
			for _, slot := range pl.regs {
				var dst mir.VReg
				if slot.kind == placeFloat {
					dst = site.vecReg(aapcsFloatArgs[slot.i], slot.w)
				} else {
					dst = site.intReg(aapcsIntArgs[slot.i], slot.w)
				}
				inRegs = append(inRegs, dst)
				c.Emit(mir.Instr{
					Op:   loadAtOp{off: slot.off, w: slot.w},
					Defs: []mir.VReg{dst},
					Uses: []mir.VReg{srcs[i]},
				})
			}
			continue
		}

		var dst mir.VReg
		switch pl.kind {
		case placeFloat:
			dst = site.vecReg(aapcsFloatArgs[pl.i], pl.w)
		case placeIndirect:
			dst = site.intReg(reg.X8, pl.w)
		default:
			dst = site.intReg(aapcsIntArgs[pl.i], pl.w)
		}
		inRegs = append(inRegs, dst)
		emitCopy(c, dst, srcs[i], pl.w)
	}

	// Every caller-saved register is a destination whether or not the call
	// names it: that is the list of places a value live across it cannot
	// be. X30 among them, which is the difference from the other
	// architecture — the link register is a register here, and BL writes it.
	defs := append([]mir.VReg(nil), inRegs[len(extraUses):]...)
	for _, r := range callerSaved {
		if site.namedInt(r) {
			continue
		}
		defs = append(defs, site.intReg(r, w64))
	}
	defs = append(defs, site.intReg(reg.X30, w64))
	for _, r := range callerSavedVec {
		if site.namedVec(r) {
			continue
		}
		defs = append(defs, site.vecReg(r, wf64))
	}

	c.Emit(mir.Instr{Op: op, Defs: defs, Uses: inRegs})

	// A result §5.5 brought back in registers, into the storage the caller
	// set aside for it. There is no ir.Def to copy into — the call returns
	// nothing, and the front end said so by writing an sret parameter — so
	// what the caller gets is the bytes, written through the address it
	// passed as that parameter.
	if sretAgg != nil {
		if len(srcs) == 0 {
			return fmt.Errorf("an sret call with no arguments")
		}
		for k := 0; k < sretAgg.n; k++ {
			var src mir.VReg
			if sretAgg.kind == aggHFA {
				src = site.vecReg(aapcsFloatArgs[k], sretAgg.w)
			} else {
				src = site.intReg(aapcsIntArgs[k], sretAgg.w)
			}
			c.Emit(mir.Instr{
				Op:   storeAtOp{off: int64(uint64(k) * sretAgg.step), w: sretAgg.w},
				Uses: []mir.VReg{src, srcs[0]},
			})
		}
	}

	// And the results back out of the registers they came back in, which
	// are the registers the arguments went in.
	var ints, floats int
	for _, result := range dsts {
		w := vr.widthOfVReg(result)
		var src mir.VReg
		if w.isFloat() {
			src = site.vecReg(aapcsFloatArgs[floats], w)
			floats++
		} else {
			src = site.intReg(aapcsIntArgs[ints], w)
			ints++
		}
		emitCopy(c, result, src, w)
	}
	return nil
}
