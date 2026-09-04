package amd64

// The calls this package makes that the module did not write.
//
// §E's four bulk-memory verbs are calls to the C library function of the
// same name: REP MOVSB is the wrong answer for every size not known to
// be large. Their IR signatures were chosen to match the C ones, which
// is why a bulk operation on parameters lowers to the call alone.

import (
	"fmt"

	"github.com/vertex-language/amd64/reg"

	"github.com/vertex-language/ir"
	"github.com/vertex-language/ir/lower/mir"
)

// libcalls is §E's verbs and the C library function each becomes.
var libcalls = map[ir.Verb]string{
	ir.VMemCpy:  memcpySym,
	ir.VMemMove: "memmove",
	ir.VMemSet:  memsetSym,
	ir.VMemCmp:  "memcmp",
}

// memsetSym is the function a zeroed ptr.alloc becomes. Named here
// rather than spelled at the use site so that the extern declaration and
// the call cannot disagree about it.
const (
	memsetSym = "memset"
	memcpySym = "memcpy"
)

// libcallSyms is every library symbol a module needs declared. Collected
// up front because an undeclared reference is an error at Finalize, by
// which point there is no instruction left to name in the message.
func libcallSyms(m *ir.Module, prefix string) []string {
	seen := map[string]bool{}
	var out []string
	add := func(s string) {
		if !seen[s] {
			seen[s] = true
			out = append(out, prefix+s)
		}
	}
	for _, fn := range m.Funcs() {
		fn.WalkInsts(func(in *ir.Inst) bool {
			if sym, ok := libcalls[in.Op().Verb]; ok {
				add(sym)
			}
			switch in.Op().Verb {
			case ir.VAlloc, ir.VAlloca:
				// Either form: zeroed means the same on both, whether
				// the size was known when the frame was planned or
				// arrives as a value.
				if in.Zeroed() {
					add(memsetSym)
				}
			}
			if copiesAggregate(in) {
				add(memcpySym)
			}
			return true
		})
	}
	return out
}

// copiesAggregate reports whether a call passes a byval aggregate in
// memory, which is the case whose lowering is a memcpy. Classified
// rather than guessed from the attribute: over-declaring would put an
// undefined memcpy in an object that never references it.
//
// Two placements are that case and not one. A stack place holds the bytes
// themselves; an indirect place holds a pointer to a copy the caller makes
// above the argument area, which is the Microsoft convention's answer for
// anything wider than eight bytes. Both copy, and missing the second turned
// every such call into an undefined memcpy at Finalize — with no instruction
// left to name in the message, which is the whole reason this pass exists.
func copiesAggregate(in *ir.Inst) bool {
	switch in.Op().Verb {
	case ir.VCall, ir.VCallInd:
	default:
		return false
	}
	places, err := classify(in.Block().Func().Module().Layout().ABI, callArgSpec(in))
	if err != nil {
		return false
	}
	for _, p := range places {
		if p.indirect || (p.kind == placeStack && p.isAggregate()) {
			return true
		}
	}
	return false
}

// iselLibcall lowers one of §E's verbs into the call it is.
func iselLibcall(c *cursor, vr *vregs, in *ir.Inst) error {
	op := in.Op()
	sym, ok := libcalls[op.Verb]
	if !ok {
		return fmt.Errorf("%s: no library function for this verb", op)
	}
	if in.Volatile() {
		// Volatile is a promise about how the bytes are touched, and
		// memcpy makes no such promise: the call is the wrong lowering
		// rather than a slow one.
		return fmt.Errorf("%s: a volatile bulk operation cannot be a library call, and no open-coded form is emitted yet", op)
	}

	args := make([]libArg, 0, 3)
	for i, a := range in.Args() {
		v, found := vr.lookup(a)
		if !found {
			return fmt.Errorf("%s: operand %d defined outside the function", op, i)
		}
		args = append(args, libArg{v: v, w: vr.widthOfVReg(v)})
	}

	// memcmp is the only one of the four with a result. The other three
	// return their destination pointer in C and the IR discards it,
	// which is a value the call leaves in RAX and nothing reads.
	var result mir.VReg
	var wantResult bool
	if len(in.Results()) == 1 {
		r, err := vr.define(in.Result(0))
		if err != nil {
			return fmt.Errorf("%s: %w", op, err)
		}
		result, wantResult = r, true
	}
	return emitLibcall(c, vr, sym, args, result, wantResult)
}

// libArg is one argument to a library call: the vreg holding it, and the
// width it is passed at.
type libArg struct {
	v mir.VReg
	w width
}

// emitLibcall emits the SysV sequence for a call to sym.
//
// Deliberately less than iselCallSeq: no stack arguments, no byval, no
// variadic count, one result at most. Each of those is true of every
// function this package invents a call to, and stating them as the shape
// rather than checking them keeps this from being a second copy of the
// general sequence. It shares callSite, where a call's real difficulty
// is.
func emitLibcall(c *cursor, vr *vregs, sym string, args []libArg, result mir.VReg, wantResult bool) error {
	abi := c.fn.Module().Layout().ABI

	// The two files counted separately, §3.2.3's way: a soft-float call
	// takes its operands in vector registers and its result in one.
	site := newCallSite(vr)
	inRegs := make([]mir.VReg, 0, len(args))
	var ints, floats int
	for i, a := range args {
		var dst mir.VReg
		switch {
		case a.w.isFloat() && floats < numFloatArgs(abi):
			dst = site.xmmReg(floatArgReg(abi, floats), a.w)
			floats++
		case !a.w.isFloat() && ints < numIntArgs(abi):
			dst = site.intReg(intArgReg(abi, ints), a.w)
			ints++
		default:
			return fmt.Errorf("libcall %s: argument %d has no register left in its file", sym, i)
		}
		inRegs = append(inRegs, dst)
		emitCopy(c, dst, a.v, a.w)
	}

	rax := site.intReg(reg.RAX, w64)

	// Every caller-saved register, exactly as a written call declares
	// them: what a library function destroys is what the ABI says, not
	// what its name suggests.
	defs := []mir.VReg{rax}
	defs = append(defs, inRegs...)
	for _, r := range regsFor(abi).callerSaved {
		if site.namedInt(r) {
			continue
		}
		defs = append(defs, site.intReg(r, w64))
	}
	for _, r := range regsFor(abi).xmm {
		if site.namedXmm(r) {
			continue
		}
		defs = append(defs, site.xmmReg(r, wf64))
	}

	c.Emit(mir.Instr{Op: callOp{sym: c.prefix + sym}, Defs: defs, Uses: inRegs})

	if wantResult {
		w := vr.widthOfVReg(result)
		src := rax
		if w.isFloat() {
			// XMM0 is both the first vector argument and the vector
			// return, so this is callSite's collision case.
			src = site.xmmReg(reg.XMM0, w)
		}
		emitCopy(c, result, src, w)
	}
	return nil
}
