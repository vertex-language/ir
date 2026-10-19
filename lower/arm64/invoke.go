package arm64

// §G3 unwinding terminators: invoke, invokeind, and resume.
//
// An invoke emits a call instruction whose address range is registered in the LSDA
// call-site table, with its unwind edge directing to a landing pad. Results are passed
// as trailing parameters to the normal target block. Values live into the landing pad
// are preserved across the call in callee-saved registers or stack spill slots.

import (
	"fmt"

	"github.com/vertex-language/arm64/reg"
	"github.com/vertex-language/ir"
	"github.com/vertex-language/ir/lower/mir"
)

// iselInvoke lowers invoke and invokeind.
func iselInvoke(fn *ir.Func, mf *mir.Func, c *cursor, vr *vregs, plan *ehPlan,
	term *ir.Inst, opts Options) error {

	ind := term.Op().Verb == ir.VInvokeInd
	what := "invoke"
	var sig *ir.Sig
	var args []*ir.Def
	var op any
	var extraUses []mir.VReg

	if ind {
		addr, ok := vr.lookup(term.Arg(0))
		if !ok {
			return fmt.Errorf("invokeind: callee defined outside the function")
		}
		// A vreg of this call's own, for the reason iselCallInd states.
		target := vr.temp(w64)
		emitCopy(c, target, addr, w64)
		extraUses = []mir.VReg{target}
		what, op, args = "invokeind", callIndOp{marker: wantsReturnMarker(term)}, term.Args()[1:]
		if t := term.NamedType(); t != nil {
			sig = t.Sig()
		}
	} else {
		sym := term.Symbol()
		if sym == nil {
			return fmt.Errorf("invoke: no callee named")
		}
		what, op, args = "invoke @"+sym.Name(), callOp{sym: sym.Name(), marker: wantsReturnMarker(term)}, term.Args()
		if callee := term.Callee(); callee != nil {
			sig = callee.Signature()
		}
	}

	targets := term.Targets()
	if len(targets) != 1 {
		return fmt.Errorf("%s: %d normal targets, want one", what, len(targets))
	}
	normal := targets[0]

	unwind := term.Unwind()
	if unwind == nil {
		return fmt.Errorf("%s: no unwind edge", what)
	}
	if plan == nil {
		return fmt.Errorf("%s: an unwind edge in a function with no pad blocks", what)
	}
	site, err := plan.site(unwind)
	if err != nil {
		return fmt.Errorf("%s: %w", what, err)
	}

	dsts, err := invokeResultVRegs(vr, what, sig, normal)
	if err != nil {
		return err
	}

	if err := iselCallSeqTo(c, vr, what, sig, args, dsts, extraUses, op, opts, site); err != nil {
		return err
	}

	// Both edges. The unwind one carries no instruction — the personality
	// branches there, not this function — but it is an edge all the same,
	// and liveness is what reads it.
	padLabel := blockLabel(fn, unwind)
	mf.Succ(c.blk, padLabel)

	moves, err := edgeCopiesTrailing(vr, normal, len(dsts))
	if err != nil {
		return fmt.Errorf("%s: %w", what, err)
	}
	emitParallelCopy(c, vr, moves)

	label := blockLabel(fn, normal.Block())
	c.Emit(mir.Instr{Op: bOp{target: label}})
	mf.Succ(c.blk, label)
	return nil
}

// invokeResultVRegs is the vregs the call's results go into: the trailing
// parameters of the normal target (§14).
func invokeResultVRegs(vr *vregs, what string, sig *ir.Sig, normal ir.BlockTarget) ([]mir.VReg, error) {
	n := 0
	if sig != nil {
		n = len(sig.Rets())
	}
	blk := normal.Block()
	params := blk.Params()
	lead := len(normal.Args())
	if len(params) != lead+n {
		return nil, fmt.Errorf("%s: @%s takes %d parameters, the edge supplies %d arguments and the call %d results",
			what, blk.Label(), len(params), lead, n)
	}
	out := make([]mir.VReg, n)
	for i := 0; i < n; i++ {
		v, ok := vr.lookup(params[lead+i])
		if !ok {
			return nil, fmt.Errorf("%s: @%s parameter %d has no vreg", what, blk.Label(), lead+i)
		}
		out[i] = v
	}
	return out, nil
}

// iselResume hands the exception back to the unwinder.
//
// _Unwind_Resume takes the exception object and does not return, which is why
// nothing follows it: the block ends here whatever the IR says comes next.
func iselResume(c *cursor, vr *vregs, term *ir.Inst, opts Options) error {
	if term.NumArgs() != 1 {
		return fmt.Errorf("resume takes one argument, not %d", term.NumArgs())
	}
	exn, ok := vr.lookup(term.Arg(0))
	if !ok {
		return fmt.Errorf("resume: the exception object is defined outside the function")
	}
	return emitLibcall(c, vr, resumeSym, opts, []mir.VReg{exn})
}

// iselPadEntry is what a pad block starts with.
//
// The personality routine enters a landing pad with the exception object in
// X0 and its selector in X1 — the two values §G3 declares as the block's
// parameters. Nothing branched here, so nothing copied them in; the pseudo
// below is where they come from as far as the allocator is concerned, and it
// emits no instruction.
func iselPadEntry(c *cursor, vr *vregs, plan *ehPlan, blk *ir.Block) error {
	i, ok := plan.padOf[blk]
	if !ok {
		return fmt.Errorf("@%s is a pad block with no plan", blk.Label())
	}
	pad := plan.pads[i]

	x0 := vr.physical(reg.X0, w64)
	x1 := vr.physical(reg.X1, w32)
	c.Emit(mir.Instr{Op: padEntryOp{pad: i}, Defs: []mir.VReg{x0, x1}})

	exn, err := vr.define(blk.Params()[0])
	if err != nil {
		return err
	}
	emitCopy(c, exn, x0, w64)

	sel, err := vr.define(blk.Params()[1])
	if err != nil {
		return err
	}
	if pad.base == 0 {
		emitCopy(c, sel, x1, w32)
		return nil
	}

	// This pad's clauses are a consecutive run of the function's type
	// table, so the personality's index becomes a clause number by
	// subtracting where the run starts. See the selector note in lsda.go.
	base := vr.temp(w32)
	c.Emit(mir.Instr{Op: constOp{imm: int64(pad.base), w: w32}, Defs: []mir.VReg{base}})
	diff := vr.temp(w32)
	c.Emit(mir.Instr{
		Op:   aluOp{verb: ir.VSub, w: w32},
		Defs: []mir.VReg{diff},
		Uses: []mir.VReg{x1, base},
	})
	if !hasCleanup(blk) {
		emitCopy(c, sel, diff, w32)
		return nil
	}

	// A pad that also cleans up is entered with a selector of zero when
	// nothing matched, and zero is what §G3 reports for that — so it has to
	// survive the subtraction rather than become a negative clause number.
	zero := vr.temp(w32)
	c.Emit(mir.Instr{Op: constOp{imm: 0, w: w32}, Defs: []mir.VReg{zero}})
	c.Emit(mir.Instr{Op: cmpImmOp{imm: 0, w: w32}, Uses: []mir.VReg{x1}})
	c.Emit(mir.Instr{
		Op:   cselOp{cond: condEQ, w: w32},
		Defs: []mir.VReg{sel},
		Uses: []mir.VReg{zero, diff},
	})
	return nil
}

func hasCleanup(blk *ir.Block) bool {
	for _, c := range blk.Clauses() {
		if c.Kind() == ir.PadCleanup {
			return true
		}
	}
	return false
}
