package amd64

import (
	"fmt"

	"github.com/vertex-language/amd64/reg"

	"github.com/vertex-language/ir"
	"github.com/vertex-language/ir/lower/mir"
)

// Tail calls: a call that replaces this frame with the callee's. See
// ir.VTailCall.
//
// The arguments are placed exactly as an ordinary call's, then the frame
// comes down and control jumps rather than calls, so the callee returns to
// this function's caller. That order is safe because the two sets barely
// meet: the teardown restores only callee-saved registers and RSP, and the
// only arguments that travel in a callee-saved register are Swift's self
// and async context -- which the teardown leaves where they were put. See
// emitTeardown.
//
// A callee with any argument in memory is refused. Placing one would mean
// writing into this function's own incoming area, which is only sound
// when it is at least as large; and a Microsoft by-reference copy lives in
// the outgoing area of the frame the teardown is about to free. Every tail
// call an async function makes passes its context and little else.

// tailTarget is where an indirect tail call's address waits while the
// frame comes down. R11 is caller-saved and an argument register under
// neither convention, so no argument is in it, the teardown does not
// restore it, and a callee is not owed its value.
const tailTarget = reg.R11Q

// iselTailCall lowers a direct tail call.
func iselTailCall(c *cursor, vr *vregs, in *ir.Inst) error {
	sym := in.Symbol()
	if sym == nil {
		return fmt.Errorf("tail_call: no callee named")
	}
	what := "tail_call @" + sym.Name()
	var sig *ir.Sig
	if callee := in.Callee(); callee != nil {
		sig = callee.Signature()
	}
	spec := callArgSpec(in)
	if err := tailArgsFitRegisters(c, what, spec, sig); err != nil {
		return err
	}
	return iselCallSeq(c, vr, what, spec, sig, in.Args(), nil, nil, tailOp{sym: sym.Name()})
}

// iselTailCallInd is iselTailCall through a pointer.
func iselTailCallInd(c *cursor, vr *vregs, in *ir.Inst) error {
	addr, ok := vr.lookup(in.Arg(0))
	if !ok {
		return fmt.Errorf("tail_callind: callee defined outside the function")
	}
	var sig *ir.Sig
	if t := in.NamedType(); t != nil {
		sig = t.Sig()
	}
	spec := callArgSpec(in)
	if err := tailArgsFitRegisters(c, "tail_callind", spec, sig); err != nil {
		return err
	}
	target := vr.physical(tailTarget, w64)
	emitCopy(c, target, addr, w64)
	return iselCallSeq(c, vr, "tail_callind", spec, sig, in.Args()[1:], nil, []mir.VReg{target}, tailIndOp{})
}

// tailArgsFitRegisters reports whether every argument of a tail call is
// placed in a register, which is the only shape this package tail calls.
func tailArgsFitRegisters(c *cursor, what string, spec []abiArg, sig *ir.Sig) error {
	if sig != nil && errorResult(sig) >= 0 {
		// The callee's error would come back in R12 to a frame that is
		// no longer there to read it. A throwing function that tail calls
		// passes its own error on, which is a different signature.
		if funcErrorResult(c.fn) < 0 {
			return fmt.Errorf("%s: a callee that may fail is tail called from a function that may not", what)
		}
	}
	places, err := classify(c.fn.Module().Layout().ABI, spec)
	if err != nil {
		return fmt.Errorf("%s: %w", what, err)
	}
	for i, pl := range places {
		if pl.kind == placeStack || pl.indirect {
			return fmt.Errorf("%s: argument %d is passed in memory, which this package does not tail call", what, i)
		}
	}
	return nil
}

// carried is the registers a tail call's arguments are sitting in when its
// teardown runs, which the teardown must not restore.
//
// Almost all of them are argument registers, which no epilogue touches.
// The ones that matter are Swift's: a receiver travels in R13 and an async
// context in R14, both callee-saved. If the teardown restored those it
// would put this function's saved copy back over the argument it had just
// placed, and the callee would read the wrong thing -- a silent wrong
// answer, not a crash.
//
// They are not restored, and that is the convention rather than a hole in
// it: a value passed forward in one of these registers is passed forward,
// and the callee owes it to whoever it returns to.
func carried(in mir.Instr, r64 func(mir.VReg) reg.R64) map[reg.R64]bool {
	out := make(map[reg.R64]bool, len(in.Uses))
	for _, u := range in.Uses {
		out[r64(u)] = true
	}
	return out
}

// isTailOp reports whether a call op replaces the frame rather than
// returning to it.
func isTailOp(op any) bool {
	switch op.(type) {
	case tailOp, tailIndOp:
		return true
	}
	return false
}
