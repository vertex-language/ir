package arm64

// The calls this package makes that the module did not write.
//
// §E's four bulk-memory verbs are calls to the C library function of the same
// name. Their IR signatures were chosen to match the C ones, so a bulk
// operation lowers to the call alone and AAPCS64 places the arguments the way
// it would place anyone's.

import (
	"fmt"

	"github.com/vertex-language/ir"
	"github.com/vertex-language/ir/lower/mir"
)

// libcalls is §E's verbs and the C library function each becomes.
var libcalls = map[ir.Verb]string{
	ir.VMemCpy:  "memcpy",
	ir.VMemMove: "memmove",
	ir.VMemSet:  "memset",
	ir.VMemCmp:  "memcmp",
}

// memsetSym is also what a zeroed allocation becomes. Named here rather than
// spelled at the use site so the extern declaration and the call cannot
// disagree about it.
const memsetSym = "memset"

// memcpySym is what a byval aggregate too large for its registers becomes.
const memcpySym = "memcpy"

// callsAggregateOnStack reports whether in is a call that puts a byval
// aggregate in the outgoing area rather than in registers.
//
// A classification error is not reported here: this pass only decides what to
// declare, and lowering the instruction will reach the same error with an
// instruction to name in the message.
func callsAggregateOnStack(in *ir.Inst, opts Options) bool {
	switch in.Op().Verb {
	case ir.VCall, ir.VCallInd:
	default:
		return false
	}
	places, err := callPlaces(in, opts)
	if err != nil {
		return false
	}
	for _, pl := range places {
		if pl.kind == placeStack && pl.isAggregate() {
			return true
		}
	}
	return false
}

// libcallSyms is every library symbol a module needs declared, prefixed.
//
// Collected up front because an undeclared reference is an error at Finalize,
// by which point there is no instruction left to name in the message.
func libcallSyms(m *ir.Module, opts Options) []string {
	prefix := opts.LibcallPrefix
	seen := map[string]bool{}
	var out []string
	for _, fn := range m.Funcs() {
		fn.WalkInsts(func(in *ir.Inst) bool {
			add := func(sym string) {
				if !seen[sym] {
					seen[sym] = true
					out = append(out, prefix+sym)
				}
			}
			if sym, ok := libcalls[in.Op().Verb]; ok {
				add(sym)
			}
			if callsAggregateOnStack(in, opts) {
				// A byval aggregate that ran out of registers is copied
				// into the outgoing area, and the copy is a memcpy. There
				// is no verb for it — the classification decides, so the
				// classification is what has to be asked.
				add(memcpySym)
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
			return true
		})
	}
	return out
}

// iselLibcall lowers one of §E's verbs into the call it is.
func iselLibcall(c *cursor, vr *vregs, in *ir.Inst, opts Options) error {
	op := in.Op()
	prefix := opts.LibcallPrefix
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
	// The general sequence, not a second copy of it. Every one of these
	// takes three arguments in registers and returns at most one value,
	// so there is nothing about them the written-call path does not
	// already do.
	return iselCallSeq(c, vr, op.String(), nil, in.Args(), in.Results(), nil, callOp{sym: prefix + sym}, opts)
}

// emitLibcall calls sym with args, discarding whatever it returns.
//
// For the calls whose operands are vregs this package built rather than values
// the module wrote — the memset behind a zeroed allocation. Every such call
// takes at most eight arguments of one file, so the places are the argument
// registers in order and there is no stack to classify.
func emitLibcall(c *cursor, vr *vregs, sym string, opts Options, args []mir.VReg) error {
	if len(args) > len(aapcsIntArgs) {
		return fmt.Errorf("libcall %s: %d arguments is more than the argument registers hold", sym, len(args))
	}
	places := make([]place, len(args))
	for i, a := range args {
		w := vr.widthOfVReg(a)
		if w.isFloat() {
			return fmt.Errorf("libcall %s: argument %d is a float, which this path does not place", sym, i)
		}
		places[i] = place{kind: placeInt, i: i, w: w}
	}
	return emitCallSeq(c, vr, places, args, nil, nil, callOp{sym: opts.LibcallPrefix + sym}, opts, nil)
}
