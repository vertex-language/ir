package verify

import (
	"errors"
	"fmt"
	"strings"

	"github.com/vertex-language/ir"
)

// Verifier sentinels: the faults only a finished module reveals. The other
// half of §19 is in ir/error.go, caught by the builder as you emit and
// sticky on the module — Module returns that first, so a caller never has
// to ask two questions to learn whether a module is sound.
var (
	// ErrNoBody is a function definition with no blocks at all. This is
	// §19.2's degenerate case: a definition whose body was never begun is
	// not a declaration, which is what ir.ImportFunc is for.
	ErrNoBody = errors.New("function definition has no body")

	// ErrNakedBody is a naked function whose body is blocks rather than
	// assembly, or is nothing at all. Naked says there is no prologue, no
	// epilogue and no frame, which leaves no lowering for an instruction
	// to have — so the body of one is ir.Func.AsmBody and nothing else.
	// This is GCC's rule too, from the other side: a naked function's body
	// may contain only asm statements.
	ErrNakedBody = errors.New("naked function's body is not assembly")

	// ErrTerminator is a block that does not end in one (§19.2). Ending in
	// more than one is unreachable from the builder, which refuses an
	// instruction after a terminator with ir.ErrTerminated, so this is only
	// ever that rule's missing half.
	ErrTerminator = errors.New("block does not end in a terminator")

	// ErrUnreachable is a block no path from the entry block reaches
	// (§19.2). Only the entry block may be unreachable-from-nothing.
	ErrUnreachable = errors.New("block is unreachable")

	// ErrEntryTarget is the entry block named as a branch target or by a
	// ptr.blockaddr (§19.17). Its inputs are the signature's parameter
	// registers, which no edge can supply.
	ErrEntryTarget = errors.New("entry block is a branch target")

	// ErrPadEdge is a pad block reached by something other than an unwind
	// edge (§19.4). Its parameters are supplied by the personality
	// routine, which is why an unwind edge carries no argument list — and
	// why a br into one would arrive with them unset.
	ErrPadEdge = errors.New("pad block reached by an ordinary edge")

	// ErrPersonality is an invoke or invokeind in a function that declares
	// no personality routine (§19.4). The unwinder has nothing to ask
	// which clauses match.
	ErrPersonality = errors.New("function unwinds without a personality")

	// ErrResume is a resume whose operand is not the exception object
	// parameter of a pad block, or is one whose pad does not dominate it
	// (§19.5). Both halves are the same rule: the value a resume hands
	// back to the unwinder is the one the personality gave a pad it
	// actually ran through.
	ErrResume = errors.New("resume does not name a dominating pad's exception parameter")

	// ErrInvokeTarget is an invoke's normal target whose parameter list is
	// not the edge's arguments followed by the callee's results (§19.16).
	// An invoke has no result list of its own — a register it defined
	// would have to dominate both edges, and on the unwind edge the call
	// did not complete — so the results arrive as the trailing parameters
	// of the block where they are live.
	ErrInvokeTarget = errors.New("invoke's normal target does not match the call")

	// ErrInvokeEdge is an invoke's normal target with a second predecessor
	// (§19.16), whether another invoke or an ordinary branch. Its trailing
	// parameters are supplied by one call, so no other edge can reach it
	// without leaving them unset.
	ErrInvokeEdge = errors.New("invoke's normal target has another predecessor")

	// ErrBlockAddr is a ptr.blockaddr naming a block that no brind in the
	// same function branches to (§19.6). §17 admits the address for brind
	// and nothing else, so the label names a block control can only fall
	// into.
	ErrBlockAddr = errors.New("blockaddr names a block no brind targets")

	// ErrInit is a global initializer whose structure is not the declared
	// type's (§19.10). §5 requires the ftype for exactly this reason:
	// there is no inference from an initializer, so the declaration is the
	// only thing the structure can be checked against.
	ErrInit = errors.New("initializer does not match the declared type")

	// ErrStructOffset is a struct whose stated field offsets are not well
	// formed (§19.18): some fields carrying at and some not, or offsets
	// that do not strictly increase, or a field that runs into its
	// successor.
	ErrStructOffset = errors.New("struct field offsets are not well formed")

	// ErrDominance is a use of a definition that does not dominate it
	// (§19.1). Single assignment itself is structural — an ir.Def is
	// created by the one thing that defines it and cannot be assigned a
	// second time — so this is the whole of the rule that survives into a
	// built module.
	ErrDominance = errors.New("definition does not dominate use")

	// ErrAsmConstraint is an inline-assembly operand list that does not
	// hold together (§8b). The template is not read here — a reference to
	// an operand that does not exist is the assembler's to catch, and
	// §8b says so — but the list around it is checkable without a target:
	// a constraint that names nothing, an output that is not a place a
	// value can be written, and a matching constraint naming an output
	// past the end of the output list.
	ErrAsmConstraint = errors.New("inline assembly's constraint list is not well formed")

	// ErrAsmGotoTarget is an asm goto whose fallthrough target's parameter
	// list is not the edge's arguments followed by the statement's outputs
	// (§19.16). It is ErrInvokeTarget's rule for the same shape: an output
	// register of the instruction's own would have to dominate the edges
	// the assembled text branches along, and on those the text did not
	// reach the end that writes it.
	ErrAsmGotoTarget = errors.New("asm goto's fallthrough target does not match its outputs")

	// ErrAsmGotoEdge is an asm goto's fallthrough target with a second
	// predecessor. The outputs arrive on this edge and on no other, so a
	// block reached another way would read them undefined — the same
	// reason ErrInvokeEdge exists.
	ErrAsmGotoEdge = errors.New("asm goto's fallthrough target has another predecessor")
)

// An Error is one verifier failure, positioned the way a §19 fault
// actually sits: in a function, in one of its blocks, at an instruction
// within that block.
type Error struct {
	Func   string // "" at module scope, where a global or a type is at fault
	Block  string // "" when the fault is the function's own
	Inst   int    // index into the block's instructions, terminator last; -1 for none
	Op     ir.Op  // the zero Op when no single instruction is at fault
	Detail string
	Err    error // one of the sentinels above
}

func (e *Error) Error() string {
	var b strings.Builder
	b.WriteString("verify: ")
	if e.Func != "" {
		b.WriteString("@" + e.Func)
		if e.Block != "" {
			b.WriteString(" @" + e.Block)
			if e.Inst >= 0 {
				fmt.Fprintf(&b, " #%d", e.Inst)
			}
		}
		b.WriteString(": ")
	}
	if e.Op.Verb != "" {
		b.WriteString(e.Op.String() + ": ")
	}
	b.WriteString(e.Err.Error())
	if e.Detail != "" {
		b.WriteString(": " + e.Detail)
	}
	return b.String()
}

func (e *Error) Unwrap() error { return e.Err }

// Errors is every fault one run found, in the order it found them.
//
// The builder is sticky and first-wins because it is a *writer*: after the
// first failure every later call is a no-op, so there is only ever one
// fault to report. A verifier is a reader of a module that is already
// finished, where the faults are independent and all of them are still
// there to be found — reporting one at a time would make fixing a module a
// sequence of runs. errors.Is reaches every sentinel in the list.
type Errors []*Error

func (es Errors) Error() string {
	switch len(es) {
	case 0:
		return "verify: no errors"
	case 1:
		return es[0].Error()
	}
	return fmt.Sprintf("%s (and %d more)", es[0], len(es)-1)
}

// Unwrap gives errors.Is and errors.As every fault in the list, not just
// the first one printed.
func (es Errors) Unwrap() []error {
	out := make([]error, len(es))
	for i, e := range es {
		out[i] = e
	}
	return out
}
