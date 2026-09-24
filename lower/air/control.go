package air

// Terminators and edges.
//
// A block parameter is a phi, and an edge into the block is an incoming
// value for each. A branch's one edge is its block's; a conditional
// branch's or a table's edge that carries arguments goes through a block
// of its own -- a trampoline, a branch and nothing else -- so that every
// phi entry names its own predecessor, which is what a phi needs when two
// edges from one block go to the same target with different arguments.
// An edge that carries none is a bare branch, which is every edge an if
// makes.

import (
	"fmt"

	am "github.com/vertex-language/air"
	"github.com/vertex-language/ir"
)

func (x *fn) term(in *ir.Inst) error {
	ts := in.Targets()
	b := x.cur
	switch in.Op().Verb {
	case ir.VBr:
		b.Br(x.edge(ts[0], false))

	case ir.VBrIf:
		then := x.edge(ts[0], true)
		els := x.edge(ts[1], true)
		b.CondBr(x.arg(in, 0), then, els)

	case ir.VBrTable:
		cases, dflt := ts[:len(ts)-1], ts[len(ts)-1]
		var cs []am.Case
		for i, t := range cases {
			cs = append(cs, am.Case{Value: am.ConstUint(am.UInt, uint64(i)), Dest: x.edge(t, true)})
		}
		b.Switch(x.arg(in, 0), x.edge(dflt, true), cs...)

	case ir.VReturn:
		if len(in.Args()) > 0 {
			return fmt.Errorf("a kernel returns nothing")
		}
		b.Ret()

	case ir.VTrap:
		// An Apple GPU has no trap an app can see: the thread ends here,
		// and a divergence from §0 is documented rather than undefined.
		b.Ret()

	case ir.VBrInd:
		return fmt.Errorf("an indirect branch: AIR has no block addresses")
	case ir.VInvoke, ir.VInvokeInd, ir.VResume:
		return fmt.Errorf("there is no unwinding on the GPU")
	case ir.VTailCall, ir.VTailCallInd:
		return fmt.Errorf("a tail call: every function is inlined into its kernel")
	case ir.VAsmGoto:
		return fmt.Errorf("inline assembly: an Apple GPU's instruction set is not published")
	default:
		return fmt.Errorf("not lowered for AIR yet")
	}
	return nil
}

// edge is the block a branch goes to for a target: the target itself, or,
// for an edge that carries arguments where more than one edge leaves the
// block, a trampoline. The arguments become the target's phis' incoming
// values from whichever block it is.
func (x *fn) edge(t ir.BlockTarget, shared bool) *am.Block {
	dest := x.blocks[t.Block()]
	args := t.Args()
	if len(args) == 0 {
		return dest
	}
	from, to := x.cur, dest
	if shared {
		from = x.k.Block(t.Block().Label() + ".edge")
		from.Br(dest)
		to = from
	}
	for i, p := range t.Block().Params() {
		x.phis[p].AddIncoming(x.v(args[i]), from)
	}
	return to
}
