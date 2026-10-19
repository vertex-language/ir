package verify

// Package verify checks §8b inline assembly constraint lists: constraints cannot
// be empty, outputs cannot be constrained to literals (imm), and matching constraint
// indices must refer to valid outputs.

import (
	"fmt"

	"github.com/vertex-language/ir"
)

func (c *checker) asm(blocks []*ir.Block) {
	for _, b := range blocks {
		for i, in := range b.All() {
			a := in.Asm()
			if a == nil {
				continue
			}
			for j, o := range a.Outs {
				c.asmConstraint(b, i, in, len(a.Outs), true, j, o.Constraint)
				if c.full() {
					return
				}
			}
			for j, arg := range a.Args {
				c.asmConstraint(b, i, in, len(a.Outs), false, j, arg.Constraint)
				if c.full() {
					return
				}
			}
			for j, name := range a.Clobbers {
				if name != "" {
					continue
				}
				c.fail(b, i, in.Op(), ErrAsmConstraint,
					"clobber %d is the empty string, which is not a register name", j)
				if c.full() {
					return
				}
			}
		}
	}
}

// asmConstraint checks one operand's constraint. nouts is the length of the
// output list, which is what a matching constraint indexes into; out says
// which list this operand is in, and j its position within that list.
func (c *checker) asmConstraint(b *ir.Block, i int, in *ir.Inst, nouts int, out bool, j int, con ir.Constraint) {
	where := fmt.Sprintf("input %d", j)
	if out {
		where = fmt.Sprintf("output %d", j)
	}

	if con.String() == "" {
		c.fail(b, i, in.Op(), ErrAsmConstraint,
			"%s has the empty constraint, which names no register class, no tie and no target letter", where)
		return
	}

	tie, tied := con.Tied()
	switch {
	case out && tied:
		// An output matching another operand would be a tie in the
		// direction the numbering does not run: %0 is decided by this
		// list, and an entry in it cannot defer to a later one.
		c.fail(b, i, in.Op(), ErrAsmConstraint,
			"%s is constrained %q, a matching constraint; only an input may match an output", where, con.String())
	case tied && tie >= nouts:
		c.fail(b, i, in.Op(), ErrAsmConstraint,
			"%s is constrained %q and there %s, so it matches an output that does not exist",
			where, con.String(), plural(nouts))
	case out && con == ir.CImm:
		// Not merely unsupported: an immediate is a literal in the
		// instruction stream, and there is nowhere for a result to go.
		c.fail(b, i, in.Op(), ErrAsmConstraint,
			"%s is constrained imm; an immediate is a literal and not a place a result can be written", where)
	}
}

// plural spells the output count the way the diagnostic reads it.
func plural(n int) string {
	switch n {
	case 0:
		return "are no outputs"
	case 1:
		return "is one output"
	}
	return fmt.Sprintf("are %d outputs", n)
}

// asmGotoTargets checks §19.16 for asm goto: outputs appear as trailing parameters
// on the single-predecessor fallthrough target block.
func (c *checker) asmGotoTargets(blocks []*ir.Block) {
	edges := edgeCounts(blocks)

	for _, b := range blocks {
		for i, in := range b.All() {
			if in.Op().Verb != ir.VAsmGoto {
				continue
			}
			a := in.Asm()
			targets := in.Targets()
			if a == nil || len(targets) == 0 || targets[0].Block() == nil {
				continue // ir.ErrPoison, sticky and already returned
			}
			fall := targets[0]
			blk := fall.Block()

			if len(a.Outs) > 0 && edges[blk] > 1 {
				c.fail(b, i, in.Op(), ErrAsmGotoEdge,
					"@%s is reached by %d edges; the outputs arrive on one of them",
					blk.Label(), edges[blk])
				if c.full() {
					return
				}
			}

			args, params := fall.Args(), blk.Params()
			if len(params) != len(args)+len(a.Outs) {
				c.fail(b, i, in.Op(), ErrAsmGotoTarget,
					"@%s takes %d parameters; the edge supplies %d arguments and the statement declares %d outputs",
					blk.Label(), len(params), len(args), len(a.Outs))
				if c.full() {
					return
				}
				continue // nothing left to line up
			}
			for j, o := range a.Outs {
				p := params[len(args)+j]
				if o.Type == p.Type() {
					continue
				}
				c.fail(b, i, in.Op(), ErrAsmGotoTarget,
					"@%s parameter %d is %s, output %d is %s",
					blk.Label(), len(args)+j, p.Type(), j, o.Type)
				if c.full() {
					return
				}
			}
		}
	}
}
