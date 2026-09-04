package verify

// §8b's constraint list.
//
// Nothing here reads the template. §8b puts that at the backend that
// assembles it, which is the only layer that knows what `%w0` means or
// which letter is a register class on this target, and this file keeps to
// what is true of an operand list on every target at once.
//
// Three things are. A constraint that is the empty string names nothing —
// not a class, not a tie, not a target letter — and there is no reading of
// it under which the operand has a place to live. An output constrained
// `imm` asks the assembler to write to a literal. And a matching
// constraint is a number into the output list, so a number past its end is
// the one arity fault §8b keeps at this layer: every backend reads a tie
// with ir.Constraint.Tied and then has to do something with an index that
// is out of range, and the only honest something is to refuse the module
// before a backend has to guess.
//
// That last one is why this rule exists at all. An out-of-range tie is not
// a crash anywhere: each backend falls back to treating the input as an
// ordinary operand in a register of its own, the template's %1 then names
// a register the author did not mean, and the object is wrong with no
// diagnostic. GCC rejects the same constraint outright.

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

// asmGotoTargets is §19.16 for the terminator form.
//
// An asm goto's outputs are the trailing parameters of its fallthrough
// target, which is invoke's rule applied to the same shape: the instruction
// defines no register of its own, because one would have to dominate the
// edges the assembled text branches along, and on those the text did not
// reach the end that writes it. The fallthrough edge is the one on which it
// did.
//
// So the target's parameter list is the edge's arguments followed by one
// parameter per output, and the target is reached by that edge alone —
// another predecessor would arrive with the outputs unset.
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
