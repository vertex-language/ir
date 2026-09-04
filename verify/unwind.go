package verify

import "github.com/vertex-language/ir"

// §19.4 (pads and personalities), §19.5 (resume), and §19.16 (an
// invoke's normal target).
//
// One clause of §19.4 is not here: that an unwind edge names a pad block
// at all. The builder checks it at emission with ir.ErrPlacement, because
// an unwind edge is a *ir.Block passed to Invoke rather than a
// BlockTarget, so the block is in hand and its kind is already known.
// What survives into a finished module is the other direction — who else
// reaches that pad — which no single call site can see.

// unwind runs the three rules in the order their faults nest: an edge
// into a pad first, since it makes every later question about that pad
// ambiguous, then the personality the unwinder needs to run one at all,
// then resume, then the shape of the normal edge.
func (c *checker) unwind(f *ir.Func, blocks []*ir.Block, dt *domTree) {
	c.padEdges(blocks)
	if c.full() {
		return
	}
	c.personality(f, blocks)
	if c.full() {
		return
	}
	c.resumes(blocks, dt)
	if c.full() {
		return
	}
	c.invokeTargets(f, blocks)
}

// padEdges is §19.4's reachability clause: a pad block is reached only by
// unwind edges.
//
// A pad's two parameters — the exception object and the personality's
// selector — are supplied by the personality routine, not by an argument
// list, which is why ir.Invoke takes the pad as a plain block and not as
// a BlockTarget. An ordinary branch to one would arrive with both unset,
// and there is no syntax that could set them. An invoke's *normal* edge
// naming a pad is the same fault and is caught here too, since it is a
// BlockTarget like any other.
func (c *checker) padEdges(blocks []*ir.Block) {
	for _, b := range blocks {
		term := b.Term()
		for _, t := range term.Targets() {
			if t.Block() == nil || !t.Block().IsPad() {
				continue
			}
			c.fail(b, instIndex(b, term), term.Op(), ErrPadEdge,
				"@%s is a pad block, reachable only by an unwind edge", t.Block().Label())
			if c.full() {
				return
			}
		}
		for _, l := range term.Labels() {
			if !l.IsPad() {
				continue
			}
			c.fail(b, instIndex(b, term), term.Op(), ErrPadEdge,
				"@%s is a pad block, and a brind label is an ordinary edge", l.Label())
			if c.full() {
				return
			}
		}
	}
}

// personality is §19.4's last clause: a function containing an invoke or
// an invokeind declares one.
//
// The fault is reported at the first invoke rather than against the
// function, because that is the instruction whose unwind edge has nothing
// to run. A function with a pad block but no invoke needs no personality
// and gets no fault here: nothing can reach that pad, which is §19.2's
// finding, already made.
func (c *checker) personality(f *ir.Func, blocks []*ir.Block) {
	if f.PersonalityFn() != nil {
		return
	}
	for _, b := range blocks {
		for i, in := range b.All() {
			if !isInvoke(in.Op()) {
				continue
			}
			c.fail(b, i, in.Op(), ErrPersonality,
				"@%s unwinds but declares no personality routine", f.Name())
			return
		}
	}
}

// resumes is §19.5: a resume takes the ptr parameter of a pad block that
// dominates it.
//
// Two halves, one rule. The operand has to be a pad's exception object —
// not some other pointer the unwinder never handed out — and the pad it
// came from has to be one control actually ran through, which is what
// dominance says. A resume in a block some other pad's edge can also
// reach would hand the unwinder an object from a pad that did not fire.
//
// §19.1 finds that second half too, and does not make this redundant: a
// pad parameter is a definition like any other, so a use it does not
// dominate is undominated. The two faults say different things about one
// instruction — that the value is not visible here, and that a resume's
// operand has to come from the pad that fired — so both are reported.
func (c *checker) resumes(blocks []*ir.Block, dt *domTree) {
	for _, b := range blocks {
		for i, in := range b.All() {
			if in.Op() != (ir.Op{Type: ir.TypeNone, Verb: ir.VResume}) {
				continue
			}
			d := in.Arg(0)
			if d == nil {
				continue // a zero Value is ir.ErrPoison, sticky and already returned
			}
			pad := d.Block()
			if pad == nil || !pad.IsPad() || !d.IsParam() || d.Index() != 0 {
				c.fail(b, i, in.Op(), ErrResume,
					"%%%s is not the exception object parameter of a pad block", d)
			} else if dt.reachable(b) && !dt.dominates(pad, b) {
				c.fail(b, i, in.Op(), ErrResume,
					"the pad @%s does not dominate @%s", pad.Label(), b.Label())
			}
			if c.full() {
				return
			}
		}
	}
}

// invokeTargets is §19.16: an invoke's normal target declares exactly the
// edge's arguments followed by the callee's results, and no other edge
// reaches it.
//
// This is the arity rule the builder cannot defer like every other
// branch's, and ir.Invoke deliberately does not try: its normal target
// has more parameters than the edge supplies arguments, so the check that
// serves a br would reject every correct invoke. The trailing parameters
// are the call's results, which is where an invoke keeps them — a result
// register of its own would have to dominate both edges, and on the
// unwind edge the call did not complete.
func (c *checker) invokeTargets(f *ir.Func, blocks []*ir.Block) {
	edges := edgeCounts(blocks)

	for _, b := range blocks {
		for i, in := range b.All() {
			if !isInvoke(in.Op()) {
				continue
			}
			sig := calleeSig(in)
			targets := in.Targets()
			if sig == nil || len(targets) == 0 || targets[0].Block() == nil {
				continue // ir.ErrPoison or ir.ErrSignature, sticky and already returned
			}
			normal := targets[0]
			blk := normal.Block()

			if edges[blk] > 1 {
				c.fail(b, i, in.Op(), ErrInvokeEdge,
					"@%s is reached by %d edges; the call supplies its trailing parameters on one of them",
					blk.Label(), edges[blk])
				if c.full() {
					return
				}
			}

			args, rets, params := normal.Args(), sig.Rets(), blk.Params()
			if len(params) != len(args)+len(rets) {
				c.fail(b, i, in.Op(), ErrInvokeTarget,
					"@%s takes %d parameters; the edge supplies %d arguments and the callee returns %d",
					blk.Label(), len(params), len(args), len(rets))
				if c.full() {
					return
				}
				continue // the per-parameter comparison below has nothing to line up
			}

			for j, a := range args {
				if a.Type() == params[j].Type() {
					continue
				}
				c.fail(b, i, in.Op(), ErrInvokeTarget,
					"@%s parameter %d is %s, the edge supplies %s",
					blk.Label(), j, params[j].Type(), a.Type())
				if c.full() {
					return
				}
			}
			for j, r := range rets {
				p := params[len(args)+j]
				if r.Type == p.Type() {
					continue
				}
				c.fail(b, i, in.Op(), ErrInvokeTarget,
					"@%s parameter %d is %s, result %d of the callee is %s",
					blk.Label(), len(args)+j, p.Type(), j, r.Type)
				if c.full() {
					return
				}
			}
		}
	}
}

// edgeCounts is how many edges reach each block, counting duplicates: two
// invokes in one block naming one normal target are two edges, and
// Block.Preds would report the one block they share.
func edgeCounts(blocks []*ir.Block) map[*ir.Block]int {
	n := make(map[*ir.Block]int, len(blocks))
	for _, b := range blocks {
		for _, s := range b.Succs() {
			n[s]++
		}
	}
	return n
}

// isInvoke reports whether op is one of the two calls with an unwind edge.
func isInvoke(op ir.Op) bool {
	return op.Type == ir.TypeNone && (op.Verb == ir.VInvoke || op.Verb == ir.VInvokeInd)
}

// calleeSig is the signature an invoke calls through: the callee's own for
// an invoke, and the func typedef's for an invokeind.
func calleeSig(in *ir.Inst) *ir.Sig {
	if c := in.Callee(); c != nil {
		return c.Signature()
	}
	t := in.NamedType()
	if t == nil || t.Kind() != ir.KindFunc {
		return nil
	}
	return t.Sig()
}
