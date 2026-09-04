package arm64

import (
	"fmt"

	"github.com/vertex-language/ir"
	"github.com/vertex-language/ir/lower/mir"
)

// iselBlockAddr lowers §D3's blockaddr: a block's address, which is ADRP and
// ADD like any other address here.
//
// The block's label has to be a symbol for that to work. A bare label folds
// into a same-section reference and leaves no trace, and the page of an
// address is not something this layer can fold — nothing here has assigned
// the section a load address yet. labeledBlocks is what promotes them.
func iselBlockAddr(c *cursor, vr *vregs, in *ir.Inst) error {
	op := in.Op()
	labels := in.Labels()
	if len(labels) != 1 {
		return fmt.Errorf("%s: %d blocks named, want exactly one", op, len(labels))
	}
	dst, err := vr.define(in.Result(0))
	if err != nil {
		return fmt.Errorf("%s: %w", op, err)
	}
	blk := labels[0]
	c.Emit(mir.Instr{
		Op:   blockAddrOp{label: blockLabel(blk.Func(), blk)},
		Defs: []mir.VReg{dst},
	})
	return nil
}

// iselBrInd lowers §G2's computed goto: BR through a register.
//
// The named labels are declared as successors and nothing else is done with
// them. They are what liveness needs — a value the target block reads has to
// still be live at a branch that can reach it — and the address itself came
// from a blockaddr, which is the only thing §D3 says a blockaddr is for.
func iselBrInd(fn *ir.Func, mf *mir.Func, c *cursor, vr *vregs, term *ir.Inst) error {
	ptr, ok := vr.lookup(term.Arg(0))
	if !ok {
		return fmt.Errorf("brind: pointer defined outside the function")
	}
	for _, b := range term.Labels() {
		mf.Succ(c.blk, blockLabel(fn, b))
	}
	// Into a vreg of this branch's own, so the allocator is free to place
	// it anywhere the successors do not need.
	target := vr.temp(w64)
	emitCopy(c, target, ptr, w64)
	c.Emit(mir.Instr{Op: brIndOp{}, Uses: []mir.VReg{target}})
	return nil
}

// iselBrTable lowers §G2's br_table.
//
// One unsigned compare covers both ends of the range: a negative selector read
// as unsigned is a very large one, so the branch that catches an index past
// the last entry catches a negative one too.
func iselBrTable(fn *ir.Func, mf *mir.Func, c *cursor, vr *vregs, term *ir.Inst) error {
	selector, ok := vr.lookup(term.Arg(0))
	if !ok {
		return fmt.Errorf("br_table: selector defined outside the function")
	}
	targets := term.Targets()
	if len(targets) == 0 {
		return fmt.Errorf("br_table: no targets")
	}

	// The last target is the default; the rest are the table.
	labels := make([]string, len(targets)-1)
	for i := range labels {
		l, err := edgeTarget(fn, mf, c, vr, targets[i], fmt.Sprintf("case%d", i))
		if err != nil {
			return err
		}
		labels[i] = l
		mf.Succ(c.blk, l)
	}
	dflt, err := edgeTarget(fn, mf, c, vr, targets[len(targets)-1], "default")
	if err != nil {
		return err
	}
	mf.Succ(c.blk, dflt)

	// The scratch registers are the allocator's, named as destinations of
	// the terminator: what is free at a branch is what allocation decides,
	// and a destination there interferes with everything live into the
	// successors — which is what keeps this sequence off a value they want.
	c.Emit(mir.Instr{
		Op: brTableOp{
			id:      fmt.Sprintf("%s.table", c.blk.Label),
			targets: labels,
			dflt:    dflt,
		},
		Defs: []mir.VReg{vr.temp(w64), vr.temp(w64), vr.temp(w64)},
		Uses: []mir.VReg{selector},
	})
	return nil
}

// labeledBlocks is every block a blockaddr names, which has to be a symbol
// rather than a bare label: a page reference needs one to survive Finalize,
// and the page of an address depends on where the section loads.
//
// A jump table's targets are not here. Its entries are distances within one
// section, which fold into the bytes and leave no relocation to need a symbol.
func labeledBlocks(mf *mir.Func) map[string]bool {
	out := map[string]bool{}
	for _, mb := range mf.Blocks {
		for _, in := range mb.Instrs {
			switch op := in.Op.(type) {
			case blockAddrOp:
				out[op.label] = true
			case asmOp:
				// An asm goto's labels are branched to by text this
				// package assembled rather than emitted, so the reference
				// arrives as a symbol reference and needs a symbol to
				// resolve against. It is Local, so the distance still folds
				// and no relocation survives.
				for _, l := range op.emitted {
					out[l] = true
				}
			}
		}
	}
	return out
}
