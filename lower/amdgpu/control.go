package amdgpu

import (
	"fmt"

	"github.com/vertex-language/ir"
	"github.com/vertex-language/ir/lower/mir"
)

// A cursor is the block isel is filling, and the ability to start another.
type cursor struct {
	fn   *ir.Func
	mf   *mir.Func
	blk  *mir.Block
	base string
	n    int

	// masked says the function will be structurized, so an i1 copy
	// must keep the bits of lanes it is not copying for.
	masked bool
}

func (x *fnState) cursor(blk *mir.Block) *cursor {
	return &cursor{fn: x.fn, mf: x.mf, blk: blk, base: blk.Label, masked: x.divergent}
}

func (c *cursor) Emit(in mir.Instr) { c.blk.Emit(in) }

// open starts a block under the one isel began in, and does not move the cursor.
func (c *cursor) open(kind string) *mir.Block {
	c.n++
	return c.mf.NewBlock(fmt.Sprintf("%s.%s%d", c.base, kind, c.n))
}

// blockLabel is the section label one ir block gets.
func blockLabel(fn *ir.Func, blk *ir.Block) string {
	if blk.IsEntry() {
		return fn.Name()
	}
	return fn.Name() + "." + blk.Label()
}

// emitCopy is a register-to-register move, marked as a copy so the
// allocator can coalesce it away — except a lane mask in a function
// that will be structurized, which is a read-modify-write under exec
// and cannot share a register with its source.
func emitCopy(c *cursor, dst, src mir.VReg, w width) {
	if c.masked && w == s64 {
		c.Emit(mir.Instr{Op: movOp{w: w, masked: true}, Defs: []mir.VReg{dst}, Uses: []mir.VReg{src, dst}})
		return
	}
	c.Emit(mir.Instr{Op: movOp{w: w}, Defs: []mir.VReg{dst}, Uses: []mir.VReg{src}, Copy: true})
}

// selectBlock selects one VIR block into its MIR block.
func (x *fnState) selectBlock(c *cursor, blk *ir.Block) error {
	for _, in := range blk.Insts() {
		if err := x.selectInst(c, in); err != nil {
			return fmt.Errorf("%s: %w", in.Op(), err)
		}
	}
	term := blk.Term()
	if err := x.selectTerm(c, term); err != nil {
		return fmt.Errorf("%s: %w", term.Op(), err)
	}
	return nil
}

func (x *fnState) selectTerm(c *cursor, in *ir.Inst) error {
	ts := in.Targets()
	switch in.Op().Verb {
	case ir.VBr:
		dest, err := x.edgeTarget(c, ts[0], "edge")
		if err != nil {
			return err
		}
		c.Emit(mir.Instr{Op: branchOp{target: dest}})
		c.mf.Succ(c.blk, dest)

	case ir.VBrIf:
		// Uniform or not: a cbranchOp is a scalar branch on the mask
		// when the function has no divergent branch, and a predicate
		// assignment once it is structurized.
		m, err := x.vr.use(in.Arg(0))
		if err != nil {
			return err
		}
		then, err := x.edgeTarget(c, ts[0], "then")
		if err != nil {
			return err
		}
		els, err := x.edgeTarget(c, ts[1], "else")
		if err != nil {
			return err
		}
		c.Emit(mir.Instr{Op: cbranchOp{then: then, els: els}, Uses: []mir.VReg{m}})
		c.mf.Succ(c.blk, then)
		c.mf.Succ(c.blk, els)

	case ir.VReturn:
		if len(in.Args()) > 0 {
			return fmt.Errorf("a kernel returns nothing")
		}
		c.Emit(mir.Instr{Op: endpgmOp{}})

	case ir.VTrap:
		c.Emit(mir.Instr{Op: trapOp{}})

	case ir.VBrTable:
		return x.brTable(c, in)
	case ir.VBrInd:
		return fmt.Errorf("brind has no lowering on a target that branches the whole wave")
	case ir.VInvoke, ir.VInvokeInd, ir.VResume:
		return fmt.Errorf("there is no unwinding on the device")
	default:
		return fmt.Errorf("not lowered")
	}
	return nil
}

// edgeTarget is the label a branch should name for one target: the block
// itself when the edge carries no arguments, and a block made to assign
// them when it does.
func (x *fnState) edgeTarget(c *cursor, t ir.BlockTarget, kind string) (string, error) {
	moves, err := x.edgeCopies(t)
	if err != nil {
		return "", err
	}
	dest := blockLabel(x.fn, t.Block())
	if len(moves) == 0 {
		return dest, nil
	}
	edge := c.open(kind)
	ec := x.cursor(edge)
	x.emitParallelCopy(ec, moves)
	ec.Emit(mir.Instr{Op: branchOp{target: dest}})
	x.mf.Succ(edge, dest)
	return edge.Label, nil
}

type copyPair struct {
	dst, src mir.VReg
	w        width
}

func (x *fnState) edgeCopies(t ir.BlockTarget) ([]copyPair, error) {
	blk := t.Block()
	args := t.Args()
	params := blk.Params()
	if len(args) != len(params) {
		return nil, fmt.Errorf("@%s takes %d parameters, the edge supplies %d", blk.Label(), len(params), len(args))
	}
	var out []copyPair
	for i, a := range args {
		src, err := x.vr.use(a)
		if err != nil {
			return nil, fmt.Errorf("@%s: edge argument %d: %w", blk.Label(), i, err)
		}
		dst, ok := x.vr.lookup(params[i])
		if !ok {
			return nil, fmt.Errorf("@%s: parameter %d has no vreg", blk.Label(), i)
		}
		if src == dst {
			continue
		}
		out = append(out, copyPair{dst: dst, src: src, w: x.vr.widthOfVReg(dst)})
	}
	return out, nil
}

// emitParallelCopy issues a set of simultaneous assignments as a
// sequence, breaking a cycle through a temporary. The fourth copy of
// this in the tree, and the one that should make it lower/mir's.
func (x *fnState) emitParallelCopy(c *cursor, moves []copyPair) {
	pending := append([]copyPair(nil), moves...)
	for len(pending) > 0 {
		i := readyMove(pending)
		if i < 0 {
			tmp := x.vr.temp(pending[0].w)
			emitCopy(c, tmp, pending[0].src, pending[0].w)
			pending[0].src = tmp
			continue
		}
		m := pending[i]
		emitCopy(c, m.dst, m.src, m.w)
		pending = append(pending[:i], pending[i+1:]...)
	}
}

func readyMove(pending []copyPair) int {
	for i, m := range pending {
		blocked := false
		for j, other := range pending {
			if i != j && other.src == m.dst {
				blocked = true
				break
			}
		}
		if !blocked {
			return i
		}
	}
	return -1
}
