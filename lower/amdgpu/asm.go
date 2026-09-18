package amdgpu

// §G4. The template is GCC's, in the same %0/%1 convention lower/asmtmpl
// parses, and the text goes through the amdgpu repo's assembler into
// the section the function is being emitted into, after each reference
// becomes the register the operand was assigned. Constraints:
//
//   - "v", "reg", "r": the operand's own register, which is a VGPR for
//     every value but an i1 and an SGPR pair for one.
//   - "s": a scalar. An i1 is one already; any other value is read out
//     of its VGPR's first active lane into a temporary SGPR before the
//     text, and an output written back into the VGPR after — the asm
//     sees a uniform value, which is what it asked for.
//
// Clobbers may name vcc, m0, scc, exec, cc and memory, none of which
// this backend keeps live across an instruction of its own; a named
// register the allocator hands out cannot be reserved, and is refused.

import (
	"fmt"
	"strings"

	"github.com/vertex-language/amdgpu/asm"

	"github.com/vertex-language/ir"
	"github.com/vertex-language/ir/lower/asmtmpl"
	"github.com/vertex-language/ir/lower/mir"
)

// asmOp is the template with its operands: Defs the outputs, Uses the
// inputs, each already in the register the constraint asked for.
type asmOp struct {
	template string
	refs     []asmtmpl.Ref
	nouts    int
}

func (x *fnState) asmInst(c *cursor, in *ir.Inst) error {
	a := in.Asm()
	for _, cl := range a.Clobbers {
		switch cl {
		case "memory", "cc", "vcc", "m0", "scc", "exec":
		default:
			return fmt.Errorf("asm clobbers %q, which the allocator cannot keep clear; name vcc, m0, scc, exec, cc or memory", cl)
		}
	}
	nops := len(a.Outs) + len(a.Args)
	refs, err := asmtmpl.Parse(a.Template, nops, nil)
	if err != nil {
		return err
	}
	for _, r := range refs {
		if r.IsLabel() {
			return fmt.Errorf("a label reference in an asm that is not asm goto")
		}
		if r.Modifier != 0 {
			return fmt.Errorf("modifier %q; a register here has one spelling", r.Modifier)
		}
	}

	// Outputs: the vreg the text writes, and where it goes after.
	type writeBack struct {
		from, to mir.VReg
		w        width
	}
	var defs []mir.VReg
	var after []writeBack
	for i, o := range a.Outs {
		d, err := x.vr.define(in.Result(i))
		if err != nil {
			return err
		}
		scalar, err := asmScalar(o.Constraint, o.Type)
		if err != nil {
			return fmt.Errorf("asm output %d: %w", i, err)
		}
		if !scalar {
			defs = append(defs, d)
			continue
		}
		w := x.vr.widthOfVReg(d)
		sw := s32
		if w == v64 {
			sw = s64
		}
		t := x.vr.temp(sw)
		defs = append(defs, t)
		after = append(after, writeBack{from: t, to: d, w: w})
	}
	// Inputs: read into a scalar where asked.
	var uses []mir.VReg
	for i, g := range a.Args {
		v, err := x.vr.use(g.Def)
		if err != nil {
			return err
		}
		scalar, err := asmScalar(g.Constraint, g.Def.Type())
		if err != nil {
			return fmt.Errorf("asm input %d: %w", i, err)
		}
		if !scalar {
			uses = append(uses, v)
			continue
		}
		switch x.vr.widthOfVReg(v) {
		case v32:
			t := x.vr.temp(s32)
			x.emit(c, "v_readfirstlane_b32", rs(t), rs(v), def(0), use(0))
			v = t
		case v64:
			t := x.vr.temp(s64)
			x.emit(c, "v_readfirstlane_b32", rs(t), rs(v), defLo(0), useLo(0))
			x.emit(c, "v_readfirstlane_b32", rs(t), rs(v, t), defHi(0), useHi(0))
			v = t
		}
		uses = append(uses, v)
	}
	c.Emit(mir.Instr{Op: asmOp{template: a.Template, refs: refs, nouts: len(defs)}, Defs: defs, Uses: uses})
	for _, wb := range after {
		if wb.w == v64 {
			x.movPair(c, wb.to, wb.from)
		} else {
			x.emit(c, "v_mov_b32", rs(wb.to), rs(wb.from), def(0), use(0))
		}
	}
	return nil
}

// asmScalar reads a constraint: whether the operand goes in an SGPR
// where it would not otherwise be.
func asmScalar(c ir.Constraint, t ir.RegType) (bool, error) {
	switch c.String() {
	case "reg", "r", "v", "":
		return false, nil
	case "s":
		return t != ir.TypeI1, nil
	case "mem", "imm":
		return false, fmt.Errorf("constraint %q: an asm operand here is a register", c)
	}
	return false, fmt.Errorf("constraint %q is not one this backend knows (v, s, reg)", c)
}

// asmEmit expands the template and assembles it into the section.
func (e *emitter) asmEmit(op asmOp, in mir.Instr) error {
	name := func(v mir.VReg) string { return e.reg(v).String() }
	text, err := asmtmpl.Expand(op.template, op.refs, func(r asmtmpl.Ref) (string, error) {
		if r.Operand < op.nouts {
			return name(in.Defs[r.Operand]), nil
		}
		return name(in.Uses[r.Operand-op.nouts]), nil
	})
	if err != nil {
		return err
	}
	// The assembler takes one instruction a line; the template may
	// separate them with semicolons or newlines.
	text = strings.ReplaceAll(text, ";", "\n")
	if err := asm.AssembleFragment(e.text, text, asm.Options{Features: e.text.Module().Features()}); err != nil {
		return fmt.Errorf("asm: %w", err)
	}
	return nil
}
