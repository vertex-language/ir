package ptx

// §G4. PTX text is what CUDA inline assembly already contains, in the
// same %0/%1 template convention lower/asmtmpl parses: each reference
// becomes the register the operand lives in, %% becomes %, and the text
// goes out through Body.Raw, unread. CUDA's constraint letters — r, l, f,
// d for the 32- and 64-bit integer and float registers, h for a halfword
// — are accepted where they agree with the operand's own type, since a
// register here has exactly one type; `reg` means the same.

import (
	"fmt"

	"github.com/vertex-language/ir"
	"github.com/vertex-language/ir/lower/asmtmpl"
	"github.com/vertex-language/ptx"
)

func (x *fn) asm(in *ir.Inst) error {
	a := in.Asm()
	if len(a.Clobbers) > 0 {
		// PTX has no register to clobber and no condition codes; "memory"
		// is honoured by there being nothing here that moves a load
		// across an instruction it cannot see through.
		for _, c := range a.Clobbers {
			if c != "memory" && c != "cc" {
				return fmt.Errorf("asm clobbers %q; PTX names no physical register", c)
			}
		}
	}
	ops := make([]ptx.Reg, 0, len(a.Outs)+len(a.Args))
	for i, o := range a.Outs {
		if err := checkConstraint(o.Constraint, o.Type, true); err != nil {
			return fmt.Errorf("asm output %d: %w", i, err)
		}
		ops = append(ops, x.v(in.Result(i)))
	}
	for i, g := range a.Args {
		if err := checkConstraint(g.Constraint, g.Def.Type(), false); err != nil {
			return fmt.Errorf("asm input %d: %w", i, err)
		}
		ops = append(ops, x.v(g.Def))
	}
	refs, err := asmtmpl.Parse(a.Template, len(ops), nil)
	if err != nil {
		return err
	}
	text, err := asmtmpl.Expand(a.Template, refs, func(r asmtmpl.Ref) (string, error) {
		if r.IsLabel() {
			return "", fmt.Errorf("a label reference in an asm that is not asm goto")
		}
		if r.Modifier != 0 {
			return "", fmt.Errorf("modifier %q; PTX registers have one view", r.Modifier)
		}
		return ops[r.Operand].Text(), nil
	})
	if err != nil {
		return err
	}
	x.b.Raw(text)
	return nil
}

// checkConstraint admits the constraints that name a register of the
// operand's own type. A tied constraint (a digit) and a memory constraint
// are refused: PTX operands are registers, and an output that is also an
// input is two registers here.
func checkConstraint(c ir.Constraint, t ir.RegType, out bool) error {
	s := c.String()
	if out && len(s) > 0 && s[0] == '=' {
		s = s[1:]
	}
	switch s {
	case "reg":
		return nil
	case "r":
		if t == ir.TypeI32 {
			return nil
		}
	case "l":
		if t == ir.TypeI64 || t == ir.TypePtr {
			return nil
		}
	case "f":
		if t == ir.TypeF32 {
			return nil
		}
	case "d":
		if t == ir.TypeF64 {
			return nil
		}
	case "mem":
		return fmt.Errorf("a memory operand; PTX asm operands are registers")
	}
	return fmt.Errorf("constraint %q on a %s operand", c, t)
}
