package arm64

import (
	"fmt"
	"strings"

	arm64asm "github.com/vertex-language/arm64"
	"github.com/vertex-language/arm64/asm"
	"github.com/vertex-language/arm64/reg"

	"github.com/vertex-language/ir"
	"github.com/vertex-language/ir/lower/asmtmpl"
	"github.com/vertex-language/ir/lower/mir"
	"github.com/vertex-language/ir/lower/regalloc"
)

// §G4, inline assembly.
//
// The template is not lowered. It is checked here, carried through register
// allocation as an ordinary instruction with ordinary defs and uses, expanded
// once the allocator has said where each operand went, and handed to the
// assembler in the sibling arm64 module — the same assembler and the same ISA
// table every other instruction in this file reaches through its typed helper.
//
// What makes that work with no new machinery in mir or regalloc is that every
// GCC constraint feature turns out to be something the allocator already does:
//
//   - A register class is the operand's VIR type, which vregs already
//     classifies. An f64 is in the vector file whether its constraint letter
//     says `w` or not.
//   - A clobber is a vreg pinned to the clobbered register and listed among
//     the instruction's Defs. Interference then puts an edge from it to
//     everything live across the instruction, which is what a clobber means.
//     usedCalleeSaved reads the same assignment map, so clobbering x19 saves
//     x19 in the prologue without anything here saying so.
//   - Early clobber is already the default: mir's interference gives every
//     def an edge to every use of its own instruction unless the instruction
//     is a copy. So `"=&r"` is correct and `"=r"` is conservative, costing a
//     register and never correctness.
//   - A tied operand is one vreg appearing in both Uses and Defs. The
//     interference builder drops the self-edge, liveness already covers both
//     roles, and the two share a register by construction rather than by a
//     coalescing pass. The input is copied into it first, so a value that
//     outlives the asm is not overwritten.
//
// The one thing genuinely absent is a fixed-register constraint, and AArch64
// does not have one: GCC reaches a named register here through a register asm
// variable, which is a frontend construct with no place in this IR.

// asmOperand is one constrained operand: the register standing for it, the
// width it is read at, and whether the template wants it spelled as an
// address.
type asmOperand struct {
	vreg mir.VReg
	w    width
	mem  bool
}

// iselAsm lowers §G4's asm and asm goto.
// iselAsm lowers one asm or asm goto. outs is nil for the instruction
// form, whose outputs are its own results; for the terminator form it is the
// fallthrough target's trailing parameters, which is where §14 puts them.
func iselAsm(fn *ir.Func, c *cursor, vr *vregs, in *ir.Inst, id int, outs []mir.VReg) error {
	a := in.Asm()
	if a == nil {
		return fmt.Errorf("asm: no template")
	}

	labels, emitted := asmLabels(fn, in)
	ops := make([]asmOperand, 0, len(a.Outs)+len(a.Args))

	// One vreg per physical register this instruction names. Only the
	// clobber list can name one here — AArch64 has no fixed-register
	// constraint — but `"x9", "w9"` is one register written twice, and two
	// vregs pinned to it are two live values in one place, which the
	// allocator refuses. The amd64 backend needs the same map for a larger
	// reason: there, a syscall's "=a" and "a" are the same register.
	site := map[int]mir.VReg{}
	pinInt := func(p reg.X) mir.VReg {
		key := int(p)
		if v, ok := site[key]; ok {
			return v
		}
		v := vr.physical(p, w64)
		site[key] = v
		return v
	}
	pinVec := func(p reg.V) mir.VReg {
		key := 1000 + int(p)
		if v, ok := site[key]; ok {
			return v
		}
		v := vr.physicalVec(p, wf64)
		site[key] = v
		return v
	}

	// Outputs first, which is the numbering GCC's templates use.
	for i, o := range a.Outs {
		w, ok := widthOf(o.Type)
		if !ok {
			return fmt.Errorf("asm: output %d is %s, which does not fit a register", i, o.Type)
		}
		v, err := asmOutVReg(vr, in, outs, i)
		if err != nil {
			return fmt.Errorf("asm: %w", err)
		}
		ops = append(ops, asmOperand{vreg: v, w: w, mem: isMemConstraint(o.Constraint)})
	}

	// Then the inputs, in declaration order.
	for i, arg := range a.Args {
		src, ok := vr.lookup(arg.Def)
		if !ok {
			return fmt.Errorf("asm: input %d defined outside the function", i)
		}
		if tie, tied := tiedOutput(arg.Constraint, len(a.Outs)); tied {
			// One vreg in both roles. The copy is what makes it safe: the
			// instruction writes this register, and the value that came in
			// may be live afterwards.
			out := ops[tie]
			emitCopy(c, out.vreg, src, out.w)
			ops = append(ops, out)
			continue
		}
		ops = append(ops, asmOperand{
			vreg: src, w: vr.widthOfVReg(src), mem: isMemConstraint(arg.Constraint),
		})
	}

	refs, err := asmtmpl.Parse(a.Template, len(ops), labels)
	if err != nil {
		return fmt.Errorf("asm: %w", err)
	}

	// Defs are the outputs and the clobbers; Uses are the inputs. A tied
	// operand is in both, and mir drops the self-edge.
	defs := make([]mir.VReg, 0, len(a.Outs)+len(a.Clobbers))
	for i := range a.Outs {
		defs = append(defs, ops[i].vreg)
	}
	uses := make([]mir.VReg, 0, len(a.Args))
	for i := range a.Args {
		uses = append(uses, ops[len(a.Outs)+i].vreg)
	}

	seen := map[mir.VReg]bool{}
	for _, v := range defs {
		seen[v] = true
	}
	for _, name := range a.Clobbers {
		v, err := clobberVReg(pinInt, pinVec, name)
		if err != nil {
			return fmt.Errorf("asm: %w", err)
		}
		if v >= 0 && !seen[v] {
			seen[v] = true
			defs = append(defs, v)
		}
	}

	c.Emit(mir.Instr{
		Op: asmOp{
			template: a.Template, refs: refs, ops: ops,
			labels: labels, emitted: emitted, id: id,
		},
		Defs: defs,
		Uses: uses,
	})
	return nil
}

// iselAsmGoto lowers §G4's terminator form.
//
// It is the asm instruction followed by a branch to the fallthrough target.
// The labels the template names are not targets of anything this package
// emits — the assembled text branches to them itself — so all this has to do
// is make sure they are reachable in the MIR CFG, or liveness would conclude
// that everything the labelled blocks read is dead.
func iselAsmGoto(fn *ir.Func, mf *mir.Func, c *cursor, vr *vregs, term *ir.Inst) error {
	targets := term.Targets()
	if len(targets) != 1 {
		return fmt.Errorf("asm goto: %d fallthrough targets, want one", len(targets))
	}

	// The outputs are the fallthrough target's trailing parameters (§14), so
	// the asm writes straight into the vregs those parameters already have.
	// Nothing copies them anywhere: the value is defined here and live in the
	// one block that declares it, which is the whole point of binding them to
	// an edge rather than to a register of the terminator's own.
	outs, err := asmGotoOutVRegs(vr, term, targets[0])
	if err != nil {
		return err
	}
	if err := iselAsm(fn, c, vr, term, vr.nextAsmID(), outs); err != nil {
		return err
	}

	// Every label is a successor. They take no parameters — §14 says so —
	// so there are no edge copies to make for them.
	for _, b := range term.Labels() {
		mf.Succ(c.blk, blockLabel(fn, b))
	}

	moves, err := edgeCopiesTrailing(vr, targets[0], len(outs))
	if err != nil {
		return err
	}
	emitParallelCopy(c, vr, moves)

	label := blockLabel(fn, targets[0].Block())
	c.Emit(mir.Instr{Op: bOp{target: label}})
	mf.Succ(c.blk, label)
	return nil
}

// asmLabels is an asm goto's block labels: the names a template may use, and
// what each one is called in the emitted text.
//
// The template names them by label rather than by number, so the slice's order
// matters only to the diagnostic that lists them.
func asmLabels(fn *ir.Func, in *ir.Inst) ([]string, map[string]string) {
	blocks := in.Labels()
	if len(blocks) == 0 {
		return nil, nil
	}
	names := make([]string, len(blocks))
	emitted := make(map[string]string, len(blocks))
	for i, b := range blocks {
		names[i] = b.Label()
		emitted[b.Label()] = blockLabel(fn, b)
	}
	return names, emitted
}

// isMemConstraint reports whether the operand is an address the template will
// dereference: GCC's `m`, and AArch64's `Q`, which is the same thing narrowed
// to a base register with no offset.
func isMemConstraint(c ir.Constraint) bool {
	s := strings.TrimLeft(c.String(), "=+&%")
	return s == "mem" || s == "m" || s == "Q"
}

// tiedOutput reads a matching constraint: a run of digits naming an output
// this input must share a register with. `"+r"` on an output is the other
// spelling of the same thing and is not handled here, because this IR gives an
// output and an input separate entries and `+` would have to name which.
//
// The digits are read by ir.Constraint.Tied, so that this package, the other
// two backends and the verifier all agree on what a tie is rather than each
// deciding again. An index past the end of the output list is a §8b fault the
// verifier reports, and the bound below is kept for a caller that skipped it:
// falling back to an ordinary operand is the conservative half of a wrong
// answer, and a diagnostic is the right one.
func tiedOutput(c ir.Constraint, nouts int) (int, bool) {
	n, ok := c.Tied()
	if !ok || n >= nouts {
		return 0, false
	}
	return n, true
}

// clobberVReg is the vreg standing for one clobbered register, or -1 for the
// two pseudo-registers.
func clobberVReg(pinInt func(reg.X) mir.VReg, pinVec func(reg.V) mir.VReg,
	name string) (mir.VReg, error) {
	switch name {
	case "memory":
		// A barrier for a scheduler this package does not have. Recording it
		// is what makes it a no-op rather than a thing that was dropped:
		// instructions are emitted in the order isel produced them and
		// nothing reorders across anything, so the guarantee already holds.
		return -1, nil
	case "cc":
		// NZCV is not a vreg here, and does not need to be. A comparison and
		// its consumer are emitted as one adjacent pair by iselCompare, and
		// brif re-tests a value rather than reading flags a predecessor set,
		// so no flag value is ever live across another instruction for an
		// asm to destroy. A peephole that fused a compare into a branch
		// would end that, and would have to model flags to do it.
		return -1, nil
	}

	r, ok := reg.Lookup(name)
	if !ok {
		return -1, fmt.Errorf("clobber %q is not a register on this target", name)
	}
	switch v := r.(type) {
	case reg.X:
		return pinInt(v), nil
	case reg.W:
		// The 32-bit view of a register is the register. Clobbering w9 makes
		// x9 unusable across the instruction, not the low half of it.
		return pinInt(reg.X(v)), nil
	case reg.Xsp:
		return -1, fmt.Errorf("clobber %q: the stack pointer is not one this package can spare", name)
	case reg.V:
		return pinVec(v), nil
	case reg.Q:
		return pinVec(reg.V(v)), nil
	case reg.D:
		return pinVec(reg.V(v)), nil
	case reg.S:
		return pinVec(reg.V(v)), nil
	case reg.H:
		return pinVec(reg.V(v)), nil
	case reg.B:
		return pinVec(reg.V(v)), nil
	}
	return -1, fmt.Errorf("clobber %q is a register this package cannot reserve", name)
}

// emitAsm expands one template and assembles it into the current section.
func emitAsm(text *arm64asm.Section, fn *ir.Func, op asmOp,
	assigned map[mir.VReg]regalloc.PhysReg) error {

	expanded, err := asmtmpl.Expand(op.template, op.refs, func(r asmtmpl.Ref) (string, error) {
		if r.IsLabel() {
			return op.emitted[r.Label], nil
		}
		o := op.ops[r.Operand]
		name, err := asmRegName(o, assigned[o.vreg], r.Modifier)
		if err != nil {
			return "", err
		}
		if o.mem {
			return "[" + name + "]", nil
		}
		return name, nil
	})
	if err != nil {
		return fmt.Errorf("%s: asm: %w", fn.Name(), err)
	}

	// Each expansion gets its own prefix, because two expansions of one
	// template in one function both contain the same `1:` and they are
	// different labels.
	return asm.AssembleFragment(text, expanded, asm.Options{
		File:        "<asm in " + fn.Name() + ">",
		LabelPrefix: fmt.Sprintf(".Lasm.%s.%d", fn.Name(), op.id),
	})
}

// asmRegName spells one allocated register the way the template's modifier
// asks for it.
//
// The default follows the operand's own width, which is what makes an
// unmodified `%0` right for an i32 and for an i64 without the template saying
// which. `%w0` and `%x0` override it, and are how one template serves both —
// the reason GCC has the modifiers at all.
func asmRegName(o asmOperand, p regalloc.PhysReg, mod byte) (string, error) {
	n := int(p)
	switch mod {
	case 0:
		switch o.w {
		case w32:
			return reg.W(n).String(), nil
		case w64:
			return reg.X(n).String(), nil
		case wf32:
			return reg.S(n).String(), nil
		default:
			return reg.D(n).String(), nil
		}
	case 'w':
		return reg.W(n).String(), nil
	case 'x':
		return reg.X(n).String(), nil
	case 'b':
		return reg.B(n).String(), nil
	case 'h':
		return reg.H(n).String(), nil
	case 's':
		return reg.S(n).String(), nil
	case 'd':
		return reg.D(n).String(), nil
	case 'q':
		return reg.Q(n).String(), nil
	}
	return "", fmt.Errorf("%%%c is not an operand modifier this target has; "+
		"it takes w, x, b, h, s, d and q", mod)
}

// §3b and §7's other body: assembly with no operands.
//
// Neither goes near mir, regalloc, or asmtmpl. There is nothing to allocate
// and nothing to substitute — the text is handed to the assembler as written,
// which is what makes these the two forms that needed no backend machinery
// beyond a place to put the bytes.

// emitModuleAsm assembles one module-level block where it was declared. The
// index is only there to keep two blocks' numeric local labels apart.
func emitModuleAsm(text *arm64asm.Section, a *ir.ModuleAsm, idx int) error {
	return asm.AssembleFragment(text, a.Text(), asm.Options{
		File:        fmt.Sprintf("<module asm %d>", idx),
		LabelPrefix: fmt.Sprintf(".Lmasm.%d", idx),
	})
}

// emitAsmBody assembles a naked function's body, under the function's own
// symbol.
//
// The symbol is defined and sized here exactly as a lowered function's is; the
// difference is only that the bytes between the two came from the assembler
// rather than from isel. No prologue and no epilogue, which is what naked
// means — a body that returns does so because its text says so.
func emitAsmBody(text *arm64asm.Section, fn *ir.Func, body string) error {
	text.Label(fn.Name(), funcBinding(fn), arm64asm.Func)
	if err := asm.AssembleFragment(text, body, asm.Options{
		File:        "<asm body of " + fn.Name() + ">",
		LabelPrefix: ".Lasmbody." + fn.Name(),
	}); err != nil {
		return err
	}
	text.EndLabel(fn.Name())
	return nil
}

// asmOutVReg is the vreg one asm output is written into.
//
// The instruction form defines a register of its own; the terminator form
// does not, and outs holds the fallthrough target's trailing parameters
// instead. Both arrive here so the operand walk above does not have to know
// which form it is lowering.
func asmOutVReg(vr *vregs, in *ir.Inst, outs []mir.VReg, i int) (mir.VReg, error) {
	if outs != nil {
		if i >= len(outs) {
			return 0, fmt.Errorf("output %d has no parameter on the fallthrough edge", i)
		}
		return outs[i], nil
	}
	return vr.define(in.Result(i))
}

// asmGotoOutVRegs is the vreg of each of the fallthrough target's trailing
// parameters — §14's binding, read back.
func asmGotoOutVRegs(vr *vregs, term *ir.Inst, fall ir.BlockTarget) ([]mir.VReg, error) {
	a := term.Asm()
	if a == nil || len(a.Outs) == 0 {
		return nil, nil
	}
	blk := fall.Block()
	params := blk.Params()
	lead := len(fall.Args())
	if len(params) != lead+len(a.Outs) {
		return nil, fmt.Errorf("asm goto: @%s takes %d parameters, the edge supplies %d arguments and the statement %d outputs",
			blk.Label(), len(params), lead, len(a.Outs))
	}
	out := make([]mir.VReg, len(a.Outs))
	for i := range a.Outs {
		v, ok := vr.lookup(params[lead+i])
		if !ok {
			return nil, fmt.Errorf("asm goto: @%s parameter %d has no vreg", blk.Label(), lead+i)
		}
		out[i] = v
	}
	return out, nil
}
