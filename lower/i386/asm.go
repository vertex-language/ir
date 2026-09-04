package i386

import (
	"fmt"
	"strings"

	i386asm "github.com/vertex-language/i386"
	"github.com/vertex-language/i386/asm"
	"github.com/vertex-language/i386/reg"

	"github.com/vertex-language/ir"
	"github.com/vertex-language/ir/lower/asmtmpl"
	"github.com/vertex-language/ir/lower/mir"
	"github.com/vertex-language/ir/lower/regalloc"
)

// §G4, inline assembly. The shape is lower/arm64's and lower/amd64's; the
// reasoning for it is written in the first of those.
//
// One thing here is refused outright, and it is worth saying why rather than
// letting it half-work.
//
// An i64 is a *pair* of registers on this target — that is the structural fact
// this whole package is arranged around — and a template referring to `%0`
// where operand zero is an i64 is naming two registers with one reference. GCC
// on this architecture does not solve that either: it offers `"A"` for the
// EDX:EAX pair specifically, and otherwise expects the frontend to have split
// the value into two 32-bit operands. Substituting the low half and hoping
// would produce an instruction that assembles, runs, and is wrong about the
// top thirty-two bits, which is the worst of the three outcomes. So a 64-bit
// operand is refused by name, and a frontend that wants one splits it.

// fixedRegs are the constraint letters that name one register.
var fixedRegs = map[string]reg.R32{
	"a": reg.EAX, "b": reg.EBX, "c": reg.ECX, "d": reg.EDX,
	"S": reg.ESI, "D": reg.EDI,
}

// asmOperand is one constrained operand.
type asmOperand struct {
	vreg mir.VReg
	w    width
	mem  bool
}

// iselAsm lowers §G4's asm and asm goto.
// iselAsm lowers one asm or asm goto. outs is nil for the instruction
// form, whose outputs are its own results; for the terminator form it is the
// fallthrough target's trailing parameters, which is where §14 puts them.
func iselAsm(fn *ir.Func, c *cursor, vr *vregs, in *ir.Inst, id int, outs []value) error {
	a := in.Asm()
	if a == nil {
		return fmt.Errorf("asm: no template")
	}

	labels, emitted := asmLabels(fn, in)
	ops := make([]asmOperand, 0, len(a.Outs)+len(a.Args))

	// One vreg per physical register this instruction names, whether from a
	// constraint or from the clobber list: "=a" and "a" are the same
	// register, and two vregs pinned to it are two live values in one place.
	site := map[reg.R32]mir.VReg{}
	pin := func(p reg.R32) mir.VReg {
		if v, ok := site[p]; ok {
			return v
		}
		v := vr.physical(p)
		site[p] = v
		return v
	}

	var copyOut []struct{ dst, src mir.VReg }
	for i, o := range a.Outs {
		w, ok := widthOf(o.Type)
		if !ok {
			return fmt.Errorf("asm: output %d is %s, which does not fit a register", i, o.Type)
		}
		if w.pairs() {
			return fmt.Errorf("asm: output %d is %s, which is a register pair on this "+
				"target; one %%-reference cannot name two registers, so split it into "+
				"two i32 operands", i, o.Type)
		}
		dst, err := asmOutVReg(vr, in, outs, i)
		if err != nil {
			return fmt.Errorf("asm: %w", err)
		}
		v := dst.lo
		if p, fixed := fixedRegOf(o.Constraint); fixed {
			v = pin(p)
			copyOut = append(copyOut, struct{ dst, src mir.VReg }{dst.lo, v})
		}
		ops = append(ops, asmOperand{vreg: v, w: w, mem: isMemConstraint(o.Constraint)})
	}

	for i, arg := range a.Args {
		src, ok := vr.lookup(arg.Def)
		if !ok {
			return fmt.Errorf("asm: input %d defined outside the function", i)
		}
		if src.w.pairs() {
			return fmt.Errorf("asm: input %d is a register pair on this target; "+
				"one %%-reference cannot name two registers, so split it into two "+
				"i32 operands", i)
		}
		if tie, tied := tiedOutput(arg.Constraint, len(a.Outs)); tied {
			out := ops[tie]
			emitCopy(c, out.vreg, src.lo)
			ops = append(ops, out)
			continue
		}
		if p, fixed := fixedRegOf(arg.Constraint); fixed {
			pinned := pin(p)
			emitCopy(c, pinned, src.lo)
			ops = append(ops, asmOperand{vreg: pinned, w: src.w, mem: isMemConstraint(arg.Constraint)})
			continue
		}
		ops = append(ops, asmOperand{vreg: src.lo, w: src.w, mem: isMemConstraint(arg.Constraint)})
	}

	refs, err := asmtmpl.Parse(a.Template, len(ops), labels)
	if err != nil {
		return fmt.Errorf("asm: %w", err)
	}

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
		v, err := clobberVReg(vr, pin, name)
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

	for _, cp := range copyOut {
		emitCopy(c, cp.dst, cp.src)
	}
	return nil
}

// iselAsmGoto lowers §G4's terminator form.
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
	for _, b := range term.Labels() {
		mf.Succ(c.blk, blockLabel(fn, b))
	}

	moves, err := edgeCopiesTrailing(vr, targets[0], len(outs))
	if err != nil {
		return err
	}
	emitParallelCopy(c, vr, moves)
	dst := blockLabel(fn, targets[0].Block())
	c.Emit(mir.Instr{Op: jmpOp{target: dst}})
	mf.Succ(c.blk, dst)
	return nil
}

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

func trimModifiers(c ir.Constraint) string {
	return strings.TrimLeft(c.String(), "=+&%")
}

func isMemConstraint(c ir.Constraint) bool {
	s := trimModifiers(c)
	return s == "mem" || s == "m" || s == "o" || s == "V"
}

func fixedRegOf(c ir.Constraint) (reg.R32, bool) {
	p, ok := fixedRegs[trimModifiers(c)]
	return p, ok
}

// tiedOutput reads a matching constraint: digits naming an output this input
// must share a register with.
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
func clobberVReg(vr *vregs, pin func(reg.R32) mir.VReg, name string) (mir.VReg, error) {
	switch name {
	case "memory":
		return -1, nil
	case "cc":
		// EFLAGS is not a vreg here. A compare is emitted adjacent to
		// whatever consumes it, so no flag value is ever live across
		// another instruction for an asm to destroy.
		return -1, nil
	}

	r, ok := reg.Lookup(strings.ToLower(strings.TrimPrefix(name, "%")))
	if !ok {
		return -1, fmt.Errorf("clobber %q is not a register on this target", name)
	}
	switch v := r.(type) {
	case reg.R32:
		return pin(v), nil
	case reg.R16:
		// The narrow views of a register are the register.
		return pin(reg.R32(v)), nil
	case reg.R8:
		return pin(reg.R32(v)), nil
	case reg.Xmm:
		return vr.physicalVec(v), nil
	}
	return -1, fmt.Errorf("clobber %q is a register this package cannot reserve", name)
}

// emitAsm expands one template and assembles it into the current section.
func emitAsm(text *i386asm.Section, fn *ir.Func, op asmOp,
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
			return "(" + name + ")", nil
		}
		return name, nil
	})
	if err != nil {
		return fmt.Errorf("%s: asm: %w", fn.Name(), err)
	}

	return asm.AssembleFragment(text, expanded, asm.Options{
		File:        "<asm in " + fn.Name() + ">",
		LabelPrefix: fmt.Sprintf(".Lasm.%s.%d", fn.Name(), op.id),
	})
}

// asmRegName spells one allocated register the way the template asks for it,
// including the `%` sigil, which GCC's substitution puts in.
func asmRegName(o asmOperand, p regalloc.PhysReg, mod byte) (string, error) {
	n := int(p)
	switch mod {
	case 0:
		switch o.w {
		case w32:
			return "%" + reg.R32(n).Name(), nil
		case wf32, wf64:
			return "%" + reg.Xmm(n).Name(), nil
		}
		return "", fmt.Errorf("a register pair has no single spelling")
	case 'b':
		if n >= 4 {
			return "", fmt.Errorf("%%b names the low byte of %s, which this "+
				"architecture cannot address", reg.R32(n).Name())
		}
		return "%" + reg.R8(n).Name(), nil
	case 'w':
		return "%" + reg.R16(n).Name(), nil
	case 'k':
		return "%" + reg.R32(n).Name(), nil
	}
	return "", fmt.Errorf("%%%c is not an operand modifier this target has; it takes b, w and k", mod)
}

// §3b and §7's other body: assembly with no operands.
//
// Neither goes near mir, regalloc, or asmtmpl. There is nothing to allocate
// and nothing to substitute — the text is handed to the assembler as written,
// which is why these two forms needed no backend machinery beyond a place to
// put the bytes. The shape is lower/arm64's; see there.

// emitModuleAsm assembles one module-level block where it was declared. Each
// block is its own assembly, and the index only keeps two blocks' numeric
// local labels apart.
func emitModuleAsm(text *i386asm.Section, a *ir.ModuleAsm, idx int) error {
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
func emitAsmBody(text *i386asm.Section, fn *ir.Func, body string) error {
	text.Label(fn.Name(), funcBinding(fn), i386asm.Func)
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
func asmOutVReg(vr *vregs, in *ir.Inst, outs []value, i int) (value, error) {
	if outs != nil {
		if i >= len(outs) {
			return value{}, fmt.Errorf("output %d has no parameter on the fallthrough edge", i)
		}
		return outs[i], nil
	}
	return vr.define(in.Result(i))
}

// asmGotoOutVRegs is the vreg of each of the fallthrough target's trailing
// parameters — §14's binding, read back.
func asmGotoOutVRegs(vr *vregs, term *ir.Inst, fall ir.BlockTarget) ([]value, error) {
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
	out := make([]value, len(a.Outs))
	for i := range a.Outs {
		v, ok := vr.lookup(params[lead+i])
		if !ok {
			return nil, fmt.Errorf("asm goto: @%s parameter %d has no vreg", blk.Label(), lead+i)
		}
		out[i] = v
	}
	return out, nil
}
