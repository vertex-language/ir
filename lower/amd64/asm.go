package amd64

import (
	"fmt"
	"strings"

	amd64asm "github.com/vertex-language/amd64"
	"github.com/vertex-language/amd64/asm"
	"github.com/vertex-language/amd64/reg"

	"github.com/vertex-language/ir"
	"github.com/vertex-language/ir/lower/asmtmpl"
	"github.com/vertex-language/ir/lower/mir"
	"github.com/vertex-language/ir/lower/regalloc"
)

// §G4, inline assembly. The shape is lower/arm64's, and the reasoning for it
// is written there; what follows is only what x86-64 does differently.
//
// It has fixed-register constraints, and AArch64 does not. `"a"` means RAX and
// `"D"` means RDI, and a Linux syscall written in C names four of them. They
// are handled the way a call's argument registers are: a fresh vreg pinned to
// the register, with the value copied into it, rather than pinning the value's
// own vreg — which would constrain it everywhere it is live rather than at the
// one instruction that cares.
//
// Its registers have four spellings rather than two. `%0` is `%rax` for a
// 64-bit operand and `%eax` for a 32-bit one, and `%b0`, `%w0`, `%k0` and
// `%q0` name the byte, word, doubleword and quadword views explicitly. GCC's
// substitution includes the sigil, so the expansion is `%rax` and not `rax`.

// fixedRegs are the constraint letters that name one register.
//
// GCC's x86 letters, the six that matter: the four accumulators plus the two
// string-instruction pointers. The rest of the letter set constrains a class
// or an immediate's range, which the operand's own type already settles here.
var fixedRegs = map[string]reg.R64{
	"a": reg.RAX, "b": reg.RBX, "c": reg.RCX, "d": reg.RDX,
	"S": reg.RSI, "D": reg.RDI,
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
func iselAsm(fn *ir.Func, c *cursor, vr *vregs, in *ir.Inst, id int, outs []mir.VReg) error {
	a := in.Asm()
	if a == nil {
		return fmt.Errorf("asm: no template")
	}

	labels, emitted := asmLabels(fn, in)
	ops := make([]asmOperand, 0, len(a.Outs)+len(a.Args))

	// One vreg per physical register named by this instruction, whether the
	// name came from a constraint or from the clobber list.
	//
	// A syscall is the case that forces it: "=a" for the result and "a" for
	// the number are the same register, and two vregs pinned to it are two
	// live values in one place, which the allocator refuses — correctly.
	// They are one value, read and then written, and one vreg is how that is
	// spelled. It is the same rule a call site follows for the registers its
	// arguments and results share.
	site := map[reg.R64]mir.VReg{}
	pin := func(p reg.R64, w width) mir.VReg {
		if v, ok := site[p]; ok {
			return v
		}
		v := vr.physical(p, w)
		site[p] = v
		return v
	}

	// Outputs first, which is the numbering GCC's templates use. A fixed
	// register becomes a pinned vreg the instruction defines, copied out
	// afterwards into the register the result actually lives in.
	var copyOut []struct {
		dst, src mir.VReg
		w        width
	}
	for i, o := range a.Outs {
		w, ok := widthOf(o.Type)
		if !ok {
			return fmt.Errorf("asm: output %d is %s, which does not fit a register", i, o.Type)
		}
		dst, err := asmOutVReg(vr, in, outs, i)
		if err != nil {
			return fmt.Errorf("asm: %w", err)
		}
		v := dst
		if p, fixed := fixedRegOf(o.Constraint); fixed {
			v = pin(p, w)
			copyOut = append(copyOut, struct {
				dst, src mir.VReg
				w        width
			}{dst, v, w})
		}
		ops = append(ops, asmOperand{vreg: v, w: w, mem: isMemConstraint(o.Constraint)})
	}

	for i, arg := range a.Args {
		src, ok := vr.lookup(arg.Def)
		if !ok {
			return fmt.Errorf("asm: input %d defined outside the function", i)
		}
		if tie, tied := tiedOutput(arg.Constraint, len(a.Outs)); tied {
			out := ops[tie]
			emitCopy(c, out.vreg, src, out.w)
			ops = append(ops, out)
			continue
		}
		w := vr.widthOfVReg(src)
		if p, fixed := fixedRegOf(arg.Constraint); fixed {
			pinned := pin(p, w)
			emitCopy(c, pinned, src, w)
			ops = append(ops, asmOperand{vreg: pinned, w: w, mem: isMemConstraint(arg.Constraint)})
			continue
		}
		ops = append(ops, asmOperand{vreg: src, w: w, mem: isMemConstraint(arg.Constraint)})
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
		// A register named by a constraint and again in the clobber list is
		// one register, already among the defs.
		if v >= 0 && !seen[v] {
			seen[v] = true
			defs = append(defs, v)
		}
	}

	c.Emit(mir.Instr{
		Op: asmOp{
			template: a.Template, refs: refs, ops: ops,
			labels: labels, emitted: emitted, id: id,
			nouts: len(a.Outs),
		},
		Defs: defs,
		Uses: uses,
	})

	for _, cp := range copyOut {
		emitCopy(c, cp.dst, cp.src, cp.w)
	}
	return nil
}

// iselAsmGoto lowers §G4's terminator form: the asm instruction, then a
// branch to the fallthrough target.
//
// The labels the template names are not targets of anything this package
// emits — the assembled text branches to them itself — so all this does is
// make them successors in the MIR CFG, without which liveness would conclude
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
	for _, b := range term.Labels() {
		mf.Succ(c.blk, blockLabel(fn, b))
	}

	moves, err := edgeCopiesTrailing(vr, targets[0], len(outs))
	if err != nil {
		return err
	}
	emitParallelCopy(c, moves)
	dst := blockLabel(fn, targets[0].Block())
	c.Emit(mir.Instr{Op: jmpOp{target: dst}})
	mf.Succ(c.blk, dst)
	return nil
}

// asmLabels is an asm goto's block labels: the names a template may use, and
// what each is called in the emitted text.
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

// trimModifiers drops the leading characters that say how an operand is used
// rather than where it lives.
func trimModifiers(c ir.Constraint) string {
	return strings.TrimLeft(c.String(), "=+&%")
}

func isMemConstraint(c ir.Constraint) bool {
	s := trimModifiers(c)
	return s == "mem" || s == "m" || s == "o" || s == "V"
}

// fixedRegOf reads a constraint that names one register.
func fixedRegOf(c ir.Constraint) (reg.R64, bool) {
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
func clobberVReg(vr *vregs, pin func(reg.R64, width) mir.VReg, name string) (mir.VReg, error) {
	switch name {
	case "memory":
		// A barrier for a scheduler this package does not have.
		return -1, nil
	case "cc":
		// EFLAGS is not a vreg here, and does not need to be. A compare is
		// emitted adjacent to whatever consumes it — iselBlock skips a
		// fused compare in the ordinary walk and re-emits it at the branch
		// or select that reads it — so no flag value is ever live across
		// another instruction for an asm to destroy. A pass that moved a
		// compare away from its consumer would end that.
		return -1, nil
	}

	r, ok := reg.Lookup(strings.ToLower(strings.TrimPrefix(name, "%")))
	if !ok {
		return -1, fmt.Errorf("clobber %q is not a register on this target", name)
	}
	switch v := r.(type) {
	case reg.R64:
		return pin(v, w64), nil
	case reg.R32:
		// The narrow views of a register are the register: clobbering eax
		// makes rax unusable across the instruction, not its low half.
		return pin(reg.R64(v), w64), nil
	case reg.R16:
		return pin(reg.R64(v), w64), nil
	case reg.R8:
		if v >= reg.AH {
			return -1, fmt.Errorf("clobber %q: a high-byte register is not one this package allocates", name)
		}
		return pin(reg.R64(v), w64), nil
	case reg.Xmm:
		return vr.physicalXmm(v, wf64), nil
	}
	return -1, fmt.Errorf("clobber %q is a register this package cannot reserve", name)
}

// emitAsm expands one template and assembles it into the current section.
func emitAsm(text *amd64asm.Section, fn *ir.Func, in mir.Instr, op asmOp,
	assigned map[mir.VReg]regalloc.PhysReg) error {

	// Which vreg each operand ended up in, read off the instruction rather
	// than off the op. See asmOp.nouts.
	vregOf := func(i int) (mir.VReg, bool) {
		if i < op.nouts {
			if i < len(in.Defs) {
				return in.Defs[i], true
			}
			return 0, false
		}
		if i-op.nouts < len(in.Uses) {
			return in.Uses[i-op.nouts], true
		}
		return 0, false
	}

	expanded, err := asmtmpl.Expand(op.template, op.refs, func(r asmtmpl.Ref) (string, error) {
		if r.IsLabel() {
			return op.emitted[r.Label], nil
		}
		o := op.ops[r.Operand]
		v, ok := vregOf(r.Operand)
		if !ok {
			return "", fmt.Errorf("operand %%%d names no register", r.Operand)
		}
		name, err := asmRegName(o, assigned[v], r.Modifier)
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
// including the `%` — GCC's substitution puts the sigil in, so a template
// writes `%0` and gets `%rax`.
func asmRegName(o asmOperand, p regalloc.PhysReg, mod byte) (string, error) {
	n := int(p)
	switch mod {
	case 0:
		switch o.w {
		case w32:
			return "%" + reg.R32(n).String(), nil
		case w64:
			return "%" + reg.R64(n).String(), nil
		case wf32, wf64, wv128:
			return "%" + reg.Xmm(n).String(), nil
		}
	case 'b':
		return "%" + reg.R8(n).String(), nil
	case 'w':
		return "%" + reg.R16(n).String(), nil
	case 'k':
		return "%" + reg.R32(n).String(), nil
	case 'q':
		return "%" + reg.R64(n).String(), nil
	case 'h':
		return "", fmt.Errorf("%%h names a high-byte register, which this package does not allocate")
	}
	return "", fmt.Errorf("%%%c is not an operand modifier this target has; it takes b, w, k and q", mod)
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
func emitModuleAsm(text *amd64asm.Section, a *ir.ModuleAsm, idx int) error {
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
func emitAsmBody(text *amd64asm.Section, fn *ir.Func, body string) error {
	text.Label(fn.Name(), funcBinding(fn), amd64asm.Func)
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
