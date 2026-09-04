package ir

// asm is the one instruction with no fixed operand shape and no semantics this
// IR can state. It exists because glibc, musl, and every kernel are unbuildable
// without it, and it is spelled as an escape hatch rather than folded into the
// general rule.

import "strings"

// A Constraint is an operand constraint. The string form is the escape hatch for
// target-specific constraint letters and tied operands.
type Constraint struct{ s string }

var (
	CReg = Constraint{"reg"}
	CMem = Constraint{"mem"}
	CImm = Constraint{"imm"}
)

// CStr is a target-specific constraint string.
func CStr(s string) Constraint { return Constraint{s} }

func (c Constraint) String() string { return c.s }

// IsKeyword reports whether the constraint is one of reg, mem, imm rather than a
// target-specific string.
func (c Constraint) IsKeyword() bool {
	return c.s == "reg" || c.s == "mem" || c.s == "imm"
}

// maxTie bounds the index a matching constraint can name, so that a string of
// digits no operand list could match saturates instead of wrapping. The value
// is out of range for every real instruction, which is the answer wanted: the
// verifier rejects it as naming an output that does not exist rather than as a
// number it could not read.
const maxTie = 1 << 20

// Tied reads a matching constraint: GCC's run of digits naming the output an
// input shares a register with. "0" and "+&0" are ties to output 0; "r", "=r"
// and "" are not ties at all.
//
// The modifiers come off first because they say how an operand is used and not
// which one it is. Whether the index names an output that exists is a question
// about the operand lists rather than about this string, so §8b makes it the
// verifier's — which is what lets every backend read a tie the same way without
// each one re-deciding what a tie is.
func (c Constraint) Tied() (int, bool) {
	s := strings.TrimLeft(c.s, "=+&%")
	if s == "" {
		return 0, false
	}
	n := 0
	for i := 0; i < len(s); i++ {
		if s[i] < '0' || s[i] > '9' {
			return 0, false
		}
		if n = n*10 + int(s[i]-'0'); n > maxTie {
			n = maxTie
		}
	}
	return n, true
}

// An AsmOut is one constrained output register.
type AsmOut struct {
	Type       RegType
	Constraint Constraint
}

// An AsmArg is one constrained input register.
type AsmArg struct {
	Def        *Def
	Constraint Constraint
}

// An Asm is the payload of an asm or asm goto instruction.
type Asm struct {
	// Template is the assembly text, in GCC's spelling: %0, %1, … number the
	// outputs and then the inputs, %% is a literal %, %l[label] names an asm
	// goto label, and a letter before the digits (%w0) is a target-specific
	// modifier. GCC's spelling rather than a new one because the assembly
	// this instruction exists to carry is already written in it — see §8b.
	//
	// Nothing here parses it. The backend that assembles it does, which is
	// where a reference to an operand that does not exist is caught.
	Template string
	Volatile bool
	Outs     []AsmOut
	Args     []AsmArg
	Clobbers []string
}

// An AsmStmt builds an asm instruction. Emit finishes it.
type AsmStmt struct {
	b   *Builder
	a   Asm
	bad bool
}

// Asm begins an inline-assembly instruction.
func (b *Builder) Asm(template string) *AsmStmt {
	return &AsmStmt{b: b, a: Asm{Template: template}}
}

// Volatile marks the statement volatile.
func (s *AsmStmt) Volatile() *AsmStmt { s.a.Volatile = true; return s }

// Out declares a constrained output.
func (s *AsmStmt) Out(t RegType, c Constraint) *AsmStmt {
	s.a.Outs = append(s.a.Outs, AsmOut{Type: t, Constraint: c})
	return s
}

// In declares a constrained input.
func (s *AsmStmt) In(v Value, c Constraint) *AsmStmt {
	d := defOf(v)
	if d == nil {
		s.b.fail(Op{TypeNone, VAsm}, ErrPoison, "asm input is a zero Value")
		s.bad = true
		return s
	}
	s.a.Args = append(s.a.Args, AsmArg{Def: d, Constraint: c})
	return s
}

// Clobber lists clobbered registers. The two pseudo-registers "memory" and "cc"
// are admitted alongside target register names.
func (s *AsmStmt) Clobber(names ...string) *AsmStmt {
	s.a.Clobbers = append(s.a.Clobbers, names...)
	return s
}

// Emit finishes the statement and returns its outputs.
func (s *AsmStmt) Emit() Results {
	op := Op{TypeNone, VAsm}
	if s.bad {
		return Results{}
	}
	res := make([]RegType, len(s.a.Outs))
	for i, o := range s.a.Outs {
		res[i] = o.Type
	}
	args := make([]*Def, len(s.a.Args))
	for i, a := range s.a.Args {
		args[i] = a.Def
	}
	asm := s.a
	in := s.b.emit(op, res, args, &imm{asm: &asm})
	return s.b.results(op, in)
}

// An AsmGotoStmt builds an asm goto terminator. To finishes it.
type AsmGotoStmt struct {
	b   *Builder
	a   Asm
	bad bool
}

// AsmGoto begins asm goto, the terminator form. It is implicitly volatile.
//
// Its outputs are the trailing parameters of the fallthrough target, exactly
// as an invoke's results are the trailing parameters of its normal target,
// and for the same reason: a register the instruction defined would have to
// dominate every edge, and on an edge the assembled text branched along, the
// text did not reach the end that writes the output. The fallthrough edge is
// the one on which it did — which is what GCC means when it says an asm
// goto's outputs are valid on the fallthrough path.
//
// So Out declares a type and a constraint here, and does not return a value:
// the value is the target block's parameter, live where it is defined and
// nowhere else. §19.16's arity rule covers it the way it covers invoke.
func (b *Builder) AsmGoto(template string) *AsmGotoStmt {
	return &AsmGotoStmt{b: b, a: Asm{Template: template, Volatile: true}}
}

// Out declares a constrained output.
//
// It yields no Value. The output arrives as a parameter of the fallthrough
// target, after the arguments that target's edge carries, and the block
// declares one parameter for each — which is the same shape Invoke uses and
// the same one §19.16 checks.
func (s *AsmGotoStmt) Out(t RegType, c Constraint) *AsmGotoStmt {
	s.a.Outs = append(s.a.Outs, AsmOut{Type: t, Constraint: c})
	return s
}

// In declares a constrained input.
func (s *AsmGotoStmt) In(v Value, c Constraint) *AsmGotoStmt {
	d := defOf(v)
	if d == nil {
		s.b.fail(Op{TypeNone, VAsmGoto}, ErrPoison, "asm goto input is a zero Value")
		s.bad = true
		return s
	}
	s.a.Args = append(s.a.Args, AsmArg{Def: d, Constraint: c})
	return s
}

// Clobber lists clobbered registers.
func (s *AsmGotoStmt) Clobber(names ...string) *AsmGotoStmt {
	s.a.Clobbers = append(s.a.Clobbers, names...)
	return s
}

// To finishes the statement with its fallthrough target and label list.
//
// The labels take no parameters — §14 says so, and there is nothing to
// supply them with: an edge the assembled text branches along carries no
// argument list this IR could write. The fallthrough target is the exception
// and the reason Out exists: it takes the edge's arguments and then one
// parameter per declared output.
func (s *AsmGotoStmt) To(fallthru BlockTarget, labels ...*Block) {
	op := Op{TypeNone, VAsmGoto}
	if s.bad || !s.b.checkTargetPlus(op, fallthru, len(s.a.Outs)) {
		return
	}
	for _, l := range labels {
		if l == nil {
			s.b.fail(op, ErrPoison, "nil label")
			return
		}
		if l.fn != s.b.blk.fn {
			panic("ir: asm goto label @" + l.label + " from another function")
		}
	}
	args := make([]*Def, len(s.a.Args))
	for i, a := range s.a.Args {
		args[i] = a.Def
	}
	asm := s.a
	s.b.emit(op, nil, args, &imm{
		asm: &asm, targets: []BlockTarget{fallthru}, labels: labels,
	})
}

// —— assembly that is not an instruction ——
//
// The two forms with no operands, no allocation, and nothing for isel to
// select: a module-scope block of assembly, and the body of a naked function.
// Both are text handed to the assembler at a point in the output; the only
// difference between them is that the second defines a symbol first.

// A ModuleAsm is a module-scope block of assembly (§3).
//
// It declares no symbol here. Whatever it defines, it defines to the linker
// and not to this module — a name it emits is not in the module-scope value
// namespace, cannot be called, and cannot be the target of an alias. That is
// the whole of what "module-level asm" means, and the reason this item has no
// Name: giving it one would be claiming knowledge of the text.
type ModuleAsm struct{ text string }

func (a *ModuleAsm) ItemKind() ItemKind { return ItemAsm }

// Text is the assembly, verbatim.
func (a *ModuleAsm) Text() string { return a.text }

// Asm appends a module-scope block of assembly, in declaration order with the
// rest of the module's items. Nothing here parses it; the backend does.
func (m *Module) Asm(text string) *ModuleAsm {
	a := &ModuleAsm{text: text}
	m.items = append(m.items, a)
	return a
}

// Asms returns the module-scope assembly blocks, in declaration order.
//
// Each block is its own assembly: a section it opens it closes, and nothing a
// block sets up — the section stack, the numeric local labels, the absolute
// symbols — reaches the next one. Order between two of them decides where the
// second's bytes fall relative to the first's and nothing else.
func (m *Module) Asms() []*ModuleAsm {
	var out []*ModuleAsm
	for _, it := range m.items {
		if a, ok := it.(*ModuleAsm); ok {
			out = append(out, a)
		}
	}
	return out
}

// AsmBody gives a function a body of assembly instead of blocks, and marks it
// naked.
//
// Naked implies it rather than requiring it to be stated first, because the
// two are the same fact: a function whose body is assembly has no prologue to
// emit, no epilogue to reach, and no frame for a lowering to lay out — there
// is nothing else "naked" could mean here. GCC's rule is the same one from the
// other side, where a naked function's body may contain only asm statements.
//
// The template is not a template. It takes no operands, so it has no %0 to
// substitute and no numbering to get wrong: a %% in it is whatever the
// assembler makes of it, and the parameters this function's signature declares
// are reachable only where the calling convention put them, which is what a
// naked function is written to know.
func (f *Func) AsmBody(text string) *Func {
	if f.m.err != nil {
		return f
	}
	if len(f.blocks) > 0 {
		f.m.fail(f.name, "", Op{}, ErrPlacement, "asm body in a function with blocks")
		return f
	}
	if f.hasAsmBody {
		f.m.fail(f.name, "", Op{}, ErrDuplicate, "a second asm body")
		return f
	}
	f.asmBody = text
	f.hasAsmBody = true
	f.naked = true
	f.frozen = true
	return f
}

// AsmBodyText reports the function's assembly body and whether it has one.
// Named -Text and not AsmBody for the reason SectionAttr is: Go allows one
// method name per receiver, not one per direction.
func (f *Func) AsmBodyText() (string, bool) { return f.asmBody, f.hasAsmBody }
