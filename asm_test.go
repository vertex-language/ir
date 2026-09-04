package ir_test

// §3b's module-level asm and §7's asm body: the two forms of assembly that
// are not instructions.
//
// What is testable here is the surface and its rules. Whether the text
// assembles is the backend's question and is asked in ir/lower; what this
// file asks is whether a module can hold the two forms, whether they keep
// their order, and whether the builder refuses the shapes that have no
// meaning — a function with both kinds of body, a parameter declared after
// the body that would have used it.

import (
	"errors"
	"strings"
	"testing"

	"github.com/vertex-language/ir"
)

// A module-level block is an item like any other, in declaration order with
// the rest.
func TestModuleAsmIsAnItem(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	m.Asm("first")
	m.Global("g", ir.RW, ir.StoreI32.FType()).Export()
	m.Asm("second")

	if err := m.Err(); err != nil {
		t.Fatalf("Err = %v, want nil", err)
	}

	var kinds []ir.ItemKind
	for _, it := range m.Items() {
		kinds = append(kinds, it.ItemKind())
	}
	want := []ir.ItemKind{ir.ItemAsm, ir.ItemGlobal, ir.ItemAsm}
	if len(kinds) != len(want) {
		t.Fatalf("Items has %d entries, want %d", len(kinds), len(want))
	}
	for i := range want {
		if kinds[i] != want[i] {
			t.Errorf("item %d is kind %v, want %v", i, kinds[i], want[i])
		}
	}

	asms := m.Asms()
	if len(asms) != 2 || asms[0].Text() != "first" || asms[1].Text() != "second" {
		t.Errorf("Asms = %v, want the two blocks in declaration order", asms)
	}
}

// A module-level block declares no symbol, so a function may take the name
// its text defines without a duplicate-name failure. Whether the two names
// collide is the linker's question and not this module's.
func TestModuleAsmDeclaresNoSymbol(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	m.Asm(".globl f\nf:\n\tret")
	fn := m.Func("f").Export().ReturnsI32()
	e := fn.Entry()
	e.Return(e.I32.Const(0))

	if err := m.Err(); err != nil {
		t.Errorf("Err = %v, want nil", err)
	}
	if m.Lookup("f") != fn {
		t.Error("the module-level block took the name out of the value namespace")
	}
}

// An asm body implies naked: the two are the same fact, so stating one states
// the other.
func TestAsmBodyImpliesNaked(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	fn := m.Func("f").Export()
	fn.ParamI64("a")
	fn.ReturnsI64()
	fn.AsmBody("ret")

	if err := m.Err(); err != nil {
		t.Fatalf("Err = %v, want nil", err)
	}
	if !fn.IsNaked() {
		t.Error("IsNaked is false; a function whose body is assembly is naked")
	}
	body, ok := fn.AsmBodyText()
	if !ok || body != "ret" {
		t.Errorf("AsmBodyText = %q, %v; want %q, true", body, ok, "ret")
	}
	if len(fn.Blocks()) != 0 {
		t.Errorf("Blocks has %d entries; a body of assembly has none", len(fn.Blocks()))
	}
}

// The shapes with no meaning.
func TestAsmBodySentinels(t *testing.T) {
	for _, tc := range []struct {
		name string
		want error
		emit func(m *ir.Module)
	}{
		{"a block after the body", ir.ErrPlacement, func(m *ir.Module) {
			fn := m.Func("f").Export().ReturnsI32()
			fn.AsmBody("ret")
			fn.Entry()
		}},
		{"a body after the blocks", ir.ErrPlacement, func(m *ir.Module) {
			fn := m.Func("f").Export().ReturnsI32()
			e := fn.Entry()
			e.Return(e.I32.Const(0))
			fn.AsmBody("ret")
		}},
		{"two bodies", ir.ErrDuplicate, func(m *ir.Module) {
			fn := m.Func("f").Export().ReturnsI32()
			fn.AsmBody("ret")
			fn.AsmBody("ret")
		}},
		{"a parameter after the body", ir.ErrFrozen, func(m *ir.Module) {
			fn := m.Func("f").Export().ReturnsI32()
			fn.AsmBody("ret")
			fn.ParamI32("late")
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			m := ir.NewModule("t", ir.X86_64Linux)
			tc.emit(m)
			if err := m.Err(); !errors.Is(err, tc.want) {
				t.Errorf("Err = %v, want %v", err, tc.want)
			}
		})
	}
}

// —— §8b's instruction, and §14's terminator form ——

// The payload survives the builder intact: the template verbatim, the two
// operand lists in declaration order, the clobbers, and one result per
// declared output. Everything a backend reads it for is here, and nothing
// at this layer has looked at the template.
func TestAsmInstructionCarriesItsPayload(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	fn := m.Func("f").Export()
	a := fn.ParamI64("a")
	b := fn.ParamI64("b")
	fn.ReturnsI64()

	e := fn.Entry()
	r := e.Asm("addq %2, %0").Volatile().
		Out(ir.TypeI64, ir.CStr("=r")).
		In(a, ir.CStr("0")).
		In(b, ir.CReg).
		Clobber("cc", "memory").Emit()
	e.Return(r.I64(0))

	if err := m.Err(); err != nil {
		t.Fatalf("Err = %v, want nil", err)
	}
	if r.Len() != 1 {
		t.Fatalf("Len = %d, want one result for the one declared output", r.Len())
	}

	in := e.Insts()[0]
	if in.Op().Verb != ir.VAsm {
		t.Fatalf("Verb = %s, want asm", in.Op().Verb)
	}
	asm := in.Asm()
	if asm == nil {
		t.Fatal("Asm is nil")
	}
	if asm.Template != "addq %2, %0" {
		t.Errorf("Template = %q, want it verbatim", asm.Template)
	}
	if !asm.Volatile {
		t.Error("Volatile is false")
	}
	if len(asm.Outs) != 1 || asm.Outs[0].Type != ir.TypeI64 || asm.Outs[0].Constraint.String() != "=r" {
		t.Errorf("Outs = %+v, want one i64 constrained =r", asm.Outs)
	}
	if len(asm.Args) != 2 {
		t.Fatalf("Args has %d entries, want two", len(asm.Args))
	}
	if asm.Args[0].Def != a.Def() || asm.Args[1].Def != b.Def() {
		t.Error("the inputs are not the two parameters in declaration order")
	}
	if got := strings.Join(asm.Clobbers, ","); got != "cc,memory" {
		t.Errorf("Clobbers = %q, want the two in order", got)
	}
	if in.NumArgs() != 2 {
		t.Errorf("NumArgs = %d; the declared inputs are the instruction's operands", in.NumArgs())
	}
}

// asm goto is a terminator, implicitly volatile, with no outputs. Its
// fallthrough is a target and its labels are a separate list, which is what
// §14 gives it instead of the parameters an ordinary edge carries.
func TestAsmGotoIsATerminator(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	fn := m.Func("f").Export().NoUnwind()
	a := fn.ParamI64("a")
	fn.ReturnsI32()

	e := fn.Entry()
	fall := fn.Block("fall")
	taken := fn.Block("taken")

	e.AsmGoto("testq %0, %0\n\tjz %l[taken]").In(a, ir.CReg).Clobber("cc").
		To(fall.To(), taken)
	fall.Return(fall.I32.Const(0))
	taken.Return(taken.I32.Const(1))

	if err := m.Err(); err != nil {
		t.Fatalf("Err = %v, want nil", err)
	}

	term := e.Term()
	if term == nil || term.Op().Verb != ir.VAsmGoto {
		t.Fatalf("the entry block's terminator is %v, want asm goto", term)
	}
	if !term.Asm().Volatile {
		t.Error("Volatile is false; §14 makes asm goto implicitly volatile")
	}
	if n := len(term.Asm().Outs); n != 0 {
		t.Errorf("Outs has %d entries; asm goto has none", n)
	}
	if ts := term.Targets(); len(ts) != 1 || ts[0].Block() != fall {
		t.Errorf("Targets = %v, want the one fallthrough @fall", ts)
	}
	if ls := term.Labels(); len(ls) != 1 || ls[0] != taken {
		t.Errorf("Labels = %v, want [@taken]", ls)
	}

	// The labels are edges: a block reached only through one is reachable,
	// which is what keeps liveness from concluding that what it reads is
	// dead.
	if ps := taken.Preds(); len(ps) != 1 || ps[0] != e {
		t.Errorf("@taken's preds = %v, want [@entry]", ps)
	}
}

// asm goto's outputs are the trailing parameters of its fallthrough target,
// which is invoke's rule and for the same reason: the terminator defines no
// register, because one would have to dominate the edges the assembled text
// branches along, and on those the text never reached the end that writes it.
func TestAsmGotoOutputsBindToTheFallthrough(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	fn := m.Func("f").Export().NoUnwind()
	a := fn.ParamI64("a")
	fn.ReturnsI64()

	e := fn.Entry()
	fall := fn.Block("fall")
	carried := fall.ParamI64("carried")
	out := fall.ParamI64("out")
	taken := fn.Block("taken")

	e.AsmGoto("testq %0, %0\n\tjz %l[taken]").
		Out(ir.TypeI64, ir.CStr("=r")).
		In(a, ir.CReg).
		To(fall.To(a), taken)
	fall.Return(fall.I64.Add(carried, out))
	taken.Return(taken.I64.Const(0))

	if err := m.Err(); err != nil {
		t.Fatalf("Err = %v, want nil", err)
	}

	term := e.Term()
	if n := len(term.Results()); n != 0 {
		t.Errorf("the terminator defines %d registers; the outputs are the target's", n)
	}
	outs := term.Asm().Outs
	if len(outs) != 1 || outs[0].Type != ir.TypeI64 || outs[0].Constraint.String() != "=r" {
		t.Errorf("Outs = %+v, want one i64 constrained =r", outs)
	}
	// The edge's own argument comes first and the output after it, which is
	// what makes the two distinguishable in the block header.
	if ps := fall.Params(); len(ps) != 2 || ps[0] != carried.Def() || ps[1] != out.Def() {
		t.Errorf("@fall's parameters are not the edge's argument then the output")
	}
}

// The arity the builder refuses: a fallthrough target whose parameter list
// is not the edge's arguments followed by the outputs.
func TestAsmGotoOutputArity(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	fn := m.Func("f").Export().NoUnwind()
	e := fn.Entry()
	fall := fn.Block("fall")
	taken := fn.Block("taken")

	e.AsmGoto("nop").Out(ir.TypeI64, ir.CStr("=r")).To(fall.To(), taken)
	fall.Return()
	taken.Return()

	if err := m.Err(); !errors.Is(err, ir.ErrArity) {
		t.Errorf("Err = %v, want %v", err, ir.ErrArity)
	}
}

// A zero Value is the residue of an earlier failure, and reaches both forms
// through the same door.
func TestAsmRefusesAZeroValue(t *testing.T) {
	for _, tc := range []struct {
		name string
		emit func(e *ir.Block)
	}{
		{"asm", func(e *ir.Block) {
			e.Asm("nop").In(ir.I64{}, ir.CReg).Emit()
		}},
		{"asm goto", func(e *ir.Block) {
			e.AsmGoto("nop").In(ir.I64{}, ir.CReg).To(e.To())
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			m := ir.NewModule("t", ir.X86_64Linux)
			fn := m.Func("f").Export()
			e := fn.Entry()
			tc.emit(e)

			if err := m.Err(); !errors.Is(err, ir.ErrPoison) {
				t.Errorf("Err = %v, want %v", err, ir.ErrPoison)
			}
		})
	}
}

// A nil label is the same kind of residue, at the list §14 gives asm goto
// instead of parameters.
func TestAsmGotoRefusesANilLabel(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	fn := m.Func("f").Export()
	e := fn.Entry()
	out := fn.Block("out")

	e.AsmGoto("nop").To(out.To(), nil)
	out.Return()

	if err := m.Err(); !errors.Is(err, ir.ErrPoison) {
		t.Errorf("Err = %v, want %v", err, ir.ErrPoison)
	}
}

// A label from another function is not a fault a module can hold: there is
// no edge to record, and no later pass that could notice. It panics where
// it is written.
func TestAsmGotoPanicsOnAForeignLabel(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	other := m.Func("other").Export()
	elsewhere := other.Entry()
	elsewhere.Return()

	fn := m.Func("f").Export()
	e := fn.Entry()
	out := fn.Block("out")

	defer func() {
		if recover() == nil {
			t.Error("no panic; a label in another function has no edge to make")
		}
	}()
	e.AsmGoto("nop").To(out.To(), elsewhere)
}

// The constraint keywords, the escape hatch, and the one reading of a
// string that is not target-specific: a matching constraint's digits.
func TestConstraints(t *testing.T) {
	for _, c := range []ir.Constraint{ir.CReg, ir.CMem, ir.CImm} {
		if !c.IsKeyword() {
			t.Errorf("%s.IsKeyword is false", c)
		}
	}
	for _, s := range []string{"=r", "0", "Q", "", "regs"} {
		if ir.CStr(s).IsKeyword() {
			t.Errorf("CStr(%q).IsKeyword is true; only reg, mem and imm are keywords", s)
		}
	}
	if got := ir.CStr("=&r").String(); got != "=&r" {
		t.Errorf("String = %q, want the string back verbatim", got)
	}

	for _, tc := range []struct {
		in   string
		n    int
		tied bool
	}{
		{"0", 0, true},
		{"1", 1, true},
		{"+&0", 0, true},
		{"=2", 2, true},
		{"12", 12, true},
		{"r", 0, false},
		{"=r", 0, false},
		{"", 0, false},
		{"0r", 0, false},
		{"=+&%", 0, false},
		{"reg", 0, false},
		// A run of digits no operand list could match saturates rather
		// than wrapping, so the verifier sees an index out of range and
		// not a small number it would have accepted.
		{"99999999999999999999", 1 << 20, true},
	} {
		n, tied := ir.CStr(tc.in).Tied()
		if n != tc.n || tied != tc.tied {
			t.Errorf("CStr(%q).Tied() = %d, %v; want %d, %v", tc.in, n, tied, tc.n, tc.tied)
		}
	}
}
