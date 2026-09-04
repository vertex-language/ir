package verify_test

// §8b's constraint list. The template is not read at this layer, so every
// fault here is in the list around it — the ones no target could disagree
// about.

import (
	"errors"
	"strings"
	"testing"

	"github.com/vertex-language/ir"
	"github.com/vertex-language/ir/verify"
)

// asmFault is wantFault without the position, for the cases whose position
// is decided by how many instructions the setup needed rather than by the
// rule under test.
func asmFault(t *testing.T, err error) *verify.Error {
	t.Helper()

	if err == nil {
		t.Fatalf("verify: no error, want %v", verify.ErrAsmConstraint)
	}
	if !errors.Is(err, verify.ErrAsmConstraint) {
		t.Fatalf("verify: %v, want %v", err, verify.ErrAsmConstraint)
	}
	var e *verify.Error
	if !errors.As(err, &e) {
		t.Fatalf("verify: %v is not a *verify.Error", err)
	}
	return e
}

// The shape the rules exist to admit: an output, an input tied to it, and
// the two pseudo-register clobbers. A tie naming an output that does exist
// is what a matching constraint is for, and nothing here may refuse it.
func TestAsmWellFormedConstraints(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	fn := m.Func("f").Export()
	a := fn.ParamI64("a")
	fn.ReturnsI64()

	e := fn.Entry()
	r := e.Asm("add $1, %0").Volatile().
		Out(ir.TypeI64, ir.CStr("=r")).
		In(a, ir.CStr("0")).
		Clobber("cc", "memory").Emit()
	e.Return(r.I64(0))

	if err := verify.Module(m); err != nil {
		t.Fatalf("verify.Module: %v", err)
	}
}

// A matching constraint naming an output past the end of the list. This is
// the rule that earns the others: a backend reads the tie, finds it out of
// range, and falls back to an ordinary operand in a register of its own, so
// the template's %1 names a register the author did not mean and the object
// is wrong with nothing said. GCC refuses the same constraint.
func TestAsmTieToNonexistentOutput(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	fn := m.Func("f").Export()
	a := fn.ParamI64("a")
	fn.ReturnsI64()

	e := fn.Entry()
	r := e.Asm("mov %1, %0").Out(ir.TypeI64, ir.CStr("=r")).In(a, ir.CStr("2")).Emit()
	e.Return(r.I64(0))

	got := wantFault(t, verify.Module(m), verify.ErrAsmConstraint, "entry", 0)
	if !strings.Contains(got.Detail, "input 0") || !strings.Contains(got.Detail, "is one output") {
		t.Errorf("Detail = %q, want it to name the input and the output count", got.Detail)
	}
}

// The count the diagnostic quotes is the output list's, so that the fix —
// which output did you mean — is legible from the message.
func TestAsmTieCountsTheOutputs(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	fn := m.Func("f").Export()
	a := fn.ParamI64("a")
	fn.ReturnsI64()

	e := fn.Entry()
	r := e.Asm("nop").
		Out(ir.TypeI64, ir.CStr("=r")).
		Out(ir.TypeI64, ir.CStr("=r")).
		In(a, ir.CStr("5")).Emit()
	e.Return(r.I64(0))

	got := wantFault(t, verify.Module(m), verify.ErrAsmConstraint, "entry", 0)
	if !strings.Contains(got.Detail, "are 2 outputs") {
		t.Errorf("Detail = %q, want it to count the outputs", got.Detail)
	}
}

// The same fault with no outputs at all, which is how a hand-written "0"
// most often arrives.
func TestAsmTieWithNoOutputs(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	fn := m.Func("f").Export()
	a := fn.ParamI64("a")

	e := fn.Entry()
	e.Asm("nop").In(a, ir.CStr("0")).Emit()
	e.Return()

	got := wantFault(t, verify.Module(m), verify.ErrAsmConstraint, "entry", 0)
	if !strings.Contains(got.Detail, "are no outputs") {
		t.Errorf("Detail = %q, want it to say there are no outputs", got.Detail)
	}
}

// An output constrained imm: an immediate is a literal in the instruction
// stream, so there is nowhere for a result to be written.
func TestAsmImmediateOutput(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	fn := m.Func("f").Export()
	fn.ReturnsI64()

	e := fn.Entry()
	r := e.Asm("nop").Out(ir.TypeI64, ir.CImm).Emit()
	e.Return(r.I64(0))

	got := wantFault(t, verify.Module(m), verify.ErrAsmConstraint, "entry", 0)
	if !strings.Contains(got.Detail, "output 0") {
		t.Errorf("Detail = %q, want it to name the output", got.Detail)
	}
}

// An output whose constraint is a matching one. The numbering runs the
// other way: %0 is decided by the output list, and an entry in that list
// cannot defer to a later one.
func TestAsmMatchingConstraintOnAnOutput(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	fn := m.Func("f").Export()
	fn.ReturnsI64()

	e := fn.Entry()
	r := e.Asm("nop").Out(ir.TypeI64, ir.CStr("=r")).Out(ir.TypeI64, ir.CStr("0")).Emit()
	e.Return(r.I64(0))

	got := wantFault(t, verify.Module(m), verify.ErrAsmConstraint, "entry", 0)
	if !strings.Contains(got.Detail, "only an input may match an output") {
		t.Errorf("Detail = %q, want it to say a match is an input's to make", got.Detail)
	}
}

// The empty string names no register class, no tie and no target letter.
// Both operand lists refuse it, and so does the clobber list.
func TestAsmEmptyNames(t *testing.T) {
	for _, tc := range []struct {
		name  string
		build func(*ir.Block) ir.Results
		want  string
	}{
		{"an output", func(e *ir.Block) ir.Results {
			return e.Asm("nop").Out(ir.TypeI64, ir.CStr("")).Emit()
		}, "output 0"},
		{"an input", func(e *ir.Block) ir.Results {
			return e.Asm("nop").Out(ir.TypeI64, ir.CStr("=r")).
				In(e.I64.Const(1), ir.CStr("")).Emit()
		}, "input 0"},
		{"a clobber", func(e *ir.Block) ir.Results {
			return e.Asm("nop").Out(ir.TypeI64, ir.CStr("=r")).Clobber("").Emit()
		}, "clobber 0"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			m := ir.NewModule("t", ir.X86_64Linux)
			fn := m.Func("f").Export()
			fn.ReturnsI64()

			e := fn.Entry()
			r := tc.build(e)
			e.Return(r.I64(0))

			got := asmFault(t, verify.Module(m))
			if !strings.Contains(got.Detail, tc.want) {
				t.Errorf("Detail = %q, want it to name %s", got.Detail, tc.want)
			}
		})
	}
}

// asm goto reaches the same rules as its block's terminator, which All puts
// last and the fault's position names.
func TestAsmGotoConstraintsAreChecked(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	fn := m.Func("f").Export().NoUnwind()
	a := fn.ParamI64("a")
	fn.ReturnsI32()

	e := fn.Entry()
	fall := fn.Block("fall")
	taken := fn.Block("taken")

	e.AsmGoto("test %0, %0\n\tjz %l[taken]").In(a, ir.CStr("0")).To(fall.To(), taken)
	fall.Return(fall.I32.Const(0))
	taken.Return(taken.I32.Const(1))

	got := wantFault(t, verify.Module(m), verify.ErrAsmConstraint, "entry", 0)
	if got.Op.Verb != ir.VAsmGoto {
		t.Errorf("Op = %s, want asm goto", got.Op)
	}
	if !strings.Contains(got.Detail, "are no outputs") {
		t.Errorf("Detail = %q; asm goto has no outputs for a tie to name", got.Detail)
	}
}

// §19.16 for the terminator form: the fallthrough target's parameters are
// the edge's arguments followed by the outputs.
func TestAsmGotoTargetArity(t *testing.T) {
	// The builder's own arity check catches a target that is short, so what
	// the verifier is left to catch is a type that does not match — and a
	// second predecessor, which the builder cannot see.
	m := ir.NewModule("t", ir.X86_64Linux)
	fn := m.Func("f").Export().NoUnwind()
	e := fn.Entry()
	fall := fn.Block("fall")
	fall.ParamI32("wrong")
	taken := fn.Block("taken")

	e.AsmGoto("nop").Out(ir.TypeI64, ir.CStr("=r")).To(fall.To(), taken)
	fall.Return()
	taken.Return()

	got := wantFault(t, verify.Module(m), verify.ErrAsmGotoTarget, "entry", 0)
	if !strings.Contains(got.Detail, "output 0 is i64") {
		t.Errorf("Detail = %q, want it to name the output's type", got.Detail)
	}
}

// The outputs arrive on the fallthrough edge and on no other, so a target
// something else also branches to would read them unset.
func TestAsmGotoTargetHasOnePredecessor(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	fn := m.Func("f").Export().NoUnwind()
	a := fn.ParamI64("a")
	e := fn.Entry()
	fall := fn.Block("fall")
	out := fall.ParamI64("out")
	other := fn.Block("other")
	taken := fn.Block("taken")

	e.AsmGoto("nop").Out(ir.TypeI64, ir.CStr("=r")).To(fall.To(), taken)
	fall.Br(other.To())
	other.Br(fall.To(a)) // a second edge into @fall
	taken.Return()
	_ = out

	if err := verify.Module(m); !errors.Is(err, verify.ErrAsmGotoEdge) {
		t.Errorf("verify = %v, want %v", err, verify.ErrAsmGotoEdge)
	}
}
