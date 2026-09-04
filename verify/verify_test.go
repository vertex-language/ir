package verify_test

import (
	"errors"
	"strings"
	"testing"

	"github.com/vertex-language/ir"
	"github.com/vertex-language/ir/verify"
)

// The sticky builder error comes back first and alone. §19.3 is the rule
// this proves the shape of: a branch's arguments against its target's
// parameters is a deferred builder check, not one of this package's, so a
// module carrying one never reaches a §19 rule here — even though this
// function is also unsound in a way this package does check.
func TestModuleReturnsStickyErrorFirst(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	fn := m.Func("f").Export()
	a := fn.ParamI32("a")
	fn.ReturnsI32()

	entry := fn.Entry()
	join := fn.Block("join")
	join.ParamI32("r")
	fn.Block("orphan").Return(a) // an ErrUnreachable this run must not reach

	entry.Br(join.To(a, a)) // two arguments for one parameter
	join.Return(a)

	err := verify.Module(m)
	if err == nil {
		t.Fatal("verify.Module: no error")
	}
	if !errors.Is(err, ir.ErrArity) {
		t.Fatalf("verify.Module = %v, want ir.ErrArity", err)
	}
	if errors.Is(err, verify.ErrUnreachable) {
		t.Error("verify.Module reported a §19 fault on a module with a sticky builder error")
	}
	var e *verify.Error
	if errors.As(err, &e) {
		t.Errorf("verify.Module = %T, want the builder's own *ir.Error", e)
	}
}

// Every fault, not the first: three unreachable blocks are three errors,
// and errors.Is reaches all of them through Errors.Unwrap.
func TestModuleReportsEveryFault(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	fn := m.Func("f").Export()
	a := fn.ParamI32("a")
	fn.ReturnsI32()

	fn.Entry().Return(a)
	for _, name := range []string{"x", "y", "z"} {
		fn.Block(name).Return(a)
	}

	err := verify.Module(m)
	var es verify.Errors
	if !errors.As(err, &es) {
		t.Fatalf("verify.Module = %v (%T), want verify.Errors", err, err)
	}
	if len(es) != 3 {
		t.Fatalf("verify.Module found %d faults, want 3: %v", len(es), es)
	}
	for i, want := range []string{"x", "y", "z"} {
		if es[i].Block != want {
			t.Errorf("errs[%d].Block = %q, want %q", i, es[i].Block, want)
		}
		if !errors.Is(es[i], verify.ErrUnreachable) {
			t.Errorf("errs[%d] = %v, want ErrUnreachable", i, es[i])
		}
	}
	if got := es.Error(); !strings.Contains(got, "and 2 more") {
		t.Errorf("Errors.Error() = %q, want it to say how many it is not printing", got)
	}
}

// MaxErrors stops the run, and stops it at the cap rather than somewhere
// past it.
func TestMaxErrors(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	fn := m.Func("f").Export()
	a := fn.ParamI32("a")
	fn.ReturnsI32()

	fn.Entry().Return(a)
	for _, name := range []string{"x", "y", "z"} {
		fn.Block(name).Return(a)
	}

	err := verify.Options{MaxErrors: 2}.Module(m)
	var es verify.Errors
	if !errors.As(err, &es) {
		t.Fatalf("Options.Module = %v (%T), want verify.Errors", err, err)
	}
	if len(es) != 2 {
		t.Errorf("Options.Module found %d faults, want the cap of 2", len(es))
	}
}

// Func is the same check on one function, for a pass that just rewrote it
// and does not want to pay for the rest of the module.
func TestFunc(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	sound := m.Func("sound").Export()
	a := sound.ParamI32("a")
	sound.ReturnsI32()
	sound.Entry().Return(a)

	broken := m.Func("broken").Export()
	b := broken.ParamI32("b")
	broken.ReturnsI32()
	broken.Entry().Return(b)
	broken.Block("orphan").Return(b)

	if err := verify.Func(sound); err != nil {
		t.Errorf("verify.Func(sound) = %v, want nil", err)
	}
	err := verify.Func(broken)
	if !errors.Is(err, verify.ErrUnreachable) {
		t.Fatalf("verify.Func(broken) = %v, want ErrUnreachable", err)
	}
	var e *verify.Error
	if errors.As(err, &e); e.Func != "broken" {
		t.Errorf("Func = %q, want %q", e.Func, "broken")
	}
}

// The position an Error prints is the position it carries: function,
// block, instruction index, mnemonic, then the sentinel.
func TestErrorText(t *testing.T) {
	e := &verify.Error{
		Func:   "f",
		Block:  "join",
		Inst:   2,
		Op:     ir.Op{Type: ir.TypeI32, Verb: ir.VAdd},
		Detail: "%x is defined in @then",
		Err:    verify.ErrDominance,
	}
	want := "verify: @f @join #2: i32.add: definition does not dominate use: %x is defined in @then"
	if got := e.Error(); got != want {
		t.Errorf("Error() = %q, want %q", got, want)
	}
}
