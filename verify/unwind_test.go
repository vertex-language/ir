package verify_test

// §19.4, §19.5, and §19.16: one case per rule, plus the shape they all
// have to accept — an invoke whose normal edge carries a value, whose
// results arrive as the target's trailing parameters, and whose pad
// resumes.

import (
	"errors"
	"testing"

	"github.com/vertex-language/ir"
	"github.com/vertex-language/ir/verify"
)

// unwinder builds a module with a personality and a callee to invoke, and
// hands back the function under construction.
func unwinder(t *testing.T, withPersonality bool) (*ir.Module, *ir.Func, ir.Callee) {
	t.Helper()

	m := ir.NewModule("t", ir.X86_64Linux)
	pers := m.ImportFunc("__gxx_personality_v0", ir.NewSig())
	callee := m.ImportFunc("mayThrow", ir.NewSig().Param(ir.TypeI32).Ret(ir.TypeI32))

	fn := m.Func("f").Export()
	if withPersonality {
		fn.Personality(pers)
	}
	return m, fn, callee
}

func TestUnwindAccepts(t *testing.T) {
	m, fn, callee := unwinder(t, true)
	a := fn.ParamI32("a")
	fn.ReturnsI32()

	entry := fn.Entry()
	cont := fn.Block("cont")
	x := cont.ParamI32("x") // the edge's own argument
	r := cont.ParamI32("r") // the call's result
	pad := fn.Pad("pad", ir.Cleanup)

	entry.Invoke(callee, []ir.Value{a}, cont.To(a), pad)
	cont.Return(cont.I32.Add(x, r))
	pad.Resume(pad.Exn())

	if err := verify.Module(m); err != nil {
		t.Fatalf("verify.Module: %v", err)
	}
}

// §19.4: a pad block is reached only by unwind edges. The branch here is
// well formed as a branch — it supplies a ptr and an i32 for the pad's
// two parameters, so the builder has nothing to say about it — and is
// still a value the personality routine was supposed to hand over.
func TestPadReachedByBranch(t *testing.T) {
	m, fn, _ := unwinder(t, true)
	a := fn.ParamI32("a")
	fn.ReturnsI32()

	entry := fn.Entry()
	pad := fn.Pad("pad", ir.Cleanup)

	entry.Br(pad.To(entry.Ptr.Const(), entry.I32.Const(0)))
	pad.Return(a)

	e := wantFault(t, verify.Module(m), verify.ErrPadEdge, "entry", 2)
	if e.Op.Verb != ir.VBr {
		t.Errorf("Op = %s, want br", e.Op)
	}
}

// §19.4: a function that unwinds declares a personality. The fault is at
// the invoke, which is the instruction with nothing to run.
func TestInvokeWithoutPersonality(t *testing.T) {
	m, fn, callee := unwinder(t, false)
	a := fn.ParamI32("a")
	fn.ReturnsI32()

	entry := fn.Entry()
	cont := fn.Block("cont")
	cont.ParamI32("r")
	pad := fn.Pad("pad", ir.Cleanup)

	entry.Invoke(callee, []ir.Value{a}, cont.To(), pad)
	cont.Return(a)
	pad.Resume(pad.Exn())

	e := wantFault(t, verify.Module(m), verify.ErrPersonality, "entry", 0)
	if e.Op.Verb != ir.VInvoke {
		t.Errorf("Op = %s, want invoke", e.Op)
	}
}

// §19.5: resume takes a pad's exception object, not any other pointer.
func TestResumeOfNonPadValue(t *testing.T) {
	m, fn, _ := unwinder(t, true)
	fn.ParamI32("a")
	fn.ReturnsI32()

	entry := fn.Entry()
	entry.Resume(entry.Ptr.Const())

	wantFault(t, verify.Module(m), verify.ErrResume, "entry", 1)
}

// §19.5: the pad has to dominate the resume. §19.1 reports the same
// instruction — a pad parameter is a definition like any other, so a use
// it does not dominate is undominated — and the two say different things
// about it, so both are reported.
func TestResumeOfNonDominatingPad(t *testing.T) {
	m, fn, callee := unwinder(t, true)
	a := fn.ParamI32("a")
	fn.ReturnsI32()

	entry := fn.Entry()
	cont := fn.Block("cont")
	cont.ParamI32("r")
	pad := fn.Pad("pad", ir.Cleanup)

	entry.Invoke(callee, []ir.Value{a}, cont.To(), pad)
	cont.Resume(pad.Exn()) // @pad does not dominate @cont
	pad.Trap()

	err := verify.Module(m)
	if !errors.Is(err, verify.ErrResume) {
		t.Fatalf("verify.Module = %v, want ErrResume", err)
	}
	if !errors.Is(err, verify.ErrDominance) {
		t.Errorf("verify.Module = %v, want ErrDominance alongside it", err)
	}
	var es verify.Errors
	if !errors.As(err, &es) {
		t.Fatalf("verify.Module = %T, want verify.Errors", err)
	}
	for _, e := range es {
		if e.Block != "cont" {
			t.Errorf("fault in @%s, want @cont: %v", e.Block, e)
		}
	}
}

// §19.16: the normal target's parameter list is the edge's arguments
// followed by the callee's results. Here it is missing the result.
func TestInvokeNormalTargetArity(t *testing.T) {
	m, fn, callee := unwinder(t, true)
	a := fn.ParamI32("a")
	fn.ReturnsI32()

	entry := fn.Entry()
	cont := fn.Block("cont")
	x := cont.ParamI32("x")
	pad := fn.Pad("pad", ir.Cleanup)

	entry.Invoke(callee, []ir.Value{a}, cont.To(a), pad) // 1 argument, 1 result, 1 parameter
	cont.Return(x)
	pad.Resume(pad.Exn())

	wantFault(t, verify.Module(m), verify.ErrInvokeTarget, "entry", 0)
}

// §19.16 again: the arity is right and the trailing parameter's type is
// not the callee's result type.
func TestInvokeNormalTargetResultType(t *testing.T) {
	m, fn, callee := unwinder(t, true)
	a := fn.ParamI32("a")
	fn.ReturnsI32()

	entry := fn.Entry()
	cont := fn.Block("cont")
	x := cont.ParamI32("x")
	cont.ParamI64("r") // the callee returns i32
	pad := fn.Pad("pad", ir.Cleanup)

	entry.Invoke(callee, []ir.Value{a}, cont.To(a), pad)
	cont.Return(x)
	pad.Resume(pad.Exn())

	e := wantFault(t, verify.Module(m), verify.ErrInvokeTarget, "entry", 0)
	if e.Detail == "" {
		t.Error("Detail is empty; a type fault has to say which parameter")
	}
}

// §19.16's last clause: no other edge reaches the normal target, or its
// trailing parameters would be unset on that edge. The second edge here
// is a perfectly ordinary branch out of the pad, supplying both
// parameters — which is exactly the confusion the rule forbids, since one
// of them is supposed to be the call's result.
func TestInvokeNormalTargetSecondPredecessor(t *testing.T) {
	m, fn, callee := unwinder(t, true)
	a := fn.ParamI32("a")
	fn.ReturnsI32()

	entry := fn.Entry()
	cont := fn.Block("cont")
	x := cont.ParamI32("x")
	cont.ParamI32("r")
	pad := fn.Pad("pad", ir.Cleanup)

	entry.Invoke(callee, []ir.Value{a}, cont.To(a), pad)
	cont.Return(x)
	pad.Br(cont.To(a, a))

	wantFault(t, verify.Module(m), verify.ErrInvokeEdge, "entry", 0)
}
