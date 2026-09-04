package verify_test

// One case per §19 rule this package implements, each expected to fail
// with a named sentinel at a named position — and one that is expected to
// pass, since a verifier that rejects everything is also a verifier that
// finds every fault.
//
// These are Go functions rather than a testdata directory of .vir files
// because there is nothing to load one with: ir/text is a printer and this
// repo has no reader, deliberately. A module is built by calling methods,
// here as everywhere else.

import (
	"errors"
	"testing"

	"github.com/vertex-language/ir"
	"github.com/vertex-language/ir/verify"
)

// wantFault checks that err is one Error naming sentinel, in block, at
// instruction idx, and returns it for a caller with more to ask.
func wantFault(t *testing.T, err error, sentinel error, block string, idx int) *verify.Error {
	t.Helper()

	if err == nil {
		t.Fatalf("verify: no error, want %v", sentinel)
	}
	if !errors.Is(err, sentinel) {
		t.Fatalf("verify: %v, want %v", err, sentinel)
	}
	var e *verify.Error
	if !errors.As(err, &e) {
		t.Fatalf("verify: %v is not a *verify.Error", err)
	}
	if e.Block != block {
		t.Errorf("Block = %q, want %q", e.Block, block)
	}
	if e.Inst != idx {
		t.Errorf("Inst = %d, want %d", e.Inst, idx)
	}
	return e
}

// sum1n is the loop from ir/lower/amd64's own tests: block parameters, a
// back edge, and a value defined in one block and used in another. Every
// rule here has to accept it.
func sum1n() *ir.Module {
	m := ir.NewModule("t", ir.X86_64Linux)
	fn := m.Func("sum1n").Export()
	n := fn.ParamI32("n")
	fn.ReturnsI32()

	entry := fn.Entry()
	loop := fn.Block("loop")
	i := loop.ParamI32("i")
	acc := loop.ParamI32("acc")
	body := fn.Block("body")
	exit := fn.Block("exit")

	entry.Br(loop.To(entry.I32.Const(1), entry.I32.Const(0)))
	loop.BrIf(loop.I32.SLt(n, i), exit.To(), body.To())
	body.Br(loop.To(body.I32.Add(i, body.I32.Const(1)), body.I32.Add(acc, i)))
	exit.Return(acc)

	return m
}

func TestModuleAcceptsSoundFunction(t *testing.T) {
	if err := verify.Module(sum1n()); err != nil {
		t.Fatalf("verify.Module: %v", err)
	}
}

// §19.2, the degenerate case: a definition with no body at all.
func TestNoBody(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	m.Func("f").Export().ReturnsI32()

	e := wantFault(t, verify.Module(m), verify.ErrNoBody, "", -1)
	if e.Func != "f" {
		t.Errorf("Func = %q, want %q", e.Func, "f")
	}
}

// §19.19: a naked function's body is assembly. Blocks for a body would be
// asking a backend to lower instructions into a function with no frame to
// lower them against.
func TestNakedWithBlocks(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	fn := m.Func("f").Export().Naked().ReturnsI32()
	e := fn.Entry()
	e.Return(e.I32.Const(0))

	fault := wantFault(t, verify.Module(m), verify.ErrNakedBody, "", -1)
	if fault.Func != "f" {
		t.Errorf("Func = %q, want %q", fault.Func, "f")
	}
}

// The same rule from the other side: naked and nothing at all. It reports
// ErrNakedBody rather than ErrNoBody, whose advice — that this should have
// been an import — is the wrong advice here.
func TestNakedWithNothing(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	m.Func("f").Export().Naked().ReturnsI32()

	wantFault(t, verify.Module(m), verify.ErrNakedBody, "", -1)
}

// A function whose body is assembly has no blocks, no registers and no CFG,
// so every rule in this package is about something that is not there.
func TestAsmBodyVerifies(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	fn := m.Func("f").Export().ReturnsI32()
	fn.AsmBody("movl $0, %eax\n\tret")

	if err := verify.Module(m); err != nil {
		t.Fatalf("verify.Module: %v", err)
	}
}

// §19.2: every block ends in exactly one terminator.
func TestMissingTerminator(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	fn := m.Func("f").Export()
	a := fn.ParamI32("a")
	fn.ReturnsI32()

	entry := fn.Entry()
	entry.I32.Add(a, a) // and then nothing

	wantFault(t, verify.Module(m), verify.ErrTerminator, "entry", -1)
}

// §19.2: only the entry block may be unreachable-from-nothing.
func TestUnreachableBlock(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	fn := m.Func("f").Export()
	a := fn.ParamI32("a")
	fn.ReturnsI32()

	entry := fn.Entry()
	orphan := fn.Block("orphan")

	entry.Return(a)
	orphan.Return(a) // terminated, well typed, and reached by nothing

	wantFault(t, verify.Module(m), verify.ErrUnreachable, "orphan", -1)
}

// §19.17: the entry block is not the target of any branch. The fault is
// reported against the block that branches, not against the entry block,
// since that is the one an author has to edit.
func TestEntryIsBranchTarget(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	fn := m.Func("f").Export()
	a := fn.ParamI32("a")
	fn.ReturnsI32()

	entry := fn.Entry()
	again := fn.Block("again")

	entry.BrIf(entry.I32.SLt(a, a), again.To(), again.To())
	again.Br(entry.To())

	e := wantFault(t, verify.Module(m), verify.ErrEntryTarget, "again", 0)
	if e.Op.Verb != ir.VBr {
		t.Errorf("Op = %s, want br", e.Op)
	}
}

// §19.17 again: ptr.blockaddr makes no edge, so nothing about the CFG
// reveals this one.
func TestEntryBlockAddr(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	fn := m.Func("f").Export()
	a := fn.ParamI32("a")
	fn.ReturnsI32()

	entry := fn.Entry()
	entry.Ptr.BlockAddr(entry)
	entry.Return(a)

	e := wantFault(t, verify.Module(m), verify.ErrEntryTarget, "entry", 0)
	if e.Op != (ir.Op{Type: ir.TypePtr, Verb: ir.VBlockAddr}) {
		t.Errorf("Op = %s, want ptr.blockaddr", e.Op)
	}
}

// §19.1: a definition dominates every use. @then defines x on one of the
// two paths into @join, so a use of x in @join is undominated — the
// canonical fault block parameters exist to make unnecessary.
func TestDominance(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	fn := m.Func("f").Export()
	a := fn.ParamI32("a")
	fn.ReturnsI32()

	entry := fn.Entry()
	then := fn.Block("then")
	join := fn.Block("join")

	entry.BrIf(entry.I32.SLt(a, a), then.To(), join.To())
	x := then.I32.Add(a, a)
	then.Br(join.To())
	join.Return(x)

	e := wantFault(t, verify.Module(m), verify.ErrDominance, "join", 0)
	if e.Op.Verb != ir.VReturn {
		t.Errorf("Op = %s, want return", e.Op)
	}
}

// A definition in a block that *does* dominate the use is fine, including
// across a back edge — the same shape as the fault above with the branch
// the other way round.
func TestDominanceAcceptsDominatingDefinition(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	fn := m.Func("f").Export()
	a := fn.ParamI32("a")
	fn.ReturnsI32()

	entry := fn.Entry()
	next := fn.Block("next")

	x := entry.I32.Add(a, a)
	entry.Br(next.To())
	next.Return(x)

	if err := verify.Module(m); err != nil {
		t.Fatalf("verify.Module: %v", err)
	}
}
