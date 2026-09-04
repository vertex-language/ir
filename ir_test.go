package ir_test

// The builder's own tests: the sticky error, which is the discipline
// every other package in this repo relies on, and walk.go's traversal,
// which is what ir/verify reads a function's shape through.
//
// This is an external test package on purpose. Everything here is
// reachable through the exported surface or it is not reachable at all —
// the same standard ir/verify is held to, and for the same reason: a
// rule that needs an unexported field is a missing method.

import (
	"errors"
	"testing"

	"github.com/vertex-language/ir"
)

// —— the sticky error ——

// A module with nothing wrong has no error, which is worth stating
// before every test that a particular thing sets one.
func TestModuleErrIsNilWhenNothingFailed(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	fn := m.Func("f").Export()
	a := fn.ParamI32("a")
	fn.ReturnsI32()
	fn.Entry().Return(fn.Entry().I32.Add(a, a))

	if err := m.Err(); err != nil {
		t.Errorf("Err = %v, want nil", err)
	}
}

// The builder is first-wins: after a failure every later call is a
// no-op, so a second fault cannot overwrite the first. That is what
// makes the reported error the one that actually happened rather than
// the last consequence of it.
func TestStickyErrorIsFirstWins(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	fn := m.Func("f").Export()
	fn.ReturnsI32()

	entry := fn.Entry()
	entry.Return(entry.I32.Const(0))
	entry.Return(entry.I32.Const(1)) // into an already-terminated block

	first := m.Err()
	if !errors.Is(first, ir.ErrTerminated) {
		t.Fatalf("Err = %v, want ErrTerminated", first)
	}

	// A different fault afterwards changes nothing.
	entry.Ptr.Alloc(8, 3) // 3 is not a power of two
	if got := m.Err(); got != first {
		t.Errorf("Err = %v after a second fault, want the first one: %v", got, first)
	}
}

// Every sentinel the builder can raise, reached through the call that
// raises it. A sentinel nothing reaches is a sentinel that has stopped
// describing anything.
func TestBuilderSentinels(t *testing.T) {
	for _, tc := range []struct {
		name string
		want error
		emit func(m *ir.Module)
	}{
		{"terminated", ir.ErrTerminated, func(m *ir.Module) {
			fn := m.Func("f").Export()
			fn.ReturnsI32()
			e := fn.Entry()
			e.Return(e.I32.Const(0))
			e.Return(e.I32.Const(1))
		}},
		{"align", ir.ErrAlign, func(m *ir.Module) {
			fn := m.Func("f").Export()
			fn.ReturnsI32()
			e := fn.Entry()
			e.Ptr.Alloc(8, 3)
			e.Return(e.I32.Const(0))
		}},
		{"align exceeds width", ir.ErrAlign, func(m *ir.Module) {
			fn := m.Func("f").Export()
			p := fn.ParamPtr("p")
			fn.ReturnsI32()
			e := fn.Entry()
			e.Return(e.I32.Load(p, ir.Align(8))) // an i32 access is four bytes
		}},
		{"placement", ir.ErrPlacement, func(m *ir.Module) {
			fn := m.Func("f").Export()
			fn.ReturnsI32()
			e := fn.Entry()
			later := fn.Block("later")
			e.Br(later.To())
			later.Ptr.Alloc(8, 8) // §19.6: the entry block only
			later.Return(later.I32.Const(0))
		}},
		{"duplicate", ir.ErrDuplicate, func(m *ir.Module) {
			m.Global("g", ir.RW, ir.StoreI32.FType()).Export()
			m.Global("g", ir.RW, ir.StoreI32.FType()).Export()
		}},
		{"poison", ir.ErrPoison, func(m *ir.Module) {
			fn := m.Func("f").Export()
			fn.ReturnsI32()
			var zero ir.I32 // never defined
			fn.Entry().Return(zero)
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

// §19.3 is the builder's, but deferred: a forward branch may name a
// block whose parameters are not declared yet, so the check cannot run
// at the call and runs when Err is asked instead.
func TestDeferredBranchArgumentCheck(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	fn := m.Func("f").Export()
	a := fn.ParamI32("a")
	fn.ReturnsI32()

	entry := fn.Entry()
	join := fn.Block("join")
	entry.Br(join.To(a, a)) // two arguments, declared below as one parameter
	r := join.ParamI32("r")
	join.Return(r)

	if err := m.Err(); !errors.Is(err, ir.ErrArity) {
		t.Errorf("Err = %v, want ErrArity", err)
	}
}

// The same shape with the arity right is accepted, which is the half
// that makes the check worth deferring rather than refusing forward
// branches outright.
func TestForwardBranchToLaterParameters(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	fn := m.Func("f").Export()
	a := fn.ParamI32("a")
	fn.ReturnsI32()

	entry := fn.Entry()
	join := fn.Block("join")
	entry.Br(join.To(a))
	r := join.ParamI32("r")
	join.Return(r)

	if err := m.Err(); err != nil {
		t.Errorf("Err = %v, want nil", err)
	}
}

// —— traversal ——

// diamond builds entry -> {then, else} -> join, which has every shape
// the CFG walkers are asked about: a fork, a merge, and a block reached
// two ways.
func diamond(t *testing.T) (*ir.Func, map[string]*ir.Block) {
	t.Helper()

	m := ir.NewModule("t", ir.X86_64Linux)
	fn := m.Func("f").Export()
	a := fn.ParamI32("a")
	fn.ReturnsI32()

	entry := fn.Entry()
	thenB := fn.Block("then")
	elseB := fn.Block("else")
	join := fn.Block("join")

	entry.BrIf(entry.I32.SLt(a, a), thenB.To(), elseB.To())
	thenB.Br(join.To())
	elseB.Br(join.To())
	join.Return(a)

	if err := m.Err(); err != nil {
		t.Fatalf("building: %v", err)
	}
	return fn, map[string]*ir.Block{
		"entry": entry, "then": thenB, "else": elseB, "join": join,
	}
}

func labels(bs []*ir.Block) []string {
	out := make([]string, len(bs))
	for i, b := range bs {
		out[i] = b.Label()
	}
	return out
}

func TestSuccs(t *testing.T) {
	_, b := diamond(t)

	if got := labels(b["entry"].Succs()); len(got) != 2 {
		t.Errorf("entry.Succs() = %v, want two", got)
	}
	if got := labels(b["join"].Succs()); len(got) != 0 {
		t.Errorf("join.Succs() = %v, want none — it returns", got)
	}
}

func TestPreds(t *testing.T) {
	_, b := diamond(t)

	if got := b["entry"].Preds(); len(got) != 0 {
		t.Errorf("entry.Preds() = %v, want none", labels(got))
	}
	if got := b["join"].Preds(); len(got) != 2 {
		t.Errorf("join.Preds() = %v, want two", labels(got))
	}
	if got := b["then"].Preds(); len(got) != 1 || got[0] != b["entry"] {
		t.Errorf("then.Preds() = %v, want [entry]", labels(got))
	}
}

// RPO visits a block only after some path from the entry has reached it,
// which is the property every dominance computation is built on.
func TestRPO(t *testing.T) {
	fn, b := diamond(t)

	order := fn.RPO()
	if len(order) != 4 {
		t.Fatalf("RPO visited %d blocks, want 4: %v", len(order), labels(order))
	}
	if order[0] != b["entry"] {
		t.Errorf("RPO starts at %s, want entry", order[0].Label())
	}
	pos := map[*ir.Block]int{}
	for i, blk := range order {
		pos[blk] = i
	}
	if pos[b["join"]] < pos[b["then"]] || pos[b["join"]] < pos[b["else"]] {
		t.Errorf("RPO put join before an arm: %v", labels(order))
	}
}

// An unreachable block is not in RPO — which is how ir/verify finds one
// without a second walk.
func TestRPOSkipsUnreachable(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	fn := m.Func("f").Export()
	a := fn.ParamI32("a")
	fn.ReturnsI32()

	entry := fn.Entry()
	orphan := fn.Block("orphan")
	entry.Return(a)
	orphan.Return(a)

	for _, b := range fn.RPO() {
		if b == orphan {
			t.Error("RPO reached a block no edge enters")
		}
	}
	if len(fn.Blocks()) != 2 {
		t.Errorf("Blocks() = %v, want both — an unreachable block is still declared", labels(fn.Blocks()))
	}
}

// Uses and ReplaceUses are the rewriting surface a pass works through.
func TestReplaceUses(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	fn := m.Func("f").Export()
	a := fn.ParamI32("a")
	b := fn.ParamI32("b")
	fn.ReturnsI32()

	entry := fn.Entry()
	sum := entry.I32.Add(a, a)
	entry.Return(sum)

	if got := len(fn.Uses(a.Def())); got != 2 {
		t.Errorf("a has %d uses, want 2", got)
	}
	if n := fn.ReplaceUses(a.Def(), b.Def()); n != 2 {
		t.Errorf("ReplaceUses replaced %d, want 2", n)
	}
	if got := len(fn.Uses(a.Def())); got != 0 {
		t.Errorf("a has %d uses after replacement, want 0", got)
	}
	if got := len(fn.Uses(b.Def())); got != 2 {
		t.Errorf("b has %d uses after replacement, want 2", got)
	}
}

// WalkUses reaches every operand slot, including a branch's arguments —
// which is the one place a use is not an instruction operand, and the
// one ir/verify's dominance check would otherwise miss.
func TestWalkUsesReachesBranchArguments(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	fn := m.Func("f").Export()
	a := fn.ParamI32("a")
	fn.ReturnsI32()

	entry := fn.Entry()
	join := fn.Block("join")
	r := join.ParamI32("r")
	entry.Br(join.To(a))
	join.Return(r)

	var seen int
	fn.WalkUses(func(u ir.Use) bool {
		if u.Def() == a.Def() {
			seen++
		}
		return true
	})
	if seen != 1 {
		t.Errorf("WalkUses saw a used %d times, want 1 — the branch argument", seen)
	}
}
