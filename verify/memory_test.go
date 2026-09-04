package verify_test

// §19.6's second clause. The first clause and §19.7–9 are the builder's;
// verify/memory.go says which sentinel catches each of them and why.

import (
	"testing"

	"github.com/vertex-language/ir"
	"github.com/vertex-language/ir/verify"
)

// A block address that no brind branches to. @target is reached by an
// ordinary branch, so this is the blockaddr rule alone and not §19.2
// wearing its clothes.
func TestBlockAddrWithoutBrind(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	fn := m.Func("f").Export()
	a := fn.ParamI32("a")
	fn.ReturnsI32()

	entry := fn.Entry()
	target := fn.Block("target")

	entry.Ptr.BlockAddr(target)
	entry.Br(target.To())
	target.Return(a)

	e := wantFault(t, verify.Module(m), verify.ErrBlockAddr, "entry", 0)
	if e.Op != (ir.Op{Type: ir.TypePtr, Verb: ir.VBlockAddr}) {
		t.Errorf("Op = %s, want ptr.blockaddr", e.Op)
	}
}

// The same address, with the brind that gives it somewhere to be used.
func TestBlockAddrWithBrind(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	fn := m.Func("f").Export()
	a := fn.ParamI32("a")
	fn.ReturnsI32()

	entry := fn.Entry()
	target := fn.Block("target")

	entry.BrInd(entry.Ptr.BlockAddr(target), target)
	target.Return(a)

	if err := verify.Module(m); err != nil {
		t.Fatalf("verify.Module: %v", err)
	}
}
