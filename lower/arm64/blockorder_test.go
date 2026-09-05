package arm64_test

import (
	"testing"

	"github.com/vertex-language/ir"
	arm64lower "github.com/vertex-language/ir/lower/arm64"
	"github.com/vertex-language/ir/verify"
)

// Selection order is not the order the blocks were built in.
//
// A frontend that splits a block partway through lowering — around an
// overflow check, say — appends the halves after blocks that already
// branch to them, and the function's block list stops matching its
// graph. Nothing in the IR says the list has to match: verify walks
// reverse postorder to check dominance, and the spec asks for no
// particular order. So a backend that selects in list order will meet
// a use before its definition and has no vreg for it, which is what
// this holds it against.
func TestBlocksAreSelectedInGraphOrder(t *testing.T) {
	m := ir.NewModule("blockorder", ir.AArch64MacOS)
	fn := m.Func("pick").Export().NoUnwind()
	n := fn.ParamI64("n")
	fn.ReturnsI64()

	// Built in an order that is deliberately not the graph's: the
	// join and its edges exist before the block that computes what
	// they carry.
	entry := fn.Entry()
	join := fn.Block("join")
	yes := fn.Block("yes")
	no := fn.Block("no")
	compute := fn.Block("compute")

	answer := join.ParamI64("answer")
	join.Return(answer)

	// entry decides nothing yet; it goes to the block that computes.
	entry.Br(compute.To())

	// compute defines the value the edges carry, and is last in the
	// list while being early in the graph.
	sum := compute.I64.Add(n, compute.I64.Const(1))
	compute.BrIf(compute.I64.Eq(n, compute.I64.Const(0)), yes.To(), no.To())

	yes.Br(join.To(sum))
	no.Br(join.To(no.I64.Const(0)))

	if err := verify.Module(m); err != nil {
		t.Fatalf("the module is not valid IR: %v", err)
	}
	if _, err := arm64lower.Lower(m, arm64lower.Options{}); err != nil {
		t.Fatalf("a valid module was refused because of the order its blocks were built in: %v", err)
	}
}
