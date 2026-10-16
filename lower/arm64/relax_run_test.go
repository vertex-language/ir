package arm64_test

// Branch relaxation: conditional branches whose targets are past B.cond's
// and CBZ's ±1MB.

import (
	"fmt"
	"testing"

	"github.com/vertex-language/ir"
)

// One block of well over a megabyte, laid out between each kind of
// conditional branch this package writes and the block it names: the
// B.cond of a compare, the CBNZ of an i1 that no compare made, and a jump
// table's range check. Each has to be written as the inverse over a B, or
// the object does not assemble.
//
// The layout is what puts the big block in the way. A branch falls through
// into its else arm, a jump table ends the chain of blocks laid out
// together, and what is left goes in the order it was declared -- so big
// follows table, and neg, flagged and dflt follow big.
func TestRunFarBranches(t *testing.T) {
	const steps = 45000 // about 32 bytes each: a megabyte and a half

	m := ir.NewModule("t", ir.AArch64Linux)
	fn := m.Func("_far").Export()
	x := fn.ParamI64("x")
	flag := fn.ParamI1("flag")
	sel := fn.ParamI32("sel")
	fn.ReturnsI64()

	entry := fn.Entry()
	mid := fn.Block("mid")
	table := fn.Block("table")
	big := fn.Block("big")
	neg := fn.Block("neg")
	flagged := fn.Block("flagged")
	dflt := fn.Block("dflt")

	entry.BrIf(entry.I64.SLt(x, entry.I64.Const(0)), neg.To(), mid.To())

	t1 := mid.I64.Add(mid.I64.Mul(x, mid.I64.Const(3)), mid.I64.Const(1))
	mid.BrIf(flag, flagged.To(), table.To())

	t2 := table.I64.Xor(t1, table.I64.Const(5))
	table.BrTable(sel, []ir.BlockTarget{big.To()}, dflt.To())

	// 0x9E3779B97F4A7C15, as the int64 it is in a register.
	const k = -0x61c8864680b583eb
	// In blocks of a few hundred steps joined by jumps, which the layout
	// keeps together: one block of them all is quadratic in the peephole
	// pass, and slow for nothing this test is about.
	b, v := big, ir.I64(t2)
	for i := 0; i < steps; i++ {
		if i > 0 && i%300 == 0 {
			next := fn.Block(fmt.Sprintf("big%d", i))
			b.Br(next.To())
			b = next
		}
		v = b.I64.Xor(b.I64.Mul(v, b.I64.Const(k)), b.I64.Const(int64(i*7919+1)))
	}
	b.Return(v)

	neg.Return(neg.I64.Const(-1))
	flagged.Return(t1)
	dflt.Return(dflt.I64.Add(t2, dflt.I64.Const(7)))

	// The chain, computed here.
	chain := func(x int64) int64 {
		u := (uint64(x)*3 + 1) ^ 5
		for i := 0; i < steps; i++ {
			u = u*0x9E3779B97F4A7C15 ^ uint64(i*7919+1)
		}
		return int64(u)
	}

	got := runNative(t, m, `
#include <stdio.h>
long far(long, _Bool, int);
int main(void) {
    printf("%ld %ld %ld %ld %ld\n", far(-5, 0, 0), far(2, 1, 0), far(2, 0, 9), far(2, 0, 0), far(40, 0, 0));
    return 0;
}
`)
	want := fmt.Sprintf("-1 7 9 %d %d\n", chain(2), chain(40))
	if got != want {
		t.Errorf("printed %q, want %q", got, want)
	}
}
