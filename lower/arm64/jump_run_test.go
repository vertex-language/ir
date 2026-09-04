package arm64_test

// §G2's computed control flow: the jump table and the computed goto.

import (
	"fmt"
	"testing"

	"github.com/vertex-language/ir"
)

// A br_table over five cases, including the out-of-range ends the single
// unsigned compare has to catch at both.
func TestRunBrTable(t *testing.T) {
	m := ir.NewModule("t", ir.AArch64Linux)
	fn := m.Func("_pick").Export()
	sel := fn.ParamI32("sel")
	fn.ReturnsI64()
	entry := fn.Entry()

	out := fn.Block("out")
	r := out.ParamI64("r")
	cases := make([]ir.BlockTarget, 5)
	for i := range cases {
		b := fn.Block(fmt.Sprintf("case%d", i))
		b.Br(out.To(b.I64.Const(int64(i+1) * 11)))
		cases[i] = b.To()
	}
	dflt := fn.Block("dflt")
	dflt.Br(out.To(dflt.I64.Const(-1)))

	entry.BrTable(sel, cases, dflt.To())
	out.Return(r)

	got := runNative(t, m, `
#include <stdio.h>
long pick(int);
static int fail = 0;
static void chk(int i, long got, long want) {
    if (got != want) { printf("pick(%d): got %ld want %ld\n", i, got, want); fail = 1; }
}
int main(void) {
    for (int i = 0; i < 5; i++) chk(i, pick(i), (long)(i + 1) * 11);
    chk(5, pick(5), -1);
    chk(-1, pick(-1), -1);
    chk(1000, pick(1000), -1);
    // A negative selector read as unsigned is a very large one, which is
    // what makes the single compare enough.
    chk(-2147483648, pick(-2147483647 - 1), -1);
    printf("%s\n", fail ? "MISMATCH" : "ok");
    return 0;
}
`)
	if got != "ok\n" {
		t.Errorf("printed %q, want %q", got, "ok\n")
	}
}

// A br_table whose edges carry block arguments, which each need a block of
// their own to assign them in.
func TestRunBrTableWithArgs(t *testing.T) {
	m := ir.NewModule("t", ir.AArch64Linux)
	fn := m.Func("_pickarg").Export()
	sel := fn.ParamI32("sel")
	base := fn.ParamI64("base")
	fn.ReturnsI64()
	entry := fn.Entry()

	out := fn.Block("out")
	r := out.ParamI64("r")

	cases := []ir.BlockTarget{
		out.To(entry.I64.Add(base, entry.I64.Const(1))),
		out.To(entry.I64.Add(base, entry.I64.Const(2))),
		out.To(entry.I64.Mul(base, entry.I64.Const(10))),
	}
	entry.BrTable(sel, cases, out.To(entry.I64.Const(0)))
	out.Return(r)

	got := runNative(t, m, `
#include <stdio.h>
long pickarg(int, long);
int main(void) {
    printf("%ld %ld %ld %ld\n", pickarg(0, 5), pickarg(1, 5), pickarg(2, 5), pickarg(9, 5));
    return 0;
}
`)
	if got != "6 7 50 0\n" {
		t.Errorf("printed %q, want %q", got, "6 7 50 0\n")
	}
}

// blockaddr and brind: a computed goto through a table of block addresses
// this function built itself.
func TestRunBrInd(t *testing.T) {
	m := ir.NewModule("t", ir.AArch64Linux)
	fn := m.Func("_jgoto").Export()
	which := fn.ParamI64("which")
	fn.ReturnsI64()
	entry := fn.Entry()

	one := fn.Block("one")
	two := fn.Block("two")
	out := fn.Block("out")
	r := out.ParamI64("r")

	// The two addresses into a frame slot each, then one selected and
	// jumped to — which is a table, built the way a frontend would.
	slots := entry.Ptr.Alloc(16, 8)
	entry.Ptr.Store(entry.Ptr.BlockAddr(one), slots)
	entry.Ptr.Store(entry.Ptr.BlockAddr(two),
		entry.Ptr.Add(slots, entry.I64.Const(8)))
	target := entry.Ptr.Load(entry.Ptr.Add(slots, entry.I64.Mul(which, entry.I64.Const(8))))
	entry.BrInd(target, one, two)

	one.Br(out.To(one.I64.Const(111)))
	two.Br(out.To(two.I64.Const(222)))
	out.Return(r)

	got := runNative(t, m, `
#include <stdio.h>
long jgoto(long);
int main(void) { printf("%ld %ld\n", jgoto(0), jgoto(1)); return 0; }
`)
	if got != "111 222\n" {
		t.Errorf("printed %q, want %q", got, "111 222\n")
	}
}
