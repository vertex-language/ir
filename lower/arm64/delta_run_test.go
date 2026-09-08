package arm64_test

// A relative pointer emitted from the IR, across a real link.
//
// `RelocInit(a).Minus(b)` is a four-byte field holding one symbol minus
// another. It is not an address: a table of these is position-independent
// where a table of addresses is not, which is why every descriptor Swift's
// runtime reads is built out of them.
//
// There is no symbol at an arbitrary offset inside a global, so the field's
// own address is spelled as the global's symbol plus the field's offset --
// carried as a negative addend, which is what swiftc's own conformance
// descriptors do entry by entry.

import (
	"testing"

	"github.com/vertex-language/ir"
)

func TestRunRelativePointer(t *testing.T) {
	m := ir.NewModule("t", ir.AArch64Linux)

	// The thing a relative pointer points at: an ordinary global, and a
	// function, which is in another section entirely.
	target := m.Global("_target", ir.RO, ir.StoreI64.FType()).Init(ir.Lit(ir.Int(7)))
	target.Export()

	fn := m.Func("_code").Export()
	fn.ReturnsI64()
	fn.Entry().Return(fn.Entry().I64.Const(3))

	// Two relative pointers side by side. The first is at offset zero of
	// the table and the second at offset four, so the second's addend is
	// minus four: the subtrahend names the table, and the field it is
	// written in is four bytes further along.
	table := m.Global("_table", ir.RO, ir.Array(2, ir.StoreI32.FType()))
	table.Export()
	table.Init(ir.List(
		ir.RelocInit(target).Minus(table),
		ir.RelocInit(fn).Minus(table).Plus(ir.Int(-4)),
	))

	got := runNative(t, m, `
#include <stdio.h>
#include <stdint.h>
extern int32_t table[2];
extern long target;
extern long code(void) __asm__("_code");
int main(void) {
    char *base = (char *)table;
    int ok = 0;
    ok += ((char *)(base + 0) + table[0]) == (char *)&target;
    ok += ((char *)(base + 4) + table[1]) == (char *)code;
    printf("%s %d\n", ok == 2 ? "ok" : "WRONG", ok);
    return 0;
}
`)
	if got != "ok 2\n" {
		t.Errorf("printed %q, want %q", got, "ok 2\n")
	}
}
