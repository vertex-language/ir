package amd64_test

// The indirect control flow: callind, brind and br_table, which arrived
// together under one smoke test — and that is how a jump table that
// destroyed two live registers went unnoticed.

import (
	"bytes"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	elfobj "github.com/vertex-language/elf/obj"

	"github.com/vertex-language/ir"
)

// A three-way switch. Two cases in the table and a default the range
// check branches to.
//
// One unsigned compare covers both ends of the range: a negative
// selector read as unsigned is a very large one, so the JAE that catches
// an index past the last entry catches a negative index too.
func TestLowerBrTable(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	fn := m.Func("sw").Export()
	sel := fn.ParamI32("sel")
	fn.ReturnsI32()

	entry := fn.Entry()
	a := fn.Block("a")
	b := fn.Block("b")
	def := fn.Block("def")

	entry.BrTable(sel, []ir.BlockTarget{a.To(), b.To()}, def.To())
	a.Return(a.I32.Const(10))
	b.Return(b.I32.Const(20))
	def.Return(def.I32.Const(0))

	tb, raw := lowerText(t, m)

	// No prologue: a br_table needs a scratch register and nothing else,
	// and here there is a caller-saved one going spare.
	want := []byte{
		0x81, 0xff, 0x02, 0x00, 0x00, 0x00, // cmp edi, 2
		0x0f, 0x83, 0x16, 0x00, 0x00, 0x00, // jae +22 (to "sw.def")
		0x48, 0x8d, 0x05, 0x00, 0x00, 0x00, 0x00, // lea rax, [rip + sw.table]
		0xff, 0x24, 0xf8, // jmp [rax + rdi*8]
		0xb8, 0x0a, 0x00, 0x00, 0x00, // sw.a: mov eax, 10
		0xc3,                         // ret
		0xb8, 0x14, 0x00, 0x00, 0x00, // sw.b: mov eax, 20
		0xc3,                         // ret
		0xb8, 0x00, 0x00, 0x00, 0x00, // sw.def: mov eax, 0
		0xc3, // ret
	}
	if !bytes.Equal(tb, want) {
		t.Errorf(".text bytes = % x, want % x", tb, want)
	}

	// The table is in .rodata, one eightbyte per case, and it is all
	// relocation: the bytes are zero in the object and the linker writes
	// the block addresses into them.
	ro := section(t, raw, ".rodata")
	if len(ro) != 16 {
		t.Errorf(".rodata is %d bytes, want 16 — two entries of eight", len(ro))
	}
	objdumpHasRelocs(t, "sw", raw,
		"R_X86_64_PC32", "sw.table",
		"R_X86_64_64", "sw.a", "sw.b")
}

// The regression test for the clobber. Eight values live into both
// successors fill every caller-saved register, so the scratch cannot be
// one — and that it is RBX, which the prologue below is saving, is the
// proof: hard-coded scratch would have cost nothing here and been wrong.
//
// A prefix, because what regressed is the prologue and the lookup; the
// summation after says nothing about the table.
func TestLowerBrTableScratchIsAllocated(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	fn := m.Func("live8").Export()
	sel := fn.ParamI32("sel")
	fn.ReturnsI64()

	entry := fn.Entry()
	join := fn.Block("join")
	def := fn.Block("def")

	vals := make([]ir.I64, 8)
	for i := range vals {
		vals[i] = entry.I64.Const(int64(i + 1))
	}
	entry.BrTable(sel, []ir.BlockTarget{join.To(), def.To()}, def.To())

	// Both successors read every value, so all eight are live across the
	// terminator and none of their registers is reusable as scratch.
	sum := func(b *ir.Block) ir.I64 {
		acc := vals[0]
		for _, v := range vals[1:] {
			acc = b.I64.Add(acc, v)
		}
		return acc
	}
	join.Return(sum(join))
	def.Return(sum(def))

	tb, raw := lowerText(t, m)

	prologue := []byte{
		0x55,             // push rbp
		0x48, 0x8b, 0xec, // mov rbp, rsp
		0x48, 0x81, 0xec, 0x10, 0x00, 0x00, 0x00, // sub rsp, 16
		0x48, 0x89, 0x5d, 0xf8, // mov [rbp-8], rbx   (the scratch register)
	}
	if !bytes.HasPrefix(tb, prologue) {
		t.Errorf(".text does not start with the prologue that saves the scratch register\n"+
			"got  % x\nwant % x", tb[:min(len(tb), len(prologue))], prologue)
	}

	// And the lookup itself uses that register, not one of the eight.
	lookup := []byte{
		0x48, 0x8d, 0x1d, 0x00, 0x00, 0x00, 0x00, // lea rbx, [rip + live8.table]
		0xff, 0x24, 0xfb, // jmp [rbx + rdi*8]
	}
	if !bytes.Contains(tb, lookup) {
		t.Errorf(".text does not contain the table lookup through RBX\ngot % x", tb)
	}
	objdumpHas(t, "live8", raw, "jmpq")
}

// A br_table whose edges carry block arguments. Each case that supplies
// one gets its own block to make the assignment in, and the table points
// at that block rather than at the shared destination — the same edge
// splitting a brif does, applied to a table with more than two exits.
func TestLowerBrTableBlockArguments(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	fn := m.Func("swa").Export()
	sel := fn.ParamI32("sel")
	x := fn.ParamI32("x")
	y := fn.ParamI32("y")
	fn.ReturnsI32()

	entry := fn.Entry()
	join := fn.Block("join")
	r := join.ParamI32("r")
	def := fn.Block("def")

	entry.BrTable(sel, []ir.BlockTarget{join.To(x), join.To(y)}, def.To())
	join.Return(r)
	def.Return(def.I32.Const(0))

	tb, raw := lowerText(t, m)

	// join is a bare ret: r is in EAX because both edges put it there,
	// which is milestone 30's return copy biasing the parameter.
	want := []byte{
		0x81, 0xff, 0x02, 0x00, 0x00, 0x00, // cmp edi, 2
		0x0f, 0x83, 0x0b, 0x00, 0x00, 0x00, // jae +11 (to "swa.def")
		0x48, 0x8d, 0x05, 0x00, 0x00, 0x00, 0x00, // lea rax, [rip + swa.table]
		0xff, 0x24, 0xf8, // jmp [rax + rdi*8]
		0xc3,                         // swa.join: ret
		0xb8, 0x00, 0x00, 0x00, 0x00, // swa.def: mov eax, 0
		0xc3,       // ret
		0x8b, 0xc6, // edge 0: mov eax, esi   (r = x)
		0xe9, 0xf2, 0xff, 0xff, 0xff, // jmp -14 (to "swa.join")
		0x8b, 0xc2, // edge 1: mov eax, edx   (r = y)
		0xe9, 0xeb, 0xff, 0xff, 0xff, // jmp -21 (to "swa.join")
	}
	if !bytes.Equal(tb, want) {
		t.Errorf(".text bytes = % x, want % x", tb, want)
	}

	// The table names the edge blocks, not the join they both reach.
	objdumpHasRelocs(t, "swa", raw, "swa.table_edge_01", "swa.table_edge_12")
}

// brind: a jump to an address in a value, with the blocks it may reach
// named so the CFG knows about the edges.
//
// The pointer is already in RDI, so the copy into the vreg the jump
// reads coalesced away and the whole terminator is two bytes.
func TestLowerBrInd(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	fn := m.Func("bi").Export()
	p := fn.ParamPtr("p")
	fn.ReturnsI32()

	entry := fn.Entry()
	a := fn.Block("a")
	b := fn.Block("b")

	entry.BrInd(p, a, b)
	a.Return(a.I32.Const(1))
	b.Return(b.I32.Const(2))

	tb, raw := lowerText(t, m)

	want := []byte{
		0xff, 0xe7, // jmp rdi
		0xb8, 0x01, 0x00, 0x00, 0x00, // bi.a: mov eax, 1
		0xc3,                         // ret
		0xb8, 0x02, 0x00, 0x00, 0x00, // bi.b: mov eax, 2
		0xc3, // ret
	}
	if !bytes.Equal(tb, want) {
		t.Errorf(".text bytes = % x, want % x", tb, want)
	}
	objdumpHas(t, "bi", raw, "jmpq")
}

// section is one named section's bytes out of a written object.
func section(t *testing.T, object []byte, name string) []byte {
	t.Helper()
	f, err := elfobj.NewFile(bytes.NewReader(object))
	if err != nil {
		t.Fatalf("NewFile: %v", err)
	}
	sec := f.Section(name)
	if sec == nil {
		t.Fatalf("%s section not found", name)
	}
	b, err := sec.Data()
	if err != nil {
		t.Fatalf("%s Data: %v", name, err)
	}
	return b
}

// objdumpHasRelocs is objdumpHas against the relocation table rather than
// the disassembly: skipped, not failed, when objdump is not on PATH.
func objdumpHasRelocs(t *testing.T, name string, object []byte, want ...string) {
	t.Helper()

	objdump, err := exec.LookPath("objdump")
	if err != nil {
		t.Skip("objdump not on PATH; skipping the relocation cross-check")
	}
	path := filepath.Join(t.TempDir(), name+".o")
	if err := os.WriteFile(path, object, 0o644); err != nil {
		t.Fatal(err)
	}
	out, err := exec.Command(objdump, "-r", path).CombinedOutput()
	if err != nil {
		t.Fatalf("objdump -r: %v\n%s", err, out)
	}
	for _, w := range want {
		if !strings.Contains(string(out), w) {
			t.Errorf("objdump -r output missing %q\n%s", w, out)
		}
	}
}
