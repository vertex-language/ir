package arm64_test

// Tests for the arm64 lowerer.
//
// Expectations are instruction words rather than bytes. Every A64 instruction
// is four bytes and a word is what a disassembler prints, so a word is what a
// reader can check against the ARM ARM; bytes would be the same information
// with the endianness applied by hand.

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	arm64elf "github.com/vertex-language/arm64/obj/elf"
	elfobj "github.com/vertex-language/elf/obj"

	"github.com/vertex-language/ir"
	arm64lower "github.com/vertex-language/ir/lower/arm64"
	"github.com/vertex-language/ir/verify"
)

// lowerWords lowers m, writes it as ELF, reads the object back, and returns
// @.text as instruction words along with the whole object for a disassembly
// cross-check.
func lowerWords(t *testing.T, m *ir.Module) (words []uint32, object []byte) {
	t.Helper()

	if err := verify.Module(m); err != nil {
		t.Fatalf("verify.Module: %v", err)
	}
	o, err := arm64lower.Lower(m, arm64lower.Options{})
	if err != nil {
		t.Fatalf("Lower: %v", err)
	}

	var buf bytes.Buffer
	if err := arm64elf.Write(&buf, o); err != nil {
		t.Fatalf("elf.Write: %v", err)
	}
	f, err := elfobj.NewFile(bytes.NewReader(buf.Bytes()))
	if err != nil {
		t.Fatalf("NewFile: %v", err)
	}
	sec := f.Section(".text")
	if sec == nil {
		t.Fatal(".text section not found")
	}
	tb, err := sec.Data()
	if err != nil {
		t.Fatalf(".text Data: %v", err)
	}
	if len(tb)%4 != 0 {
		t.Fatalf(".text is %d bytes, which is not a whole number of instructions", len(tb))
	}
	for i := 0; i < len(tb); i += 4 {
		words = append(words, binary.LittleEndian.Uint32(tb[i:]))
	}
	return words, buf.Bytes()
}

func equalWords(t *testing.T, got, want []uint32) {
	t.Helper()
	if len(got) == len(want) {
		same := true
		for i := range got {
			if got[i] != want[i] {
				same = false
				break
			}
		}
		if same {
			return
		}
	}
	t.Errorf(".text = %s, want %s", hexWords(got), hexWords(want))
}

func hexWords(w []uint32) string {
	var b strings.Builder
	b.WriteByte('[')
	for i, v := range w {
		if i > 0 {
			b.WriteByte(' ')
		}
		fmt.Fprintf(&b, "%08x", v)
	}
	b.WriteByte(']')
	return b.String()
}

// objdumpHas is the best-effort disassembly cross-check: skipped, not failed,
// when objdump is not on PATH, since the word comparison is authoritative.
func objdumpHas(t *testing.T, name string, object []byte, mnemonics ...string) {
	t.Helper()

	objdump, err := exec.LookPath("objdump")
	if err != nil {
		t.Skip("objdump not on PATH; skipping the disassembly cross-check")
	}
	path := filepath.Join(t.TempDir(), name+".o")
	if err := os.WriteFile(path, object, 0o644); err != nil {
		t.Fatal(err)
	}
	out, err := exec.Command(objdump, "-d", path).CombinedOutput()
	if err != nil {
		t.Fatalf("objdump -d: %v\n%s", err, out)
	}
	for _, want := range mnemonics {
		if !strings.Contains(string(out), want) {
			t.Errorf("objdump -d output missing %q\n%s", want, out)
		}
	}
}

// The simplest function there is: two parameters added and returned.
//
// No prologue. X30 holds the return address in a register rather than on the
// stack, so a leaf that stores nothing has nothing to save — which is the
// clearest difference from the other architecture, where the same function
// still pays for a frame the moment it needs one at all.
func TestLowerAdd(t *testing.T) {
	m := ir.NewModule("t", ir.AArch64Linux)
	fn := m.Func("add2").Export()
	a := fn.ParamI32("a")
	b := fn.ParamI32("b")
	fn.ReturnsI32()
	entry := fn.Entry()
	entry.Return(entry.I32.Add(a, b))

	got, raw := lowerWords(t, m)
	equalWords(t, got, []uint32{
		0x0b010002, // add w2, w0, w1
		0x2a0203e0, // mov w0, w2
		0xd65f03c0, // ret
	})
	objdumpHas(t, "add2", raw, "add", "ret")
}

// Three addresses is the shape here: the destination is named separately from
// both operands, so an add of a value into itself needs no copy first.
func TestLowerAddInPlace(t *testing.T) {
	m := ir.NewModule("t", ir.AArch64Linux)
	fn := m.Func("dbl").Export()
	a := fn.ParamI64("a")
	fn.ReturnsI64()
	entry := fn.Entry()
	entry.Return(entry.I64.Add(a, a))

	got, raw := lowerWords(t, m)
	equalWords(t, got, []uint32{
		0x8b000001, // add x1, x0, x0
		0xaa0103e0, // mov x0, x1
		0xd65f03c0, // ret
	})
	objdumpHas(t, "dbl", raw, "add")
}

// A comparison is always its own instruction. ADD does not set the flags here
// — ADDS does — so there is no fusing to decide about and every §B result is
// materialized with CSET.
func TestLowerCompareAndBranch(t *testing.T) {
	m := ir.NewModule("t", ir.AArch64Linux)
	fn := m.Func("min").Export()
	a := fn.ParamI32("a")
	b := fn.ParamI32("b")
	fn.ReturnsI32()

	entry := fn.Entry()
	join := fn.Block("join")
	r := join.ParamI32("r")
	entry.BrIf(entry.I32.SLt(a, b), join.To(a), join.To(b))
	join.Return(r)

	got, raw := lowerWords(t, m)
	equalWords(t, got, []uint32{
		0x6b01001f, // cmp w0, w1        (subs wzr, w0, w1)
		0x9a9fa7e2, // cset x2, lt
		0x7100005f, // cmp w2, #0
		0x54000061, // b.ne +12          (the "then" edge)
		0x14000003, // b +12             (the "else" edge)
		0xd65f03c0, // join: ret         (r is already in w0)
		0x17ffffff, // then edge: b -4   (r = a, already in place)
		0x2a0103e0, // else edge: mov w0, w1
		0x17fffffd, // b -12
	})
	objdumpHas(t, "min", raw, "cmp", "cset", "b.ne")
}

// A load through a pointer, and the frame that is not needed for it.
func TestLowerLoad(t *testing.T) {
	m := ir.NewModule("t", ir.AArch64Linux)
	fn := m.Func("ld").Export()
	p := fn.ParamPtr("p")
	fn.ReturnsI64()
	entry := fn.Entry()
	entry.Return(entry.I64.Load(p))

	got, raw := lowerWords(t, m)
	equalWords(t, got, []uint32{
		0xf9400001, // ldr x1, [x0]
		0xaa0103e0, // mov x0, x1
		0xd65f03c0, // ret
	})
	objdumpHas(t, "ld", raw, "ldr")
}

// The frame, and the record at the top of it.
//
// STP with pre-index is the push: it stores the pair and moves SP in one
// instruction. The epilogue takes SP back from X29 rather than adding the
// frame size, which is the same instruction count and is right whether or not
// anything moved SP in between.
func TestLowerFrame(t *testing.T) {
	m := ir.NewModule("t", ir.AArch64Linux)
	fn := m.Func("slot").Export()
	v := fn.ParamI64("v")
	fn.ReturnsI64()
	entry := fn.Entry()
	p := entry.Ptr.Alloc(8, 8)
	entry.I64.Store(v, p)
	entry.Return(entry.I64.Load(p))

	got, raw := lowerWords(t, m)
	equalWords(t, got, []uint32{
		0xa9bf7bfd, // stp x29, x30, [sp, #-16]!
		0x910003fd, // mov x29, sp
		0xd10043ff, // sub sp, sp, #16
		0xd10023a1, // sub x1, x29, #8      (the slot's address)
		0xf9000020, // str x0, [x1]
		0xf9400020, // ldr x0, [x1]
		0x910003bf, // mov sp, x29
		0xa8c17bfd, // ldp x29, x30, [sp], #16
		0xd65f03c0, // ret
	})
	objdumpHas(t, "slot", raw, "stp", "ldp", "ret")
}

// A64 has no instruction that takes a 64-bit literal, so a constant is a MOVZ
// and however many MOVKs the value needs.
func TestLowerConstant(t *testing.T) {
	m := ir.NewModule("t", ir.AArch64Linux)
	fn := m.Func("k").Export()
	fn.ReturnsI32()
	entry := fn.Entry()
	entry.Return(entry.I32.Const(100))

	got, raw := lowerWords(t, m)
	equalWords(t, got, []uint32{
		0x52800c80, // mov w0, #100
		0xd65f03c0, // ret
	})
	objdumpHas(t, "k", raw, "mov")
}

// §A's division is the instruction and the guards §A's trap needs, which A64
// does not raise for itself: SDIV by zero gives zero and INT_MIN/-1 gives
// INT_MIN, both quietly.
//
// Structure rather than an exact word list. What matters is that the divide
// is guarded on both counts and that the guard ends somewhere that stops —
// which is one BRK, reached by two branches. TestRunDivisionTraps is the
// check that it really stops.
func TestLowerDivisionGuards(t *testing.T) {
	m := ir.NewModule("t", ir.AArch64Linux)
	fn := m.Func("d").Export()
	a := fn.ParamI64("a")
	b := fn.ParamI64("b")
	fn.ReturnsI64()
	entry := fn.Entry()
	entry.Return(entry.I64.SDiv(a, b))

	got, raw := lowerWords(t, m)

	const (
		brk   = 0xd4200000 // brk #0
		sdiv  = 0x9ac00c00 // sdiv xd, xn, xm — the opcode, less its registers
		opMsk = 0xffe0fc00
	)
	var brks, divs int
	for _, w := range got {
		switch {
		case w == brk:
			brks++
		case w&opMsk == sdiv:
			divs++
		}
	}
	if brks != 1 {
		t.Errorf("%d BRKs, want one: %s", brks, hexWords(got))
	}
	if divs != 1 {
		t.Errorf("%d SDIVs, want one: %s", divs, hexWords(got))
	}
	objdumpHas(t, "d", raw, "sdiv", "brk")
}

// The layout block is checked before anything is selected: a module built for
// the other architecture names a different ABI and is refused by name.
func TestLowerRefusesForeignLayout(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	fn := m.Func("f").Export()
	fn.ReturnsI32()
	entry := fn.Entry()
	entry.Return(entry.I32.Const(0))

	if _, err := arm64lower.Lower(m, arm64lower.Options{}); err == nil {
		t.Error("Lower should refuse a module whose layout names the sysv ABI")
	}
}
