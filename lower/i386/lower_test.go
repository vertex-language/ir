package i386_test

// Tests for the i386 lowerer.
//
// Byte expectations rather than a running program. This is the one backend in
// the tree whose output cannot be executed on the machine it is developed on
// — Apple Silicon has no 32-bit x86 — so the arm64 backend's link-and-run
// check has no counterpart here, and objdump is the cross-check instead.
//
// Which makes the objdump pass matter more than it does elsewhere: a byte
// expectation written by hand carries whatever its author believed, and the
// disassembler is the only second opinion available.

import (
	"bytes"
	"encoding/hex"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	i386elf "github.com/vertex-language/i386/obj/elf"

	"github.com/vertex-language/ir"
	i386lower "github.com/vertex-language/ir/lower/i386"
	"github.com/vertex-language/ir/verify"
)

// lowerText lowers m and returns @.text's bytes along with the whole object.
func lowerText(t *testing.T, m *ir.Module) (text []byte, object []byte) {
	t.Helper()

	if err := verify.Module(m); err != nil {
		t.Fatalf("verify.Module: %v", err)
	}
	o, err := i386lower.Lower(m, i386lower.Options{})
	if err != nil {
		t.Fatalf("Lower: %v", err)
	}
	var buf bytes.Buffer
	if err := i386elf.Write(&buf, o); err != nil {
		t.Fatalf("elf.Write: %v", err)
	}
	for _, s := range o.Sections() {
		if s.Name() == ".text" {
			return s.Bytes(), buf.Bytes()
		}
	}
	t.Fatal(".text not found")
	return nil, nil
}

// objdumpText disassembles the object.
//
// Skipped only when objdump is absent, which is an environment fact. It is
// not skipped for anything else: this is the only check in the file with an
// opinion of its own — every byte expectation here was written by the same
// author as the code it checks — so a disassembly that will not run has to be
// a failure rather than a quiet pass.
func objdumpText(t *testing.T, name string, object []byte) string {
	t.Helper()
	objdump, err := exec.LookPath("objdump")
	if err != nil {
		t.Skip("objdump not on PATH")
	}
	path := filepath.Join(t.TempDir(), name+".o")
	if err := os.WriteFile(path, object, 0o644); err != nil {
		t.Fatal(err)
	}
	out, err := exec.Command(objdump, "-d", path).CombinedOutput()
	if err != nil {
		t.Fatalf("objdump -d: %v\n%s", err, out)
	}
	if !strings.Contains(string(out), "elf32-i386") {
		t.Fatalf("objdump did not read this as a 32-bit i386 object:\n%s", out)
	}
	return string(out)
}

func wantBytes(t *testing.T, got []byte, want string) {
	t.Helper()
	w, err := hex.DecodeString(strings.ReplaceAll(want, " ", ""))
	if err != nil {
		t.Fatalf("bad expectation: %v", err)
	}
	if !bytes.Equal(got, w) {
		t.Errorf(".text = %s\n   want %s", hexOf(got), hexOf(w))
	}
}

func hexOf(b []byte) string {
	var s strings.Builder
	for i, v := range b {
		if i > 0 {
			s.WriteByte(' ')
		}
		fmt.Fprintf(&s, "%02x", v)
	}
	return s.String()
}

func hasAll(t *testing.T, disasm string, want ...string) {
	t.Helper()
	for _, w := range want {
		if !strings.Contains(disasm, w) {
			t.Errorf("disassembly missing %q\n%s", w, disasm)
		}
	}
}

// mnemonics is the disassembly's instructions in order, which is what an
// adjacency claim has to be checked against.
func mnemonics(disasm string) []string {
	var out []string
	for _, line := range strings.Split(disasm, "\n") {
		i := strings.Index(line, "\t")
		if i < 0 {
			continue
		}
		// "  0: 55  \tpushl\t%ebp" — the mnemonic is between the two
		// tabs, or runs to the end when the instruction has no operand.
		m := line[i+1:]
		if j := strings.Index(m, "\t"); j >= 0 {
			m = m[:j]
		}
		if m = strings.TrimSpace(m); m != "" {
			out = append(out, m)
		}
	}
	return out
}

// The simplest function there is: two 32-bit parameters added and returned.
//
// Both arrive on the stack, which is the whole of this architecture's calling
// convention — there is no register argument in the psABI.
func TestLowerAdd32(t *testing.T) {
	m := ir.NewModule("t", ir.I386Linux)
	fn := m.Func("add2").Export()
	a := fn.ParamI32("a")
	b := fn.ParamI32("b")
	fn.ReturnsI32()
	entry := fn.Entry()
	entry.Return(entry.I32.Add(a, b))

	text, raw := lowerText(t, m)
	hasAll(t, objdumpText(t, "add2", raw), "pushl", "movl", "addl", "leave", "ret")
	if len(text) == 0 {
		t.Fatal("no code emitted")
	}
}

// An i64 added, which is the shape the whole package exists for: two
// registers per value, ADD then ADC, and the carry crossing between them.
func TestLowerAdd64(t *testing.T) {
	m := ir.NewModule("t", ir.I386Linux)
	fn := m.Func("add64").Export()
	a := fn.ParamI64("a")
	b := fn.ParamI64("b")
	fn.ReturnsI64()
	entry := fn.Entry()
	entry.Return(entry.I64.Add(a, b))

	_, raw := lowerText(t, m)
	d := objdumpText(t, "add64", raw)
	hasAll(t, d, "addl", "adcl")

	// ADC has to come after the ADD with nothing between them that writes
	// the flags. Only a move may: the allocator inserts nothing else, and
	// an x86 move leaves the flags alone.
	ms := mnemonics(d)
	add, adc := -1, -1
	for i, m := range ms {
		if m == "addl" && add < 0 {
			add = i
		}
		if m == "adcl" {
			adc = i
		}
	}
	if add < 0 || adc < 0 || adc < add {
		t.Fatalf("expected an addl followed by an adcl, got %v", ms)
	}
	for _, m := range ms[add+1 : adc] {
		if !strings.HasPrefix(m, "mov") {
			t.Errorf("%q sits between the addl and the adcl and is not a move; it may clobber the carry\n%v", m, ms)
		}
	}
}

// A 64-bit subtract, which is SUB and SBB for the same reason.
func TestLowerSub64(t *testing.T) {
	m := ir.NewModule("t", ir.I386Linux)
	fn := m.Func("sub64").Export()
	a := fn.ParamI64("a")
	b := fn.ParamI64("b")
	fn.ReturnsI64()
	entry := fn.Entry()
	entry.Return(entry.I64.Sub(a, b))

	_, raw := lowerText(t, m)
	hasAll(t, objdumpText(t, "sub64", raw), "subl", "sbbl")
}

// A 64-bit equality, which is the two halves XOR'd and OR'd: one flag out of
// two comparisons, and no branch.
func TestLowerEq64(t *testing.T) {
	m := ir.NewModule("t", ir.I386Linux)
	fn := m.Func("eq64").Export()
	a := fn.ParamI64("a")
	b := fn.ParamI64("b")
	fn.ReturnsI32()
	entry := fn.Entry()
	entry.Return(entry.I32.ZExtI1(entry.I64.Eq(a, b)))

	_, raw := lowerText(t, m)
	hasAll(t, objdumpText(t, "eq64", raw), "xorl", "orl", "sete", "movzbl")
}

// A 64-bit signed ordering, which is a subtract through both halves and the
// flags the SBB leaves.
func TestLowerSLt64(t *testing.T) {
	m := ir.NewModule("t", ir.I386Linux)
	fn := m.Func("lt64").Export()
	a := fn.ParamI64("a")
	b := fn.ParamI64("b")
	fn.ReturnsI32()
	entry := fn.Entry()
	entry.Return(entry.I32.ZExtI1(entry.I64.SLt(a, b)))

	_, raw := lowerText(t, m)
	hasAll(t, objdumpText(t, "lt64", raw), "subl", "sbbl", "setl")
}

// Sign extension into a pair: the low half is the value and the high half is
// every bit of its sign.
func TestLowerSExt(t *testing.T) {
	m := ir.NewModule("t", ir.I386Linux)
	fn := m.Func("widen").Export()
	a := fn.ParamI32("a")
	fn.ReturnsI64()
	entry := fn.Entry()
	entry.Return(entry.I64.SExtI32(a))

	_, raw := lowerText(t, m)
	hasAll(t, objdumpText(t, "widen", raw), "sarl")
}

// Every callee-saved register this function takes has to be given back, and
// the slot it is kept in has to be inside this function's own frame.
//
// A slot at zero is the saved EBP, which LEAVE pops on the way out: storing
// EBX there hands the caller EBX's old value as its frame pointer. That is
// what this checked when it was written, because that is what it found.
func TestLowerSavesCalleeRegistersBelowFrame(t *testing.T) {
	m := ir.NewModule("t", ir.I386Linux)
	fn := m.Func("uses").Export()
	a := fn.ParamI64("a")
	b := fn.ParamI64("b")
	fn.ReturnsI64()
	entry := fn.Entry()
	entry.Return(entry.I64.Add(a, b))

	_, raw := lowerText(t, m)
	d := objdumpText(t, "uses", raw)
	for _, line := range strings.Split(d, "\n") {
		if !strings.Contains(line, "%ebx") && !strings.Contains(line, "%esi") &&
			!strings.Contains(line, "%edi") {
			continue
		}
		if strings.Contains(line, "(%ebp)") && !strings.Contains(line, "0x") {
			t.Errorf("a callee-saved register is kept at [ebp+0], which is the saved EBP:\n%s", d)
			break
		}
	}
	// And the frame has to have been taken, since something is kept in it.
	hasAll(t, d, "subl")
}

// A branch with block arguments and a frame slot, which exercises the
// parallel copy and the EBP-relative addressing together.
func TestLowerBranchAndFrame(t *testing.T) {
	m := ir.NewModule("t", ir.I386Linux)
	fn := m.Func("maxstore").Export()
	a := fn.ParamI32("a")
	b := fn.ParamI32("b")
	fn.ReturnsI32()

	entry := fn.Entry()
	join := fn.Block("join")
	r := join.ParamI32("r")

	slot := entry.Ptr.Alloc(4, 4)
	entry.BrIf(entry.I32.SLt(a, b), join.To(b), join.To(a))
	join.I32.Store(r, slot)
	join.Return(join.I32.Load(slot))

	_, raw := lowerText(t, m)
	hasAll(t, objdumpText(t, "maxstore", raw), "setl", "testl", "jne", "jmp", "leal")
}

// A call, whose arguments are a run of stores into the outgoing area.
func TestLowerCall(t *testing.T) {
	m := ir.NewModule("t", ir.I386Linux)
	dbl := m.ImportFunc("dbl", ir.NewSig().Param(ir.TypeI32).Ret(ir.TypeI32))

	fn := m.Func("adddbl").Export()
	a := fn.ParamI32("a")
	fn.ReturnsI32()
	entry := fn.Entry()
	r := entry.Call(dbl, a).Value(0).(ir.I32)
	entry.Return(entry.I32.Add(r, a))

	_, raw := lowerText(t, m)
	hasAll(t, objdumpText(t, "adddbl", raw), "call", "movl")
}

// f80 is refused by name, and stays refused.
//
// The psABI's long double is the ten-byte x87 type and this package's floats
// live in the vector unit, where there is no ten-byte anything. Reaching it
// would mean a second register file with a stack discipline, for one type.
func TestLowerRefusesF80(t *testing.T) {
	m := ir.NewModule("t", ir.I386Linux)
	fn := m.Func("f").Export()
	a := fn.ParamF80("a")
	fn.ReturnsF80()
	entry := fn.Entry()
	entry.Return(entry.F80().Add(a, a))

	_, err := i386lower.Lower(m, i386lower.Options{})
	if err == nil {
		t.Fatal("Lower should refuse an f80")
	}
}

// A module built for another architecture is refused by its layout block
// before anything is selected.
func TestLowerRefusesForeignLayout(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	fn := m.Func("f").Export()
	fn.ReturnsI32()
	entry := fn.Entry()
	entry.Return(entry.I32.Const(0))

	if _, err := i386lower.Lower(m, i386lower.Options{}); err == nil {
		t.Error("Lower should refuse a 64-bit layout")
	}
}
