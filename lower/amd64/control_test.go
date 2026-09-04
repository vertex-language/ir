package amd64_test

import (
	"bytes"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	amd64elf "github.com/vertex-language/amd64/obj/elf"
	elfcore "github.com/vertex-language/elf"
	elfobj "github.com/vertex-language/elf/obj"
	"github.com/vertex-language/ir"
	amd64lower "github.com/vertex-language/ir/lower/amd64"
)

func buildAdd(t *testing.T) *ir.Module {
	t.Helper()

	m := ir.NewModule("t", ir.X86_64Linux)
	fn := m.Func("add").Export()
	a := fn.ParamI32("a")
	b := fn.ParamI32("b")
	fn.ReturnsI32()

	entry := fn.Entry()
	entry.Return(entry.I32.Add(a, b))

	return m
}

func TestLowerRoundTrip(t *testing.T) {
	o, err := amd64lower.Lower(buildAdd(t), amd64lower.Options{OptLevel: amd64lower.O0})
	if err != nil {
		t.Fatalf("Lower: %v", err)
	}

	var buf bytes.Buffer
	if err := amd64elf.Write(&buf, o); err != nil {
		t.Fatalf("elf.Write: %v", err)
	}

	f, err := elfobj.NewFile(bytes.NewReader(buf.Bytes()))
	if err != nil {
		t.Fatalf("NewFile: %v", err)
	}

	if f.Machine != elfcore.EM_X86_64 {
		t.Errorf("Machine = %v, want EM_X86_64", f.Machine)
	}

	text := f.Section(".text")
	if text == nil {
		t.Fatal(".text section not found")
	}
	tb, err := text.Data()
	if err != nil {
		t.Fatalf(".text Data: %v", err)
	}

	// mov eax, edi ; add eax, esi ; ret — a is EDI, b is ESI under SysV;
	want := []byte{
		0x8b, 0xc7, // mov eax, edi
		0x03, 0xc6, // add eax, esi
		0xc3, // ret
	}
	if !bytes.Equal(tb, want) {
		t.Errorf(".text bytes = % x, want % x", tb, want)
	}

	syms, err := f.Symbols()
	if err != nil {
		t.Fatalf("Symbols: %v", err)
	}
	var add *elfobj.Symbol
	for _, s := range syms {
		if s.Name == "add" {
			add = s
		}
	}
	if add == nil {
		t.Fatal(`symbol "add" not found`)
	}
	if add.Bind != elfcore.STB_GLOBAL {
		t.Errorf("add.Bind = %v, want STB_GLOBAL", add.Bind)
	}
	if add.Type != elfcore.STT_FUNC {
		t.Errorf("add.Type = %v, want STT_FUNC", add.Type)
	}

	objdump, err := exec.LookPath("objdump")
	if err != nil {
		t.Skip("objdump not on PATH; skipping the disassembly cross-check")
	}

	dir := t.TempDir()
	path := filepath.Join(dir, "add.o")
	if err := os.WriteFile(path, buf.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}

	cmd := exec.Command(objdump, "-d", path)
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("objdump -d: %v\n%s", err, out)
	}
	for _, want := range []string{"movl", "addl", "ret"} {
		if !strings.Contains(string(out), want) {
			t.Errorf("objdump -d output missing %q\n%s", want, out)
		}
	}
}

func buildMax(t *testing.T) *ir.Module {
	t.Helper()

	m := ir.NewModule("t", ir.X86_64Linux)
	fn := m.Func("max").Export()
	a := fn.ParamI32("a")
	b := fn.ParamI32("b")
	fn.ReturnsI32()

	entry := fn.Entry()
	thenB := fn.Block("then")
	elseB := fn.Block("else")

	entry.BrIf(entry.I32.SLt(a, b), thenB.To(), elseB.To())
	thenB.Return(b)
	elseB.Return(a)

	return m
}

func TestLowerBranchRoundTrip(t *testing.T) {
	o, err := amd64lower.Lower(buildMax(t), amd64lower.Options{})
	if err != nil {
		t.Fatalf("Lower: %v", err)
	}

	var buf bytes.Buffer
	if err := amd64elf.Write(&buf, o); err != nil {
		t.Fatalf("elf.Write: %v", err)
	}

	f, err := elfobj.NewFile(bytes.NewReader(buf.Bytes()))
	if err != nil {
		t.Fatalf("NewFile: %v", err)
	}

	text := f.Section(".text")
	if text == nil {
		t.Fatal(".text section not found")
	}
	tb, err := text.Data()
	if err != nil {
		t.Fatalf(".text Data: %v", err)
	}
	// cmp edi, esi ; jl then ; jmp els ; then: mov eax, esi ; ret ; els: mov eax, edi ; ret
	want := []byte{
		0x3b, 0xfe, // cmp edi, esi
		0x0f, 0x8c, 0x05, 0x00, 0x00, 0x00, // jl +5 (to "then")
		0xe9, 0x03, 0x00, 0x00, 0x00, // jmp +3 (to "else")
		0x8b, 0xc6, // then: mov eax, esi
		0xc3,       // ret
		0x8b, 0xc7, // else: mov eax, edi
		0xc3, // ret
	}
	if !bytes.Equal(tb, want) {
		t.Errorf(".text bytes = % x, want % x", tb, want)
	}

	syms, err := f.Symbols()
	if err != nil {
		t.Fatalf("Symbols: %v", err)
	}
	var max *elfobj.Symbol
	for _, s := range syms {
		if s.Name == "max" {
			max = s
		}
	}
	if max == nil {
		t.Fatal(`symbol "max" not found`)
	}
	if max.Value != 0 {
		t.Errorf("max.Value = %d, want 0", max.Value)
	}

	objdump, err := exec.LookPath("objdump")
	if err != nil {
		t.Skip("objdump not on PATH")
	}
	dir := t.TempDir()
	path := filepath.Join(dir, "max.o")
	if err := os.WriteFile(path, buf.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}
	out, err := exec.Command(objdump, "-d", path).CombinedOutput()
	if err != nil {
		t.Fatalf("objdump -d: %v\n%s", err, out)
	}
	for _, want := range []string{"cmpl", "jl", "jmp", "ret"} {
		if !strings.Contains(string(out), want) {
			t.Errorf("objdump -d output missing %q\n%s", want, out)
		}
	}
}

func buildMin(t *testing.T) *ir.Module {
	t.Helper()

	m := ir.NewModule("t", ir.X86_64Linux)
	fn := m.Func("min").Export()
	a := fn.ParamI32("a")
	b := fn.ParamI32("b")
	fn.ReturnsI32()

	entry := fn.Entry()
	thenB := fn.Block("then")
	elseB := fn.Block("else")
	join := fn.Block("join")
	r := join.ParamI32("r")

	entry.BrIf(entry.I32.SLt(a, b), thenB.To(), elseB.To())
	thenB.Br(join.To(a))
	elseB.Br(join.To(b))
	join.Return(r)

	return m
}

func TestLowerBlockParamRoundTrip(t *testing.T) {
	o, err := amd64lower.Lower(buildMin(t), amd64lower.Options{})
	if err != nil {
		t.Fatalf("Lower: %v", err)
	}

	var buf bytes.Buffer
	if err := amd64elf.Write(&buf, o); err != nil {
		t.Fatalf("elf.Write: %v", err)
	}

	f, err := elfobj.NewFile(bytes.NewReader(buf.Bytes()))
	if err != nil {
		t.Fatalf("NewFile: %v", err)
	}

	text := f.Section(".text")
	if text == nil {
		t.Fatal(".text section not found")
	}
	tb, err := text.Data()
	if err != nil {
		t.Fatalf(".text Data: %v", err)
	}

	want := []byte{
		0x3b, 0xfe, // cmp edi, esi
		0x0f, 0x8c, 0x05, 0x00, 0x00, 0x00, // jl +5 (to "then")
		0xe9, 0x05, 0x00, 0x00, 0x00, // jmp +5 (to "else")
		0xe9, 0x07, 0x00, 0x00, 0x00, // then: jmp +7 (to "join"; r is a)
		0x8b, 0xfe, // else: mov edi, esi   (b into r)
		0xe9, 0x00, 0x00, 0x00, 0x00, // jmp +0 (to "join")
		0x8b, 0xc7, // join: mov eax, edi   (r into the return register)
		0xc3, // ret
	}
	if !bytes.Equal(tb, want) {
		t.Errorf(".text bytes = % x, want % x", tb, want)
	}

	objdump, err := exec.LookPath("objdump")
	if err != nil {
		t.Skip("objdump not on PATH")
	}
	dir := t.TempDir()
	path := filepath.Join(dir, "min.o")
	if err := os.WriteFile(path, buf.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}
	out, err := exec.Command(objdump, "-d", path).CombinedOutput()
	if err != nil {
		t.Fatalf("objdump -d: %v\n%s", err, out)
	}
	for _, want := range []string{"cmpl", "jl", "jmp", "ret"} {
		if !strings.Contains(string(out), want) {
			t.Errorf("objdump -d output missing %q\n%s", want, out)
		}
	}
}

// Tests lowering of two simultaneous block arguments.
func TestLowerTwoBlockArguments(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	fn := m.Func("swap").Export()
	a := fn.ParamI32("a")
	b := fn.ParamI32("b")
	fn.ReturnsI32()

	entry := fn.Entry()
	join := fn.Block("join")
	x := join.ParamI32("x")
	y := join.ParamI32("y")

	entry.Br(join.To(b, a))
	join.Return(join.I32.Sub(x, y))

	o, err := amd64lower.Lower(m, amd64lower.Options{})
	if err != nil {
		t.Fatalf("Lower: %v", err)
	}

	var buf bytes.Buffer
	if err := amd64elf.Write(&buf, o); err != nil {
		t.Fatalf("elf.Write: %v", err)
	}
	f, err := elfobj.NewFile(bytes.NewReader(buf.Bytes()))
	if err != nil {
		t.Fatalf("NewFile: %v", err)
	}
	text := f.Section(".text")
	if text == nil {
		t.Fatal(".text section not found")
	}
	tb, err := text.Data()
	if err != nil {
		t.Fatalf(".text Data: %v", err)
	}

	want := []byte{
		0xe9, 0x00, 0x00, 0x00, 0x00, // jmp +0 (to "join")
		0x8b, 0xc6, // join: mov eax, esi   (x, which is b)
		0x2b, 0xc7, // sub eax, edi         (minus y, which is a)
		0xc3, // ret
	}
	if !bytes.Equal(tb, want) {
		t.Errorf(".text bytes = % x, want % x", tb, want)
	}

	objdump, err := exec.LookPath("objdump")
	if err != nil {
		t.Skip("objdump not on PATH")
	}
	dir := t.TempDir()
	path := filepath.Join(dir, "swap.o")
	if err := os.WriteFile(path, buf.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}
	out, err := exec.Command(objdump, "-d", path).CombinedOutput()
	if err != nil {
		t.Fatalf("objdump -d: %v\n%s", err, out)
	}
	for _, want := range []string{"movl", "subl", "ret"} {
		if !strings.Contains(string(out), want) {
			t.Errorf("objdump -d output missing %q\n%s", want, out)
		}
	}
}

func buildSum1N(t *testing.T) *ir.Module {
	t.Helper()

	m := ir.NewModule("t", ir.X86_64Linux)
	fn := m.Func("sum1n").Export()
	n := fn.ParamI32("n")
	fn.ReturnsI32()

	entry := fn.Entry()
	loop := fn.Block("loop")
	i := loop.ParamI32("i")
	acc := loop.ParamI32("acc")
	exit := fn.Block("exit")
	done := fn.Block("done")
	r := done.ParamI32("r")
	body := fn.Block("body")

	entry.Br(loop.To(entry.I32.Const(1), entry.I32.Const(0)))
	loop.BrIf(loop.I32.SLt(n, i), exit.To(), body.To())
	exit.Br(done.To(acc))
	body.Br(loop.To(body.I32.Add(i, body.I32.Const(1)), body.I32.Add(acc, i)))
	done.Return(r)

	return m
}

func TestLowerLoopRoundTrip(t *testing.T) {
	o, err := amd64lower.Lower(buildSum1N(t), amd64lower.Options{})
	if err != nil {
		t.Fatalf("Lower: %v", err)
	}

	var buf bytes.Buffer
	if err := amd64elf.Write(&buf, o); err != nil {
		t.Fatalf("elf.Write: %v", err)
	}

	f, err := elfobj.NewFile(bytes.NewReader(buf.Bytes()))
	if err != nil {
		t.Fatalf("NewFile: %v", err)
	}
	text := f.Section(".text")
	if text == nil {
		t.Fatal(".text section not found")
	}
	tb, err := text.Data()
	if err != nil {
		t.Fatalf(".text Data: %v", err)
	}

	want := []byte{
		0xb8, 0x01, 0x00, 0x00, 0x00, // mov eax, 1   (i = 1)
		0xb9, 0x00, 0x00, 0x00, 0x00, // mov ecx, 0   (acc = 0)
		0xe9, 0x00, 0x00, 0x00, 0x00, // jmp +0 (to "loop")
		0x3b, 0xf8, // cmp edi, eax   (n vs i)
		0x0f, 0x8c, 0x05, 0x00, 0x00, 0x00, // jl +5 (to "exit")
		0xe9, 0x08, 0x00, 0x00, 0x00, // jmp +8 (to "body")
		0xe9, 0x00, 0x00, 0x00, 0x00, // exit: jmp +0 (to "done"; r is acc)
		0x8b, 0xc1, // done: mov eax, ecx
		0xc3,                         // ret
		0xba, 0x01, 0x00, 0x00, 0x00, // body: mov edx, 1
		0x8b, 0xf0, // mov esi, eax   (new_i base = i)
		0x03, 0xf2, // add esi, edx   (new_i = i+1)
		0x8b, 0xd1, // mov edx, ecx   (new_acc base = acc)
		0x03, 0xd0, // add edx, eax   (new_acc = acc+i)
		0x8b, 0xc6, // mov eax, esi   (i = new_i)
		0x8b, 0xca, // mov ecx, edx   (acc = new_acc)
		0xe9, 0xd5, 0xff, 0xff, 0xff, // jmp -43 (back to "loop")
	}
	if !bytes.Equal(tb, want) {
		t.Errorf(".text bytes = % x, want % x", tb, want)
	}

	objdump, err := exec.LookPath("objdump")
	if err != nil {
		t.Skip("objdump not on PATH")
	}
	dir := t.TempDir()
	path := filepath.Join(dir, "sum1n.o")
	if err := os.WriteFile(path, buf.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}
	out, err := exec.Command(objdump, "-d", path).CombinedOutput()
	if err != nil {
		t.Fatalf("objdump -d: %v\n%s", err, out)
	}
	for _, want := range []string{"cmpl", "jl", "jmp", "addl", "ret"} {
		if !strings.Contains(string(out), want) {
			t.Errorf("objdump -d output missing %q\n%s", want, out)
		}
	}
}

// Tests conditional branch on an i1 value.
func TestLowerBrIfOnValue(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	fn := m.Func("f").Export()
	c := fn.ParamI1("c")
	fn.ReturnsI32()

	entry := fn.Entry()
	thenB := fn.Block("then")
	elseB := fn.Block("else")
	entry.BrIf(c, thenB.To(), elseB.To())
	thenB.Return(thenB.I32.Const(1))
	elseB.Return(elseB.I32.Const(0))

	tb, raw := lowerText(t, m)

	want := []byte{
		0x40, 0x84, 0xff, // test dil, dil
		0x0f, 0x85, 0x05, 0x00, 0x00, 0x00, // jne then
		0xe9, 0x06, 0x00, 0x00, 0x00, // jmp else
		0xb8, 0x01, 0x00, 0x00, 0x00, // mov eax, 1
		0xc3,                         // ret
		0xb8, 0x00, 0x00, 0x00, 0x00, // mov eax, 0
		0xc3, // ret
	}
	if !bytes.Equal(tb, want) {
		t.Errorf(".text bytes = % x, want % x", tb, want)
	}
	objdumpHas(t, "f", raw, "testb", "jne")
}

// Tests conditional branch whose targets carry block arguments.
func TestLowerBrIfBlockArguments(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	fn := m.Func("min").Export()
	a := fn.ParamI32("a")
	b := fn.ParamI32("b")
	fn.ReturnsI32()

	entry := fn.Entry()
	join := fn.Block("join")
	r := join.ParamI32("r")

	entry.BrIf(entry.I32.SLt(a, b), join.To(a), join.To(b))
	join.Return(r)

	tb, raw := lowerText(t, m)

	// r is in EAX, not EDI. The move that puts a return value in its
	// register is a copy isel emits, so the coalescer can see it and
	// bias r towards the register it ends up in — and once r is EAX,
	// the join block's own move is a mov eax, eax and goes away. The
	// two edges pay for it instead, which is where the values differ
	// anyway: neither edge was free before.
	want := []byte{
		0x3b, 0xfe, // cmp edi, esi
		0x0f, 0x8c, 0x06, 0x00, 0x00, 0x00, // jl +6 (to "min.then")
		0xe9, 0x08, 0x00, 0x00, 0x00, // jmp +8 (to "min.else")
		0xc3,       // join: ret   (r is already in EAX)
		0x8b, 0xc7, // min.then: mov eax, edi   (r = a)
		0xe9, 0xf8, 0xff, 0xff, 0xff, // jmp -8 (to "join")
		0x8b, 0xc6, // min.else: mov eax, esi   (r = b)
		0xe9, 0xf1, 0xff, 0xff, 0xff, // jmp -15 (to "join")
	}
	if !bytes.Equal(tb, want) {
		t.Errorf(".text bytes = % x, want % x", tb, want)
	}

	objdumpHas(t, "min", raw, "cmpl", "jl", "movl", "jmp", "ret")
}

// Tests a back edge that permutes parameters.
func TestLowerPermutedBackEdge(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	fn := m.Func("rot").Export()
	n := fn.ParamI32("n")
	x0 := fn.ParamI32("x")
	y0 := fn.ParamI32("y")
	fn.ReturnsI32()

	entry := fn.Entry()
	loop := fn.Block("loop")
	i := loop.ParamI32("i")
	x := loop.ParamI32("x")
	y := loop.ParamI32("y")
	body := fn.Block("body")
	exit := fn.Block("exit")

	entry.Br(loop.To(entry.I32.Const(0), x0, y0))
	loop.BrIf(loop.I32.SLt(i, n), body.To(), exit.To())
	body.Br(loop.To(body.I32.Add(i, body.I32.Const(1)), y, x))
	exit.Return(x)

	tb, raw := lowerText(t, m)

	want := []byte{
		0xb8, 0x00, 0x00, 0x00, 0x00, // mov eax, 0     (i = 0)
		0xe9, 0x00, 0x00, 0x00, 0x00, // jmp +0 (to "loop")
		0x3b, 0xc7, // loop: cmp eax, edi   (i vs n)
		0x0f, 0x8c, 0x05, 0x00, 0x00, 0x00, // jl +5 (to "body")
		0xe9, 0x15, 0x00, 0x00, 0x00, // jmp +21 (to "exit")
		0xb9, 0x01, 0x00, 0x00, 0x00, // body: mov ecx, 1
		0x44, 0x8b, 0xc0, // mov r8d, eax    (i+1 base = i)
		0x44, 0x03, 0xc1, // add r8d, ecx    (i+1)
		0x41, 0x8b, 0xc0, // mov eax, r8d    (i = i+1)
		0x87, 0xd6, // xchg esi, edx   (x, y = y, x)
		0xe9, 0xde, 0xff, 0xff, 0xff, // jmp -34 (back to "loop")
		0x8b, 0xc6, // exit: mov eax, esi  (return x)
		0xc3, // ret
	}
	if !bytes.Equal(tb, want) {
		t.Errorf(".text bytes = % x, want % x", tb, want)
	}

	objdumpHas(t, "rot", raw, "cmpl", "jl", "addl", "xchgl", "ret")
}

// Tests compare conditions that can fuse with brif.
func TestLowerCompareConditions(t *testing.T) {
	for _, tc := range []struct {
		verb     string
		cmp      func(b *ir.Block, a, c ir.I32) ir.I1
		opcode   byte   // the second byte of the 0f 8x near jump
		mnemonic string // what objdump calls it
	}{
		{"eq", func(b *ir.Block, a, c ir.I32) ir.I1 { return b.I32.Eq(a, c) }, 0x84, "je"},
		{"ne", func(b *ir.Block, a, c ir.I32) ir.I1 { return b.I32.Ne(a, c) }, 0x85, "jne"},
		{"slt", func(b *ir.Block, a, c ir.I32) ir.I1 { return b.I32.SLt(a, c) }, 0x8c, "jl"},
		{"sle", func(b *ir.Block, a, c ir.I32) ir.I1 { return b.I32.SLe(a, c) }, 0x8e, "jle"},
		{"ult", func(b *ir.Block, a, c ir.I32) ir.I1 { return b.I32.ULt(a, c) }, 0x82, "jb"},
		{"ule", func(b *ir.Block, a, c ir.I32) ir.I1 { return b.I32.ULe(a, c) }, 0x86, "jbe"},
	} {
		t.Run(tc.verb, func(t *testing.T) {
			m := ir.NewModule("t", ir.X86_64Linux)
			fn := m.Func("f").Export()
			a := fn.ParamI32("a")
			b := fn.ParamI32("b")
			fn.ReturnsI32()

			entry := fn.Entry()
			yes := fn.Block("yes")
			no := fn.Block("no")

			entry.BrIf(tc.cmp(entry, a, b), yes.To(), no.To())
			yes.Return(yes.I32.Const(1))
			no.Return(no.I32.Const(0))

			tb, raw := lowerText(t, m)

			want := []byte{
				0x3b, 0xfe, // cmp edi, esi
				0x0f, tc.opcode, 0x05, 0x00, 0x00, 0x00, // jcc +5 (to "f.yes")
				0xe9, 0x06, 0x00, 0x00, 0x00, // jmp +6 (to "f.no")
				0xb8, 0x01, 0x00, 0x00, 0x00, // yes: mov eax, 1
				0xc3,                         // ret
				0xb8, 0x00, 0x00, 0x00, 0x00, // no: mov eax, 0
				0xc3, // ret
			}
			if !bytes.Equal(tb, want) {
				t.Errorf(".text bytes = % x, want % x", tb, want)
			}

			objdumpHas(t, "f", raw, "cmpl", tc.mnemonic, "ret")
		})
	}
}

// Tests collision resolution when a user block is named "then".
func TestLowerEdgeLabelCollision(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	fn := m.Func("f").Export()
	a := fn.ParamI32("a")
	b := fn.ParamI32("b")
	fn.ReturnsI32()

	entry := fn.Entry()
	thenB := fn.Block("then")
	p := thenB.ParamI32("p")
	elseB := fn.Block("else")

	entry.BrIf(entry.I32.SLt(a, b), thenB.To(b), elseB.To())
	thenB.Return(p)
	elseB.Return(a)

	tb, raw := lowerText(t, m)

	// p is in EAX for the reason r is in TestLowerBrIfBlockArguments:
	// the return's copy is a copy, so p coalesces with it. The move
	// that was in "then" is in the edge block instead.
	want := []byte{
		0x3b, 0xfe, // cmp edi, esi
		0x0f, 0x8c, 0x09, 0x00, 0x00, 0x00, // jl +9 (to "f.then.2", the edge)
		0xe9, 0x01, 0x00, 0x00, 0x00, // jmp +1 (to "f.else")
		0xc3,       // then: ret   (p is already in EAX)
		0x8b, 0xc7, // else: mov eax, edi
		0xc3,       // ret
		0x8b, 0xc6, // f.then.2: mov eax, esi   (p, which is b)
		0xe9, 0xf5, 0xff, 0xff, 0xff, // jmp -11 (to "f.then")
	}
	if !bytes.Equal(tb, want) {
		t.Errorf(".text bytes = % x, want % x", tb, want)
	}

	objdumpHas(t, "f", raw, "cmpl", "jl", "movl", "ret")
}

// Tests a permuted backedge loop carrying mixed widths.
func TestLowerMixedWidthBackEdge(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	fn := m.Func("rot64").Export()
	n := fn.ParamI32("n")
	x0 := fn.ParamI64("x")
	y0 := fn.ParamI64("y")
	fn.ReturnsI64()

	entry := fn.Entry()
	loop := fn.Block("loop")
	i := loop.ParamI32("i")
	x := loop.ParamI64("x")
	y := loop.ParamI64("y")
	body := fn.Block("body")
	exit := fn.Block("exit")

	entry.Br(loop.To(entry.I32.Const(0), x0, y0))
	loop.BrIf(loop.I32.SLt(i, n), body.To(), exit.To())
	body.Br(loop.To(body.I32.Add(i, body.I32.Const(1)), y, x))
	exit.Return(x)

	tb, raw := lowerText(t, m)

	want := []byte{
		0xb8, 0x00, 0x00, 0x00, 0x00, // mov eax, 0      (i = 0, 32-bit)
		0xe9, 0x00, 0x00, 0x00, 0x00, // jmp +0 (to "loop")
		0x3b, 0xc7, // loop: cmp eax, edi   (i vs n, 32-bit)
		0x0f, 0x8c, 0x05, 0x00, 0x00, 0x00, // jl +5 (to "body")
		0xe9, 0x16, 0x00, 0x00, 0x00, // jmp +22 (to "exit")
		0xb9, 0x01, 0x00, 0x00, 0x00, // body: mov ecx, 1
		0x44, 0x8b, 0xc0, // mov r8d, eax    (i+1 base = i)
		0x44, 0x03, 0xc1, // add r8d, ecx    (i+1)
		0x41, 0x8b, 0xc0, // mov eax, r8d    (i = i+1, 32-bit)
		0x48, 0x87, 0xd6, // xchg rsi, rdx   (x, y = y, x, 64-bit)
		0xe9, 0xdd, 0xff, 0xff, 0xff, // jmp -35 (back to "loop")
		0x48, 0x8b, 0xc6, // exit: mov rax, rsi  (return x, 64-bit)
		0xc3, // ret
	}
	if !bytes.Equal(tb, want) {
		t.Errorf(".text bytes = % x, want % x", tb, want)
	}

	objdumpHas(t, "rot64", raw, "cmpl", "jl", "addl", "xchgq", "movq")
}

// Tests the Trap terminator.
func TestLowerTrap(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	fn := m.Func("die").Export()
	fn.ReturnsI32()
	fn.Entry().Trap()

	tb, raw := lowerText(t, m)
	if want := []byte{0x0f, 0x0b}; !bytes.Equal(tb, want) {
		t.Errorf(".text bytes = % x, want % x", tb, want)
	}
	objdumpHas(t, "die", raw, "ud2")
}
