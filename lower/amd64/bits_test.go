package amd64_test

// Milestone 28: §A6, and the Options field that made three quarters of
// it reachable.
//
// popcnt, clz and ctz are ordinary integer verbs whose instructions
// carry CPUID bits, so lowering one is a question about the processor
// and not only about the module. Options.Features is where the caller
// answers it.

import (
	"bytes"
	"os"
	"path/filepath"
	"testing"

	elfobj "github.com/vertex-language/elf/obj"

	"github.com/vertex-language/amd64/feature"
	amd64elf "github.com/vertex-language/amd64/obj/elf"

	"github.com/vertex-language/ir"
	amd64lower "github.com/vertex-language/ir/lower/amd64"
	"github.com/vertex-language/ir/verify"
)

// lowerFor lowers m for a named processor and returns @.text's bytes,
// read back out of a written ELF the way lowerText does.
func lowerFor(t *testing.T, m *ir.Module, set feature.Set) []byte {
	t.Helper()

	if err := verify.Module(m); err != nil {
		t.Fatalf("verify.Module: %v", err)
	}
	o, err := amd64lower.Lower(m, amd64lower.Options{Features: set})
	if err != nil {
		t.Fatalf("Lower: %v", err)
	}
	var buf bytes.Buffer
	if err := amd64elf.Write(&buf, o); err != nil {
		t.Fatalf("elf.Write: %v", err)
	}
	// Written to a file as well, so a failing test can be disassembled
	// by hand out of the temp directory.
	os.WriteFile(filepath.Join(t.TempDir(), "f.o"), buf.Bytes(), 0o644)

	f, err := elfobj.NewFile(bytes.NewReader(buf.Bytes()))
	if err != nil {
		t.Fatalf("elfobj.NewFile: %v", err)
	}
	for _, s := range f.Sections {
		if s.Name != ".text" {
			continue
		}
		d, err := s.Data()
		if err != nil {
			t.Fatalf("section data: %v", err)
		}
		return d
	}
	t.Fatal("no .text section")
	return nil
}

// The counting three, each one instruction whose destination is not also
// a source — the rare unary shape in a table where a unary operation is
// usually in place.
func TestLowerBitCounting(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)

	a := m.Func("pc").Export()
	x := a.ParamI64("a")
	a.ReturnsI64()
	a.Entry().Return(a.Entry().I64.Popcnt(x))

	b := m.Func("cl").Export()
	y := b.ParamI32("a")
	b.ReturnsI32()
	b.Entry().Return(b.Entry().I32.Clz(y))

	c := m.Func("ct").Export()
	z := c.ParamI32("a")
	c.ReturnsI32()
	c.Entry().Return(c.Entry().I32.Ctz(z))

	got := lowerFor(t, m, feature.NewSet(feature.V3))

	want := []byte{
		0xf3, 0x48, 0x0f, 0xb8, 0xc7, // popcnt rax, rdi
		0xc3,                   // ret
		0xf3, 0x0f, 0xbd, 0xc7, // lzcnt eax, edi
		0xc3,                   // ret
		0xf3, 0x0f, 0xbc, 0xc7, // tzcnt eax, edi
		0xc3, // ret
	}
	if !bytes.Equal(got, want) {
		t.Errorf(".text bytes = % x, want % x", got, want)
	}
}

// bswap is the opposite shape and the one §A6 verb that is baseline: it
// reverses a register in place, so it is the ALU table's form with one
// operand and needs the move first when the destination is elsewhere.
func TestLowerByteSwap(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	fn := m.Func("bs").Export()
	a := fn.ParamI64("a")
	fn.ReturnsI64()
	fn.Entry().Return(fn.Entry().I64.Bswap(a))

	tb, raw := lowerText(t, m)

	want := []byte{
		0x48, 0x8b, 0xc7, // mov rax, rdi
		0x48, 0x0f, 0xc8, // bswap rax
		0xc3, // ret
	}
	if !bytes.Equal(tb, want) {
		t.Errorf(".text bytes = % x, want % x", tb, want)
	}
	objdumpHas(t, "bs", raw, "bswapq")
}

// A gated verb lowers on every processor: to its instruction where the
// processor has it, and where it does not, to the sequence LLVM writes for
// the baseline -- BSR and BSF with a conditional move for zero, and the SWAR
// sum for popcnt. Emitting the instruction anyway would be the one wrong
// answer: LZCNT on a processor without it decodes as BSR, which answers a
// different question and says nothing.
func TestGatedVerbsFallBackOffTarget(t *testing.T) {
	var (
		popcnt = []byte{0xf3, 0x0f, 0xb8}
		lzcnt  = []byte{0xf3, 0x0f, 0xbd}
		tzcnt  = []byte{0xf3, 0x0f, 0xbc}
		bsr    = []byte{0x0f, 0xbd}
		bsf    = []byte{0x0f, 0xbc}
		imul   = []byte{0x69} // imul r32, r/m32, imm32: the byte sum
	)
	for _, tc := range []struct {
		name    string
		emit    func(b *ir.Block, x ir.I32) ir.I32
		set     feature.Set
		want    []byte
		wantNot []byte
	}{
		{"popcnt on the baseline", func(b *ir.Block, x ir.I32) ir.I32 { return b.I32.Popcnt(x) }, feature.Default(), imul, popcnt},
		{"popcnt on v2", func(b *ir.Block, x ir.I32) ir.I32 { return b.I32.Popcnt(x) }, feature.NewSet(feature.V2), popcnt, nil},
		{"clz on v2", func(b *ir.Block, x ir.I32) ir.I32 { return b.I32.Clz(x) }, feature.NewSet(feature.V2), bsr, lzcnt},
		{"clz on v3", func(b *ir.Block, x ir.I32) ir.I32 { return b.I32.Clz(x) }, feature.NewSet(feature.V3), lzcnt, nil},
		{"ctz on the baseline", func(b *ir.Block, x ir.I32) ir.I32 { return b.I32.Ctz(x) }, feature.Default(), bsf, tzcnt},
		{"ctz on v3", func(b *ir.Block, x ir.I32) ir.I32 { return b.I32.Ctz(x) }, feature.NewSet(feature.V3), tzcnt, nil},
	} {
		t.Run(tc.name, func(t *testing.T) {
			m := ir.NewModule("t", ir.X86_64Linux)
			fn := m.Func("f").Export()
			x := fn.ParamI32("a")
			fn.ReturnsI32()
			fn.Entry().Return(tc.emit(fn.Entry(), x))

			text := lowerFor(t, m, tc.set)
			if !bytes.Contains(text, tc.want) {
				t.Errorf(".text is % x; want % x in it", text, tc.want)
			}
			if tc.wantNot != nil && bytes.Contains(text, tc.wantNot) {
				t.Errorf(".text is % x; % x is an instruction this processor does not have", text, tc.wantNot)
			}
		})
	}
}

// The zero Options is the baseline, which is what a caller who says
// nothing about the processor gets -- and not an empty set, which would
// describe no processor at all. So popcnt is the sequence and not POPCNT.
func TestLowerDefaultsToTheBaseline(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	fn := m.Func("f").Export()
	x := fn.ParamI64("a")
	fn.ReturnsI64()
	fn.Entry().Return(fn.Entry().I64.Popcnt(x))

	var unset feature.Set
	if text := lowerFor(t, m, unset); bytes.Contains(text, []byte{0xf3, 0x48, 0x0f, 0xb8}) {
		t.Errorf(".text is % x; POPCNT is not a baseline instruction", text)
	}
}

// The float verbs whose instructions are SSE4.1's and FMA3's are calls to
// the C library on a processor without them, which is what they are
// defined to be.
func TestFloatVerbsCallTheLibraryOffTarget(t *testing.T) {
	for _, tc := range []struct {
		name string
		emit func(b *ir.Block, x ir.F64) ir.F64
		sym  string
	}{
		{"trunc", func(b *ir.Block, x ir.F64) ir.F64 { return b.F64.Trunc(x) }, "trunc"},
		{"floor", func(b *ir.Block, x ir.F64) ir.F64 { return b.F64.Floor(x) }, "floor"},
		{"ceil", func(b *ir.Block, x ir.F64) ir.F64 { return b.F64.Ceil(x) }, "ceil"},
		{"nearest", func(b *ir.Block, x ir.F64) ir.F64 { return b.F64.Nearest(x) }, "nearbyint"},
		{"fma", func(b *ir.Block, x ir.F64) ir.F64 { return b.F64.FMA(x, x, x) }, "fma"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			m := ir.NewModule("t", ir.X86_64Linux)
			fn := m.Func("f").Export()
			x := fn.ParamF64("a")
			fn.ReturnsF64()
			fn.Entry().Return(tc.emit(fn.Entry(), x))

			if err := verify.Module(m); err != nil {
				t.Fatalf("verify.Module: %v", err)
			}
			o, err := amd64lower.Lower(m, amd64lower.Options{})
			if err != nil {
				t.Fatalf("Lower: %v", err)
			}
			if sym, ok := o.Symbol(tc.sym); !ok || sym.Defined() {
				t.Errorf("no undefined %s in the object; the baseline has no instruction for it", tc.sym)
			}
		})
	}
}
