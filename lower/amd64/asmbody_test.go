package amd64_test

// §3b's module-level asm and §7's asm body.
//
// These two are the one place in this package where an expected-bytes test is
// the right test rather than a lazy one. The objection to byte comparison for
// §G4 is that the answer depends on what the allocator chose, so the expected
// bytes would be encoding the allocator's answer in order to check the
// assembler's. Here there is no allocator: the text has no operands, so the
// bytes are a function of the text alone and nothing else. What that buys is
// the one claim about a naked function worth making — that its .text is its
// body exactly, with nothing added before or after it.
//
// The expected bytes below are clang's, for the same three instructions.

import (
	"bytes"
	"strings"
	"testing"

	elfcore "github.com/vertex-language/elf"

	"github.com/vertex-language/ir"
)

// A naked function is its body and nothing else: no prologue, no epilogue,
// and no alignment padding around it.
func TestAsmBodyIsExactlyItsText(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	fn := m.Func("nadd").Export().NoUnwind()
	fn.ParamI64("a")
	fn.ParamI64("b")
	fn.ReturnsI64()
	fn.AsmBody("movq %rdi, %rax\n\taddq %rsi, %rax\n\tret")

	f := lowerFile(t, m)

	want := []byte{
		0x48, 0x89, 0xf8, // movq %rdi, %rax
		0x48, 0x01, 0xf0, // addq %rsi, %rax
		0xc3, // ret
	}
	if got := sectionData(t, f, ".text"); !bytes.Equal(got, want) {
		t.Errorf(".text = % x, want % x", got, want)
	}

	sym := symbol(t, f, "nadd")
	if sym.Bind != elfcore.STB_GLOBAL {
		t.Errorf("nadd.Bind = %v, want STB_GLOBAL", sym.Bind)
	}
	if sym.Type != elfcore.STT_FUNC {
		t.Errorf("nadd.Type = %v, want STT_FUNC", sym.Type)
	}
	if sym.Size != uint64(len(want)) {
		t.Errorf("nadd.Size = %d, want %d", sym.Size, len(want))
	}
}

// A module-level block between two lowered functions. Its bytes land where it
// was declared, and the function after it is still in .text — a fragment that
// left the section switched would put that one somewhere else.
func TestModuleAsmBetweenFunctions(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)

	before := m.Func("before").Export().NoUnwind().ReturnsI64()
	e := before.Entry()
	e.Return(e.I64.Const(1))

	m.Asm(".globl seven\nseven:\n\tmovq $7, %rax\n\tret")

	after := m.Func("after").Export().NoUnwind().ReturnsI64()
	e2 := after.Entry()
	e2.Return(e2.I64.Const(2))

	f := lowerFile(t, m)

	sev := symbol(t, f, "seven")
	if sev.Bind != elfcore.STB_GLOBAL {
		t.Errorf("seven.Bind = %v, want STB_GLOBAL", sev.Bind)
	}

	// Declaration order is emission order: before, then the block, then
	// after. Each symbol's value is its offset in .text.
	b, a := symbol(t, f, "before"), symbol(t, f, "after")
	if !(b.Value < sev.Value && sev.Value < a.Value) {
		t.Errorf("offsets are before=%d seven=%d after=%d; want them in declaration order",
			b.Value, sev.Value, a.Value)
	}
	if sec := f.Section(".text"); sec == nil || a.Shndx != sev.Shndx || a.Shndx != b.Shndx {
		t.Errorf("the three symbols are in sections %d, %d and %d; want one .text",
			b.Shndx, sev.Shndx, a.Shndx)
	}
}

// A block that switches sections and switches back. The data lands in
// .rodata and .text keeps only what the lowered function put there.
func TestModuleAsmSectionSwitch(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)

	m.Asm(".pushsection .rodata\n" +
		".globl answer\n" +
		".p2align 3\n" +
		"answer:\n" +
		"\t.quad 42\n" +
		".popsection")

	fn := m.Func("one").Export().NoUnwind().ReturnsI64()
	e := fn.Entry()
	e.Return(e.I64.Const(1))

	f := lowerFile(t, m)

	want := []byte{42, 0, 0, 0, 0, 0, 0, 0}
	if got := sectionData(t, f, ".rodata"); !bytes.Equal(got, want) {
		t.Errorf(".rodata = % x, want % x", got, want)
	}
	if got := symbol(t, f, "answer"); got.Shndx == symbol(t, f, "one").Shndx {
		t.Error("answer is in the same section as the lowered function; the switch did not take")
	}
}

// A body the assembler refuses reaches the caller with the position the
// assembler gave it, rather than becoming a module error with a section
// offset for a location.
func TestAsmBodyRefused(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	fn := m.Func("bad").Export().NoUnwind()
	fn.AsmBody("nosuchinsn %rax, %rbx")

	err := lowerAsmErr(t, m)
	if err == nil {
		t.Fatal("lowered; the body is not assembly")
	}
	if !strings.Contains(err.Error(), "nosuchinsn") {
		t.Errorf("error %q does not name the mnemonic it refused", err)
	}
}

// Each module-level block is its own assembly: the section stack of one does
// not reach the next.
func TestModuleAsmBlocksAreIndependent(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	m.Asm(".pushsection .rodata")
	m.Asm(".popsection")

	err := lowerAsmErr(t, m)
	if err == nil {
		t.Fatal("lowered; the pop is in a block that pushed nothing")
	}
	if !strings.Contains(err.Error(), "no section pushed") {
		t.Errorf("error %q does not say the stack was empty", err)
	}
}
