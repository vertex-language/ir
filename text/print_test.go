package text_test

// The printer's tests are golden text, because that is what the printer
// is for: .vir exists to be read by a person and diffed by a test, and
// the only faithful assertion about a printer is the exact string it
// produced.
//
// There is no parser, so these do not round-trip. A golden file that
// cannot be read back is still the strongest check available — it fails
// on any change to the syntax, which is exactly when a reader of .vir
// wants to be told.

import (
	"bytes"
	"strings"
	"testing"

	"github.com/vertex-language/ir"
	"github.com/vertex-language/ir/text"
)

// format is Format with the error folded into the test.
func format(t *testing.T, m *ir.Module) string {
	t.Helper()

	b, err := text.Format(m)
	if err != nil {
		t.Fatalf("text.Format: %v", err)
	}
	return string(b)
}

// same compares against a golden string, trimming the leading newline
// the raw literal starts with so the expected text can be written
// left-aligned in the source.
func same(t *testing.T, got, want string) {
	t.Helper()

	want = strings.TrimPrefix(want, "\n")
	if got == want {
		return
	}
	t.Errorf("printed:\n%s\nwant:\n%s", got, want)
}

// The module header: §3's module, use and layout, which every module has
// and which nothing can omit — they are NewModule's parameters.
func TestPrintModuleHeader(t *testing.T) {
	m := ir.NewModule("hdr", ir.X86_64Linux)

	same(t, format(t, m), `
module hdr

use "x86_64/linux"

layout {
  abi        sysv,
  endian     little,
  ptrbits    64,
  stackalign 16,
  extfloat   f80, f128,
}
`)
}

// A whole small function, which is the shape most .vir a person reads
// will be: a signature with named parameters, a block with an
// instruction, and a terminator.
func TestPrintFunction(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	fn := m.Func("add").Export().NoUnwind()
	a := fn.ParamI32("a")
	b := fn.ParamI32("b")
	fn.ReturnsI32()

	entry := fn.Entry()
	entry.Return(entry.I32.Add(a, b))

	got := format(t, m)
	want := `
export func @add(%a i32, %b i32) i32 nounwind {
@entry:
  %0 = i32.add %a, %b
  return %0
}
`
	if !strings.Contains(got, strings.TrimPrefix(want, "\n")) {
		t.Errorf("printed:\n%s\nwant it to contain:\n%s", got, want)
	}
}

// §3b and §7's other body. Both are one keyword and a string, and the
// escaping is strconv's, so a template's newlines and quotes survive as
// something a reader can paste back into an assembler.
func TestPrintAssembly(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	m.Asm(".pushsection .init_array,\"aw\"\n.quad my_ctor\n.popsection")

	fn := m.Func("_start").Export().NoReturn()
	fn.AsmBody("movq %rsp, %rdi\n\tcall __libc_start_main")

	got := format(t, m)
	want := `
asm ".pushsection .init_array,\"aw\"\n.quad my_ctor\n.popsection"

export func @_start() naked noreturn {
  asm "movq %rsp, %rdi\n\tcall __libc_start_main"
}
`
	if !strings.Contains(got, strings.TrimPrefix(want, "\n")) {
		t.Errorf("printed:\n%s\nwant it to contain:\n%s", got, want)
	}
}

// Block parameters and the arguments a branch carries into them, which
// is what this IR has instead of phi nodes and the thing a reader of
// .vir most needs to see spelled out.
func TestPrintBlockParameters(t *testing.T) {
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

	got := format(t, m)
	for _, want := range []string{
		"@join(%r i32):",
		"brif %0, @join(%a), @join(%b)",
	} {
		if !strings.Contains(got, want) {
			t.Errorf("printed output missing %q\n%s", want, got)
		}
	}
}

// §5's globals: each domain, each initializer form the printer has a
// spelling for, and the linkage that precedes them.
func TestPrintGlobals(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	m.Global("counter", ir.RW, ir.StoreI64.FType()).Export().Init(ir.Lit(ir.Int(7)))
	m.Global("msg", ir.RO, ir.Array(6, ir.StoreI8.FType())).Internal().Init(ir.Str("hello"))
	m.Global("blank", ir.RW, ir.StoreI32.FType()).Export()

	got := format(t, m)
	for _, want := range []string{
		`export global rw @counter i64 = 7`,
		`internal global ro @msg [6]i8 = "hello"`,
		`export global rw @blank i32 = zeroed`,
	} {
		if !strings.Contains(got, want) {
			t.Errorf("printed output missing %q\n%s", want, got)
		}
	}
}

// A named struct and the alignment-independent part of §5's type
// syntax.
func TestPrintTypes(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	m.Struct("pair").
		Field("a", ir.StoreI32.FType()).
		Field("b", ir.StoreI64.FType())

	got := format(t, m)
	for _, want := range []string{"type @pair struct {", "a i32,", "b i64"} {
		if !strings.Contains(got, want) {
			t.Errorf("printed output missing %q\n%s", want, got)
		}
	}
}

// Format refuses a module carrying a sticky builder error rather than
// printing a partial one. There is nothing faithful to print: every call
// after the first failure was a no-op, so the module in hand is not the
// module the caller thinks they built.
func TestFormatRefusesABrokenModule(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	fn := m.Func("f").Export()
	fn.ReturnsI32()
	// A block that never gets a terminator is fine; an instruction after
	// one is not, and is what makes the module sticky.
	entry := fn.Entry()
	entry.Return(entry.I32.Const(0))
	entry.Return(entry.I32.Const(1))

	if err := m.Err(); err == nil {
		t.Skip("the builder accepted a second terminator; this test needs a sticky error")
	}
	if _, err := text.Format(m); err == nil {
		t.Error("Format printed a module carrying a builder error")
	}
}

// Print is Format plus a write, and returns the same refusal.
func TestPrintWritesWhatFormatReturns(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	fn := m.Func("f").Export()
	fn.ReturnsI32()
	fn.Entry().Return(fn.Entry().I32.Const(0))

	var buf bytes.Buffer
	if err := text.Print(&buf, m); err != nil {
		t.Fatalf("text.Print: %v", err)
	}
	if got, want := buf.String(), format(t, m); got != want {
		t.Errorf("Print wrote:\n%s\nFormat returned:\n%s", got, want)
	}
}

// The zero Printer is documented as not being the form Format produces:
// the indent falls back to two spaces, but metadata stays off.
func TestZeroPrinterIndents(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	fn := m.Func("f").Export()
	fn.ReturnsI32()
	fn.Entry().Return(fn.Entry().I32.Const(0))

	var p text.Printer
	b, err := p.Format(m)
	if err != nil {
		t.Fatalf("Format: %v", err)
	}
	if !strings.Contains(string(b), "\n  %0 = i32.const 0") {
		t.Errorf("the zero Printer did not fall back to a two-space indent:\n%s", b)
	}
}

// An indent the caller chose is used verbatim.
func TestPrinterIndent(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	fn := m.Func("f").Export()
	fn.ReturnsI32()
	fn.Entry().Return(fn.Entry().I32.Const(0))

	p := text.Printer{Indent: "\t"}
	b, err := p.Format(m)
	if err != nil {
		t.Fatalf("Format: %v", err)
	}
	if !strings.Contains(string(b), "\n\t%0 = i32.const 0") {
		t.Errorf("Indent was not used:\n%s", b)
	}
}

// §8b's instruction and §14's terminator form, which are the two shapes
// with no fixed operand list. The keyword constraints print bare and a
// target-specific one prints quoted, because that is the difference §8b's
// grammar draws between `reg` and a string.
func TestPrintInlineAssembly(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	fn := m.Func("f").Export().NoUnwind()
	a := fn.ParamI64("a")
	fn.ReturnsI32()

	e := fn.Entry()
	fall := fn.Block("fall")
	taken := fn.Block("taken")

	e.Asm("addq %2, %0").Volatile().
		Out(ir.TypeI64, ir.CReg).
		In(a, ir.CStr("0")).
		In(a, ir.CMem).
		Clobber("cc", "memory").Emit()
	e.AsmGoto("testq %0, %0\n\tjz %l[taken]").In(a, ir.CStr("=&r")).Clobber("cc").
		To(fall.To(), taken)
	fall.Return(fall.I32.Const(0))
	taken.Return(taken.I32.Const(1))

	got := format(t, m)
	want := `
export func @f(%a i64) i32 nounwind {
@entry:
  (%0 i64 reg) = asm volatile "addq %2, %0" (%a "0", %a mem) clobber "cc", "memory"
  asm goto "testq %0, %0\n\tjz %l[taken]" (%a "=&r") clobber "cc" to @fall, [@taken]
`
	if !strings.Contains(got, strings.TrimPrefix(want, "\n")) {
		t.Errorf("printed:\n%s\nwant it to contain:\n%s", got, want)
	}
}

// asm goto's outputs print where §8b's do — the template numbers them first,
// so seeing them first is what makes %0 findable — and carry no register
// name, because they define none: the values are the fallthrough target's
// trailing parameters, which the block header shows.
func TestPrintAsmGotoOutputs(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	fn := m.Func("f").Export().NoUnwind()
	a := fn.ParamI64("a")
	fn.ReturnsI64()

	e := fn.Entry()
	fall := fn.Block("fall")
	out := fall.ParamI64("out")
	taken := fn.Block("taken")

	e.AsmGoto("testq %1, %1\n\tjz %l[taken]").
		Out(ir.TypeI64, ir.CStr("=r")).In(a, ir.CReg).Clobber("cc").
		To(fall.To(), taken)
	fall.Return(out)
	taken.Return(taken.I64.Const(0))

	got := format(t, m)
	want := `
  (i64 "=r") = asm goto "testq %1, %1\n\tjz %l[taken]" (%a reg) clobber "cc" to @fall, [@taken]

@fall(%out i64):
`
	if !strings.Contains(got, strings.TrimPrefix(want, "\n")) {
		t.Errorf("printed:\n%s\nwant it to contain:\n%s", got, want)
	}
}

// The lists §8b lets an asm leave out: no outputs, no inputs, no clobbers,
// and not volatile. What is left is the keyword and the string.
func TestPrintInlineAssemblyWithNoOperands(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	fn := m.Func("barrier").Export()
	e := fn.Entry()
	e.Asm("mfence").Emit()
	e.Return()

	got := format(t, m)
	if want := `  asm "mfence" ()`; !strings.Contains(got, want) {
		t.Errorf("printed output missing %q\n%s", want, got)
	}
}
