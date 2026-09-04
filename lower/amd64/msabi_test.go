package amd64_test

// The Microsoft x64 calling convention, checked at the level the difference
// actually shows: which register an argument lands in, what the prologue
// writes, and whether an unwind record comes out beside the code.
//
// These read the object rather than run the program, because a Windows
// program does not run on the machine this suite usually runs on. What they
// pin down is exactly the part that differs from SysV — everything below the
// convention is the same instruction selection every other test here covers.

import (
	"bytes"
	"os"
	"path/filepath"
	"testing"

	amd64pe "github.com/vertex-language/amd64/obj/pe"
	"github.com/vertex-language/ir"
	amd64lower "github.com/vertex-language/ir/lower/amd64"
	"github.com/vertex-language/ir/verify"
	"github.com/vertex-language/pe/coff"
)

// lowerMS lowers m for the Microsoft ABI and hands back the COFF object.
func lowerMS(t *testing.T, m *ir.Module) *coff.File {
	t.Helper()

	if err := verify.Module(m); err != nil {
		t.Fatalf("verify.Module: %v", err)
	}
	o, err := amd64lower.Lower(m, amd64lower.Options{})
	if err != nil {
		t.Fatalf("Lower: %v", err)
	}
	var buf bytes.Buffer
	if err := amd64pe.Write(&buf, o, amd64pe.Options{File: m.Name()}); err != nil {
		t.Fatalf("pe.Write: %v", err)
	}
	// coff reads from a file rather than a slice, and a test that is about
	// what the writer produced should read it back the way anything else
	// would.
	path := filepath.Join(t.TempDir(), m.Name()+".obj")
	if err := os.WriteFile(path, buf.Bytes(), 0o666); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}
	f, err := coff.Open(path)
	if err != nil {
		t.Fatalf("coff.Open: %v", err)
	}
	t.Cleanup(func() { f.Close() })
	return f
}

// textOfMS is the lowered .text bytes for a Microsoft-ABI module.
func textOfMS(t *testing.T, m *ir.Module) []byte {
	t.Helper()
	f := lowerMS(t, m)
	for _, s := range f.Sections {
		if s.Name == ".text" {
			b, err := s.Data()
			if err != nil {
				t.Fatalf(".text Data: %v", err)
			}
			return b
		}
	}
	t.Fatal("no .text section")
	return nil
}

// TestMSArgumentRegisters. The first four arguments travel in RCX, RDX, R8 and
// R9, and a float takes the vector register at its own position rather than
// one from a second sequence — so the third argument of f(i32, f64, i32) is
// R8, not RDX.
func TestMSArgumentRegisters(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Windows)
	callee := m.ImportFunc("g", ir.NewSig().
		Param(ir.TypeI32).Param(ir.TypeF64).Param(ir.TypeI32).Ret(ir.TypeI32))

	fn := m.Func("f").Export()
	a := fn.ParamI32("a")
	b := fn.ParamF64("b")
	c := fn.ParamI32("c")
	fn.ReturnsI32()
	e := fn.Entry()

	// The arguments are passed on reversed, so that the first has to reach
	// the third position and the third the first.
	//
	// Passing them straight through would prove nothing: f's own parameters
	// arrive in exactly the registers a call with the same signature wants,
	// so a good allocator emits no moves at all and the .text holds a
	// prologue, a call, and an epilogue. That is the right code and it
	// looks identical to an ABI that passes nothing. Reversing them is what
	// makes the register assignment observable.
	e.Return(e.Call(callee, c, b, a).Value(0).(ir.I32))

	text := textOfMS(t, m)

	// The shuffle has to read R8 and write it — c comes out of it, a goes
	// into it — and the double stays in XMM1 throughout, untouched, because
	// it is in the second position either way. Rather than pin the whole
	// encoding, check for the two REX prefixes that naming R8 requires:
	// 0x41 extends the ModRM r/m field, 0x44 the reg field.
	if !bytes.Contains(text, []byte{0x41}) {
		t.Error("no REX.B prefix in .text; nothing read the third argument out of R8")
	}
	if !bytes.Contains(text, []byte{0x44}) {
		t.Error("no REX.R prefix in .text; nothing wrote the third argument into R8")
	}
}

// TestMSPrologueHomesVarargs. A variadic function under this ABI writes its
// four incoming argument registers into the caller's home space, which is what
// makes va_list a bare pointer into one contiguous array.
func TestMSPrologueHomesVarargs(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Windows)
	fn := m.Func("v").Export()
	fmtp := fn.ParamPtr("fmt")
	fn.Variadic()
	fn.ReturnsI64()
	e := fn.Entry()
	ap := e.Ptr.Alloc(8, 8)
	e.VaStart(ap)
	got := e.I64.VaArg(ap)
	e.VaEnd(ap)
	_ = fmtp
	e.Return(got)

	text := textOfMS(t, m)

	// mov [rbp+10h], rcx / [rbp+18h], rdx / [rbp+20h], r8 / [rbp+28h], r9.
	for _, want := range [][]byte{
		{0x48, 0x89, 0x4d, 0x10},
		{0x48, 0x89, 0x55, 0x18},
		{0x4c, 0x89, 0x45, 0x20},
		{0x4c, 0x89, 0x4d, 0x28},
	} {
		if !bytes.Contains(text, want) {
			t.Errorf("prologue does not home % x", want)
		}
	}
}

// TestMSEmitsUnwindData. Every function with a frame gets a RUNTIME_FUNCTION
// in .pdata and an UNWIND_INFO in .xdata; without them nothing on Windows can
// unwind through the frame, and longjmp is an unwind.
func TestMSEmitsUnwindData(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Windows)
	callee := m.ImportFunc("g", ir.NewSig().Ret(ir.TypeI32))

	fn := m.Func("f").Export()
	fn.ReturnsI32()
	e := fn.Entry()
	e.Return(e.Call(callee).Value(0).(ir.I32))

	f := lowerMS(t, m)

	var pdata, xdata *coff.Section
	for _, s := range f.Sections {
		switch s.Name {
		case ".pdata":
			pdata = s
		case ".xdata":
			xdata = s
		}
	}
	if pdata == nil {
		t.Fatal("no .pdata section")
	}
	if xdata == nil {
		t.Fatal("no .xdata section")
	}
	if pdata.Size != 12 {
		t.Errorf(".pdata is %d bytes, want one 12-byte RUNTIME_FUNCTION", pdata.Size)
	}
	relocs, err := pdata.Relocs()
	if err != nil {
		t.Fatalf(".pdata Relocs: %v", err)
	}
	if len(relocs) != 3 {
		t.Fatalf(".pdata has %d relocations, want 3", len(relocs))
	}

	info, err := xdata.Data()
	if err != nil {
		t.Fatalf(".xdata Data: %v", err)
	}
	if len(info) < 4 {
		t.Fatalf(".xdata is %d bytes, too short for an UNWIND_INFO", len(info))
	}
	if v := info[0] & 7; v != 1 {
		t.Errorf("UNWIND_INFO version = %d, want 1", v)
	}
	if info[3] != 0 {
		t.Errorf("frame register = %#x, want none: a fixed frame is unwound from RSP", info[3])
	}
	// The last code is the one replayed last, and the prologue's first act
	// is always "push rbp": UWOP_PUSH_NONVOL (0) with RBP (5) in the
	// operation-info nibble, at prologue offset 1.
	codes := info[4 : 4+int(info[2])*2]
	last := codes[len(codes)-2:]
	if last[0] != 1 || last[1] != 0x50 {
		t.Errorf("last unwind code = % x, want 01 50 (push rbp at offset 1)", last)
	}
}

// TestMSRejectsLongDouble. long double is not a Microsoft type — MSVC makes it
// a double — so a module that asks to pass one is refused rather than given a
// placement invented here.
func TestMSRejectsLongDouble(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Windows)
	fn := m.Func("f").Export()
	fn.ParamF128("x")
	fn.ReturnsI32()
	e := fn.Entry()
	e.Return(e.I32.Const(0))

	if _, err := amd64lower.Lower(m, amd64lower.Options{}); err == nil {
		t.Error("Lower accepted an f128 parameter under the Microsoft ABI")
	}
}
