package amd64_test

// Milestone 10: a module's data, and the first relocation.
//
// Everything before this milestone lowered to a .text section and
// nothing else, which made a module with globals lower to an object that
// silently did not contain them. These tests read the data back out of a
// written ELF the way the .text tests read instructions out of one — with
// elfobj.NewFile, not by trusting the writer.

import (
	"bytes"
	"testing"

	elfcore "github.com/vertex-language/elf"
	elfobj "github.com/vertex-language/elf/obj"

	amd64elf "github.com/vertex-language/amd64/obj/elf"

	"github.com/vertex-language/ir"
	amd64lower "github.com/vertex-language/ir/lower/amd64"
	"github.com/vertex-language/ir/verify"
)

// lowerFile lowers m, writes it as ELF, and reads the whole object back.
func lowerFile(t *testing.T, m *ir.Module) *elfobj.File {
	t.Helper()

	if err := verify.Module(m); err != nil {
		t.Fatalf("verify.Module: %v", err)
	}
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
	return f
}

// sectionData is one section's bytes, or a failure naming what was
// missing. A .bss has none by definition, so this is for the rest.
func sectionData(t *testing.T, f *elfobj.File, name string) []byte {
	t.Helper()

	s := f.Section(name)
	if s == nil {
		t.Fatalf("%s section not found", name)
	}
	d, err := s.Data()
	if err != nil {
		t.Fatalf("%s Data: %v", name, err)
	}
	return d
}

// symbol looks one up by name.
func symbol(t *testing.T, f *elfobj.File, name string) *elfobj.Symbol {
	t.Helper()

	syms, err := f.Symbols()
	if err != nil {
		t.Fatalf("Symbols: %v", err)
	}
	for _, s := range syms {
		if s.Name == name {
			return s
		}
	}
	t.Fatalf("symbol %q not found", name)
	return nil
}

// §5's three domains against the sections they land in, and the one
// split that is not a domain's: an rw global whose initializer is all
// zeroes belongs in .bss, where it costs address space and no file
// bytes, while one with content belongs in .data.
func TestLowerGlobalSections(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	m.Global("counter", ir.RW, ir.StoreI64.FType()).Export().Init(ir.Lit(ir.Int(7)))
	m.Global("blank", ir.RW, ir.StoreI32.FType()).Export()
	m.Global("msg", ir.RO, ir.Array(6, ir.StoreI8.FType())).Export().Init(ir.Str("hello"))

	f := lowerFile(t, m)

	if got, want := sectionData(t, f, ".data"), []byte{7, 0, 0, 0, 0, 0, 0, 0}; !bytes.Equal(got, want) {
		t.Errorf(".data = % x, want % x", got, want)
	}
	// "hello" and the NUL the declared length leaves room for. The array
	// is the storage; the string fills the front of it.
	if got, want := sectionData(t, f, ".rodata"), []byte("hello\x00"); !bytes.Equal(got, want) {
		t.Errorf(".rodata = %q, want %q", got, want)
	}

	// @blank has no bytes anywhere — that is what .bss means — but it has
	// a symbol, and the symbol has its size.
	if b := symbol(t, f, "blank"); b.Size != 4 {
		t.Errorf("blank.Size = %d, want 4", b.Size)
	}
	if bss := f.Section(".bss"); bss == nil {
		t.Error(".bss section not found")
	}

	for _, tc := range []struct {
		name string
		size uint64
	}{
		{"counter", 8}, {"blank", 4}, {"msg", 6},
	} {
		s := symbol(t, f, tc.name)
		if s.Type != elfcore.STT_OBJECT {
			t.Errorf("%s.Type = %v, want STT_OBJECT", tc.name, s.Type)
		}
		if s.Bind != elfcore.STB_GLOBAL {
			t.Errorf("%s.Bind = %v, want STB_GLOBAL", tc.name, s.Bind)
		}
		if s.Size != tc.size {
			t.Errorf("%s.Size = %d, want %d", tc.name, s.Size, tc.size)
		}
	}
}

// Linkage and binding are two attributes in VIR and one in an object
// file, so this pins how the two fold together.
func TestLowerGlobalBinding(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	m.Global("shared", ir.RW, ir.StoreI32.FType()).Export().Init(ir.Lit(ir.Int(1)))
	m.Global("mine", ir.RW, ir.StoreI32.FType()).Internal().Init(ir.Lit(ir.Int(2)))
	m.Global("maybe", ir.RW, ir.StoreI32.FType()).Export().Weak().Init(ir.Lit(ir.Int(3)))

	f := lowerFile(t, m)

	for _, tc := range []struct {
		name string
		bind elfcore.SymBind
	}{
		{"shared", elfcore.STB_GLOBAL},
		{"mine", elfcore.STB_LOCAL},
		{"maybe", elfcore.STB_WEAK},
	} {
		if got := symbol(t, f, tc.name).Bind; got != tc.bind {
			t.Errorf("%s.Bind = %v, want %v", tc.name, got, tc.bind)
		}
	}
}

// ptr.getaddr, which is this repo's first relocation.
//
// RIP-relative and not absolute: the small code model puts every symbol
// within two gigabytes of the instruction naming it, so a signed 32-bit
// displacement from the next instruction reaches all of them and the
// text section needs no load-time patching to be shared.
func TestLowerGetAddr(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	counter := m.Global("counter", ir.RW, ir.StoreI64.FType()).Export().Init(ir.Lit(ir.Int(0)))

	fn := m.Func("addr_of_counter").Export()
	fn.ReturnsPtr()
	entry := fn.Entry()
	entry.Return(entry.Ptr.GetAddr(counter))

	f := lowerFile(t, m)

	want := []byte{
		0x48, 0x8d, 0x05, 0x00, 0x00, 0x00, 0x00, // lea rax, [rip+0]
		0xc3, // ret
	}
	if got := sectionData(t, f, ".text"); !bytes.Equal(got, want) {
		t.Errorf(".text = % x, want % x", got, want)
	}

	// The four zero bytes above are the hole. This is what fills it.
	text := f.Section(".text")
	relocs, err := text.Relocs()
	if err != nil {
		t.Fatalf(".text Relocs: %v", err)
	}
	if len(relocs) != 1 {
		t.Fatalf("found %d relocations in .text, want 1", len(relocs))
	}
	r := relocs[0]
	if r.Sym == nil || r.Sym.Name != "counter" {
		t.Errorf("relocation names %v, want counter", r.Sym)
	}
	if elfcore.RelocX86_64(r.Type) != elfcore.R_X86_64_PC32 {
		t.Errorf("relocation type = %v, want R_X86_64_PC32", elfcore.RelocX86_64(r.Type))
	}
	// The displacement is measured from the end of the instruction and
	// the hole sits four bytes before it, so the addend is -4. Getting
	// this wrong points the lea four bytes past its symbol.
	if r.Addend != -4 {
		t.Errorf("relocation addend = %d, want -4", r.Addend)
	}
	if r.Offset != 3 {
		t.Errorf("relocation offset = %d, want 3 — the lea's displacement field", r.Offset)
	}
}

// The address of an imported symbol is the same instruction: which
// object ends up holding it is the linker's question. What Lower has to
// get right is declaring it, so the reference has something to name.
func TestLowerGetAddrOfImport(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	errno := m.ImportGlobal("errno", ir.StoreI32.FType())

	fn := m.Func("addr_of_errno").Export()
	fn.ReturnsPtr()
	entry := fn.Entry()
	entry.Return(entry.Ptr.GetAddr(errno))

	f := lowerFile(t, m)

	if s := symbol(t, f, "errno"); !s.Undefined() {
		t.Errorf("errno names section %d; an import is defined elsewhere and must be undefined here", s.Shndx)
	}

	relocs, err := f.Section(".text").Relocs()
	if err != nil {
		t.Fatalf(".text Relocs: %v", err)
	}
	if len(relocs) != 1 || relocs[0].Sym == nil || relocs[0].Sym.Name != "errno" {
		t.Fatalf("want one relocation naming errno, got %v", relocs)
	}
}

// A global whose initializer is another symbol's address: the data-side
// twin of ptr.getaddr, and absolute rather than relative because a
// global holds an address, not a distance from itself.
func TestLowerRelocInitializer(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	target := m.Global("target", ir.RW, ir.StoreI32.FType()).Export().Init(ir.Lit(ir.Int(1)))
	m.Global("pointer", ir.RW, ir.StoreI64.FType()).Export().Init(ir.RelocInit(target))

	f := lowerFile(t, m)

	relocs, err := f.Section(".data").Relocs()
	if err != nil {
		t.Fatalf(".data Relocs: %v", err)
	}
	if len(relocs) != 1 {
		t.Fatalf("found %d relocations in .data, want 1", len(relocs))
	}
	if r := relocs[0]; elfcore.RelocX86_64(r.Type) != elfcore.R_X86_64_64 {
		t.Errorf("relocation type = %v, want R_X86_64_64", elfcore.RelocX86_64(r.Type))
	}
}

// What §5 has that this milestone does not emit, each refused by name.
func TestLowerRejectsUnsupportedGlobals(t *testing.T) {
	for _, tc := range []struct {
		name  string
		build func(m *ir.Module)
	}{
		{"tls", func(m *ir.Module) {
			m.Global("t", ir.TLS, ir.StoreI32.FType()).Export()
		}},
		{"f80 literal", func(m *ir.Module) {
			m.Global("f", ir.RW, ir.StoreF80.FType()).Export().
				Init(ir.Lit(ir.Float(1.5)))
		}},
		{"float literal in an integer", func(m *ir.Module) {
			m.Global("f", ir.RW, ir.StoreI32.FType()).Export().
				Init(ir.Lit(ir.Float(1.5)))
		}},
		{"section", func(m *ir.Module) {
			m.Global("s", ir.RW, ir.StoreI32.FType()).Export().
				Section(".mine").Init(ir.Lit(ir.Int(1)))
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			m := ir.NewModule("t", ir.X86_64Linux)
			tc.build(m)
			if _, err := amd64lower.Lower(m, amd64lower.Options{}); err == nil {
				t.Errorf("Lower should refuse a %s global", tc.name)
			}
		})
	}
}

// A float literal, as the bit pattern its format gives it. f32 and f64 are
// the two widths a value here can have; f128 is the one a value cannot, and
// is written all the same because §5's storage widths are not §17's register
// types — a module may hold a __float128 it only ever memcpys.
func TestLowerFloatGlobals(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	m.Global("single", ir.RW, ir.StoreF32.FType()).Export().
		Init(ir.Lit(ir.Float(1.5)))
	m.Global("double", ir.RW, ir.StoreF64.FType()).Export().
		Init(ir.Lit(ir.Float(-2.75)))
	m.Global("quad", ir.RW, ir.StoreF128.FType()).Export().
		Init(ir.Lit(ir.Float(1.5)))

	f := lowerFile(t, m)
	data := sectionData(t, f, ".data")

	for _, tc := range []struct {
		name string
		want []byte
	}{
		{"single", []byte{0x00, 0x00, 0xc0, 0x3f}},
		{"double", []byte{0, 0, 0, 0, 0, 0, 0x06, 0xc0}},
		// binary128 1.5: the leading mantissa bit at the top of 112,
		// and 16383 in the exponent.
		{"quad", []byte{
			0, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0x80, 0xff, 0x3f,
		}},
	} {
		sym := symbol(t, f, tc.name)
		got := data[sym.Value : sym.Value+uint64(len(tc.want))]
		if !bytes.Equal(got, tc.want) {
			t.Errorf("%s = % x, want % x", tc.name, got, tc.want)
		}
	}
}

// ── milestone 17: the ABI layout table ────────────────────────────────

// A struct laid out by the psABI's rules: each field at the next offset
// satisfying its own alignment, and the whole padded to a multiple of
// the struct's alignment so an array of it keeps every element aligned.
//
// { i8, i64, i32 } is the shape that shows all three effects at once —
// seven bytes of padding after the byte, and four at the end.
func TestLowerStructInitializer(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	rec := m.Struct("rec").
		Field("tag", ir.StoreI8.FType()).
		Field("big", ir.StoreI64.FType()).
		Field("n", ir.StoreI32.FType())
	m.Global("r", ir.RW, rec.FType()).Export().
		Init(ir.List(ir.Lit(ir.Int(1)), ir.Lit(ir.Int(2)), ir.Lit(ir.Int(3))))

	f := lowerFile(t, m)

	want := []byte{
		0x01, 0, 0, 0, 0, 0, 0, 0, // tag at 0, then padding to 8
		0x02, 0, 0, 0, 0, 0, 0, 0, // big at 8
		0x03, 0, 0, 0 /*      */, 0, 0, 0, 0, // n at 16, then the tail
	}
	if got := sectionData(t, f, ".data"); !bytes.Equal(got, want) {
		t.Errorf(".data = % x, want % x", got, want)
	}
	if got := symbol(t, f, "r").Size; got != 24 {
		t.Errorf("r.Size = %d, want 24", got)
	}
}

// A named-field initializer names some fields and leaves the rest zero,
// which is what the form is for.
func TestLowerPartialStructInitializer(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	rec := m.Struct("rec").
		Field("tag", ir.StoreI8.FType()).
		Field("big", ir.StoreI64.FType()).
		Field("n", ir.StoreI32.FType())
	m.Global("p", ir.RW, rec.FType()).Export().
		Init(ir.Fields(ir.FieldVal{Name: "n", Init: ir.Lit(ir.Int(9))}))

	f := lowerFile(t, m)

	want := make([]byte, 24)
	want[16] = 9 // n's offset, and nothing else set
	if got := sectionData(t, f, ".data"); !bytes.Equal(got, want) {
		t.Errorf(".data = % x, want % x", got, want)
	}
}

// An array needs no padding between elements: an element's own size
// already includes whatever keeps the next one aligned.
func TestLowerArrayInitializer(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	m.Global("a", ir.RW, ir.Array(3, ir.StoreI16.FType())).Export().
		Init(ir.List(ir.Lit(ir.Int(0x1111)), ir.Lit(ir.Int(0x2222)), ir.Lit(ir.Int(0x3333))))

	f := lowerFile(t, m)

	want := []byte{0x11, 0x11, 0x22, 0x22, 0x33, 0x33}
	if got := sectionData(t, f, ".data"); !bytes.Equal(got, want) {
		t.Errorf(".data = % x, want % x", got, want)
	}
}

// A global is aligned to its type whether or not the declaration says
// so. §5's align attribute raises a global's alignment; it does not
// supply the one the type already requires, and emitting nothing would
// put a struct wanting eight bytes wherever the previous global ended.
func TestLowerGlobalAlignment(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	// Three bytes, so anything following it would land at offset 3.
	m.Global("odd", ir.RW, ir.Array(3, ir.StoreI8.FType())).Export().
		Init(ir.Str("ab"))
	m.Global("wide", ir.RW, ir.StoreI64.FType()).Export().Init(ir.Lit(ir.Int(0x7f)))

	f := lowerFile(t, m)

	wide := symbol(t, f, "wide")
	if wide.Value%8 != 0 {
		t.Errorf("wide is at offset %d, which is not 8-aligned", wide.Value)
	}
	if got := sectionData(t, f, ".data"); len(got) != 16 {
		t.Errorf(".data is %d bytes, want 16 — three, padded to eight, plus eight", len(got))
	}
}

// ptr.alloc of a named type, which needed exactly this table: a type
// states no size and no alignment of its own, and both are the ABI's to
// compute from it.
func TestLowerAllocType(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	rec := m.Struct("rec").
		Field("tag", ir.StoreI8.FType()).
		Field("big", ir.StoreI64.FType()).
		Field("n", ir.StoreI32.FType())

	fn := m.Func("f").Export()
	fn.ReturnsPtr()
	entry := fn.Entry()
	entry.Return(entry.Ptr.AllocType(rec))

	f := lowerFile(t, m)

	want := []byte{
		0x55,             // push rbp
		0x48, 0x8b, 0xec, // mov rbp, rsp
		0x48, 0x81, 0xec, 0x20, 0x00, 0x00, 0x00, // sub rsp, 32   (24, rounded)
		0x48, 0x8d, 0x45, 0xe8, // lea rax, [rbp-24]
		0xc9, // leave
		0xc3, // ret
	}
	if got := sectionData(t, f, ".text"); !bytes.Equal(got, want) {
		t.Errorf(".text = % x, want % x", got, want)
	}
}

// A packed struct has no padding between its fields and aligns to one,
// which is the one case where an initializer's bytes are the fields
// concatenated and nothing else.
func TestLowerPackedStructInitializer(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	rec := m.Struct("rec").
		Field("tag", ir.StoreI8.FType()).
		Field("big", ir.StoreI64.FType()).
		Pack()
	m.Global("p", ir.RW, rec.FType()).Export().
		Init(ir.List(ir.Lit(ir.Int(1)), ir.Lit(ir.Int(2))))

	f := lowerFile(t, m)

	want := []byte{
		0x01,                      // tag at 0
		0x02, 0, 0, 0, 0, 0, 0, 0, // big at 1, unaligned and unpadded
	}
	if got := sectionData(t, f, ".data"); !bytes.Equal(got, want) {
		t.Errorf(".data = % x, want % x", got, want)
	}
	if got := symbol(t, f, "p").Size; got != 9 {
		t.Errorf("p.Size = %d, want 9", got)
	}
}

// A union initializer names one member and the rest of the union's
// storage is zero — every member begins at offset zero, which is what a
// union is.
func TestLowerUnionInitializer(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	u := m.Union("u").
		Field("b", ir.StoreI8.FType()).
		Field("q", ir.StoreI64.FType())
	m.Global("v", ir.RW, u.FType()).Export().
		Init(ir.Fields(ir.FieldVal{Name: "b", Init: ir.Lit(ir.Int(0x7f))}))

	f := lowerFile(t, m)

	want := make([]byte, 8)
	want[0] = 0x7f
	if got := sectionData(t, f, ".data"); !bytes.Equal(got, want) {
		t.Errorf(".data = % x, want % x", got, want)
	}
}

// An array of a struct: the element's own tail padding is what keeps the
// second element aligned, so the initializer emits no padding between
// them and the bytes are two whole elements.
func TestLowerArrayOfStructInitializer(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	rec := m.Struct("rec").
		Field("tag", ir.StoreI8.FType()).
		Field("n", ir.StoreI32.FType())
	elem := func(tag, n int64) ir.Init {
		return ir.List(ir.Lit(ir.Int(tag)), ir.Lit(ir.Int(n)))
	}
	m.Global("a", ir.RW, ir.Array(2, rec.FType())).Export().
		Init(ir.List(elem(1, 2), elem(3, 4)))

	f := lowerFile(t, m)

	want := []byte{
		0x01, 0, 0, 0, 0x02, 0, 0, 0,
		0x03, 0, 0, 0, 0x04, 0, 0, 0,
	}
	if got := sectionData(t, f, ".data"); !bytes.Equal(got, want) {
		t.Errorf(".data = % x, want % x", got, want)
	}
}
