package amd64_test

// §5's comdat attribute: a definition the linker keeps once however many
// objects carry it. A C++ frontend emits every inline function, virtual
// table and template instance this way, so a module with two comdat items
// has to come out as two groups -- and on Windows, a comdat function's
// unwind records have to follow it out of the link.

import (
	"bytes"
	"os"
	"path/filepath"
	"testing"

	elfcore "github.com/vertex-language/elf"
	"github.com/vertex-language/pe/coff"

	amd64pe "github.com/vertex-language/amd64/obj/pe"

	"github.com/vertex-language/ir"
	amd64lower "github.com/vertex-language/ir/lower/amd64"
)

func comdatModule(t *testing.T, target ir.Target) *ir.Module {
	m := ir.NewModule("t", target)
	m.Global("vtable", ir.RO, ir.Array(1, ir.StorePtr.FType())).Export().Comdat().Init(ir.List(ir.Lit(ir.Int(0))))

	inl := m.Func("inline_f").Export().Comdat()
	inl.ReturnsI32()
	e := inl.Entry()
	e.Return(e.I32.Const(7))

	main := m.Func("main").Export()
	main.ReturnsI32()
	b := main.Entry()
	// A call, so that main has a frame and the Windows object has unwind
	// records for something other than the comdat function.
	r := b.Call(inl)
	b.Return(r.Value(0))
	return m
}

func TestComdatELF(t *testing.T) {
	f := lowerFile(t, comdatModule(t, ir.X86_64Linux))
	groups, err := f.Groups()
	if err != nil {
		t.Fatalf("Groups: %v", err)
	}
	keys := map[string]bool{}
	for _, g := range groups {
		if g.Flags&elfcore.GRP_COMDAT == 0 {
			t.Errorf("group %q is not COMDAT", g.Key())
		}
		keys[g.Key()] = true
	}
	if !keys["inline_f"] || !keys["vtable"] {
		t.Errorf("groups keyed on %v, want inline_f and vtable", keys)
	}
}

func TestComdatCOFF(t *testing.T) {
	m := comdatModule(t, ir.X86_64Windows)
	o, err := amd64lower.Lower(m, amd64lower.Options{})
	if err != nil {
		t.Fatalf("Lower: %v", err)
	}
	var buf bytes.Buffer
	if err := amd64pe.Write(&buf, o); err != nil {
		t.Fatalf("pe.Write: %v", err)
	}
	path := filepath.Join(t.TempDir(), "t.obj")
	if err := os.WriteFile(path, buf.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}
	f, err := coff.Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer f.Close()

	leaders := map[string]int{}
	associated := 0
	for _, s := range f.Sections {
		c, err := s.Comdat()
		if err != nil {
			t.Fatalf("%s: %v", s.Name, err)
		}
		if c == nil {
			continue
		}
		if c.Leader != nil {
			leaders[c.Leader.Name]++
		}
		if c.Associated != nil {
			associated++
		}
	}
	if leaders["inline_f"] != 1 || leaders["vtable"] != 1 {
		t.Errorf("COMDAT sections elected on %v, want one each on inline_f and vtable", leaders)
	}
	// inline_f is a leaf and gets no unwind records; main is not comdat.
	// So nothing is associative here, and that is the claim: an ordinary
	// function's records stay in the shared .pdata.
	if associated != 0 {
		t.Errorf("%d associative sections, want none for a leaf comdat function", associated)
	}
}
