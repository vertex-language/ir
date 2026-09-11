package amd64

// This backend's half of §5: the layout table, the three sections, and how an
// address is spelled. The walk itself is in lower/globals, which three
// backends share — what an aggregate's braces mean is the IR's shape and not
// this architecture's.

import (
	"fmt"

	amd64asm "github.com/vertex-language/amd64"

	"github.com/vertex-language/ir"
	"github.com/vertex-language/ir/lower/globals"
)

// globalTarget adapts this package's assembler to the shared walk.
type globalTarget struct {
	layout
	am       *amd64asm.Module
	tlsModel tlsModel
}

func (t globalTarget) Section(k globals.Kind) globals.Section {
	return globalSection{t.am.Section(sectionKind(k))}
}

// sectionKind is the assembler's spelling of a global's load-time behaviour.
// It is separate from Section because NamedSection needs the same answer: the
// name decides where a global goes, and the kind still decides what it is.
func sectionKind(k globals.Kind) amd64asm.SectionKind {
	switch k {
	case globals.ROData:
		return amd64asm.ROData
	case globals.RelROData:
		return amd64asm.RelROData
	case globals.BSS:
		return amd64asm.BSS
	}
	return amd64asm.Data
}

// NamedSection places a global in a section the module named -- §5's section
// attribute, and the whole of how an Objective-C image is laid out.
//
// The kind still decides the section's load-time behaviour, because that is
// what the *contents* are: a named BSS section holds a size and no bytes
// however it is spelled. The name decides where the bytes go, and on Mach-O
// it carries the type and attributes too, in as(1)'s own syntax:
//
//	__DATA,__objc_selrefs,literal_pointers,no_dead_strip
//
// The writer reads that; nothing here has to.
func (t globalTarget) NamedSection(name string, k globals.Kind) globals.Section {
	return globalSection{t.am.SectionNamed(name, sectionKind(k))}
}

// ComdatSection places a global in a section of its own that the linker
// keeps once: the assembler's COMDAT section, elected on the global.
func (t globalTarget) ComdatSection(k globals.Kind, leader string) globals.Section {
	return globalSection{t.am.ComdatSection(sectionKind(k).String(), sectionKind(k), leader)}
}

// layout is this target's answers to globals.Layout: the shape of a type,
// which §2's symbolic constants need and which has nothing to do with the
// sections globalTarget also carries. isel resolves a sizeof with one of
// these and no assembler at all.
type layout struct{}

func (layout) SizeAlign(t ir.FType) (uint64, uint64, error) { return sizeAlign(t) }

func (layout) FieldOffsets(t *ir.Type) ([]uint64, error) { return fieldOffsets(t) }

// PtrBytes is eight: an x86-64 address is a quadword.
func (globalTarget) PtrBytes() uint64 { return 8 }

type globalSection struct{ s *amd64asm.Section }

func (g globalSection) Align(n int)       { g.s.Align(n) }
func (g globalSection) Close(name string) { g.s.EndLabel(name) }
func (g globalSection) Byte(v byte)       { g.s.Byte(v) }
func (g globalSection) Long(v uint32)     { g.s.Long(v) }
func (g globalSection) Quad(v uint64)     { g.s.Quad(v) }
func (g globalSection) Ascii(s string)    { g.s.Ascii(s) }
func (g globalSection) Zero(n int)        { g.s.Zero(n) }

func (g globalSection) Object(name string, b globals.Binding) {
	var binding amd64asm.Binding
	switch b {
	case globals.Weak:
		binding = amd64asm.Weak
	case globals.Global:
		binding = amd64asm.Global
	default:
		binding = amd64asm.Local
	}
	g.s.Label(name, binding, amd64asm.ObjectSym)
}

// PtrTo places an eight-byte absolute reference. This assembler's SymRef
// carries an addend, so an offset from a symbol is expressible here.
func (g globalSection) PtrTo(sym string, addend int64) error {
	g.s.Ref(amd64asm.Ref(sym, amd64asm.RefAbs64).Add(addend))
	return nil
}

// Delta places a four-byte field holding to - from. This backend's
// assembler folds a same-section pair through LabelDiff and has no way to
// spell one it cannot fold, so a difference that needs a relocation is
// refused by name rather than dropped.
func (g globalSection) Delta(to, from string, addend int64) error {
	return fmt.Errorf("a symbol difference between %s and %s needs a relocation this backend does not emit yet", to, from)
}

// lowerGlobals writes every global definition in m into am.
func lowerGlobals(am *amd64asm.Module, m *ir.Module) error {
	return globals.Lower(globalTarget{am: am, tlsModel: tlsModelFor(m)}, m)
}

// funcBinding is a function's binding in this assembler's spelling. Functions
// go through the same translation globals do — the walk names a binding, this
// adapter spells it — so that a static function is a local symbol and an
// inline definition emitted in several units is a weak one.
func funcBinding(fn *ir.Func) amd64asm.Binding {
	switch globals.FuncBinding(fn) {
	case globals.Weak:
		return amd64asm.Weak
	case globals.Local:
		return amd64asm.Local
	}
	return amd64asm.Global
}

// The PE thread-local section, and why the name has a dollar in it.
//
// COFF orders the pieces of a section by what follows the '$' in the input
// section's name and then drops the suffix, which is how the CRT arranges
// the TLS template: _tls_used declares the bounds in .tls$AAA and .tls$ZZZ,
// and everything a program declares thread-local lands between them in
// .tls$. Naming the section .tls outright puts it outside those bounds, and
// the loader then copies a block that does not contain it.
const peTLSSection = ".tls$"

// TLSSection is the thread-local template, under the one model this backend
// implements: PE's static TLS.
//
// The kind is ignored. PE has one template section and the loader copies
// all of it, so a zeroed thread-local occupies it like any other rather
// than costing address space alone.
//
// It is the ABI that decides, not the container, because the container is
// chosen after lowering and the sequence at the use site has to match the
// storage. ELF's four models and Mach-O's are a different question with
// different relocations, so a module that is not Microsoft's gets nil and
// globals.Lower refuses the declaration.
func (t globalTarget) TLSSection(globals.Kind) globals.Section {
	if t.tlsModel != tlsPE {
		return nil
	}
	return globalSection{t.am.SectionNamed(peTLSSection, amd64asm.Data)}
}

// tlsModel is which thread-local model a module gets, which is a fact about
// its ABI.
type tlsModel uint8

const (
	tlsNone tlsModel = iota
	tlsPE
)

func tlsModelFor(m *ir.Module) tlsModel {
	if m.Layout().ABI == abiMS {
		return tlsPE
	}
	return tlsNone
}

// needsTLSIndex reports whether this module reaches a thread-local, which is
// what makes the CRT's _tls_index a symbol it references.
//
// Declaring one is enough — a module that never reads it still emits the
// storage — and so is importing one, which is how a thread-local defined in
// another unit is reached. An import carries no domain, having no storage
// here to put anywhere, so what marks it is the model attribute instead.
func needsTLSIndex(m *ir.Module) bool {
	if tlsModelFor(m) != tlsPE {
		return false
	}
	for _, g := range m.Globals() {
		if g.Domain() == ir.TLS {
			return true
		}
	}
	for _, g := range m.GlobalImports() {
		if g.TLSModelAttr() != 0 {
			return true
		}
	}
	return false
}
