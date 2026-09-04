package amd64

// This backend's half of §5: the layout table, the three sections, and how an
// address is spelled. The walk itself is in lower/globals, which three
// backends share — what an aggregate's braces mean is the IR's shape and not
// this architecture's.

import (
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
	var kind amd64asm.SectionKind
	switch k {
	case globals.ROData:
		kind = amd64asm.ROData
	case globals.BSS:
		kind = amd64asm.BSS
	default:
		kind = amd64asm.Data
	}
	return globalSection{t.am.Section(kind)}
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
// It is the ABI that decides, not the container, because the container is
// chosen after lowering and the sequence at the use site has to match the
// storage. ELF's four models and Mach-O's are a different question with
// different relocations, so a module that is not Microsoft's gets nil and
// globals.Lower refuses the declaration.
func (t globalTarget) TLSSection() globals.Section {
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
