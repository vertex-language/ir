package arm64

// This backend's half of §5: the layout table, the three sections, and how an
// address is spelled. The walk itself is in lower/globals, which three
// backends share — what an aggregate's braces mean is the IR's shape and not
// this architecture's.

import (
	"fmt"

	arm64asm "github.com/vertex-language/arm64"

	"github.com/vertex-language/ir"
	"github.com/vertex-language/ir/lower/globals"
)

// globalTarget adapts this package's assembler to the shared walk.
type globalTarget struct {
	layout
	am *arm64asm.Module
}

func (t globalTarget) Section(k globals.Kind) globals.Section {
	var kind arm64asm.SectionKind
	switch k {
	case globals.ROData:
		kind = arm64asm.ROData
	case globals.BSS:
		kind = arm64asm.BSS
	default:
		kind = arm64asm.Data
	}
	return globalSection{t.am.Section(kind)}
}

// layout is this target's answers to globals.Layout: the shape of a type,
// which §2's symbolic constants need and which has nothing to do with the
// sections globalTarget also carries. isel resolves a sizeof with one of
// these and no assembler at all.
type layout struct{}

func (layout) SizeAlign(t ir.FType) (uint64, uint64, error) { return sizeAlign(t) }

func (layout) FieldOffsets(t *ir.Type) ([]uint64, error) {
	offsets, _, _, err := structLayout(t, 0)
	return offsets, err
}

// PtrBytes is eight: an AArch64 address is a doubleword.
func (globalTarget) PtrBytes() uint64 { return 8 }

type globalSection struct{ s *arm64asm.Section }

func (g globalSection) Align(n int)       { g.s.Align(n) }
func (g globalSection) Close(name string) { g.s.EndLabel(name) }
func (g globalSection) Byte(v byte)       { g.s.Byte(v) }
func (g globalSection) Long(v uint32)     { g.s.Long(v) }
func (g globalSection) Quad(v uint64)     { g.s.Quad(v) }
func (g globalSection) Ascii(s string)    { g.s.Ascii(s) }
func (g globalSection) Zero(n int)        { g.s.Zero(n) }

func (g globalSection) Object(name string, b globals.Binding) {
	var binding arm64asm.Binding
	switch b {
	case globals.Weak:
		binding = arm64asm.Weak
	case globals.Global:
		binding = arm64asm.Global
	default:
		binding = arm64asm.Local
	}
	g.s.Label(name, binding, arm64asm.ObjectSym)
}

// PtrTo places an eight-byte absolute reference.
//
// No addend. arm64/obj's Ref takes a name and a kind and nothing else, so an
// offset from a symbol is expressible in the format and not in this API —
// refused rather than silently dropped.
func (g globalSection) PtrTo(sym string, addend int64) error {
	if addend != 0 {
		return fmt.Errorf("an addend on an address initializer is not emitted yet")
	}
	g.s.Ref(sym, arm64asm.RefAbs64)
	return nil
}

// lowerGlobals writes every global definition in m into am.
func lowerGlobals(am *arm64asm.Module, m *ir.Module) error {
	return globals.Lower(globalTarget{am: am}, m)
}

// funcBinding is a function's binding in this assembler's spelling. Functions
// go through the same translation globals do — the walk names a binding, this
// adapter spells it — so that a static function is a local symbol and an
// inline definition emitted in several units is a weak one.
func funcBinding(fn *ir.Func) arm64asm.Binding {
	switch globals.FuncBinding(fn) {
	case globals.Weak:
		return arm64asm.Weak
	case globals.Local:
		return arm64asm.Local
	}
	return arm64asm.Global
}

// TLSSection is nil: this target has no TLS model yet, so globals.Lower
// refuses a thread-local rather than emitting storage nothing can reach.
// See lower.go's list of what §D3 is still missing here.
func (t globalTarget) TLSSection() globals.Section { return nil }
