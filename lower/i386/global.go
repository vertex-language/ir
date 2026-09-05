package i386

// This backend's half of §5: the layout table, the three sections, and how an
// address is spelled. The walk itself is in lower/globals, which three
// backends share — what an aggregate's braces mean is the IR's shape and not
// this architecture's.

import (
	"fmt"
	"math"

	i386asm "github.com/vertex-language/i386"

	"github.com/vertex-language/ir"
	"github.com/vertex-language/ir/lower/globals"
)

// globalTarget adapts this package's assembler to the shared walk.
type globalTarget struct {
	layout
	am *i386asm.Module
}

func (t globalTarget) Section(k globals.Kind) globals.Section {
	var kind i386asm.SectionKind
	switch k {
	case globals.ROData:
		kind = i386asm.ROData
	case globals.RelROData:
		kind = i386asm.RelROData
	case globals.BSS:
		kind = i386asm.BSS
	default:
		kind = i386asm.Data
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

// PtrBytes is four, which is the whole difference this target makes to §5: an
// address initializer fills a long here and a quad on the other two.
func (globalTarget) PtrBytes() uint64 { return 4 }

type globalSection struct{ s *i386asm.Section }

func (g globalSection) Align(n int)       { g.s.Align(n) }
func (g globalSection) Close(name string) { g.s.EndLabel(name) }
func (g globalSection) Byte(v byte)       { g.s.Byte(v) }
func (g globalSection) Long(v uint32)     { g.s.Long(v) }
func (g globalSection) Quad(v uint64)     { g.s.Quad(v) }
func (g globalSection) Ascii(s string)    { g.s.Ascii(s) }
func (g globalSection) Zero(n int)        { g.s.Zero(n) }

func (g globalSection) Object(name string, b globals.Binding) {
	var binding i386asm.Binding
	switch b {
	case globals.Weak:
		binding = i386asm.Weak
	case globals.Global:
		binding = i386asm.Global
	default:
		binding = i386asm.Local
	}
	g.s.Label(name, binding, i386asm.Object)
}

// PtrTo places a four-byte absolute reference.
//
// The addend is an int32 here because the field it lands in is four bytes,
// which is the same reason the pointer is: an offset that does not fit is
// refused rather than truncated into a different address.
func (g globalSection) PtrTo(sym string, addend int64) error {
	if addend < math.MinInt32 || addend > math.MaxInt32 {
		return fmt.Errorf("an addend of %d does not fit a 32-bit address field", addend)
	}
	g.s.Ref(i386asm.Ref(sym, i386asm.RefAbs32).Plus(int32(addend)))
	return nil
}

// lowerGlobals writes every global definition in m into am.
func lowerGlobals(am *i386asm.Module, m *ir.Module) error {
	return globals.Lower(globalTarget{am: am}, m)
}

// funcBinding is a function's binding in this assembler's spelling. Functions
// go through the same translation globals do — the walk names a binding, this
// adapter spells it — so that a static function is a local symbol and an
// inline definition emitted in several units is a weak one.
func funcBinding(fn *ir.Func) i386asm.Binding {
	switch globals.FuncBinding(fn) {
	case globals.Weak:
		return i386asm.Weak
	case globals.Local:
		return i386asm.Local
	}
	return i386asm.Global
}

// TLSSection is nil: this target has no TLS model yet, so globals.Lower
// refuses a thread-local rather than emitting storage nothing can reach.
// See lower.go's list of what §D3 is still missing here.
func (t globalTarget) TLSSection(globals.Kind) globals.Section { return nil }
