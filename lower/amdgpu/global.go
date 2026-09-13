package amdgpu

// This backend's half of §5: the layout table, the sections, and how an
// address is spelled. The walk itself is lower/globals.
//
// Device globals are ordinary ELF sections in the code object — .rodata,
// .data, .bss — that the loader maps into device memory, and a kernel
// reaches one through s_getpc_b64 and a pair of rel32 halves. Workgroup
// storage is different: LDS is not a section at all but a size in the
// kernel descriptor, and a shared global is an offset into that size.
// SharedSection hands the walk a section that measures and remembers
// offsets and keeps no bytes, since a shared global has no initializer
// to keep.

import (
	"fmt"

	amdgpuasm "github.com/vertex-language/amdgpu"
	"github.com/vertex-language/amdgpu/operand"

	"github.com/vertex-language/ir"
	"github.com/vertex-language/ir/lower/globals"
)

type globalTarget struct {
	layout
	l *lowerer
}

// layout is this target's answers to globals.Layout.
type layout struct{}

func (layout) SizeAlign(t ir.FType) (uint64, uint64, error) { return sizeAlign(t) }
func (layout) FieldOffsets(t *ir.Type) ([]uint64, error)    { return fieldOffsets(t) }

func (globalTarget) PtrBytes() uint64 { return 8 }

func (t globalTarget) Section(k globals.Kind) globals.Section {
	return globalSection{t.l.am.Section(sectionKind(k))}
}

func sectionKind(k globals.Kind) amdgpuasm.SectionKind {
	switch k {
	case globals.ROData:
		return amdgpuasm.ROData
	case globals.RelROData:
		return amdgpuasm.RelROData
	case globals.BSS:
		return amdgpuasm.BSS
	}
	return amdgpuasm.Data
}

// TLSSection is nil: a device has no threads in the sense tls means.
func (globalTarget) TLSSection(globals.Kind) globals.Section { return nil }

// SharedSection is LDS: offsets, no bytes.
func (t globalTarget) SharedSection() globals.Section {
	if t.l.lds == nil {
		t.l.lds = map[string]uint32{}
	}
	return &ldsSection{l: t.l}
}

type globalSection struct{ s *amdgpuasm.Section }

func (g globalSection) Align(n int)       { g.s.Align(n) }
func (g globalSection) Close(name string) { g.s.EndLabel(name) }
func (g globalSection) Byte(v byte)       { g.s.Byte(v) }
func (g globalSection) Long(v uint32)     { g.s.Long(v) }
func (g globalSection) Quad(v uint64)     { g.s.Quad(v) }
func (g globalSection) Ascii(s string)    { g.s.Ascii(s) }
func (g globalSection) Zero(n int)        { g.s.Zero(n) }

func (g globalSection) Object(name string, b globals.Binding) {
	binding := amdgpuasm.Local
	switch b {
	case globals.Weak:
		binding = amdgpuasm.Weak
	case globals.Global:
		binding = amdgpuasm.Global
	}
	g.s.Label(name, binding, amdgpuasm.ObjectSym)
}

func (g globalSection) PtrTo(sym string, addend int64) error {
	g.s.Ref(operand.Ref(sym, amdgpuasm.RefAbs64).WithAddend(addend))
	return nil
}

func (g globalSection) Delta(to, from string, addend int64) error {
	return fmt.Errorf("a symbol difference between %s and %s needs a relocation this backend does not emit yet", to, from)
}

// ldsSection lays workgroup storage out: each object at its alignment,
// one after another, the total being what every kernel asks for.
type ldsSection struct {
	l    *lowerer
	name string
}

func (s *ldsSection) Align(n int) {
	if n > 1 {
		s.l.ldsSize = alignUp32(s.l.ldsSize, uint32(n))
	}
}

func (s *ldsSection) Object(name string, _ globals.Binding) {
	s.name = name
	s.l.lds[name] = s.l.ldsSize
}

func (s *ldsSection) Close(string)     {}
func (s *ldsSection) Byte(byte)        { s.l.ldsSize++ }
func (s *ldsSection) Long(uint32)      { s.l.ldsSize += 4 }
func (s *ldsSection) Quad(uint64)      { s.l.ldsSize += 8 }
func (s *ldsSection) Ascii(str string) { s.l.ldsSize += uint32(len(str)) }
func (s *ldsSection) Zero(n int)       { s.l.ldsSize += uint32(n) }
func (s *ldsSection) PtrTo(sym string, _ int64) error {
	return fmt.Errorf("@%s: workgroup storage has no initializer to hold the address of %s", s.name, sym)
}
func (s *ldsSection) Delta(to, from string, _ int64) error {
	return fmt.Errorf("@%s: workgroup storage has no initializer to hold %s-%s", s.name, to, from)
}
