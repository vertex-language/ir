package ptx

// This backend's half of §5: the layout table, the state spaces, and how
// bytes become a declaration. The walk itself is lower/globals, which
// every backend shares.
//
// PTX has no sections and no byte stream. A global is a declaration with
// a state space, an alignment, an element type, a length and an
// initializer, so the adapter collects the bytes the walk emits for one
// object and turns them into a .b8 array when the object is closed. An
// object with no non-zero byte gets no initializer at all — PTX zeroes
// .global and .const storage by default, which is what BSS means here.
//
// An address in the bytes is a relocation the walk hands over by symbol,
// and a .b8 array cannot hold one. Such an object is emitted as a .u64
// array instead, with generic(sym)+addend where the walk placed the
// address — which needs every address on an 8-byte boundary and the
// object's size a multiple of 8. A vtable is that shape; a packed struct
// holding a pointer is not, and is refused.

import (
	"fmt"
	"strconv"

	"github.com/vertex-language/ir"
	"github.com/vertex-language/ir/lower/globals"
	"github.com/vertex-language/ptx"
)

type globalTarget struct {
	layout
	l *lowerer
}

// layout is this target's answers to globals.Layout.
type layout struct{}

func (layout) SizeAlign(t ir.FType) (uint64, uint64, error) { return sizeAlign(t) }
func (layout) FieldOffsets(t *ir.Type) ([]uint64, error)   { return fieldOffsets(t) }

func (globalTarget) PtrBytes() uint64 { return 8 }

func (t globalTarget) Section(k globals.Kind) globals.Section {
	space := ptx.Global
	if k == globals.ROData || k == globals.RelROData {
		space = ptx.Const
	}
	return &section{l: t.l, space: space}
}

// TLSSection is nil: a device has no threads in the sense tls means.
func (globalTarget) TLSSection(globals.Kind) globals.Section { return nil }

// SharedSection is workgroup storage: .shared, never initialized.
func (t globalTarget) SharedSection() globals.Section {
	return &section{l: t.l, space: ptx.Shared}
}

// A section collects one object at a time. The walk opens it with
// Object, writes bytes, and Closes it, and the close is what emits.
type section struct {
	l     *lowerer
	space ptx.Space

	name    string
	binding globals.Binding
	align   int
	buf     []byte
	relocs  []reloc
	sym     ir.Symbol
	err     error
}

type reloc struct {
	off    int
	sym    string
	addend int64
}

func (s *section) Align(n int) { s.align = n }

func (s *section) Object(name string, b globals.Binding) {
	s.name, s.binding = name, b
	s.buf, s.relocs = nil, nil
}

func (s *section) Byte(v byte)    { s.buf = append(s.buf, v) }
func (s *section) Zero(n int)     { s.buf = append(s.buf, make([]byte, n)...) }
func (s *section) Ascii(v string) { s.buf = append(s.buf, v...) }
func (s *section) Long(v uint32) {
	s.buf = append(s.buf, byte(v), byte(v>>8), byte(v>>16), byte(v>>24))
}
func (s *section) Quad(v uint64) {
	s.Long(uint32(v))
	s.Long(uint32(v >> 32))
}

func (s *section) PtrTo(sym string, addend int64) error {
	s.relocs = append(s.relocs, reloc{off: len(s.buf), sym: sym, addend: addend})
	s.Quad(0)
	return nil
}

func (s *section) Delta(to, from string, addend int64) error {
	return fmt.Errorf("@%s: a symbol difference has no PTX initializer", s.name)
}

// Close emits the object as a declaration.
func (s *section) Close(name string) {
	v := &ptx.Var{Space: s.space, Align: s.align, Type: ptx.B8, Name: symName(name)}
	switch s.binding {
	case globals.Global:
		v.Linkage = ptx.Visible
	case globals.Weak:
		v.Linkage = ptx.Weak
	}
	size := len(s.buf)
	if size == 0 {
		size = 1
	}
	v.Len = size

	switch {
	case s.space == ptx.Shared:
		// Zeroed by §19.21, and .shared takes no initializer anyway.
	case len(s.relocs) > 0:
		if size%8 != 0 {
			s.l.fail(fmt.Errorf("lower: @%s holds an address and is %d bytes; a PTX address initializer needs a multiple of 8", name, size))
			return
		}
		v.Type = ptx.B64
		v.Len = size / 8
		v.Init = s.wordsInit(name)
	case nonZero(s.buf):
		v.Init = bytesInit(s.buf)
	}
	// The symbol the walk named is the global's; record it under the
	// module's symbol so getaddr can find the space it lives in.
	if sym := s.l.m.Lookup(name); sym != nil {
		s.l.vars[sym] = v
	}
	s.l.pm.Add(v)
}

func nonZero(b []byte) bool {
	for _, c := range b {
		if c != 0 {
			return true
		}
	}
	return false
}

// bytesInit is a .b8 initializer list.
type bytesInit []byte

func (b bytesInit) InitText() string {
	out := make([]byte, 0, 6*len(b)+2)
	out = append(out, '{')
	for i, c := range b {
		if i > 0 {
			out = append(out, ',', ' ')
		}
		out = append(out, '0', 'x')
		const hex = "0123456789abcdef"
		out = append(out, hex[c>>4], hex[c&15])
	}
	return string(append(out, '}'))
}

// wordsInit is a .b64 initializer list with addresses where the walk put
// them.
type wordsInit []string

func (w wordsInit) InitText() string {
	s := "{"
	for i, e := range w {
		if i > 0 {
			s += ", "
		}
		s += e
	}
	return s + "}"
}

func (s *section) wordsInit(name string) ptx.Init {
	at := map[int]reloc{}
	for _, r := range s.relocs {
		if r.off%8 != 0 {
			s.l.fail(fmt.Errorf("lower: @%s holds an address at byte %d; a PTX address initializer needs it on an 8-byte boundary", name, r.off))
			return nil
		}
		at[r.off] = r
	}
	var out wordsInit
	for off := 0; off < len(s.buf); off += 8 {
		if r, ok := at[off]; ok {
			e := "generic(" + symName(r.sym) + ")"
			if r.addend != 0 {
				e += "+" + strconv.FormatInt(r.addend, 10)
			}
			out = append(out, e)
			continue
		}
		var v uint64
		for i := 7; i >= 0; i-- {
			v = v<<8 | uint64(s.buf[off+i])
		}
		out = append(out, "0x"+strconv.FormatUint(v, 16))
	}
	return out
}

// declareGlobalImport declares a datum another module defines. PTX's
// .extern needs the space and a size; the size is the type's.
func (l *lowerer) declareGlobalImport(g *ir.GlobalImport) {
	size, align, err := sizeAlign(g.Type())
	if err != nil {
		l.fail(fmt.Errorf("lower: import @%s is %s: %w", g.Name(), g.Type(), err))
		return
	}
	if size == 0 {
		size = 1
	}
	v := &ptx.Var{Linkage: ptx.Extern, Space: ptx.Global, Align: int(align), Type: ptx.B8, Name: symName(g.Name()), Len: int(size)}
	l.vars[g] = v
	l.pm.Add(v)
}

// fail records the first error the walk's adapter hit; globals.Lower's
// Section interface returns nothing from Close, so the error surfaces
// after the walk.
func (l *lowerer) fail(err error) {
	if l.err == nil {
		l.err = err
	}
}
