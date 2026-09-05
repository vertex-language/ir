// Package globals lays a module's §5 declarations down as bytes.
//
// One copy of a walk that three backends were going to need. The initializer
// walk is the IR's shape rather than any target's: what an aggregate's braces
// mean, which fields a partial initializer leaves zero, how much padding sits
// between two struct members. None of that is architecture, and the two
// backends that had written it independently agreed line for line except
// where one of them had a bug.
//
// What is architecture is behind Target: the layout table the walk measures
// with, the three sections it writes into, and how an address is spelled as a
// relocation. Each backend supplies about forty lines of adapter.
package globals

import (
	"fmt"
	"math"

	"github.com/vertex-language/ir"
)

// A Binding is a symbol's binding. The object formats agree on these three
// and the assemblers spell them differently, so the walk names them and the
// adapter translates.
type Binding uint8

const (
	Local Binding = iota
	Global
	Weak
)

// A Kind is one of the three sections a global's bytes can go in.
type Kind uint8

const (
	ROData Kind = iota
	Data
	BSS
)

// A Layout is what one architecture says about the shape of a type. It is
// the one piece of real architecture in the walk — an eightbyte aligns to
// eight on the 64-bit targets and to four on i386 — and it is also the whole
// of what §2's symbolic constants need, which is why it is named apart from
// the rest: an isel resolving a sizeof has a Layout and no sections.
type Layout interface {
	// SizeAlign measures a type on this target.
	SizeAlign(ir.FType) (size, align uint64, err error)

	// FieldOffsets is where each of a struct's fields begins.
	FieldOffsets(*ir.Type) ([]uint64, error)
}

// A Target is what one architecture supplies.
type Target interface {
	Layout

	// Section is the builder for one of the three sections.
	Section(Kind) Section

	// TLSSection is where a global in domain tls goes, or nil where this
	// target has no TLS model.
	//
	// A nil answer is what makes the walk refuse a thread-local rather
	// than put one somewhere nothing can reach it from. The storage and
	// the sequence that finds it are one feature: a target that emits the
	// bytes and cannot lower ptr.tlsaddr has produced a global no
	// instruction can address, which is worse than not compiling.
	//
	// The kind is Data or BSS, decided the same way it is for an ordinary
	// global. A target with one thread-local section ignores it; Mach-O
	// has two, __thread_data and __thread_bss, and a zeroed template
	// belongs in the second for the same reason .bss exists.
	TLSSection(Kind) Section

	// PtrBytes is how many bytes an address initializer fills.
	PtrBytes() uint64
}

// A TLSDescriptors target needs something beside a thread-local's
// template, under the name the program uses.
//
// Mach-O is why it exists. A thread-local there is two symbols: the
// template, which no instruction names, and a three-word descriptor,
// which every access goes through. One section and one symbol cannot
// say that, and the shape is the container's rather than the walk's —
// so the walk asks, and a target that answers nothing gets the single
// symbol it wanted.
type TLSDescriptors interface {
	// TLSTemplateName is what to call the template, given the name the
	// program declared. It is a second symbol, so it needs a second
	// name; Mach-O's convention is the declared name with $tlv$init
	// after it.
	TLSTemplateName(name string) string

	// TLSDescriptor writes what stands beside the template, under the
	// declared name. template is what TLSTemplateName returned, already
	// emitted.
	TLSDescriptor(g *ir.Global, template string) error
}

// A Section is the part of an assembler's section builder this walk uses.
type Section interface {
	Align(n int)

	// Object opens a data symbol and Close ends it, which is what gives
	// the symbol its size.
	Object(name string, b Binding)
	Close(name string)

	Byte(v byte)
	Long(v uint32)
	Quad(v uint64)
	Ascii(s string)
	Zero(n int)

	// PtrTo places a pointer-sized absolute reference to sym plus addend.
	// An addend the assembler's API cannot express is an error rather than
	// a silently dropped offset.
	PtrTo(sym string, addend int64) error
}

// Lower writes every global definition in m, in declaration order.
//
// Nothing here reorders or packs: a global's address is the linker's to
// choose, and the only thing this can get wrong is the alignment it asks for.
func Lower(t Target, m *ir.Module) error {
	for _, g := range m.Globals() {
		if err := lowerGlobal(t, g); err != nil {
			return err
		}
	}
	return nil
}

func lowerGlobal(t Target, g *ir.Global) error {
	tls := g.Domain() == ir.TLS
	if tls && t.TLSSection(sectionFor(g)) == nil {
		// A thread-local is not a section and an offset; it is an entry
		// in a TLS block, reached through a model with its own
		// relocations and its own sequence at the use site. That is
		// §D3's tlsaddr as much as it is §5's declaration, and a target
		// with no answer to the first has no business emitting the
		// second.
		return fmt.Errorf("lower: @%s is in domain tls, which needs a TLS model and its relocations", g.Name())
	}
	if _, ok := g.ComdatAttr(); ok {
		return fmt.Errorf("lower: @%s asks for a comdat group, which is not emitted yet", g.Name())
	}
	if g.SectionAttr() != "" {
		return fmt.Errorf("lower: @%s asks for section %q; only the domain's own section is emitted yet", g.Name(), g.SectionAttr())
	}

	// The name the bytes are emitted under, and how visible it is.
	// Both change for a thread-local on a target that wants a
	// descriptor: the template becomes a second, local symbol, and the
	// declared name goes to the descriptor the program actually reaches.
	name, binding := g.Name(), bindingFor(g)
	var desc TLSDescriptors

	sec := t.Section(sectionFor(g))
	if tls {
		sec = t.TLSSection(sectionFor(g))
		if d, ok := t.(TLSDescriptors); ok {
			desc = d
			name = d.TLSTemplateName(g.Name())
			binding = Local
		}
	}

	// Alignment before the label: it is a property of the address the
	// label is about to name. The type's own alignment when the
	// declaration states none — §5's align attribute raises a global's
	// alignment, it does not supply the one the type already requires.
	_, natural, err := t.SizeAlign(g.Type())
	if err != nil {
		return fmt.Errorf("lower: @%s is %s: %w", g.Name(), g.Type(), err)
	}
	if a := g.AlignAttr(); a > natural {
		natural = a
	}
	sec.Align(int(natural))
	sec.Object(name, binding)

	if err := emitInit(t, sec, g, g.Type(), g.Initializer()); err != nil {
		return err
	}
	sec.Close(name)

	if desc != nil {
		return desc.TLSDescriptor(g, name)
	}
	return nil
}

// sectionFor is the section a global's bytes belong in.
//
// §5's three domains are not three sections. ro is .rodata and tls is refused
// above, but rw splits: a global whose initializer is all zeroes belongs in
// .bss, where it costs address space and no file bytes, and one with any
// content in .data. That split is a property of the initializer rather than
// the domain, which is why it is decided here and not by a Domain method.
func sectionFor(g *ir.Global) Kind {
	if g.Domain() == ir.RO {
		// Read-only, but only once something has written it: an
		// initializer naming another symbol is a relocation, and a
		// relocation has to be applied. .rodata is mapped read-only
		// from the file and __TEXT,__const is inside the text
		// segment, so a table of pointers placed there either faults
		// when the loader rebases it or is silently left holding
		// whatever was in the file. A vtable is exactly this shape,
		// and clang puts one in __DATA,__const for the same reason.
		//
		// Data rather than a section of its own. The right answer is
		// relro -- .data.rel.ro on ELF, __DATA_CONST on Mach-O --
		// which is written once by the loader and then made read-only
		// again; expressing that is a Kind here, a section in each
		// assembler and a mapping in each container writer, and none
		// of those exist. Data is correct and less protected, which
		// is the trade being made until they do.
		if hasReloc(g.Initializer()) {
			return Data
		}
		return ROData
	}
	// A thread-local's bytes are the template every thread's copy is made
	// from, and the split is the same one an ordinary global gets: all
	// zeroes costs address space alone, anything else costs file bytes.
	// Which two sections those are is TLSSection's answer.

	if k := g.Initializer().Kind(); k == ir.InitZeroed || k == 0 {
		return BSS
	}
	return Data
}

// FuncBinding is a function's symbol binding, on the same terms bindingFor
// gives a global's: linkage and binding are two attributes in VIR and one in
// an object file, so weak wins.
//
// A function that states no linkage is Global, where a global that states none
// is Local. The asymmetry is deliberate and is what every module written
// before this existed assumes: a function is a module's interface unless
// ir.Internal says otherwise, and a hand-built module that never calls Export
// still expects its entry point to be callable.
func FuncBinding(fn *ir.Func) Binding {
	switch {
	case fn.IsWeak():
		return Weak
	case fn.Linkage() == ir.Internal:
		return Local
	}
	return Global
}

// bindingFor is a global's symbol binding: §5's linkage and binding are two
// attributes in VIR and one in an object file, so weak wins.
func bindingFor(g *ir.Global) Binding {
	switch {
	case g.Binding() == ir.Weak:
		return Weak
	case g.Linkage() == ir.Export:
		return Global
	}
	return Local
}

// emitInit writes exactly SizeAlign(t) bytes of initializer.
//
// Exactly, which is what the aggregate cases depend on: a struct pads between
// its fields by the difference between where the next starts and where the
// last ended, and that arithmetic is only sound if every nested call wrote its
// type's full width.
func emitInit(tg Target, sec Section, g *ir.Global, t ir.FType, init ir.Init) error {
	size, _, err := tg.SizeAlign(t)
	if err != nil {
		return fmt.Errorf("lower: @%s is %s: %w", g.Name(), t, err)
	}

	switch init.Kind() {
	case ir.InitZeroed, 0:
		sec.Zero(int(size))
		return nil

	case ir.InitLiteral:
		r := resolveAlias(t)
		if r.Kind() != ir.FTypeScalar {
			return fmt.Errorf("lower: @%s is %s; a literal initializer fills one scalar", g.Name(), t)
		}
		return emitLiteral(tg, sec, g, t, r.Scalar(), size, init.Const())

	case ir.InitString:
		str := init.String_()
		if _, ok := arrayBytes(t); !ok {
			return fmt.Errorf("lower: @%s is %s; a string initializer fills an array of i8", g.Name(), t)
		}
		if uint64(len(str)) > size {
			return fmt.Errorf("lower: @%s is %s; the string is %d bytes", g.Name(), t, len(str))
		}
		sec.Ascii(str)
		sec.Zero(int(size - uint64(len(str))))
		return nil

	case ir.InitRelocKind:
		// The address of another symbol, resolved by the linker: the
		// data-side twin of ptr.getaddr, absolute because a global holds
		// an address and not a distance from itself.
		r := init.Reloc()
		if r.Sym == nil {
			return fmt.Errorf("lower: @%s: a reloc initializer with no symbol", g.Name())
		}
		if r.Minus != nil {
			return fmt.Errorf("lower: @%s: a symbol difference initializer is not emitted yet", g.Name())
		}
		if size != tg.PtrBytes() {
			return fmt.Errorf("lower: @%s is %s; an address fills %d bytes on this target", g.Name(), t, tg.PtrBytes())
		}
		off, err := addend(tg, r)
		if err != nil {
			return fmt.Errorf("lower: @%s: %w", g.Name(), err)
		}
		if err := sec.PtrTo(r.Sym.Name(), off); err != nil {
			return fmt.Errorf("lower: @%s: %w", g.Name(), err)
		}
		return nil

	case ir.InitList:
		return emitList(tg, sec, g, t, init, size)

	case ir.InitFields:
		return emitFields(tg, sec, g, t, init, size)
	}
	return fmt.Errorf("lower: @%s: initializer form %v is not emitted", g.Name(), init.Kind())
}

// emitLiteral writes one scalar literal, which is where the initializer's
// kind and the declared type's have to agree.
//
// A float is written as its bit pattern rather than converted at emission
// time: what a section takes is bytes, and what §5 states is a value, and the
// step between them is the format's own encoding and not the assembler's.
func emitLiteral(tg Target, sec Section, g *ir.Global, t ir.FType, st ir.StoreType, size uint64, c ir.Const) error {
	switch st {
	case ir.StoreF32, ir.StoreF64, ir.StoreF80, ir.StoreF128:
		if c.Kind() != ir.ConstFloat {
			return fmt.Errorf("lower: @%s is %s; its initializer is not a float literal", g.Name(), t)
		}
		return emitFloat(sec, g, t, st, size, c.Float())
	}
	v, err := ConstInt(tg, c)
	if err != nil {
		return fmt.Errorf("lower: @%s is %s; %w", g.Name(), t, err)
	}
	return emitScalar(sec, g, size, uint64(v))
}

// emitFloat writes a float literal at its declared width.
//
// f80 is refused rather than encoded. Its ten bytes are the x87 register
// file's own format, no backend in this tree holds an f80 in a register, and
// a global nothing can load is worse than one that says so.
func emitFloat(sec Section, g *ir.Global, t ir.FType, st ir.StoreType, size uint64, v float64) error {
	switch st {
	case ir.StoreF32:
		if size < 4 {
			return fmt.Errorf("lower: @%s is %s; an f32 fills four bytes", g.Name(), t)
		}
		sec.Long(math.Float32bits(float32(v)))
		sec.Zero(int(size - 4))
		return nil

	case ir.StoreF64:
		if size < 8 {
			return fmt.Errorf("lower: @%s is %s; an f64 fills eight bytes", g.Name(), t)
		}
		sec.Quad(math.Float64bits(v))
		sec.Zero(int(size - 8))
		return nil

	case ir.StoreF128:
		if size < 16 {
			return fmt.Errorf("lower: @%s is %s; an f128 fills sixteen bytes", g.Name(), t)
		}
		lo, hi := float128Bits(v)
		sec.Quad(lo)
		sec.Quad(hi)
		sec.Zero(int(size - 16))
		return nil
	}
	return fmt.Errorf("lower: @%s is %s; an f80 initializer is x87's own ten-byte format, which no backend here holds in a register", g.Name(), t)
}

// float128Bits widens a double to binary128, which is exact: every value a
// double names, subnormals included, is a normal number in the wider format.
//
// The mantissa is the double's fifty-two bits at the top of binary128's
// hundred and twelve, so it moves left by sixty and straddles the two words.
// The exponent is rebiased, except at the two ends: the all-ones exponent
// stays all ones, so an infinity and a NaN carry across unchanged, and a
// subnormal is normalized on the way, its leading one shifted up out of the
// mantissa and paid for in the exponent.
func float128Bits(v float64) (lo, hi uint64) {
	b := math.Float64bits(v)
	sign := b >> 63
	exp := (b >> 52) & 0x7ff
	man := b & 0xf_ffff_ffff_ffff

	var e uint64
	switch {
	case exp == 0x7ff:
		e = 0x7fff
	case exp != 0:
		e = exp - 1023 + 16383
	case man != 0:
		shift := uint64(0)
		for man&(1<<52) == 0 {
			man <<= 1
			shift++
		}
		man &^= 1 << 52
		e = 16383 - 1022 - shift
	}

	return man << 60, sign<<63 | e<<48 | man>>4
}

func emitScalar(sec Section, g *ir.Global, size, v uint64) error {
	switch size {
	case 1:
		sec.Byte(byte(v))
	case 2:
		sec.Byte(byte(v))
		sec.Byte(byte(v >> 8))
	case 4:
		sec.Long(uint32(v))
	case 8:
		sec.Quad(v)
	default:
		return fmt.Errorf("lower: @%s: no %d-byte scalar initializer", g.Name(), size)
	}
	return nil
}

// emitList writes a positional aggregate initializer.
func emitList(tg Target, sec Section, g *ir.Global, t ir.FType, init ir.Init, size uint64) error {
	r := resolveAlias(t)
	elems := init.Elems()

	if r.Kind() == ir.FTypeArray {
		if uint64(len(elems)) != r.Len() {
			return fmt.Errorf("lower: @%s is %s; the initializer has %d elements", g.Name(), t, len(elems))
		}
		// No padding between elements: an element's own size already
		// includes whatever keeps the next one aligned.
		for _, e := range elems {
			if err := emitInit(tg, sec, g, r.Elem(), e); err != nil {
				return err
			}
		}
		return nil
	}

	named := r.Named()
	if r.Kind() != ir.FTypeNamed || named == nil {
		return fmt.Errorf("lower: @%s is %s; a braced initializer fills an array, a struct, or a union", g.Name(), t)
	}
	switch named.Kind() {
	case ir.KindStruct:
		fields := named.Fields()
		if len(elems) != len(fields) {
			return fmt.Errorf("lower: @%s is @%s, which has %d fields; the initializer has %d elements",
				g.Name(), named.Name(), len(fields), len(elems))
		}
		return emitStruct(tg, sec, g, named, elems, size)
	case ir.KindUnion:
		fields := named.Fields()
		if len(elems) != 1 || len(fields) == 0 {
			return fmt.Errorf("lower: @%s is the union @%s; a union initializer names one member, and this one has %d",
				g.Name(), named.Name(), len(elems))
		}
		return emitUnion(tg, sec, g, fields[0], elems[0], size)
	}
	return fmt.Errorf("lower: @%s is @%s; a braced initializer fills an array, a struct, or a union", g.Name(), named.Name())
}

// emitFields writes a named-field initializer. The list may be partial —
// naming fields is what the form is for — and what it does not name is zero.
func emitFields(tg Target, sec Section, g *ir.Global, t ir.FType, init ir.Init, size uint64) error {
	r := resolveAlias(t)
	named := r.Named()
	if r.Kind() != ir.FTypeNamed || named == nil ||
		(named.Kind() != ir.KindStruct && named.Kind() != ir.KindUnion) {
		return fmt.Errorf("lower: @%s is %s; a field initializer fills a struct or a union", g.Name(), t)
	}

	fields := named.Fields()
	vals := make([]ir.Init, len(fields))
	for _, fv := range init.FieldVals() {
		i := fieldIndex(fields, fv.Name)
		if i < 0 {
			return fmt.Errorf("lower: @%s: @%s has no field %q", g.Name(), named.Name(), fv.Name)
		}
		vals[i] = fv.Init
	}

	if named.Kind() == ir.KindUnion {
		given := init.FieldVals()
		if len(given) != 1 {
			return fmt.Errorf("lower: @%s is the union @%s; a union initializer names one member, and this one names %d",
				g.Name(), named.Name(), len(given))
		}
		i := fieldIndex(fields, given[0].Name)
		return emitUnion(tg, sec, g, fields[i], given[0].Init, size)
	}
	return emitStruct(tg, sec, g, named, vals, size)
}

// emitStruct writes a struct's fields at their own offsets, padding between
// them and to the struct's full width.
func emitStruct(tg Target, sec Section, g *ir.Global, named *ir.Type, vals []ir.Init, size uint64) error {
	offsets, err := tg.FieldOffsets(named)
	if err != nil {
		return fmt.Errorf("lower: @%s is @%s: %w", g.Name(), named.Name(), err)
	}
	fields := named.Fields()

	var at uint64
	for i, f := range fields {
		// An explicit offset can put a field before the last one ended,
		// which no amount of padding expresses: refused rather than
		// written over.
		if offsets[i] < at {
			return fmt.Errorf("lower: @%s: @%s's field %q begins at %d, before the previous field ended at %d",
				g.Name(), named.Name(), f.Name, offsets[i], at)
		}
		sec.Zero(int(offsets[i] - at))

		v := vals[i]
		if v.IsZero() {
			v = ir.ZeroInit
		}
		if err := emitInit(tg, sec, g, f.Type, v); err != nil {
			return err
		}
		fsize, _, err := tg.SizeAlign(f.Type)
		if err != nil {
			return err
		}
		at = offsets[i] + fsize
	}
	sec.Zero(int(size - at))
	return nil
}

// emitUnion writes the named member and pads to the union's full width.
func emitUnion(tg Target, sec Section, g *ir.Global, f ir.Field, val ir.Init, size uint64) error {
	if err := emitInit(tg, sec, g, f.Type, val); err != nil {
		return err
	}
	fsize, _, err := tg.SizeAlign(f.Type)
	if err != nil {
		return err
	}
	sec.Zero(int(size - fsize))
	return nil
}

func fieldIndex(fields []ir.Field, name string) int {
	for i, f := range fields {
		if f.Name == name {
			return i
		}
	}
	return -1
}

// arrayBytes reports whether t is an array of i8, which is what a string
// initializer fills, and how long it is.
func arrayBytes(t ir.FType) (uint64, bool) {
	r := resolveAlias(t)
	if r.Kind() != ir.FTypeArray {
		return 0, false
	}
	e := resolveAlias(r.Elem())
	if e.Kind() != ir.FTypeScalar || e.Scalar() != ir.StoreI8 {
		return 0, false
	}
	return r.Len(), true
}

// maxNesting bounds a typedef chain, which the IR does not promise is acyclic.
const maxNesting = 64

// resolveAlias follows a typedef to the type it names.
//
// Aliased, not FType: a named type's FType is the wrapper around itself, so
// following that resolves nothing and spins until the bound. One of the two
// backends this replaces had exactly that, which is why a global of a
// typedef'd type could not be lowered on it at all.
func resolveAlias(t ir.FType) ir.FType {
	for i := 0; i < maxNesting; i++ {
		if t.Kind() != ir.FTypeNamed {
			return t
		}
		n := t.Named()
		if n == nil || n.Kind() != ir.KindAlias {
			return t
		}
		t = n.Aliased()
	}
	return t
}

// addend is a reloc initializer's displacement.
//
// Which may be one of §2's symbolic constants and usually is: &arr[3] is the
// array's symbol plus an offsetof, and Init.Plus says so in as many words.
// This used to answer zero for anything that was not already an integer,
// which silently initialized such a global to the array's own address — a
// wrong answer where there was a right one to be had, since the walk has the
// layout table that resolves it.
func addend(l Layout, r ir.Reloc) (int64, error) {
	if !r.HasAddend {
		return 0, nil
	}
	return ConstInt(l, r.Addend)
}

// hasReloc reports whether an initializer names a symbol anywhere in
// it, which is what makes its bytes something a loader has to write.
func hasReloc(i ir.Init) bool {
	switch i.Kind() {
	case ir.InitRelocKind:
		return true
	case ir.InitList:
		for _, e := range i.Elems() {
			if hasReloc(e) {
				return true
			}
		}
	case ir.InitFields:
		for _, f := range i.FieldVals() {
			if hasReloc(f.Init) {
				return true
			}
		}
	}
	return false
}
