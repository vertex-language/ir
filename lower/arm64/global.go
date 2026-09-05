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
	// darwin says the module is being built for Mach-O, which is the one
	// container this backend has a thread-local model for.
	darwin bool
	// libcallPrefix is Options.LibcallPrefix, for the one symbol this
	// package invents on a thread-local's behalf.
	libcallPrefix string
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
func lowerGlobals(am *arm64asm.Module, m *ir.Module, opts Options) error {
	darwin := opts.Variadic == VariadicDarwin
	t := globalTarget{am: am, darwin: darwin, libcallPrefix: opts.LibcallPrefix}
	if darwin && hasThreadLocal(m) {
		// Every descriptor's first word names it, and it is the loader's
		// rather than anything this module defines. Declared once here
		// rather than per descriptor: Extern is a declaration, and a
		// module makes each one once.
		am.Extern(t.bootstrapSym())
	}
	return globals.Lower(t, m)
}

// hasThreadLocal reports whether the module declares any thread-local.
func hasThreadLocal(m *ir.Module) bool {
	for _, g := range m.Globals() {
		if g.Domain() == ir.TLS {
			return true
		}
	}
	return false
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

// The thread-local sections, spelled the ELF way and translated by the
// writers as every other section here is. Mach-O's own names are
// __thread_data, __thread_bss and __thread_vars.
const (
	tlsDataSection = ".tdata"
	tlsBSSSection  = ".tbss"
	tlsDescSection = ".tlvdesc"
)

// tlvInitSuffix is what Mach-O calls a thread-local's template, beside
// the descriptor that carries the declared name. clang writes the same.
const tlvInitSuffix = "$tlv$init"

// tlvBootstrap is the thunk every descriptor starts out pointing at.
// dyld replaces it with one that knows the block once the variable has
// been reached the first time; libSystem exports it.
//
// It takes the platform's prefix for the reason Options.LibcallPrefix
// gives: the names a module wrote are already spelled the way the
// platform wants them, and this is a name this package invented, so it
// has no author to have spelled it. On Mach-O that makes the symbol
// __tlv_bootstrap, which is what libSystem.tbd exports.
const tlvBootstrap = "_tlv_bootstrap"

func (t globalTarget) bootstrapSym() string { return t.libcallPrefix + tlvBootstrap }

// TLSSection is where a thread-local's template goes, under Mach-O's
// model: the only one this backend implements.
//
// It is the ABI that decides rather than the container, because the
// container is chosen after lowering and the sequence at the use site
// has to match the storage. ELF's four models want different
// relocations and a different sequence, so a module that is not
// Darwin's gets nil and globals.Lower refuses the declaration.
func (t globalTarget) TLSSection(k globals.Kind) globals.Section {
	if !t.darwin {
		return nil
	}
	name := tlsDataSection
	kind := arm64asm.Data
	if k == globals.BSS {
		name, kind = tlsBSSSection, arm64asm.BSS
	}
	return globalSection{t.am.SectionNamed(name, kind)}
}

// TLSTemplateName is the second symbol a thread-local needs: the bytes
// every thread's copy is made from, which no instruction names.
func (globalTarget) TLSTemplateName(name string) string { return name + tlvInitSuffix }

// TLSDescriptor writes the three words every access to a thread-local
// goes through.
//
//	+0  thunk     __tlv_bootstrap, which dyld swaps for one that knows
//	              where this thread's block is
//	+8  key       dyld's, and zero until it fills it in
//	+16 offset    the template's distance from the start of the region,
//	              which the linker computes from this reference
//
// The declared name is the descriptor's, not the template's: `_counter`
// in the program is this, and `_counter$tlv$init` is the bytes. Every
// TLVP relocation names the descriptor, so a program that took the
// template's address instead would reach the same memory for every
// thread.
func (t globalTarget) TLSDescriptor(g *ir.Global, template string) error {
	if !t.darwin {
		return fmt.Errorf("lower: @%s: a thread-local descriptor has no shape on this target", g.Name())
	}
	sec := globalSection{t.am.SectionNamed(tlsDescSection, arm64asm.Data)}
	sec.Align(8)
	sec.Object(g.Name(), bindingFor(g))
	if err := sec.PtrTo(t.bootstrapSym(), 0); err != nil {
		return fmt.Errorf("lower: @%s: %w", g.Name(), err)
	}
	sec.Quad(0)
	if err := sec.PtrTo(template, 0); err != nil {
		return fmt.Errorf("lower: @%s: %w", g.Name(), err)
	}
	sec.Close(g.Name())
	return nil
}

// bindingFor is globals.bindingFor, which is unexported. A descriptor is
// as visible as the variable it describes.
func bindingFor(g *ir.Global) globals.Binding {
	switch {
	case g.Binding() == ir.Weak:
		return globals.Weak
	case g.Linkage() == ir.Export:
		return globals.Global
	}
	return globals.Local
}
