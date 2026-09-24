// Package air lowers VIR to AIR: an *air.Module, which the air package
// writes as the .metallib Metal loads.
//
// # What is different about this backend
//
// AIR is LLVM IR, so this is translation more than selection: VIR's
// values become AIR's SSA values, its blocks become blocks, and its block
// parameters become phis. Three things need more than a table.
//
// Pointers. VIR has one pointer type; AIR has no generic pointer, and every
// pointer is in device, constant, threadgroup or thread memory. Each
// pointer's space is inferred (space.go) from where it comes from --
// kernel arguments are device memory, shared globals threadgroup, read-only
// globals constant, allocations thread -- and a pointer that could be in
// two is refused by name rather than guessed. Every device function is
// inlined into its kernels first, which is what makes the inference local.
// A pointer is an i8 addrspace(n)* here, cast to the type a load or store
// wants at the access.
//
// Arguments. AIR has no special registers: a kernel's grid position is an
// argument Metal fills. The work-item verbs a kernel uses become the
// built-in arguments that answer them. A kernel's own parameters are bound
// in order: parameter i is [[buffer(i)]], a pointer as device char*, a
// scalar as constant T&, a by-value aggregate as constant bytes. A dynamic
// shared import is a [[threadgroup(n)]] argument.
//
// What AIR does not have. §0 traps on a zero divisor and on a float
// conversion out of range; Metal has no trap an app can see. So a divide
// guards its divisor (the answer to a zero one is the dividend, defined
// rather than undefined behaviour the compiler could exploit), a float
// conversion saturates, and trap ends the thread. Each is a divergence
// from §0 and is documented as one, not hidden.
//
// # Where things live
//
// lower.go is the entry point and the globals adapter. space.go infers the
// pointers' spaces. func.go is a kernel's arguments, built-ins and blocks.
// isel.go is instructions, control.go terminators and phis, gpu.go the
// work-item, wave, barrier, fence and atomic verbs. layout.go is the device
// type layout, shared with the other GPU backends.
package air

import (
	"fmt"

	am "github.com/vertex-language/air"
	"github.com/vertex-language/ir"
	"github.com/vertex-language/ir/lower/globals"
	"github.com/vertex-language/ir/lower/inline"
)

// Options say what the module is lowered for.
type Options struct {
	// Target is the deployment target; the zero value is macOS 13.0, the
	// oldest AIR (2.5) writes for, and which newer systems load.
	Target am.Target
	// Language is the MSL version the module says it is; zero is 3.0.
	Language am.Language
	// Family is the oldest GPU family it runs on; zero is Apple7 (M1).
	Family am.Family
	// FastMath is -ffast-math: fast flags on float instructions.
	FastMath bool
}

func (o Options) target() am.Target {
	if o.Target == (am.Target{}) {
		return am.Target{OS: am.MacOS, Version: am.OS(13, 0)}
	}
	return o.Target
}

func (o Options) language() am.Language {
	if o.Language == 0 {
		return am.MSL30
	}
	return o.Language
}

// Lower builds an AIR module from m: one AIR kernel per VIR kernel, with
// every device function it calls inlined into it. m itself is changed by
// the inlining.
func Lower(m *ir.Module, opts Options) (*am.Module, error) {
	if err := checkLayout(m); err != nil {
		return nil, err
	}
	kernels := func(f *ir.Func) bool { return f.Signature().CallConv() == ir.Kernel }
	if err := inline.Module(m, inline.Options{Into: kernels}); err != nil {
		return nil, fmt.Errorf("lower: %w", err)
	}
	out := am.NewModule(opts.target(), opts.language())
	if opts.Family != 0 {
		out.Family = opts.Family
	}
	out.FastMath = opts.FastMath
	l := &lowerer{m: m, out: out, globals: map[string]*am.Global{}, shared: map[ir.Symbol]int{}}

	if err := globals.Lower(globalTarget{l: l}, m); err != nil {
		return nil, fmt.Errorf("lower: %w", err)
	}
	if l.err != nil {
		return nil, l.err
	}
	for _, g := range m.GlobalImports() {
		if g.Domain() == ir.Shared {
			l.shared[g] = len(l.shared)
		}
	}
	n := 0
	for _, f := range m.Funcs() {
		if !kernels(f) {
			continue
		}
		n++
		if err := l.kernel(f); err != nil {
			return nil, fmt.Errorf("lower: @%s: %w", f.Name(), err)
		}
	}
	if n == 0 {
		return nil, fmt.Errorf("lower: module %q has no kernels; a library holds kernels", m.Name())
	}
	if err := out.Verify(); err != nil {
		return nil, fmt.Errorf("lower: %w", err)
	}
	return out, nil
}

// checkLayout refuses a module whose layout block is not the device's.
func checkLayout(m *ir.Module) error {
	l := m.Layout()
	switch {
	case m.Use() != ir.AIR64.Use():
		return fmt.Errorf("lower: module %q is for %q; this package lowers for %q", m.Name(), m.Use(), ir.AIR64.Use())
	case l.PtrBits != 64:
		return fmt.Errorf("lower: module %q declares ptrbits %d; AIR pointers are 64 bits", m.Name(), l.PtrBits)
	case l.Endian != ir.LittleEndian:
		return fmt.Errorf("lower: module %q declares %s-endian; the GPU is little-endian", m.Name(), l.Endian)
	case l.ABI != "air":
		return fmt.Errorf("lower: module %q declares abi %q; the device convention is air", m.Name(), l.ABI)
	}
	return nil
}

type lowerer struct {
	m   *ir.Module
	out *am.Module

	// globals are the module's globals by name: constant tables and
	// threadgroup storage, as bytes.
	globals map[string]*am.Global
	// shared numbers the dynamic shared imports: each is the
	// [[threadgroup(n)]] argument of every kernel that uses it.
	shared map[ir.Symbol]int

	err error
}

// ---- globals ---------------------------------------------------------------

// globalTarget is the lower/globals adapter: the walk hands it each
// global's bytes, and it makes an AIR global of them. Read-only data is a
// constant table; shared storage is threadgroup memory. Metal has no
// writable memory at program scope, so a writable global is refused.
type globalTarget struct{ l *lowerer }

func (t globalTarget) SizeAlign(ft ir.FType) (uint64, uint64, error) { return sizeAlign(ft) }
func (t globalTarget) FieldOffsets(nt *ir.Type) ([]uint64, error)    { return fieldOffsets(nt) }
func (t globalTarget) PtrBytes() uint64                              { return 8 }
func (t globalTarget) TLSSection(globals.Kind) globals.Section       { return nil }

func (t globalTarget) Section(k globals.Kind) globals.Section {
	return &section{l: t.l, kind: k}
}

func (t globalTarget) SharedSection() globals.Section {
	return &section{l: t.l, shared: true}
}

// A section collects one global's bytes between Object and Close.
type section struct {
	l      *lowerer
	kind   globals.Kind
	shared bool

	name  string
	align int
	data  []byte
	size  int // shared storage has a size and no bytes
}

func (s *section) fail(format string, args ...any) {
	if s.l.err == nil {
		s.l.err = fmt.Errorf("lower: @%s: "+format, append([]any{s.name}, args...)...)
	}
}

func (s *section) Align(n int) {
	if n > s.align {
		s.align = n
	}
}

func (s *section) Object(name string, b globals.Binding) {
	s.name, s.data, s.size = name, nil, 0
}

func (s *section) Close(name string) {
	n := len(s.data)
	if s.shared {
		n = s.size
	}
	if n == 0 {
		n = 1
	}
	elem := am.Array(am.Char, n)
	align := s.align
	if align < 1 {
		align = 1
	}
	var g *am.Global
	switch {
	case s.shared:
		g = s.l.out.Threadgroup(name, elem)
	case s.kind == globals.ROData:
		bytes := make([]*am.Const, n)
		for i := range bytes {
			v := byte(0)
			if i < len(s.data) {
				v = s.data[i]
			}
			bytes[i] = am.ConstInt(am.Char, int64(v))
		}
		g = s.l.out.Constant(name, elem, am.ConstArray(am.Char, bytes...))
	default:
		s.fail("Metal has no writable memory at program scope: a writable global is a kernel argument instead")
		return
	}
	g.Align = align
	s.l.globals[name] = g
	s.align = 0
}

func (s *section) Byte(v byte) { s.data = append(s.data, v) }
func (s *section) Long(v uint32) {
	s.data = append(s.data, byte(v), byte(v>>8), byte(v>>16), byte(v>>24))
}
func (s *section) Quad(v uint64)  { s.Long(uint32(v)); s.Long(uint32(v >> 32)) }
func (s *section) Ascii(v string) { s.data = append(s.data, v...) }
func (s *section) Zero(n int) {
	if s.shared {
		s.size += n
		return
	}
	s.data = append(s.data, make([]byte, n)...)
}

func (s *section) PtrTo(sym string, addend int64) error {
	return fmt.Errorf("@%s holds the address of @%s: a table of addresses is not lowered for AIR yet", s.name, sym)
}

func (s *section) Delta(to, from string, addend int64) error {
	return fmt.Errorf("@%s holds @%s - @%s: a symbol difference is not lowered for AIR", s.name, to, from)
}
