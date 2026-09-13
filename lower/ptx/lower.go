// Package ptx lowers VIR to a *ptx.Module: PTX, the virtual ISA every NVIDIA
// toolchain stops at.
//
// # What is different about this backend
//
// PTX is itself a virtual-register IR. Registers are unlimited and typed,
// ptxas allocates them, and a block argument is a move rather than a
// parallel assignment — a fresh temporary per edge argument costs nothing
// and ptxas coalesces it. So there is no mir here, no regalloc, no frame
// planning and no emit step: this package is instruction selection, and the
// *ptx.Module it hands back is the finished artifact the way an
// *amd64obj.Object is. ptx/text prints it; the driver's JIT or ptxas takes
// the text. Nobody outside NVIDIA emits SASS, and LLVM's own NVPTX backend
// stops exactly here.
//
// # What it must not get wrong
//
// Rounding is spelled. Every f32 and f64 add, sub, mul, fma, div and sqrt
// carries .rn; without it ptxas may contract a mul and an add into an fma
// and may use an approximate divide, and §0 says round-to-nearest-even,
// one rounding per verb. Nothing carries .ftz: denormals are IEEE.
//
// Traps are explicit. A GPU does not fault on a zero divisor or an
// out-of-range conversion; PTX's div yields an unspecified value and cvt
// saturates. So i32.sdiv is a compare and a predicated trap before the
// divide, and the non-_sat_ float-to-int rows are a range test and a trap
// before the cvt — while the _sat_ rows are the cvt alone, whose clamp and
// NaN→0 are what §C2 asks for.
//
// A trap on a GPU is sticky: the context is faulted and every later launch
// in it fails. That is "traps" as §0 means it; nothing here expects a
// signal.
//
// # Where things live
//
// lower.go is the entry point, the layout check and the SM gate. types.go
// is the VIR-type-to-PTX-type table. func.go is a function's state:
// registers, labels, parameters, the local depot. isel.go selects
// instructions, control.go selects terminators and edge moves, call.go
// builds the .param marshalling sequence, atomic.go is §H with its scopes,
// bulk.go is §E as inline loops, and global.go is the lower/globals adapter
// that turns bytes into .global, .const and .shared declarations.
package ptx

import (
	"fmt"

	"github.com/vertex-language/ir"
	"github.com/vertex-language/ir/lower/globals"
	"github.com/vertex-language/ir/lower/inline"
	"github.com/vertex-language/ptx"
)

// OptLevel is how hard Lower tries. Only O0 exists today.
type OptLevel uint8

const O0 OptLevel = iota

// Options are the things Lower has no opinion about.
type Options struct {
	OptLevel OptLevel

	// ISA is the .version directive. The zero value is 8.0, which any
	// CUDA 12 driver accepts and which has every qualifier this package
	// emits.
	ISA ptx.ISAVersion

	// SM is the .target directive. The zero value is sm_50, the oldest
	// target the ptx repo models. Verbs whose instruction needs a newer
	// SM — scoped atomics below sm_70, f64 atomics below sm_60 — are
	// refused by name against the SM chosen here.
	SM ptx.Target

	// Inline copies every device function a kernel calls into the
	// kernel first, through lower/inline, which rewrites the module it
	// is given. PTX has a calling convention, so this is only speed: a
	// call through .param space is a round trip through local memory.
	Inline bool
}

func (o Options) isa() ptx.ISAVersion {
	if o.ISA.IsZero() {
		return ptx.ISA80
	}
	return o.ISA
}

func (o Options) sm() ptx.Target {
	if o.SM.SM == 0 {
		return ptx.SM50
	}
	return o.SM
}

// Lower builds a PTX module from m.
func Lower(m *ir.Module, opts Options) (*ptx.Module, error) {
	if err := checkLayout(m); err != nil {
		return nil, err
	}
	if opts.Inline {
		kernels := func(f *ir.Func) bool { return f.Signature().CallConv() == ir.Kernel }
		if err := inline.Module(m, inline.Options{Into: kernels}); err != nil {
			return nil, fmt.Errorf("lower: %w", err)
		}
	}
	l := &lowerer{
		m:       m,
		opts:    opts,
		pm:      ptx.NewModule(opts.isa(), opts.sm(), ptx.Addr64),
		funcs:   map[ir.Symbol]*ptx.Func{},
		kernels: map[*ir.Func]*ptx.Kernel{},
		vars:    map[ir.Symbol]*ptx.Var{},
	}

	// Declarations first, so that a body can name any function or global
	// in the module regardless of order. PTX itself wants a symbol
	// declared before use, and the module's declaration order is kept
	// for everything that has bytes.
	for _, f := range m.FuncImports() {
		if err := l.declareImport(f); err != nil {
			return nil, err
		}
	}
	for _, g := range m.GlobalImports() {
		l.declareGlobalImport(g)
	}
	if err := globals.Lower(globalTarget{l: l}, m); err != nil {
		return nil, err
	}
	if l.err != nil {
		return nil, l.err
	}
	for _, f := range m.Funcs() {
		if err := l.declareFunc(f); err != nil {
			return nil, err
		}
	}
	for _, it := range m.Items() {
		switch x := it.(type) {
		case *ir.Func:
			if err := l.lowerFunc(x); err != nil {
				return nil, err
			}
		case *ir.ModuleAsm:
			return nil, fmt.Errorf("lower: a module-level asm block is not emitted yet")
		}
	}
	return l.pm, nil
}

// checkLayout refuses a module whose layout block is not the device's.
func checkLayout(m *ir.Module) error {
	l := m.Layout()
	switch {
	case m.Use() != ir.NVPTX64.Use():
		return fmt.Errorf("lower: module %q is for %q; this package lowers for %q", m.Name(), m.Use(), ir.NVPTX64.Use())
	case l.PtrBits != 64:
		return fmt.Errorf("lower: module %q declares ptrbits %d; PTX pointers are 64 bits here", m.Name(), l.PtrBits)
	case l.Endian != ir.LittleEndian:
		return fmt.Errorf("lower: module %q declares %s-endian; the device is little-endian", m.Name(), l.Endian)
	case l.ABI != "ptx":
		return fmt.Errorf("lower: module %q declares abi %q; the device convention is ptx", m.Name(), l.ABI)
	}
	return nil
}

type lowerer struct {
	m    *ir.Module
	opts Options
	pm   *ptx.Module

	funcs   map[ir.Symbol]*ptx.Func
	kernels map[*ir.Func]*ptx.Kernel
	vars    map[ir.Symbol]*ptx.Var

	err error // the first fault the globals adapter recorded
}

// needSM refuses a verb whose instruction the chosen SM does not have.
func (l *lowerer) needSM(op ir.Op, sm int, what string) error {
	if l.opts.sm().SM < sm {
		return fmt.Errorf("%s: %s needs sm_%d, and Options.SM is %s", op, what, sm, l.opts.sm())
	}
	return nil
}

// symName is a module symbol as PTX spells it. PTX identifiers are
// [A-Za-z_$][A-Za-z0-9_$]*, and VIR admits the characters cl's mangling
// uses — ?, @, ., <, > — so those are escaped as $ and two hex digits,
// which keeps distinct names distinct.
func symName(name string) string {
	ok := true
	for i := 0; i < len(name); i++ {
		c := name[i]
		if !(c == '_' || c == '$' || c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z' || (i > 0 && c >= '0' && c <= '9')) {
			ok = false
			break
		}
	}
	if ok {
		return name
	}
	const hex = "0123456789abcdef"
	out := make([]byte, 0, len(name)+8)
	for i := 0; i < len(name); i++ {
		c := name[i]
		if c == '_' || c == '$' || c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z' || (i > 0 && c >= '0' && c <= '9') {
			out = append(out, c)
			continue
		}
		out = append(out, '$', hex[c>>4], hex[c&15])
	}
	return string(out)
}
