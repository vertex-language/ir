// Package ptx lowers VIR to a *ptx.Module (NVIDIA PTX virtual ISA).
//
// # Design Notes
//
// PTX is a typed, virtual-register IR with unlimited registers allocated by
// ptxas. Block arguments map to moves coalesced by ptxas, eliminating the need
// for MIR, register allocation, or machine frame planning; this package directly
// performs instruction selection into a *ptx.Module.
//
// # Correctness Rules
//
//   - Rounding: All f32/f64 operations explicitly specify .rn (round-to-nearest-even)
//     to prevent contraction into approximate divides or unintended FMAs. Denormals
//     are preserved (no .ftz).
//   - Traps: GPU execution does not fault on division by zero or conversion
//     overflow; integer division and non-saturating conversions emit range tests
//     and predicated traps before the operation. Saturating conversions directly use
//     clamping and NaN->0 behavior.
//
// # Subsystem Layout
//
//   - lower.go: Entry point, layout validation, and SM gating.
//   - types.go: VIR-to-PTX type translation.
//   - func.go: Virtual registers, labels, parameters, and local depot.
//   - isel.go / control.go: Instruction selection and terminators.
//   - call.go: Parameter marshaling.
//   - atomic.go / bulk.go / global.go: Memory scopes, bulk loops, and declarations.
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

	// Prune leaves out every function and import no kernel calls,
	// directly or through another function: what a kernel was inlined
	// from, say, which the driver would otherwise have to link, and
	// whose calls -- to an intrinsic its caller resolves -- it cannot.
	// A function reached other than by a call is not seen.
	Prune bool
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
	keep := func(ir.Callee) bool { return true }
	if opts.Prune {
		reached := reachedFromKernels(m)
		keep = func(c ir.Callee) bool { return reached[c] }
	}
	for _, f := range m.FuncImports() {
		if !keep(f) {
			continue
		}
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
	// A body may call a function defined after it, and PTX wants the
	// callee's prototype first: one is declared ahead of every body for
	// each function called from a body that precedes its own.
	for _, f := range forwardCalled(m) {
		if !keep(f) {
			continue
		}
		if err := l.declareProto(f); err != nil {
			return nil, err
		}
	}
	for _, f := range m.Funcs() {
		if !keep(f) {
			continue
		}
		if err := l.declareFunc(f); err != nil {
			return nil, err
		}
	}
	for _, it := range m.Items() {
		switch x := it.(type) {
		case *ir.Func:
			if !keep(x) {
				continue
			}
			if err := l.lowerFunc(x); err != nil {
				return nil, err
			}
		case *ir.ModuleAsm:
			return nil, fmt.Errorf("lower: a module-level asm block is not emitted yet")
		}
	}
	return l.pm, nil
}

// reachedFromKernels is the kernels of m and every function and import
// they call, directly or not.
func reachedFromKernels(m *ir.Module) map[ir.Callee]bool {
	reached := map[ir.Callee]bool{}
	var work []*ir.Func
	for _, f := range m.Funcs() {
		if f.Signature().CallConv() == ir.Kernel {
			reached[f] = true
			work = append(work, f)
		}
	}
	for len(work) > 0 {
		f := work[len(work)-1]
		work = work[:len(work)-1]
		f.WalkInsts(func(in *ir.Inst) bool {
			if in.Op().Verb != ir.VCall {
				return true
			}
			c := in.Callee()
			if c == nil || reached[c] {
				return true
			}
			reached[c] = true
			if g, ok := c.(*ir.Func); ok {
				work = append(work, g)
			}
			return true
		})
	}
	return reached
}

// forwardCalled is the functions called from a body that comes before
// their own in the module, in module order.
func forwardCalled(m *ir.Module) []*ir.Func {
	defined := map[*ir.Func]bool{}
	needed := map[*ir.Func]bool{}
	var out []*ir.Func
	for _, f := range m.Funcs() {
		defined[f] = true
		f.WalkInsts(func(in *ir.Inst) bool {
			if in.Op().Verb != ir.VCall {
				return true
			}
			callee, ok := in.Callee().(*ir.Func)
			if ok && !defined[callee] && !needed[callee] && callee.Signature().CallConv() != ir.Kernel {
				needed[callee] = true
				out = append(out, callee)
			}
			return true
		})
	}
	return out
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
