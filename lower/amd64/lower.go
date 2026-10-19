// Package amd64 lowers VIR to a finished, immutable *amd64/obj.Object.
//
// The pipeline validates module layout and features, emits data sections (global.go),
// and lowers functions via instruction selection (isel.go), register allocation (regalloc),
// stack frame layout (frame.go), and machine code emission (emit.go).
//
// Both System V AMD64 ABI and Microsoft x64 ABI are supported.
//
// # Register Files and Types
//
// Virtual registers map to 64-bit general-purpose registers (RAX..R15) and
// 128-bit XMM registers (XMM0..XMM15). Pointers are 64-bit values.
// Extended precision f128 is supported via compiler-rt helper calls (softfloat.go).
//
// # File Organization
//
//   - types.go: MIR opcodes and operand roles
//   - vregs.go: register classes and virtual register mapping
//   - frame.go: stack frame planning and ABI argument/return classification
//   - isel*.go: instruction selection
//   - control.go: block ordering, branches, and jump tables
//   - atomic.go: §H atomic operations
//   - emit.go: binary machine code generation
package amd64

import (
	"fmt"

	amd64asm "github.com/vertex-language/amd64"
	"github.com/vertex-language/amd64/feature"
	amd64obj "github.com/vertex-language/amd64/obj"

	"github.com/vertex-language/ir"
	"github.com/vertex-language/ir/lower/inline"
	"github.com/vertex-language/ir/lower/mir"
	"github.com/vertex-language/ir/lower/regalloc"
)

// OptLevel is how hard Lower tries. O0 lowers each function as it is
// given; O1 first runs ir.Func.Optimize on it, which a frontend that
// leaves the cleanup to the backend (vsc) wants and a test of one
// lowering does not.
type OptLevel uint8

const (
	O0 OptLevel = iota
	O1
)

// Options are the things Lower has no opinion about.
type Options struct {
	OptLevel OptLevel

	// Features is the processor being compiled for. The zero value is
	// the baseline, which every AMD64 processor implements.
	Features feature.Set

	// LibcallPrefix is prepended to the name of every function this
	// package calls that the module did not declare — §E's memcpy, and
	// the soft-float routines §C's f128 cases become.
	//
	// Stated rather than derived. The names a module writes are the names
	// it gets; a name this package invents has no such author, and the
	// two platforms disagree — Mach-O prefixes a C symbol with an
	// underscore and ELF does not.
	LibcallPrefix string
}

// features is the processor Options names, or the baseline.
func (o Options) features() feature.Set {
	var unset feature.Set
	if o.Features == unset {
		return feature.Default()
	}
	return o.Features
}

// Lower builds an AMD64 object from m.
func Lower(m *ir.Module, opts Options) (*amd64obj.Object, error) {
	if err := checkLayout(m); err != nil {
		return nil, err
	}

	// What a function may not execute becomes a call (§0) -- and so does
	// what it requires that x86-64 has no instruction for. The assembler
	// is told every feature some function may use: the rows themselves
	// are gated, and a requires function's are its own.
	m.LegalizeCryptoOn(capabilitiesOf(opts.features()), lowered)
	asmFeatures := opts.features()
	for _, f := range m.Funcs() {
		asmFeatures = asmFeatures.Add(funcFeatures(opts.features(), f).Features()...)
	}

	am := amd64asm.NewModule(amd64asm.WithFeatures(asmFeatures))

	// A weak import is a reference the link may leave unresolved.
	for _, f := range m.FuncImports() {
		if f.IsWeak() {
			am.ExternWeak(f.Name())
			continue
		}
		am.Extern(f.Name())
	}
	for _, g := range m.GlobalImports() {
		if g.IsWeak() {
			am.ExternWeak(g.Name())
			continue
		}
		am.Extern(g.Name())
	}
	for _, sym := range libcallSyms(m, opts.LibcallPrefix, opts.features()) {
		am.Extern(sym)
	}
	for _, sym := range softFloatSyms(m, opts.LibcallPrefix) {
		am.Extern(sym)
	}
	if needsTLSIndex(m) {
		// The CRT's, and a reference rather than a definition: the loader
		// fills it with the slot this image was given. It is declared
		// here rather than where it is used because a section's
		// references resolve against the module, and the module is what
		// knows whether anything is thread-local at all.
		am.Extern(tlsIndexSymbol)
	}

	if err := lowerGlobals(am, m); err != nil {
		return nil, err
	}

	if opts.OptLevel >= O1 {
		// The small callees -- a rotate, a getter, a helper of a few
		// operations -- are copied into their callers before anything is
		// lowered, so that ir.Func.Optimize sees through them and no call
		// is paid for an operation or two.
		// Every body is cleaned up first, so that a callee is measured as
		// it will be lowered: a frontend's locals are slots until
		// PromoteSlots, and a getter of one load would otherwise count
		// as a function with a stack frame.
		for _, f := range m.Funcs() {
			if _, asm := f.AsmBodyText(); !asm && len(f.Blocks()) > 0 {
				f.Optimize()
			}
		}
		for _, f := range m.Funcs() {
			if err := inline.Func(f, inline.Options{Should: smallCallee}); err != nil {
				return nil, err
			}
		}
	}

	// The §V verbs a function has no instruction for are rewritten into
	// the ones it has -- after inlining, so that a small helper's byte
	// swap copied into a function that requires aes is that function's
	// PSHUFB and not the helper's baseline sequence or call.
	// The swizzle's call is an import the externs above did not see.
	imported := map[string]bool{}
	for _, f := range m.FuncImports() {
		imported[f.Name()] = true
	}
	for _, f := range m.Funcs() {
		f.ExpandVectorVerbs(expandFor(funcFeatures(opts.features(), f)))
	}
	for _, f := range m.FuncImports() {
		if !imported[f.Name()] {
			am.Extern(f.Name())
		}
	}

	// Items rather than Funcs, because a module-level asm block has a
	// position among them: it is emitted where it was declared, and that
	// is what decides where its bytes fall.
	text := am.Section(amd64asm.Text)
	nasm := 0
	for _, it := range m.Items() {
		switch x := it.(type) {
		case *ir.Func:
			// A comdat function is a section of its own, elected on its
			// name, so that the linker can discard the copies every other
			// unit emitted without touching anything beside them. Its
			// unwind records follow it (see emitUnwind).
			sec := text
			if key, comdat := x.ComdatAttr(); comdat {
				if key == "" {
					key = x.Name()
				}
				if key != x.Name() {
					return nil, fmt.Errorf("lower: @%s asks for comdat %q; a function's group is keyed on its own name", x.Name(), key)
				}
				sec = am.ComdatSection(amd64asm.Text.String(), amd64asm.Text, key)
			}
			if err := lowerFunc(am, sec, x, opts); err != nil {
				return nil, err
			}
		case *ir.ModuleAsm:
			if err := emitModuleAsm(text, x, nasm); err != nil {
				return nil, err
			}
			nasm++
		}
	}

	return am.Finalize()
}

// smallCallee is O1's inlining policy: a callee of at most smallInsts
// instructions that allocates no stack slot, which inlined into a loop
// would be one more slot for every iteration's copy of the body.
func smallCallee(_, callee *ir.Func) bool {
	n := 0
	for _, b := range callee.Blocks() {
		for _, in := range b.Insts() {
			switch in.Op().Verb {
			case ir.VAlloc, ir.VStackSave, ir.VInvoke, ir.VInvokeInd:
				return false
			}
			n++
			if n > smallInsts {
				return false
			}
		}
	}
	return true
}

// smallInsts is how many instructions a callee O1 inlines may have.
const smallInsts = 40

// checkLayout refuses a module whose layout block is not one this
// package can lower for (64-bit, little-endian, SysV ABI).
func checkLayout(m *ir.Module) error {
	l := m.Layout()
	switch {
	case l.PtrBits != 64:
		return fmt.Errorf("lower: module %q declares ptrbits %d; amd64 pointers are 64 bits", m.Name(), l.PtrBits)
	case l.Endian != ir.LittleEndian:
		return fmt.Errorf("lower: module %q declares %s-endian; amd64 is little-endian", m.Name(), l.Endian)
	case l.ABI != "sysv" && l.ABI != "ms":
		return fmt.Errorf("lower: module %q declares abi %q; only sysv and ms are supported", m.Name(), l.ABI)
	}
	return nil
}

// lowerFunc runs the full pipeline for one function: frame planning,
// isel, register allocation, and emission.
func lowerFunc(am *amd64asm.Module, text *amd64asm.Section, fn *ir.Func, opts Options) error {
	if body, ok := fn.AsmBodyText(); ok {
		return emitAsmBody(text, fn, body)
	}
	if opts.OptLevel >= O1 {
		// Lower ran ir.Func.Optimize on every function before inlining;
		// what inlining brought in is folded, merged and numbered here.
		fn.Tidy()
	}

	feat := funcFeatures(opts.features(), fn)
	fr, err := planFrame(fn, feat)
	if err != nil {
		return err
	}

	abi := fn.Module().Layout().ABI
	mf := mir.NewFunc()
	pool := regalloc.NewPool(allocatable(abi))
	pool.AddClass(xmmClass, allocatableXmm(abi))
	vr := newVRegs(mf, pool, len(fn.Params()))

	blocks := fn.Blocks()
	mbs := make([]*mir.Block, len(blocks))
	var entry *mir.Block
	for i, blk := range blocks {
		mbs[i] = mf.NewBlock(blockLabel(fn, blk))
		if blk.IsEntry() {
			entry = mbs[i]
		}
	}
	if err := classifyParams(fn, entry, vr, fr); err != nil {
		return err
	}
	if err := classifyBlockParams(fn, vr); err != nil {
		return err
	}

	// Selection order is not emission order. An instruction's result
	// gets its vreg where the instruction is selected, so a block
	// using a value has to come after the block defining it — which
	// is reverse postorder, and not necessarily the order the blocks
	// were built in. See the same comment in lower/arm64. The mir
	// blocks stay indexed by the function's own order, so what is
	// emitted is unchanged.
	uses := indexUses(fn)
	at := make(map[*ir.Block]int, len(blocks))
	for i, blk := range blocks {
		at[blk] = i
	}
	done := make([]bool, len(blocks))
	sel := func(blk *ir.Block) error {
		i, ok := at[blk]
		if !ok || done[i] {
			return nil
		}
		done[i] = true
		if err := iselBlock(fn, mf, newCursor(fn, mf, mbs[i], opts.LibcallPrefix, feat), vr, fr, blk, uses); err != nil {
			return fmt.Errorf("lower: %s: %w", fn.Name(), err)
		}
		return nil
	}
	for _, blk := range fn.RPO() {
		if err := sel(blk); err != nil {
			return err
		}
	}
	for _, blk := range blocks {
		if err := sel(blk); err != nil {
			return err
		}
	}

	assigned, err := regalloc.Spilling(mf, pool, &spiller{fr: fr})
	if err != nil {
		return fmt.Errorf("lower: %s: %w", fn.Name(), err)
	}

	saved := usedCalleeSaved(abi, pool, assigned)
	fr.reserveSaves(saved)

	start := text.Offset()
	shape, err := emit(am, text, fn, mf, assigned, fr, saved)
	if err != nil {
		return err
	}
	if abi == abiMS {
		// Windows walks a frame through .pdata and .xdata rather than
		// through the code, and a frame with no record is one nothing can
		// unwind through. See unwind.go.
		return emitUnwind(am, text, fn.Name(), text.Offset()-start, shape)
	}
	return nil
}
