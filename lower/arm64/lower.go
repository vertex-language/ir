// Package arm64 lowers VIR to a finished, immutable *arm64/obj.Object.
//
// The second architecture in this tree, and written after the first rather
// than beside it, which is why it shares mir and regalloc and shares nothing
// else yet. What is genuinely common between the two backends is visible now
// in a way it could not be before — the frame planning, the parallel copy on a
// branch edge, the shape of isel's dispatch — and hoisting it is a job for
// when both are complete rather than one.
//
// # What differs from the amd64 backend, and why
//
// Three addresses, not two. Every A64 data-processing instruction names its
// destination separately from both operands, so nothing has to be copied into
// place before an add and the two-address dance amd64's isel does is simply
// absent here.
//
// No flags on arithmetic. ADD does not set NZCV; ADDS does. So a comparison is
// always its own instruction and there is no fusing to decide about — where
// the amd64 backend asks whether a compare's only reader is a branch, this one
// materializes every §B result with CSET and lets the allocator coalesce.
//
// A fixed instruction width, which is what makes constants and addresses
// interesting. No instruction takes a 64-bit literal: a wide constant is a
// MOVZ and a run of MOVKs, and a symbol's address is ADRP plus ADD.
//
// The frame record rather than a pushed frame pointer. STP with pre-index
// stores X29 and X30 and moves SP in one instruction, and X30 is a register
// rather than a return address on the stack — a leaf function that stores
// nothing needs no prologue at all.
//
// # What is still missing
//
// §A through §C4 lower completely, as do §D, §D2, §E, §F, §G and §H. What
// does not:
//
//   - §I under the base standard's variadic convention. Apple's variant is
//     written and is what Options.Variadic selects; the base standard needs
//     a register save area in the prologue and a two-region walk in va_arg,
//     and is refused by name rather than lowered as Apple's.
//   - §D3's tlsaddr.
//   - §C2 and §C3 out of an extended float. f80 is refused outright and
//     stays refused: AArch64's long double is binary128, so f80 is a type
//     this target does not have. f128 needs a soft-float table like the
//     amd64 backend's.
//   - sret as something lowering introduces. A signature that states one is
//     honoured, and §5.5 decides what becomes of it: a result small enough
//     or homogeneous enough comes back in registers and the parameter names
//     a slot of the callee's own, and anything else comes back through the
//     caller's storage, whose address is X8 — §6.9's indirect result
//     location register, so the first real argument is still X0. What is not
//     written is a return of more values than the signature declared, which
//     would have to allocate the storage and rewrite the call: a
//     legalization pass, not a selection rule.
//   - a homogeneous aggregate of f128. Its members want Q registers, and the
//     widths here are S and D; such an aggregate keeps the by-reference
//     passing every aggregate used to get, which is wrong against clang but
//     is wrong the way it already was rather than newly so. See abi.go.
//   - a byval aggregate past the last named argument of a variadic call. It
//     would have to be copied into the list and read back out by va_arg;
//     it is refused by name.
//   - §G3, which no backend in this tree implements. §G4 is implemented;
//     see asm.go.
package arm64

import (
	"fmt"

	arm64asm "github.com/vertex-language/arm64"
	arm64obj "github.com/vertex-language/arm64/obj"

	"github.com/vertex-language/ir"
	"github.com/vertex-language/ir/lower/mir"
	"github.com/vertex-language/ir/lower/regalloc"
)

// OptLevel is how hard Lower tries. Only O0 exists today.
type OptLevel uint8

const O0 OptLevel = iota

// A VariadicABI is which of AAPCS64's two variadic conventions a target uses.
//
// They are not compatible and the difference is not a detail: under the base
// standard a variadic argument goes in the register a named one would have
// gone in, and va_list is a five-field structure walking a register save area
// the callee's prologue writes. Under Apple's variant every variadic argument
// is on the stack and va_list is a pointer. Lowering a call for the wrong one
// is a wrong call rather than a slow one, which is why this is stated rather
// than guessed from the layout block — both platforms declare abi "aapcs".
type VariadicABI uint8

const (
	// VariadicAAPCS64 is the base standard's. Not implemented: it needs
	// the register save area in the prologue and the two-region walk in
	// va_arg, neither of which is written.
	VariadicAAPCS64 VariadicABI = iota

	// VariadicDarwin is Apple's variant. Every variadic argument occupies
	// one eight-byte stack slot, in order, and va_list is a plain pointer
	// at the next one.
	VariadicDarwin
)

// Options are the things Lower has no opinion about.
type Options struct {
	OptLevel OptLevel

	// Features is the processor being compiled for. The zero value is the
	// Armv8-A baseline, which every AArch64 processor implements.
	Features arm64asm.FeatureSet

	// Variadic is which variadic convention to use. The zero value is the
	// base standard's, which is not implemented and is refused by name.
	Variadic VariadicABI

	// LibcallPrefix is prepended to the name of every function this
	// package calls that the module did not declare — §E's memcpy and its
	// neighbours.
	//
	// Stated rather than derived. Nothing else here mangles a symbol: the
	// names a module writes are the names it gets, which is why its
	// functions are already spelled the way the platform wants them. A
	// name this package invents has no such author, and the two platforms
	// disagree — Mach-O prefixes a C symbol with an underscore and ELF
	// does not.
	LibcallPrefix string
}

func (o Options) features() arm64asm.FeatureSet {
	var unset arm64asm.FeatureSet
	if o.Features == unset {
		return arm64asm.Armv8A.Set()
	}
	return o.Features
}

// Lower builds an AArch64 object from m.
func Lower(m *ir.Module, opts Options) (*arm64obj.Object, error) {
	if err := checkLayout(m); err != nil {
		return nil, err
	}

	am := arm64asm.NewModule(arm64asm.WithFeatures(opts.features()))

	for _, f := range m.FuncImports() {
		am.Extern(f.Name())
	}
	for _, g := range m.GlobalImports() {
		am.Extern(g.Name())
	}
	for _, s := range libcallSyms(m, opts) {
		am.Extern(s)
	}

	if err := lowerGlobals(am, m); err != nil {
		return nil, err
	}

	// Items rather than Funcs, because a module-level asm block has a
	// position among them: it is emitted where it was declared, and two of
	// them may be one text with a declaration in between.
	text := am.Section(arm64asm.Text)
	nasm := 0
	for _, it := range m.Items() {
		switch x := it.(type) {
		case *ir.Func:
			if err := lowerFunc(am, text, x, opts); err != nil {
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

// checkLayout refuses a module whose layout block is not one this package can
// lower for.
func checkLayout(m *ir.Module) error {
	l := m.Layout()
	switch {
	case l.PtrBits != 64:
		return fmt.Errorf("lower: module %q declares ptrbits %d; AArch64 pointers are 64 bits", m.Name(), l.PtrBits)
	case l.Endian != ir.LittleEndian:
		// AArch64 can run big-endian and almost nothing does. Refusing is
		// not a claim the architecture cannot; it is that this package
		// emits little-endian data and has not been asked for the other.
		return fmt.Errorf("lower: module %q declares %s-endian; only little-endian is supported", m.Name(), l.Endian)
	case l.ABI != "aapcs":
		return fmt.Errorf("lower: module %q declares abi %q; only aapcs is supported", m.Name(), l.ABI)
	}
	return nil
}

// lowerFunc runs the full pipeline for one function.
func lowerFunc(am *arm64asm.Module, text *arm64asm.Section, fn *ir.Func, opts Options) error {
	if body, ok := fn.AsmBodyText(); ok {
		return emitAsmBody(text, fn, body)
	}

	fr, err := planFrame(fn, opts)
	if err != nil {
		return err
	}

	mf := mir.NewFunc()
	pool := regalloc.NewPool(allocatable())
	pool.AddClass(vecClass, allocatableVec())
	vr := newVRegs(mf, pool, len(fn.Params()))

	entry := mf.NewBlock(blockLabel(fn, fn.Entry().Block()))
	if err := classifyParams(fn, entry, vr, fr); err != nil {
		return err
	}
	if err := classifyBlockParams(fn, vr); err != nil {
		return err
	}

	blocks := fn.Blocks()
	mblocks := make([]*mir.Block, len(blocks))
	mblocks[0] = entry
	for i, blk := range blocks[1:] {
		mblocks[i+1] = mf.NewBlock(blockLabel(fn, blk))
	}

	// Selection order is not emission order.
	//
	// An instruction's result gets its vreg where the instruction is
	// selected, so a block that uses a value has to be selected after
	// the block that defines it. Dominance guarantees such an order
	// exists and reverse postorder is one, but the order the blocks
	// were built in is not: a frontend that splits a block partway
	// through lowering appends the halves after blocks that already
	// branch to them, and the list stops matching the graph.
	//
	// The mir blocks stay indexed by the function's own order, so
	// what is emitted is exactly what it was before.
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
		if err := iselBlock(fn, mf, vr, fr, blk, mblocks[i], opts); err != nil {
			return fmt.Errorf("lower: %s: %w", fn.Name(), err)
		}
		return nil
	}
	for _, blk := range fn.RPO() {
		if err := sel(blk); err != nil {
			return err
		}
	}
	// Anything RPO did not reach is unreachable, and is still emitted
	// — an empty mir block would leave a label with no body behind it.
	for _, blk := range blocks {
		if err := sel(blk); err != nil {
			return err
		}
	}

	assigned, err := regalloc.Spilling(mf, pool, &spiller{fr: fr})
	if err != nil {
		return fmt.Errorf("lower: %s: regalloc: %w", fn.Name(), err)
	}

	saved := usedCalleeSaved(pool, assigned)
	savedVec := usedCalleeSavedVec(pool, assigned)
	fr.reserveSaves(saved)
	fr.reserveSavesVec(savedVec)
	if len(saved) > 0 || len(savedVec) > 0 {
		fr.force = true
	}

	if err := emit(am, text, fn, mf, assigned, fr, saved, savedVec); err != nil {
		return err
	}
	return nil
}
