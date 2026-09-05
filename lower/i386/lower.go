// Package i386 lowers VIR to a finished, immutable *i386/obj.Object.
//
// The third architecture in this tree and the first that is not 64-bit, which
// is the only thing about it that is genuinely hard.
//
// # What differs from the other two backends, and why
//
// An i64 does not fit in a register. Every other backend here maps one
// ir.Def to one vreg; this one maps an i64 to two, a low half and a high
// half, and every 64-bit verb becomes a pair of instructions over them —
// ADD then ADC, SUB then SBB, and the shifts through SHLD and SHRD. That
// choice is what the rest of the package is arranged around: `value` is a
// pair rather than a register, and isel asks for halves rather than
// operands.
//
// The pair is safe against the register allocator for a reason worth
// stating: ADC reads the carry ADD left, and the only instructions the
// allocator inserts are spills and reloads, which are MOVs — and a MOV on
// x86 does not touch the flags. On an architecture whose moves did, this
// would need the two halves welded into one MIR instruction.
//
// Every argument is on the stack. There is no register argument in the
// Intel386 psABI, so the outgoing area is the whole story of a call and
// classification is arithmetic rather than a two-file walk. A 64-bit
// argument is two slots; the return comes back in EAX, or EDX:EAX.
//
// Six allocatable registers. ESP and EBP are spoken for, which leaves EAX,
// ECX, EDX, EBX, ESI and EDI — and an i64 occupies two of the six. Spilling
// is the normal case here rather than the exceptional one.
//
// Floats in SSE2, returned through x87. The psABI's float unit is x87, whose
// registers are a stack that nothing in this tree's register allocator can
// model, so every float value here lives in an XMM register instead — eight
// flat registers that allocate like any other file. The price is a baseline
// that is no longer a 386, SSE2 having arrived with the Pentium 4, and one
// place where x87 survives anyway: the psABI returns a float in ST(0), so a
// return is a store and an FLD and taking a result back is an FSTP. See
// isel_float.go.
//
// Half of §C2 is a call. SSE2 converts between a float and a *32-bit*
// integer and nothing wider — the 64-bit forms are REX.W, which is x86-64 —
// so every conversion with a 64-bit end is a compiler-rt helper, and the
// unsigned 32-bit ones are the signed instruction with a bias around it.
// The four rounding verbs and fma are libm's for the same kind of reason:
// ROUNDSD is SSE4.1 and VFMADD is FMA3, and neither is in this baseline.
// See isel_cvt.go.
//
// # What is still missing
//
// f80 and f128, which are not values here. f80 is x87's own format and the
// register file this package does not allocate; f128 would need the
// soft-float table the amd64 backend has and this one does not. Both are
// refused by name, as source, destination or storage.
//
// Of the rest: §D3's tlsaddr and §G3's unwinding, neither of which any
// backend in this tree implements. §G4's inline assembly is implemented —
// see asm.go — with the one exception that a 64-bit operand is refused,
// because it is a register pair here and one %-reference cannot name two
// registers.
//
// # Verifying this
//
// The generated code runs, which took some arranging. There is no way to
// execute a 32-bit x86 process on the host this was written on, so the test
// harness links the output against a freestanding runtime into a multiboot
// kernel and boots it under qemu-system-i386, reading the answers off the
// emulated serial port. See run_test.go, which explains the parts.
package i386

import (
	"fmt"

	i386asm "github.com/vertex-language/i386"
	"github.com/vertex-language/i386/feature"
	i386obj "github.com/vertex-language/i386/obj"

	"github.com/vertex-language/ir"
	"github.com/vertex-language/ir/lower/mir"
	"github.com/vertex-language/ir/lower/regalloc"
)

// OptLevel is how hard Lower tries. Only O0 exists today.
type OptLevel uint8

const O0 OptLevel = iota

// Options are the things Lower has no opinion about.
type Options struct {
	OptLevel OptLevel
}

// Lower builds an Intel386 object from m.
func Lower(m *ir.Module, opts Options) (*i386obj.Object, error) {
	if err := checkLayout(m); err != nil {
		return nil, err
	}

	// SSE2, which is what this package's floats are. That moves the
	// baseline off the 386 — SSE2 arrived with the Pentium 4 — and it is
	// the price of having a flat register file to allocate rather than
	// x87's stack.
	am := i386asm.NewModule(i386asm.WithFeatures(
		feature.New(feature.I686).Add(feature.SSE, feature.SSE2)))

	for _, f := range m.FuncImports() {
		am.Extern(f.Name())
	}
	for _, g := range m.GlobalImports() {
		am.Extern(g.Name())
	}

	declareLibcalls(am, m)

	if err := lowerGlobals(am, m); err != nil {
		return nil, err
	}

	// Items rather than Funcs, because a module-level asm block has a
	// position among them: it is emitted where it was declared, and that
	// is what decides where its bytes fall.
	text := am.Section(i386asm.Text)
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
	case l.PtrBits != 32:
		return fmt.Errorf("lower: module %q declares ptrbits %d; Intel386 pointers are 32 bits", m.Name(), l.PtrBits)
	case l.Endian != ir.LittleEndian:
		return fmt.Errorf("lower: module %q declares %s-endian; only little-endian is supported", m.Name(), l.Endian)
	case l.ABI != "sysv":
		return fmt.Errorf("lower: module %q declares abi %q; only sysv is supported", m.Name(), l.ABI)
	}
	return nil
}

// lowerFunc runs the full pipeline for one function.
func lowerFunc(am *i386asm.Module, text *i386asm.Section, fn *ir.Func, opts Options) error {
	if body, ok := fn.AsmBodyText(); ok {
		return emitAsmBody(text, fn, body)
	}

	fr, err := planFrame(fn)
	if err != nil {
		return err
	}

	mf := mir.NewFunc()
	pool := regalloc.NewPool(allocatable())
	pool.AddClass(vecClass, allocatableVec())
	vr := newVRegs(mf, pool, len(fn.Params()))

	entry := mf.NewBlock(blockLabel(fn, fn.Entry().Block()))
	if err := classifyParams(fn, entry, vr); err != nil {
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

	// Selection order is not emission order. An instruction's result
	// gets its vreg where the instruction is selected, so a block
	// using a value has to come after the block defining it — which
	// is reverse postorder, and not necessarily the order the blocks
	// were built in. See the same comment in lower/arm64. The mir
	// blocks stay indexed by the function's own order, so what is
	// emitted is unchanged.
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
	for _, blk := range blocks {
		if err := sel(blk); err != nil {
			return err
		}
	}

	assigned, err := regalloc.Spilling(mf, pool, &spiller{fr: fr})
	if err != nil {
		return fmt.Errorf("lower: %s: regalloc: %w", fn.Name(), err)
	}

	// The save slots come last, after spilling has taken whatever it
	// needed: reserve hands out frame in call order, and a save reserved
	// before the spills would have to be renumbered by them.
	saved := usedCalleeSaved(pool, assigned)
	fr.reserveSaves(saved)

	if err := emit(am, text, fn, mf, assigned, fr, saved); err != nil {
		return err
	}
	return nil
}
