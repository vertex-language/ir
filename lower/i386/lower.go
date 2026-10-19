// Package i386 lowers VIR to a finished, immutable *i386/obj.Object.
//
// # Key Design Details
//
//   - 64-bit values (i64) are split into pairs of 32-bit registers (low:high),
//     using paired instructions (ADD/ADC, SUB/SBB, SHLD/SHRD).
//   - All function call arguments are passed on the stack per the Intel386 psABI;
//     returns use EAX (or EDX:EAX for i64).
//   - Six allocatable GPRs: EAX, ECX, EDX, EBX, ESI, EDI.
//   - Floating-point computation uses SSE2 (XMM0..XMM7); returns pass through x87 ST(0).
//   - Wide conversions and 64-bit divisions lower to runtime helper calls.
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
			if _, comdat := x.ComdatAttr(); comdat {
				// Emitting it as an ordinary definition would link once
				// and fail the moment a second unit carried the same
				// function; refusing keeps the failure at the compiler.
				return nil, fmt.Errorf("lower: @%s asks for a comdat group, which this target does not emit yet", x.Name())
			}
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

	restrictBytes(mf, pool)
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
