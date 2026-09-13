// Package amdgpu lowers VIR to a finished, immutable *amdgpu/obj.Object:
// an HSA code object for a CDNA or GCN processor, ready for
// amdgpu/obj/elf.WriteHSACO and hipModuleLoadData.
//
// # Milestones
//
//   - 26, a straight line. A kernel's arguments s_load out of the
//     kernarg segment, the work-item id out of v0, a global load and a
//     store through a flat address, s_waitcnt after every one of them,
//     and s_endpgm. Four register classes, split statically between
//     singles and pairs; see types.go. vector_add decodes under llvm-mc
//     and its descriptor is clang's for the same register counts.
//   - 27, uniform control flow. A branch whose condition every lane
//     agrees on is s_cmp on the lane mask and s_cbranch; a loop with a
//     uniform trip count is that twice. A branch on a divergent
//     condition is refused by name — the exec-mask lowering is the next
//     milestone, and refusing is what keeps this one honest.
//   - 28, calls. There is no device calling convention yet, so every
//     call a kernel makes is inlined first, by lower/inline, and what
//     remains — a call through a pointer, a call to an import — is
//     refused. Lower rewrites the module it is given to do this.
//
// # What is different about this target
//
// Memory is asynchronous. A load issues and returns; its result arrives
// when a counter says so, and reading the register before then reads
// whatever was there. At O0 every memory instruction is followed by an
// s_waitcnt for all of them, which is correct and slow, and is the pass
// LLVM calls SIInsertWaitcnts done as a rule rather than an analysis.
//
// The wave is the unit of execution. An i1 is a lane mask — one bit per
// work-item, held in an SGPR pair — and a compare writes one. A branch
// is scalar: the hardware branches the whole wave or none of it, which
// is why a condition that differs across lanes cannot be a branch at all
// and needs the execution mask instead.
package amdgpu

import (
	"fmt"

	amdgpuasm "github.com/vertex-language/amdgpu"
	"github.com/vertex-language/amdgpu/feature"
	amdgpuobj "github.com/vertex-language/amdgpu/obj"

	"github.com/vertex-language/ir"
	"github.com/vertex-language/ir/lower/globals"
	"github.com/vertex-language/ir/lower/inline"
	"github.com/vertex-language/ir/lower/mir"
	"github.com/vertex-language/ir/lower/regalloc"
)

// OptLevel is how hard Lower tries. Only O0 exists today.
type OptLevel uint8

const O0 OptLevel = iota

// Options are the things Lower has no opinion about.
type Options struct {
	OptLevel OptLevel

	// ASIC is the processor being compiled for. The zero value is
	// refused: unlike a CPU, there is no baseline every AMD GPU runs.
	ASIC feature.ASIC

	// Wave is the wavefront width. The zero value is the ASIC's own.
	Wave feature.WaveSize

	// Features overrides the ASIC's default feature set.
	Features feature.Set
}

// Lower builds an AMDGPU code object from m.
//
// Every call a kernel makes is inlined into it first, which changes m:
// a device function has no calling convention on this target yet, and
// its body copied into each kernel that calls it is the one way it runs.
// A device function that is not exported is left alone once its callers
// have their copies; an exported one is refused, since something outside
// the module would call it.
func Lower(m *ir.Module, opts Options) (*amdgpuobj.Object, error) {
	if err := checkLayout(m); err != nil {
		return nil, err
	}
	if err := inline.Module(m, inline.Options{Into: isKernel}); err != nil {
		return nil, fmt.Errorf("lower: %w", err)
	}
	if opts.ASIC == 0 {
		return nil, fmt.Errorf("lower: Options.ASIC names no processor; there is no AMD GPU every kernel runs on")
	}
	switch opts.ASIC.Generation() {
	case feature.GenGCN, feature.GenCDNA:
	default:
		return nil, fmt.Errorf("lower: %s is %s, and this package encodes for GFX9 and CDNA only", opts.ASIC, opts.ASIC.Generation())
	}
	mopts := []amdgpuasm.Option{amdgpuasm.WithASIC(opts.ASIC)}
	if opts.Wave != 0 {
		mopts = append(mopts, amdgpuasm.WithWaveSize(opts.Wave))
	}
	var unset feature.Set
	if opts.Features != unset {
		mopts = append(mopts, amdgpuasm.WithFeatures(opts.Features))
	}
	am := amdgpuasm.NewModule(mopts...)
	l := &lowerer{m: m, opts: opts, am: am}

	for _, f := range m.FuncImports() {
		am.Extern(f.Name())
	}
	for _, g := range m.GlobalImports() {
		am.Extern(g.Name())
	}
	if err := globals.Lower(globalTarget{l: l}, m); err != nil {
		return nil, err
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
	return am.Finalize()
}

func isKernel(f *ir.Func) bool { return f.Signature().CallConv() == ir.Kernel }

// checkLayout refuses a module whose layout block is not the device's.
func checkLayout(m *ir.Module) error {
	l := m.Layout()
	switch {
	case m.Use() != ir.AMDGCN.Use():
		return fmt.Errorf("lower: module %q is for %q; this package lowers for %q", m.Name(), m.Use(), ir.AMDGCN.Use())
	case l.PtrBits != 64:
		return fmt.Errorf("lower: module %q declares ptrbits %d; AMDGPU pointers are 64 bits", m.Name(), l.PtrBits)
	case l.Endian != ir.LittleEndian:
		return fmt.Errorf("lower: module %q declares %s-endian; the device is little-endian", m.Name(), l.Endian)
	case l.ABI != "hsa":
		return fmt.Errorf("lower: module %q declares abi %q; the device convention is hsa", m.Name(), l.ABI)
	}
	return nil
}

type lowerer struct {
	m    *ir.Module
	opts Options
	am   *amdgpuasm.Module

	// lds is workgroup storage laid out so far: each shared global's
	// offset, and the total a kernel that reaches any of them asks for.
	lds     map[string]uint32
	ldsSize uint32
}

// wave64 reports whether the wave is 64 lanes wide, which every GFX9
// and CDNA wave is.
func (l *lowerer) wave64() bool { return l.am.WaveSize() != feature.Wave32 }

// lowerFunc runs the pipeline for one function.
func (l *lowerer) lowerFunc(fn *ir.Func) error {
	if !isKernel(fn) {
		if fn.Linkage() == ir.Export {
			return fmt.Errorf("lower: @%s: an exported device function; there is no calling convention for the device yet, and only a kernel's own calls are inlined", fn.Name())
		}
		return nil
	}
	if _, ok := fn.AsmBodyText(); ok {
		return fmt.Errorf("lower: @%s: a function whose body is assembly is not emitted yet", fn.Name())
	}
	for _, blk := range fn.Blocks() {
		if blk.IsPad() {
			return fmt.Errorf("lower: @%s: @%s is a pad block; there is no unwinding on the device", fn.Name(), blk.Label())
		}
	}

	uni := analyzeUniformity(fn)
	mf := mir.NewFunc()
	pool := pool()
	vr := newVRegs(mf, pool, len(fn.Params()))

	blocks := fn.Blocks()
	mbs := make([]*mir.Block, len(blocks))
	for i, blk := range blocks {
		mbs[i] = mf.NewBlock(blockLabel(fn, blk))
	}
	k, err := l.newKernel(fn)
	if err != nil {
		return err
	}
	x := &fnState{l: l, fn: fn, mf: mf, vr: vr, uni: uni, k: k}
	if err := x.entry(mbs[0]); err != nil {
		return fmt.Errorf("lower: @%s: %w", fn.Name(), err)
	}
	// Block parameters get vregs before any block is selected, since an
	// edge into a block may be selected before the block is.
	for _, blk := range blocks {
		for _, p := range blk.Params() {
			if _, err := vr.define(p); err != nil {
				return fmt.Errorf("lower: @%s: @%s: %w", fn.Name(), blk.Label(), err)
			}
		}
	}
	// Selection in reverse postorder, so a value is selected before any
	// block that uses it; the blocks stay in the function's own order.
	at := map[*ir.Block]int{}
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
		c := newCursor(fn, mf, mbs[i])
		if err := x.selectBlock(c, blk); err != nil {
			return fmt.Errorf("lower: @%s: @%s: %w", fn.Name(), blk.Label(), err)
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

	assigned, err := regalloc.Assign(mf, pool)
	if err != nil {
		return fmt.Errorf("lower: @%s: %w; spilling needs scratch memory, which is not set up yet", fn.Name(), err)
	}
	return l.emit(x, assigned)
}
