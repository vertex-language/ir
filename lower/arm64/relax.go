package arm64

import (
	"strconv"

	arm64asm "github.com/vertex-language/arm64"

	"github.com/vertex-language/ir"
	"github.com/vertex-language/ir/lower/mir"
	"github.com/vertex-language/ir/lower/regalloc"
)

// Branch relaxation: a conditional branch that cannot reach its target.
//
// B.cond and CBZ/CBNZ hold a 19-bit word displacement, ±1MB; B holds 26
// bits, ±128MB. A function past a megabyte -- a table of a few thousand
// struct literals is one -- has conditional branches the short form cannot
// write, and such a branch is written as the inverse condition over a B:
//
//	b.ne  far        b.eq  1f
//	                 b     far
//	                 1:
//
// The decision has to be made before the branch is written rather than by
// the assembler afterwards. emit records offsets as it goes -- an invoke's
// call-site range, a pad's landing point, the function's length for its
// unwind record -- and a word inserted after the fact would leave every one
// of them wrong. So a function large enough to need it is emitted twice:
// once into a module of its own, which is thrown away and is only there to
// say where each block and each conditional branch landed, and once for
// real, with the branches that may not reach written long.

// shortReach is how far B.cond, CBZ and CBNZ reach forward: the largest
// word displacement 19 signed bits hold, in bytes. Backward reaches one word
// further, which is not worth a second constant.
const shortReach = (1<<18 - 1) * 4

// relaxEstimate is what a function is assumed to cost, in bytes, per
// machine instruction selected, when deciding whether to measure it at
// all: no op emits more than a handful of words, and a function would have
// to average sixteen to reach a megabyte below the line this draws. An
// asm op is counted by its template instead, since one may be any length.
const relaxEstimate = 64

// branchPlan is what emit is told about the conditional branches of one
// function, and, in the measuring pass, what it records about them.
type branchPlan struct {
	// far holds the conditional branches to write long, by their place
	// in the order emit writes them. The measuring pass and the real one
	// walk the same instructions in the same order, which is what makes
	// an ordinal enough to match one to the other.
	far map[int]bool

	// n is the ordinal of the next conditional branch emit writes.
	n int

	// measure is set on the pass that is thrown away: it fills blocks
	// and sites.
	measure bool
	blocks  map[string]int
	sites   []condSite
}

// condSite is one conditional branch: its offset and the block it names.
type condSite struct {
	at     int
	target string
}

// cond writes the conditional branch emit has reached, to target. write
// writes it in the short form, with the condition inverted when invert is
// set, to the label it is given.
func (bp *branchPlan) cond(text *arm64asm.Section, fn *ir.Func, target string,
	write func(invert bool, label string)) {
	if bp == nil {
		write(false, target)
		return
	}
	i := bp.n
	bp.n++
	if bp.measure {
		bp.sites = append(bp.sites, condSite{at: text.Offset(), target: target})
	}
	if !bp.far[i] {
		write(false, target)
		return
	}
	// A bare label, which is no symbol and is folded away; '#' is in no
	// block label, so it cannot meet one.
	over := fn.Name() + "#far" + strconv.Itoa(i)
	write(true, over)
	text.B(arm64asm.Label(target))
	text.Label(over)
}

// block records where a block starts, on the measuring pass.
func (bp *branchPlan) block(text *arm64asm.Section, label string) {
	if bp != nil && bp.measure {
		bp.blocks[label] = text.Offset()
	}
}

// planBranches decides which of mf's conditional branches emit writes
// long, or returns nil when the function is too small for any of them to
// need it.
func planBranches(features arm64asm.FeatureSet, fn *ir.Func, mf *mir.Func,
	assigned map[mir.VReg]regalloc.PhysReg, fr *frame, sv saves) (*branchPlan, error) {
	size := 0
	for _, mb := range mf.Blocks {
		for _, in := range mb.Instrs {
			if op, ok := in.Op.(asmOp); ok {
				size += len(op.template) + 4
				continue
			}
			size += relaxEstimate
		}
	}
	if size <= shortReach {
		return nil, nil
	}

	// The measuring pass, into a module nothing else sees. No exception
	// plan: what emit does with one is record offsets, never choose code.
	scratch := arm64asm.NewModule(arm64asm.WithFeatures(features))
	measured := &branchPlan{measure: true, blocks: map[string]int{}}
	if err := emit(scratch, scratch.Section(arm64asm.Text), fn, mf, assigned, fr, sv, nil, measured); err != nil {
		return nil, err
	}
	if scratch.Err() != nil {
		// The real pass will fail the same way, and say so in the
		// module it is writing; these offsets mean nothing.
		return nil, nil
	}

	// Writing a branch long adds one word, so between any two points the
	// real pass can grow by at most a word per conditional branch. A
	// branch is written long if it might not reach once that growth is
	// counted, which settles it in one pass rather than a fixed point.
	slack := 4 * len(measured.sites)
	far := map[int]bool{}
	for i, s := range measured.sites {
		to, ok := measured.blocks[s.target]
		if !ok {
			continue
		}
		d := to - s.at
		if d < 0 {
			d = -d
		}
		if d+slack > shortReach {
			far[i] = true
		}
	}
	if len(far) == 0 {
		return nil, nil
	}
	return &branchPlan{far: far}, nil
}
