package ir

// Optimize runs the scalar cleanups a CPU backend wants before it selects
// instructions, the ones the AIR backend runs for Apple's compiler:
//
//   - a struct's frame slot is split into its fields, and locals the
//     frontend kept in slots become registers (SplitSlots, PromoteSlots);
//   - a branch on two constants is decided -- every shift by a literal
//     has two, Swift's shift testing its amount against the width -- and
//     arithmetic on constants is done (FoldConstants);
//   - a multiply or divide by a power of two is a shift (ReducePowersOfTwo);
//   - the chains of blocks folding and inlining leave are merged
//     (MergeBlocks), and a value computed twice is computed once
//     (ThreadJoins, NumberValues).
//
// Without them every shift by a literal was a compare and two branches,
// and a block of straight-line code a run of jumps to the next
// instruction.
//
// It runs once on a function: the slot passes are not run twice. What
// changes a function after it -- inlining -- is cleaned up by Tidy.
func (f *Func) Optimize() {
	f.SplitSlots()
	f.PromoteSlots()
	f.Tidy()
}

// Tidy is Optimize without the slot passes: constants folded, powers of
// two reduced, blocks merged and values numbered again, for a function
// whose slots are already registers -- one that has had small callees
// inlined into it since Optimize ran.
func (f *Func) Tidy() {
	f.FoldConstants()
	f.ReducePowersOfTwo()
	f.MergeBlocks()
	f.ThreadJoins()
	f.NumberValues()
	f.MergeBlocks()
}
