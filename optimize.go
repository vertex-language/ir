package ir

// Optimize runs scalar optimizations prior to instruction selection:
//   - SplitSlots & PromoteSlots (SROA and mem2reg)
//   - FoldConstants (constant folding)
//   - ReducePowersOfTwo (strength reduction)
//   - MergeBlocks, ThreadJoins, NumberValues (CFG simplification and LVN)
//
// The slot promotion passes are run once; post-inlining cleanup uses Tidy.
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
