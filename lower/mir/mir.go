// Package mir is machine IR: instructions in terms of virtual registers,
// still rewritable, with no opinion about which architecture they target.
//
// It exists to prove one design fact stated in the ir/lower README: register
// allocation is a function of MIR, not of VIR and not of a finished
// obj.Object, because only MIR has both vregs and the physregs an allocator
// assigns them into. Op is `any` rather than a fixed enum so a second
// architecture's isel can define its own opcode type without this package
// ever knowing it exists.
//
// This is deliberately minimal, and it grew exactly once. A Func began as
// an ordered list of Blocks with no CFG shape recorded at all, on the
// grounds that nothing needed to ask where control went — only to walk
// every instruction in program order, which an ordered slice answers.
// Liveness is what needed to ask. A value is live at a point when some
// path from that point reaches a use, and "some path" is not a question
// an ordered slice can answer, so Block gained Succs and this package
// gained live.go.
//
// What it still does not have is predecessors, dominance, or loop
// structure. Liveness is a backward analysis and reads only successors;
// the rest is the day something needs it.
package mir

// VReg is a virtual register: an SSA-like identity with no physical home
// until a register allocator gives it one.
type VReg int

// Instr is one machine instruction in terms of virtual registers.
//
// Op is target-specific and opaque to this package on purpose — mir carries
// no opinion about what a "move" or an "add" is on any particular
// architecture, only the def/use shape a register allocator needs to see.
type Instr struct {
	Op   any
	Defs []VReg
	Uses []VReg

	// Imm is the instruction's immediate operand, if it has one. A single
	// int64 covers every immediate this milestone's isel builds; a wider
	// need is a wider field when something actually needs it.
	Imm    int64
	HasImm bool

	// Copy marks an instruction that does nothing but put Uses[0] into
	// Defs[0].
	//
	// It is the one thing about an Op this package knows without
	// interpreting it, and it is here because a register allocator has to
	// know it. Every other instruction's destination conflicts with its
	// own operands — writing the destination would destroy an operand the
	// instruction has not finished reading. A copy is the exception: its
	// destination *is* its operand's value, so the two may share a
	// register, and when they do the instruction is a no-op the target
	// can drop.
	//
	// That is what makes it possible to ask for a value in a particular
	// physical register — copy it into a vreg pinned there — without
	// paying for a move when it is already in that register.
	Copy bool
}

// Block is one straight-line run of instructions, ending in whatever
// terminator instruction its own Instrs' last entry is — mir does not
// distinguish a terminator from any other Instr.
type Block struct {
	Label  string
	Instrs []Instr

	// Succs are the blocks control can reach from this one, in no
	// significant order. A block with none is one that returns.
	//
	// Set by whoever built the block, not derived here. An Instr's Op is
	// `any` and mir does not interpret it, so this package cannot read a
	// terminator to find out where it goes — which is the same reason Op
	// is `any` in the first place, and the price of it.
	Succs []*Block
}

// Func is a function body in MIR: an ordered list of blocks.
type Func struct {
	Blocks []*Block

	next    VReg
	byLabel map[string]*Block
}

// NewFunc returns an empty function body.
func NewFunc() *Func { return &Func{byLabel: map[string]*Block{}} }

// NewVReg allocates a fresh virtual register, unique within this Func.
func (f *Func) NewVReg() VReg {
	v := f.next
	f.next++
	return v
}

// NewBlock appends an empty block labeled name and returns it for the
// caller to fill in.
func (f *Func) NewBlock(label string) *Block {
	b := &Block{Label: label}
	f.Blocks = append(f.Blocks, b)
	if f.byLabel == nil {
		f.byLabel = map[string]*Block{}
	}
	f.byLabel[label] = b
	return b
}

// Block returns the block with this label, or nil.
//
// It exists because a target's isel names its branch targets by label —
// they are what the assembler will take — and Succs wants the block. A
// label is the one identity both halves already agree on.
func (f *Func) Block(label string) *Block { return f.byLabel[label] }

// Succ records that control can reach the block labeled label from b.
// A label naming no block is ignored: it is a branch out of the function,
// which has no successor here to record.
func (f *Func) Succ(b *Block, label string) {
	if s := f.Block(label); s != nil {
		b.Succs = append(b.Succs, s)
	}
}

// NumVRegs is one more than the highest VReg this function has handed
// out, which makes it the length of a slice indexed by VReg.
func (f *Func) NumVRegs() int { return int(f.next) }

// Emit appends an instruction to the block.
func (b *Block) Emit(in Instr) { b.Instrs = append(b.Instrs, in) }
