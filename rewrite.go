package ir

// The two primitives a pass that copies code is built from: a block split
// at an instruction, and an instruction cloned under a renaming. Neither
// re-checks what the builder checked when the original was emitted; what
// they check is what changed — that the renaming answered for every
// operand, and that the split point is in the block.
//
// walk.go has the rest: ReplaceUses, Remove, RemoveBlock, SetTerm.

// SplitAfter moves everything after in — the instructions that follow it
// and the terminator — into a new block labeled label, and leaves b
// ending at in with no terminator, for the caller to supply one. The new
// block takes no parameters; a caller that needs some declares them
// before it emits anything else into the block.
//
// It is what an inliner does to the call site: the call's block ends at
// the call, the callee's copy runs, and its returns branch to the block
// holding what followed.
func (b *Block) SplitAfter(in *Inst, label string) *Block {
	f := b.fn
	if f == nil || f.m.err != nil {
		return nil
	}
	at := -1
	for i, x := range b.insts {
		if x == in {
			at = i
			break
		}
	}
	if at < 0 {
		f.m.fail(f.name, b.label, Op{}, ErrPlacement, "SplitAfter: %s is not in @%s", in.op, b.label)
		return nil
	}
	nb := f.newBlock(label)
	if f.m.err != nil {
		return nil
	}
	nb.insts = append([]*Inst(nil), b.insts[at+1:]...)
	for _, x := range nb.insts {
		x.move(nb)
	}
	if b.term != nil {
		nb.term = b.term
		b.term.move(nb)
	}
	b.insts = b.insts[:at+1]
	b.term = nil
	return nb
}

// move rehomes an instruction and its results.
func (in *Inst) move(to *Block) {
	in.blk = to
	for _, r := range in.results {
		r.blk = to
	}
}

// Clone emits into b a copy of in: the same op, immediates and metadata,
// fresh results with the original's names, operands and branch targets
// renamed through def and blk. A nil def or blk is the identity. Every
// operand must be answered for: a def that maps to nil, or to a value of
// another function, is the caller's bug and is reported as ErrPoison
// rather than emitted.
//
// A terminator becomes b's terminator; anything else is appended. The
// clone of a return is a return, which an inliner replaces on its own
// terms, since what a return means depends on where the copy is going.
func (b *Block) Clone(in *Inst, def func(*Def) *Def, blk func(*Block) *Block) *Inst {
	f := b.fn
	if f == nil || f.m.err != nil || in == nil {
		return nil
	}
	if def == nil {
		def = func(d *Def) *Def { return d }
	}
	if blk == nil {
		blk = func(x *Block) *Block { return x }
	}
	rename := func(d *Def) *Def {
		if d == nil {
			return nil
		}
		nd := def(d)
		if nd == nil || nd.fn != f {
			f.m.fail(f.name, b.label, in.op, ErrPoison, "Clone: no value for %%%s in @%s", d, f.name)
			return nil
		}
		return nd
	}
	args := make([]*Def, len(in.args))
	for i, a := range in.args {
		if args[i] = rename(a); args[i] == nil {
			return nil
		}
	}
	var im *imm
	if in.im != nil {
		c := *in.im
		if len(c.targets) > 0 {
			c.targets = make([]BlockTarget, len(in.im.targets))
			for i, t := range in.im.targets {
				nt := BlockTarget{blk: blk(t.blk), bare: t.bare}
				if len(t.args) > 0 {
					nt.args = make([]*Def, len(t.args))
					for j, a := range t.args {
						if nt.args[j] = rename(a); nt.args[j] == nil {
							return nil
						}
					}
				}
				c.targets[i] = nt
			}
		}
		if len(c.labels) > 0 {
			c.labels = make([]*Block, len(in.im.labels))
			for i, l := range in.im.labels {
				c.labels[i] = blk(l)
			}
		}
		if c.unwind != nil {
			c.unwind = blk(c.unwind)
		}
		im = &c
	}
	res := make([]RegType, len(in.results))
	for i, r := range in.results {
		res[i] = r.typ
	}
	out := b.emit(in.op, res, args, im)
	if out == nil {
		return nil
	}
	for i, r := range in.results {
		out.results[i].name = r.name
	}
	if len(in.meta) > 0 {
		out.meta = append([]Attach(nil), in.meta...)
	}
	return out
}
