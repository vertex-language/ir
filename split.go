package ir

import "sort"

// Frame-slot splitting: scalar replacement of aggregates, the part of it
// promotion needs.
//
// A frontend keeps a struct in a frame slot and reaches each field by an
// offset from it -- `ptr.add %slot, 8` then a load -- and PromoteSlots
// leaves such a slot alone, since an address computation is not a plain
// load or store. When every use of the slot is a plain access at a
// constant offset, directly or through such a `ptr.add`, and the accesses
// at different offsets do not overlap, the slot is really one variable per
// field. This pass makes it that: a slot of its own for each offset, each
// access rewritten to it, the address computations gone. PromoteSlots then
// promotes each on its own.
//
// It matters most where a pointer lives in a struct on a GPU: a device has
// no generic pointer, a pointer's space is inferred from where it came
// from, and one read back from memory has come from nowhere. Split and
// promoted, it comes from where it was made.
//
// What is left alone: a slot any other instruction touches, a volatile
// access, an offset that is not a constant, a slot sized by a named type,
// and accesses at one offset of different widths or that overlap another.

// SplitSlots splits every splittable alloc in f into one per field, and
// reports how many it split.
func (f *Func) SplitSlots() int {
	if f == nil || f.entry == nil || f.m == nil || f.m.err != nil {
		return 0
	}
	var slots []*Inst
	for _, in := range f.entry.insts {
		if in.op.Verb == VAlloc && in.im != nil && in.im.typ == nil && len(in.args) == 0 {
			slots = append(slots, in)
		}
	}
	if len(slots) == 0 {
		return 0
	}
	uses := f.slotUses()
	n := 0
	for _, alloc := range slots {
		if f.splitSlot(alloc, uses[alloc.results[0]]) {
			n++
		}
	}
	return n
}

// fieldAccess is one load or store of a slot, at an offset from it.
type fieldAccess struct {
	in    *Inst
	arg   int // which operand is the address
	off   uint64
	width uint64
}

// splitSlot splits one slot, if every use allows it.
func (f *Func) splitSlot(alloc *Inst, uses []Use) bool {
	var accesses []fieldAccess
	var adds []*Inst
	// access records in, which reads or writes through operand arg at off.
	access := func(in *Inst, idx int, off uint64) bool {
		if in.im != nil && in.im.volatile {
			return false
		}
		width, _, load, ok := accessWidth(in.op.Verb)
		if !ok {
			return false
		}
		if load && idx != 0 || !load && idx != 1 {
			return false
		}
		if width == 0 {
			width = regBytes(in.op.Type)
			if width == 0 {
				return false
			}
		}
		accesses = append(accesses, fieldAccess{in: in, arg: idx, off: off, width: width})
		return true
	}
	all := f.allUses()
	for _, u := range uses {
		in := u.Inst
		if u.Target >= 0 {
			return false
		}
		// A store of the slot's own address somewhere is its escape.
		if in.op.Verb == VStore && u.Index == 0 {
			return false
		}
		if in.op.Verb == VAdd && in.op.Type == TypePtr && u.Index == 0 && len(in.args) == 2 {
			off, ok := constOf(in.args[1])
			if !ok || off < 0 {
				return false
			}
			for _, v := range all[in.results[0]] {
				if v.Target >= 0 || v.Inst.op.Verb == VStore && v.Index == 0 {
					return false
				}
				if !access(v.Inst, v.Index, uint64(off)) {
					return false
				}
			}
			adds = append(adds, in)
			continue
		}
		if !access(in, u.Index, 0) {
			return false
		}
	}
	if len(accesses) == 0 || len(adds) == 0 {
		// Nothing reaches a field by an offset: PromoteSlots' own case.
		return false
	}
	// One width per offset, and no two fields overlapping.
	widths := map[uint64]uint64{}
	for _, a := range accesses {
		if w, ok := widths[a.off]; ok && w != a.width {
			return false
		}
		widths[a.off] = a.width
	}
	offs := make([]uint64, 0, len(widths))
	for off := range widths {
		offs = append(offs, off)
	}
	sort.Slice(offs, func(i, j int) bool { return offs[i] < offs[j] })
	for i := 1; i < len(offs); i++ {
		if offs[i-1]+widths[offs[i-1]] > offs[i] {
			return false
		}
	}
	if last := offs[len(offs)-1]; alloc.im != nil && alloc.im.size != 0 && last+widths[last] > alloc.im.size {
		return false
	}
	// A slot of its own per field, beside the old one in the entry block.
	field := map[uint64]*Def{}
	for _, off := range offs {
		cl := f.entry.Clone(alloc, nil, nil)
		if cl == nil {
			return false
		}
		cl.im.size = widths[off]
		align := widths[off]
		if align > 16 {
			align = 16
		}
		cl.im.align = align
		cl.im.hasAlign = true
		field[off] = cl.results[0]
	}
	for _, a := range accesses {
		a.in.args[a.arg] = field[a.off]
	}
	for _, add := range adds {
		if add.blk != nil {
			add.blk.Remove(add)
		}
	}
	f.entry.Remove(alloc)
	return true
}

// allUses is every use of every def in f, by the def.
func (f *Func) allUses() map[*Def][]Use {
	out := map[*Def][]Use{}
	f.WalkUses(func(u Use) bool {
		if d := u.Def(); d != nil {
			out[d] = append(out[d], u)
		}
		return true
	})
	return out
}

// regBytes is how many bytes a full-width access of a register type moves.
func regBytes(t RegType) uint64 {
	switch t {
	case TypeI32, TypeF32:
		return 4
	case TypeI64, TypeF64, TypePtr:
		return 8
	case TypeI128, TypeF128, TypeV128:
		return 16
	}
	return 0
}
