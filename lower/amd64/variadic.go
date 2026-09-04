package amd64

// §I's callee half. §3.5.7's va_list is four fields rather than a
// pointer because the arguments are in two places: a register save area
// the prologue writes, and the caller's outgoing area.
//
//	struct {
//	    uint32 gp_offset;         // 0
//	    uint32 fp_offset;         // 4
//	    void  *overflow_arg_area; // 8
//	    void  *reg_save_area;     // 16
//	}

import (
	"fmt"

	"github.com/vertex-language/ir"
	"github.com/vertex-language/ir/lower/mir"
)

// iselVaStart lowers §I's va_start: the four fields, written once.
func iselVaStart(c *cursor, vr *vregs, fr *frame, in *ir.Inst) error {
	op := in.Op()
	ap, ok := vr.lookup(in.Arg(0))
	if !ok {
		return fmt.Errorf("%s: the list is defined outside the function", op)
	}
	fn := in.Block().Func()
	if fn.Module().Layout().ABI == abiMS {
		// One field, not four. Every argument on this ABI owns an
		// eightbyte of the caller's frame, the register ones included —
		// the prologue homed those — so the whole list is one array and
		// the list is a pointer to the first argument past the named
		// ones.
		off, err := msVaTailOff(fn)
		if err != nil {
			return fmt.Errorf("%s: %w", op, err)
		}
		addr := vr.temp(w64)
		c.Emit(mir.Instr{Op: leaInOp{off: stackParamOff(off)}, Defs: []mir.VReg{addr}})
		c.Emit(mir.Instr{Op: storeOp{w: w64}, Uses: []mir.VReg{addr, ap}})
		return nil
	}
	if !fr.saveAreaSet {
		return fmt.Errorf("%s: this function has no register save area; only a variadic function has a list to start", op)
	}

	// Where the tail begins. The offsets skip this function's own named
	// parameters, which occupied registers before the tail did, and
	// fp_offset counts from the end of the integer half — hence 48.
	c.Emit(mir.Instr{
		Op:   storeImmAtOp{off: vaListGPOffset, imm: int64(fr.vaInts * 8), w: w32},
		Uses: []mir.VReg{ap},
	})
	c.Emit(mir.Instr{
		Op:   storeImmAtOp{off: vaListFPOffset, imm: int64(saveAreaGPSize + fr.vaFloats*16), w: w32},
		Uses: []mir.VReg{ap},
	})

	overflow := vr.temp(w64)
	c.Emit(mir.Instr{
		Op:   leaInOp{off: stackParamOff(int32(fr.vaOverflow))},
		Defs: []mir.VReg{overflow},
	})
	c.Emit(mir.Instr{
		Op:   storeAtOp{off: vaListOverflow, w: w64},
		Uses: []mir.VReg{ap, overflow},
	})

	regSave := vr.temp(w64)
	c.Emit(mir.Instr{Op: leaOp{off: fr.saveArea}, Defs: []mir.VReg{regSave}})
	c.Emit(mir.Instr{
		Op:   storeAtOp{off: vaListRegSave, w: w64},
		Uses: []mir.VReg{ap, regSave},
	})
	return nil
}

// iselVaEnd lowers §I's va_end, which on this ABI is nothing: the list
// is four fields in storage the caller provided, with no resource to
// release. Emitting nothing is the lowering, not the absence of one.
func iselVaEnd(in *ir.Inst) error {
	_ = in
	return nil
}

// iselVaCopy lowers §I's va_copy: twenty-four bytes from one list to
// another. Three moves rather than a memcpy — the size is a constant and
// three eightbytes is below any size worth a call.
func iselVaCopy(c *cursor, vr *vregs, in *ir.Inst) error {
	op := in.Op()
	dst, ok := vr.lookup(in.Arg(0))
	if !ok {
		return fmt.Errorf("%s: the destination is defined outside the function", op)
	}
	src, ok := vr.lookup(in.Arg(1))
	if !ok {
		return fmt.Errorf("%s: the source is defined outside the function", op)
	}
	size := int32(vaListSize)
	if in.Block().Func().Module().Layout().ABI == abiMS {
		// One pointer, not four fields: copying twenty-four bytes would
		// write sixteen past the end of an object the caller sized at
		// eight.
		size = 8
	}
	for off := int32(0); off < size; off += 8 {
		tmp := vr.temp(w64)
		c.Emit(mir.Instr{Op: loadAtOp{off: off, w: w64}, Defs: []mir.VReg{tmp}, Uses: []mir.VReg{src}})
		c.Emit(mir.Instr{Op: storeAtOp{off: off, w: w64}, Uses: []mir.VReg{dst, tmp}})
	}
	return nil
}

// iselVaArgRef lowers §I's ptr.va_arg_ref, which is how
// va_arg(ap, struct S) is said. The same classifier byval uses, and only
// the memory half of its answer: an aggregate passed in registers has
// its eightbytes scattered through the save area, and gathering them
// into one address is not written yet.
func iselVaArgRef(c *cursor, vr *vregs, in *ir.Inst) error {
	op := in.Op()
	t := in.NamedType()
	if t == nil {
		return fmt.Errorf("%s: no type named", op)
	}
	agg, err := classifyAggregate(t.FType())
	if err != nil {
		return fmt.Errorf("%s: %w", op, err)
	}
	if !agg.inMemory() {
		return fmt.Errorf("%s: %s is small enough to have been passed in registers, and gathering its eightbytes out of the save area is not written yet", op, t.FType())
	}

	ap, ok := vr.lookup(in.Arg(0))
	if !ok {
		return fmt.Errorf("%s: the list is defined outside the function", op)
	}
	dst, err := vr.define(in.Result(0))
	if err != nil {
		return fmt.Errorf("%s: %w", op, err)
	}

	// A memory-class argument is in the caller's outgoing area, so this
	// is the overflow pointer and nothing else — no branch, because
	// there is no register case to be in.
	c.Emit(mir.Instr{Op: loadAtOp{off: vaListOverflow, w: w64}, Defs: []mir.VReg{dst}, Uses: []mir.VReg{ap}})

	// SysV aligns a stack argument to eight, or stricter if the type is.
	// Emitted only when it can do something: an eight-aligned type is
	// already where it needs to be.
	if agg.align > 8 {
		if agg.align > maxAlign {
			return fmt.Errorf("%s: %s wants %d-byte alignment; the stack guarantees %d", op, t.FType(), agg.align, maxAlign)
		}
		up := vr.temp(w64)
		c.Emit(mir.Instr{Op: leaAtOp{off: int32(agg.align - 1)}, Defs: []mir.VReg{up}, Uses: []mir.VReg{dst}})
		c.Emit(mir.Instr{
			Op:   andImmOp{imm: -int64(agg.align), w: w64},
			Defs: []mir.VReg{dst},
			Uses: []mir.VReg{up},
		})
	}

	// Past it, by whole eightbytes: the overflow area gives every
	// argument a multiple of eight, the same rule that gives a stack
	// argument eight bytes on the way in.
	next := vr.temp(w64)
	c.Emit(mir.Instr{Op: leaAtOp{off: int32(alignUp(agg.size, 8))}, Defs: []mir.VReg{next}, Uses: []mir.VReg{dst}})
	c.Emit(mir.Instr{Op: storeAtOp{off: vaListOverflow, w: w64}, Uses: []mir.VReg{ap, next}})
	return nil
}

// iselVaArg lowers §I's va_arg. A branch and not a computation because
// the two places are not adjacent: an argument is in the save area while
// its file's offset is still inside its half, and in the overflow area
// afterwards.
//
//	if (offset < limit) { addr = reg_save + offset; offset += step; }
//	else                { addr = overflow; overflow += 8; }
//	result = *addr
func iselVaArg(c *cursor, vr *vregs, in *ir.Inst) error {
	op := in.Op()
	ap, ok := vr.lookup(in.Arg(0))
	if !ok {
		return fmt.Errorf("%s: the list is defined outside the function", op)
	}
	w, ok := widthOf(op.Type)
	if !ok {
		return fmt.Errorf("%s: %s is not a value this package reads from a list", op, op.Type)
	}
	result, err := vr.define(in.Result(0))
	if err != nil {
		return fmt.Errorf("%s: %w", op, err)
	}

	if in.Block().Func().Module().Layout().ABI == abiMS {
		// Read the eightbyte the list points at and step over it. Every
		// argument is one eightbyte wide here whatever its type — a
		// float in the tail was promoted to double and duplicated into
		// the integer register the prologue homed, which is the slot
		// this reads — so there is no file to pick and no branch.
		if w == wv128 {
			return fmt.Errorf("%s: %s has no place in the Microsoft calling convention", op, in.Result(0).Type())
		}
		addr := vr.temp(w64)
		c.Emit(mir.Instr{Op: loadOp{w: w64}, Defs: []mir.VReg{addr}, Uses: []mir.VReg{ap}})

		eight := vr.temp(w64)
		c.Emit(mir.Instr{Op: constOp{imm: 8, w: w64}, Defs: []mir.VReg{eight}})

		next := vr.temp(w64)
		c.Emit(mir.Instr{Op: aluOp{verb: ir.VAdd, w: w64}, Defs: []mir.VReg{next}, Uses: []mir.VReg{addr, eight}})

		c.Emit(mir.Instr{Op: storeOp{w: w64}, Uses: []mir.VReg{next, ap}})
		c.Emit(mir.Instr{Op: loadOp{w: w}, Defs: []mir.VReg{result}, Uses: []mir.VReg{addr}})
		return nil
	}

	// Which file's counter this argument walks, and how far one step is.
	// A vector slot is sixteen bytes wide in the save area even for an
	// f64, because the area holds whole XMM registers.
	offField, limit, step := int32(vaListGPOffset), int64(saveAreaGPSize), int64(8)
	if w.isFloat() {
		offField, limit, step = vaListFPOffset, saveAreaSize, 16
	}

	inReg := c.open("va_reg")
	inMem := c.open("va_mem")
	done := c.open("va_done")

	// The address, whichever arm produced it. A block parameter
	// everywhere else in this package, but the cursor's blocks are
	// MIR-level and have none — so one vreg two blocks assign.
	addr := vr.temp(w64)

	cur := vr.temp(w32)
	c.Emit(mir.Instr{Op: loadAtOp{off: offField, w: w32}, Defs: []mir.VReg{cur}, Uses: []mir.VReg{ap}})
	c.Emit(mir.Instr{Op: cmpImmOp{imm: limit - step + 1, w: w32}, Uses: []mir.VReg{cur}})
	c.Emit(mir.Instr{Op: brccOp{cond: condB, then: inReg.Label, els: inMem.Label}})
	c.mf.Succ(c.blk, inReg.Label)
	c.mf.Succ(c.blk, inMem.Label)

	// The register arm: the save area plus this file's offset, and the
	// offset advanced past what was taken.
	reg := newCursor(c.fn, c.mf, inReg, c.prefix)
	base := vr.temp(w64)
	reg.Emit(mir.Instr{Op: loadAtOp{off: vaListRegSave, w: w64}, Defs: []mir.VReg{base}, Uses: []mir.VReg{ap}})
	wide := vr.temp(w64)
	reg.Emit(mir.Instr{Op: zextOp{}, Defs: []mir.VReg{wide}, Uses: []mir.VReg{cur}})
	reg.Emit(mir.Instr{
		Op:   aluOp{verb: ir.VAdd, w: w64},
		Defs: []mir.VReg{addr},
		Uses: []mir.VReg{base, wide},
	})
	reg.Emit(mir.Instr{Op: addImmAtOp{off: offField, imm: step, w: w32}, Uses: []mir.VReg{ap}})
	reg.to(done)

	// The memory arm: the overflow pointer, advanced one eightbyte
	// whatever the width — the same rule that gives a stack argument
	// eight bytes on the way in.
	mem := newCursor(c.fn, c.mf, inMem, c.prefix)
	mem.Emit(mir.Instr{Op: loadAtOp{off: vaListOverflow, w: w64}, Defs: []mir.VReg{addr}, Uses: []mir.VReg{ap}})
	next := vr.temp(w64)
	mem.Emit(mir.Instr{Op: leaAtOp{off: 8}, Defs: []mir.VReg{next}, Uses: []mir.VReg{addr}})
	mem.Emit(mir.Instr{Op: storeAtOp{off: vaListOverflow, w: w64}, Uses: []mir.VReg{ap, next}})
	mem.to(done)

	c.blk = done
	c.Emit(mir.Instr{Op: loadAtOp{off: 0, w: w}, Defs: []mir.VReg{result}, Uses: []mir.VReg{addr}})
	return nil
}

// msVaTailOff is where a Microsoft-ABI variadic function's tail begins,
// as a byte offset into the caller's argument area.
//
// Every argument owns one eightbyte of that area and position i is at
// offset 8i, so the answer is eight times the number of positions the
// named parameters spent. Counted from the classification rather than
// from len(Params()) because a small aggregate returned in RAX spends no
// position at all: its sret parameter is a slot of this function's own,
// not something the caller passed.
func msVaTailOff(fn *ir.Func) (int32, error) {
	places, err := classifyMS(paramArgs(fn))
	if err != nil {
		return 0, err
	}
	var slots int
	for _, p := range places {
		var next int
		switch {
		case len(p.regs) > 0:
			next = p.regs[0].i + 1
		case p.kind == placeStack:
			next = int(p.off)/8 + 1
		default:
			continue // spends no position: an empty aggregate, or sret in RAX
		}
		if next > slots {
			slots = next
		}
	}
	return int32(slots * 8), nil
}
