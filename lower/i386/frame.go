package i386

import (
	"fmt"

	"github.com/vertex-language/i386/reg"

	"github.com/vertex-language/ir"
	"github.com/vertex-language/ir/lower/mir"
)

// frame is one function's stack frame.
//
// EBP-based, the way the psABI's own examples are: the saved EBP is at [ebp],
// the return address at [ebp+4], and the caller's arguments begin at [ebp+8].
// Locals are a negative displacement from EBP.
type frame struct {
	// next is how many bytes below EBP are spoken for. A running total
	// rather than a final size, because spilling adds to it.
	next uint64

	// outArgs is the outgoing argument area at the bottom of the frame.
	outArgs uint64

	slot map[*ir.Inst]int32 // ptr.alloc -> its offset from EBP

	saveAt map[reg.R32]int32

	// vaOffset is where this function's variadic arguments begin in the
	// caller's outgoing area: after whatever its named parameters put
	// there. Only meaningful when variadic is set.
	vaOffset int32
	variadic bool

	// scratchAt is the eight-byte slot a value crossing between the
	// register files passes through, reserved on first use. One per
	// function: nothing reads it across an instruction boundary.
	scratchAt  int32
	hasScratch bool
}

// scratch is the slot a value crossing between the integer and vector files
// goes through, reserved the first time one is asked for.
func (f *frame) scratch() int32 {
	if !f.hasScratch {
		f.scratchAt, f.hasScratch = f.reserve8(), true
	}
	return f.scratchAt
}

// maxAlign is the alignment ESP is required to hold at a call. Sixteen since
// the 2000s psABI revision, which is more than the architecture asks for and
// less than SSE would.
const maxAlign = 16

// size is how much the prologue subtracts from ESP after pushing EBP.
func (f *frame) size() uint64 { return alignUp(f.next+f.outArgs, maxAlign) }

// outgoing is the size of the outgoing argument area, which a dynamic
// allocation has to leave at the bottom of the frame.
func (f *frame) outgoing() int32 { return int32(f.outArgs) }

// reserve takes four more bytes of frame and returns their offset from EBP.
func (f *frame) reserve() int32 {
	f.next += 4
	return -int32(f.next)
}

// reserve8 takes eight, which a spill slot needs: it may hold a double.
func (f *frame) reserve8() int32 {
	f.next = alignUp(f.next+8, 8)
	return -int32(f.next)
}

func (f *frame) reserveOutArgs(n uint64) {
	if n > f.outArgs {
		f.outArgs = n
	}
}

func (f *frame) reserveSaves(regs []reg.R32) {
	if len(regs) == 0 {
		return
	}
	f.saveAt = make(map[reg.R32]int32, len(regs))
	for _, r := range regs {
		f.saveAt[r] = f.reserve()
	}
}

// planFrame assigns every ptr.alloc in fn a slot and returns the frame they
// add up to.
func planFrame(fn *ir.Func) (*frame, error) {
	fr := &frame{slot: map[*ir.Inst]int32{}}

	if sig := fn.Signature(); sig != nil && sig.IsVariadic() {
		// A variadic function's list starts after whatever its named
		// parameters put on the stack, which here is all of them.
		if places, err := classifySysV(paramTypes(fn)); err == nil {
			fr.variadic = true
			fr.vaOffset = int32(stackEnd(places))
		}
	}

	var off uint64
	for _, blk := range fn.Blocks() {
		for _, in := range blk.Insts() {
			switch in.Op().Verb {
			case ir.VCall:
				fr.reserveOutArgs(callStackBytes(in))
				continue
			case ir.VCallInd:
				fr.reserveOutArgs(callStackBytes(in))
				continue
			}
			// A call this package invents needs the same area, and
			// the frame is planned before isel knows it will make one.
			if n := libcallOutgoing(in); n > 0 {
				fr.reserveOutArgs(n)
			}
			if in.Op() != (ir.Op{Type: ir.TypePtr, Verb: ir.VAlloc}) {
				continue
			}
			size, align, err := allocSizeAlign(in)
			if err != nil {
				return nil, fmt.Errorf("lower: %s: %s: %w", fn.Name(), in.Op(), err)
			}
			if align > maxAlign {
				return nil, fmt.Errorf("lower: %s: ptr.alloc wants %d-byte alignment; the frame guarantees %d", fn.Name(), align, maxAlign)
			}
			off = alignUp(off+size, align)
			if off > 1<<31 {
				return nil, fmt.Errorf("lower: %s: the frame exceeds 2GB", fn.Name())
			}
			fr.slot[in] = -int32(off)
		}
	}

	fr.next = off
	return fr, nil
}

func allocSizeAlign(in *ir.Inst) (size, align uint64, err error) {
	if a, stated := in.Align(); stated && a != 0 {
		return in.Size(), a, nil
	}
	t := in.NamedType()
	if t == nil {
		return 0, 0, fmt.Errorf("neither a size and alignment nor a type")
	}
	return sizeAlign(t.FType())
}

// A place is where the psABI puts one argument: an offset into the outgoing
// area, always, since there is no register argument on this architecture.
type place struct {
	off int32
	w   width
}

// classifySysV places a list of arguments by the Intel386 psABI.
//
// The whole convention is arithmetic. Every argument is on the stack in
// declaration order, each rounded up to four bytes, so an i64 is two slots
// and everything else is one — there is no register file to run out of and
// no second queue to keep.
func classifySysV(types []ir.RegType) ([]place, error) {
	out := make([]place, len(types))
	var off int32
	for i, t := range types {
		w, ok := widthOf(t)
		if !ok {
			return nil, fmt.Errorf("%s is not a value this package passes; only i1, i32, i64, ptr, f32 and f64 are, f80 and f128 being types this target does not have", t)
		}
		out[i] = place{off: off, w: w}
		off += 4
		if w.pairs() || w == wf64 {
			// A double is eight bytes on the stack, four-aligned
			// like everything else on this psABI.
			off += 4
		}
	}
	return out, nil
}

func typesOf(defs []*ir.Def) []ir.RegType {
	out := make([]ir.RegType, len(defs))
	for i, d := range defs {
		out[i] = d.Type()
	}
	return out
}

func paramTypes(fn *ir.Func) []ir.RegType {
	ps := fn.Params()
	out := make([]ir.RegType, len(ps))
	for i, p := range ps {
		out[i] = p.Type()
	}
	return out
}

// callStackBytes is how much outgoing area one call needs.
//
// An indirect call's first operand is the callee's address, which goes in no
// argument slot: including it would shift every real argument one place right
// and reserve room the call never writes to.
func callStackBytes(in *ir.Inst) uint64 {
	args := in.Args()
	if in.Op().Verb == ir.VCallInd && len(args) > 0 {
		args = args[1:]
	}
	places, err := classifySysV(typesOf(args))
	if err != nil {
		return 0
	}
	return outgoingBytes(places)
}

// stackEnd is the first byte past the stack arguments, unrounded — where the
// next thing placed on the stack would go.
func stackEnd(places []place) uint64 {
	var n int32
	for _, p := range places {
		end := p.off + 4
		if p.w.pairs() || p.w == wf64 {
			end += 4
		}
		if end > n {
			n = end
		}
	}
	return uint64(n)
}

func outgoingBytes(places []place) uint64 {
	var n int32
	for _, p := range places {
		end := p.off + 4
		if p.w.pairs() || p.w == wf64 {
			end += 4
		}
		if end > n {
			n = end
		}
	}
	return alignUp(uint64(n), maxAlign)
}

// stackParamOff is where an argument at byte offset off in the caller's
// outgoing area ends up above this function's EBP.
//
// Eight: the saved EBP this function pushed and the return address the call
// pushed, four bytes each.
func stackParamOff(off int32) int32 { return 8 + off }

// classifyParams copies fn's parameters out of the stack slots they arrived
// in and into vregs the allocator is free to place.
func classifyParams(fn *ir.Func, entry *mir.Block, vr *vregs) error {
	places, err := classifySysV(paramTypes(fn))
	if err != nil {
		return fmt.Errorf("lower: %s: %w", fn.Name(), err)
	}

	for i, p := range fn.Params() {
		pl := places[i]
		val, err := vr.define(p)
		if err != nil {
			return fmt.Errorf("lower: %s: parameter %q: %w", fn.Name(), p.Name(), err)
		}
		if pl.w.isFloat() {
			// A float arrives as bytes on the stack like every
			// other argument; only the register it goes into is
			// different.
			entry.Emit(mir.Instr{
				Op:   fframeLoadOp{off: stackParamOff(pl.off), w: pl.w},
				Defs: []mir.VReg{val.lo},
			})
			continue
		}
		entry.Emit(mir.Instr{
			Op:   frameLoadOp{off: stackParamOff(pl.off)},
			Defs: []mir.VReg{val.lo},
		})
		if pl.w.pairs() {
			entry.Emit(mir.Instr{
				Op:   frameLoadOp{off: stackParamOff(pl.off + 4)},
				Defs: []mir.VReg{val.hi},
			})
		}
	}
	return nil
}

// classifyBlockParams allocates registers for every block's own parameters.
func classifyBlockParams(fn *ir.Func, vr *vregs) error {
	for _, blk := range fn.Blocks() {
		for _, p := range blk.Params() {
			if _, err := vr.define(p); err != nil {
				return fmt.Errorf("lower: %s: @%s parameter %q: %w", fn.Name(), blk.Label(), p.Name(), err)
			}
		}
	}
	return nil
}
