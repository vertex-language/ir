package amd64

import (
	"fmt"

	"github.com/vertex-language/amd64/reg"

	"github.com/vertex-language/ir"
	"github.com/vertex-language/ir/lower/mir"
)

// frame is one function's stack frame: how many bytes of it there are,
// and where each ptr.alloc's storage sits inside it.
type frame struct {
	// next is how many bytes below RBP are spoken for. It is a running
	// total rather than a final size because spilling adds to it.
	next uint64

	// outArgs is the outgoing argument area: the bytes at the bottom of
	// the frame that hold the arguments past the sixth of whichever call.
	outArgs uint64

	slot map[*ir.Inst]int32 // ptr.alloc -> its offset from RBP

	// saveAt is where each callee-saved register this function used gets preserved.
	saveAt map[reg.R64]int32

	// force makes a prologue even when nothing needs storage.
	force bool

	// saveArea is where §3.5.7's register save area begins, and
	// saveAreaSet says there is one. va_arg indexes it, so the prologue
	// has to put the argument registers somewhere indexable.
	saveArea    int32
	saveAreaSet bool

	// Where the variadic tail begins, which va_start writes into the
	// list: how many of each file's argument registers the named
	// parameters used, and how many bytes of the caller's area.
	vaInts, vaFloats int
	vaOverflow       uint64

	// wideSpills marks a function holding sixteen bytes in a vector
	// register — an f128 or a v128 — whose spills
	// are a whole register rather than half of one. See spiller.Slot.
	wideSpills bool

	// dynamic marks a function whose RSP moves after the prologue. It
	// rounds the outgoing area to the stack alignment: an allocation
	// hands back the address just above it, and only a multiple of
	// sixteen keeps that aligned.
	dynamic bool
}

// size is how much the prologue subtracts from RSP.
func (f *frame) size() uint64 { return alignUp(f.next+f.outgoing(), maxAlign) }

// outgoing is the outgoing argument area's size, rounded up to the stack
// alignment in a function whose RSP moves. See frame.dynamic.
func (f *frame) outgoing() uint64 {
	if f.dynamic {
		return alignUp(f.outArgs, maxAlign)
	}
	return f.outArgs
}

// reserveOutArgs widens the outgoing argument area to hold n bytes.
func (f *frame) reserveOutArgs(n uint64) {
	if n > f.outArgs {
		f.outArgs = n
	}
}

// needed reports whether this function has a prologue at all.
func (f *frame) needed() bool { return f.force || f.size() > 0 }

// reserveSaves gives each callee-saved register the function used a slot
// to be preserved in.
func (f *frame) reserveSaves(regs []reg.R64) {
	if len(regs) == 0 {
		return
	}
	f.saveAt = make(map[reg.R64]int32, len(regs))
	for _, r := range regs {
		f.saveAt[r] = f.reserve()
	}
}

// reserve takes eight more bytes of frame and returns their offset from RBP.
//
// Aligned to eight, which the eightbyte store that will use it wants anyway,
// and which Windows requires: an unwind record spells a saved register's slot
// as a scaled-by-eight offset, so a slot at four bytes past an eightbyte
// boundary is one the format cannot express.
func (f *frame) reserve() int32 {
	f.next = alignUp(f.next+8, 8)
	return -int32(f.next)
}

// reserveBytes takes size bytes of frame at the given alignment and
// returns their offset from RBP. Rounded after growing, not before: the
// frame grows downwards, so the low end is what has to land on the
// alignment and the low end is what the total names.
func (f *frame) reserveBytes(size, align uint64) (int32, error) {
	if align == 0 {
		align = 1
	}
	f.next = alignUp(f.next+size, align)
	if f.next > 1<<31 {
		return 0, fmt.Errorf("the frame exceeds 2GB")
	}
	return -int32(f.next), nil
}

// maxAlign is the strictest alignment a slot can ask for.
const maxAlign = 16

// The register save area §3.5.7 describes: the six integer argument
// registers and then the eight vector ones, in the order SysV names
// them, because va_arg indexes into it by an offset the ABI fixes and
// not by one this package is free to choose.
const (
	saveAreaInts   = 6
	saveAreaFloats = 8
	saveAreaGPSize = saveAreaInts * 8    // 48
	saveAreaFPSize = saveAreaFloats * 16 // 128
	saveAreaSize   = saveAreaGPSize + saveAreaFPSize

	// The four fields of §3.5.7's va_list: two 32-bit offsets into the
	// save area, and two pointers.
	vaListGPOffset = 0
	vaListFPOffset = 4
	vaListOverflow = 8
	vaListRegSave  = 16
	vaListSize     = 24
)

// planFrame assigns every ptr.alloc in fn a slot and returns the frame
// they add up to.
func planFrame(fn *ir.Func) (*frame, error) {
	fr := &frame{slot: map[*ir.Inst]int32{}}

	// A function with parameters SysV left on the stack reads them from
	// above RBP, which means it needs an RBP to read them from — before
	// it has stored anything of its own.
	//
	// A classification error is not reported here. classifyParams runs
	// the same function and says it properly, with the parameter's name
	// in the message; a frame planned for a function that will not lower
	// is harmless.
	if places, err := classify(fn.Module().Layout().ABI, paramArgs(fn)); err == nil {
		fr.force = usesStack(places)
	}

	// Somewhere for the prologue to put the arguments the caller passed
	// in registers: 16-aligned for the movaps its vector half uses, and
	// reserved first so §3.5.7's fixed offsets index one block.
	fr.wideSpills = holdsWideVector(fn)

	if needsSaveArea(fn) {
		fr.force = true
		if fn.Module().Layout().ABI == abiMS {
			// No area of this function's own. The four registers go into
			// the home space the caller already reserved, which is what
			// makes every argument one eightbyte of one array. There is
			// still a prologue to write them in, so the flag is set and
			// only the reservation is skipped.
			fr.saveAreaSet = true
		} else {
			at, err := fr.reserveBytes(saveAreaSize, maxAlign)
			if err != nil {
				return nil, fmt.Errorf("lower: %s: %w", fn.Name(), err)
			}
			fr.saveArea, fr.saveAreaSet = at, true

			ints, floats, stackBytes, err := namedPlaces(fn)
			if err != nil {
				return nil, fmt.Errorf("lower: %s: %w", fn.Name(), err)
			}
			fr.vaInts, fr.vaFloats, fr.vaOverflow = ints, floats, stackBytes
		}
	}

	// From wherever the save area left the total, not from zero: what
	// keeps two pieces of frame apart is that the total only grows.
	off := fr.next
	for _, blk := range fn.Blocks() {
		for _, in := range blk.Insts() {
			if softFloatCalls(in) {
				// Only the f128 operations that actually call: a
				// literal is a MOVAPS and the sign verbs are one
				// logical instruction.
				fr.force = true
				continue
			}
			if _, isLibcall := libcalls[in.Op().Verb]; isLibcall {
				// A bulk-memory verb is a call to the C library, so it
				// wants the same aligned RSP a written call does.
				fr.force = true
				if fn.Module().Layout().ABI == abiMS {
					// Except for the home space, which every call on
					// this ABI reserves whether the callee writes it or
					// not — a libcall is a call.
					fr.reserveOutArgs(msShadow)
				}
				continue
			}
			if v := in.Op().Verb; v == ir.VCall || v == ir.VCallInd {
				// A function that calls anything has a frame whether or
				// not it stores anything in one: SysV wants RSP 16-byte
				// aligned at a call instruction, entry leaves it eight
				// off that, and the prologue's push of RBP is what puts
				// it back. An indirect call is a call: the alignment the
				// callee is promised does not depend on how it was named.
				fr.force = true
				fr.reserveOutArgs(callStackBytes(in))
				continue
			}
			switch in.Op().Verb {
			case ir.VAlloca, ir.VStackRestore:
				// A dynamic frame needs an RBP to be dynamic against,
				// and an outgoing area sized to a multiple of the stack
				// alignment — see allocaOp, which hands back the space
				// above that area.
				fr.force = true
				fr.dynamic = true
				continue
			case ir.VStackSave:
				// Reading the stack pointer does not move it. Only
				// alloca and stackrestore do, and a stacksave that
				// belongs to a pair is one whose stackrestore is in the
				// same function and has already said so. It still wants
				// a frame, because a value read out of RSP is only
				// meaningful in a function that has one.
				fr.force = true
				continue
			case ir.VFrameAddr, ir.VReturnAddr:
				// Both name a place in the frame, so both need there to
				// be one.
				fr.force = true
				continue
			}
			if in.Op() != (ir.Op{Type: ir.TypePtr, Verb: ir.VAlloc}) {
				continue
			}
			size, align, err := allocSizeAlign(in)
			if err != nil {
				return nil, fmt.Errorf("lower: %s: %s: %w", fn.Name(), in.Op(), err)
			}
			switch {
			case align > maxAlign:
				return nil, fmt.Errorf("lower: %s: ptr.alloc wants %d-byte alignment; the frame guarantees %d", fn.Name(), align, maxAlign)
			}
			if in.Zeroed() {
				// The zeroing is a memset, which is a call, which needs
				// the frame a call needs. See iselAlloc.
				fr.force = true
			}

			// Grow down, then round the running total up so the slot's
			// own address is aligned: RBP is 16-aligned, so an offset
			// that is a multiple of align puts the slot on a multiple of
			// align.
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

// holdsF128 reports whether any value in fn is an f128, which is what
// makes a vector spill sixteen bytes. Parameters too: one that only
// arrives still occupies a register while it is live.
// holdsWideVector reports whether fn has a value that occupies a whole
// vector register. Both f128 and v128 do, and a spill of one is sixteen
// bytes; every other value in that file is eight or four.
func holdsWideVector(fn *ir.Func) bool {
	for _, p := range fn.Params() {
		if wideVector(p.Type()) {
			return true
		}
	}
	found := false
	fn.WalkInsts(func(in *ir.Inst) bool {
		if wideVector(in.Op().Type) {
			found = true
			return false
		}
		for _, d := range in.Results() {
			if wideVector(d.Type()) {
				found = true
				return false
			}
		}
		return true
	})
	return found
}

// needsSaveArea reports whether fn has to spill its argument registers
// for va_arg to find them. Variadic *and* naming va_start: 176 bytes
// and fourteen stores is a lot for a list nobody opens.
func needsSaveArea(fn *ir.Func) bool {
	sig := fn.Signature()
	if sig == nil || !sig.IsVariadic() {
		return false
	}
	found := false
	fn.WalkInsts(func(in *ir.Inst) bool {
		if in.Op().Verb == ir.VVaStart {
			found = true
			return false
		}
		return true
	})
	return found
}

// namedPlaces is how many of each file's argument registers fn's own
// declared parameters used, and how many bytes of the caller's outgoing
// area they occupied. It is where va_arg starts reading, in both.
func namedPlaces(fn *ir.Func) (ints, floats int, stackBytes uint64, err error) {
	places, err := classify(fn.Module().Layout().ABI, paramArgs(fn))
	if err != nil {
		return 0, 0, 0, err
	}
	for _, p := range places {
		if p.kind == placeStack {
			if end := uint64(p.off) + alignUp(p.size, 8); end > stackBytes {
				stackBytes = end
			}
			continue
		}
		for _, r := range p.regs {
			if r.kind == placeFloat {
				floats++
				continue
			}
			ints++
		}
	}
	return ints, floats, stackBytes, nil
}

// allocSizeAlign is how much storage one ptr.alloc wants and how it has
// to be aligned.
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

// callStackBytes is how much outgoing argument area one call needs.
func callStackBytes(in *ir.Inst) uint64 {
	abi := in.Block().Func().Module().Layout().ABI
	places, err := classify(abi, callArgSpec(in))
	if err != nil {
		return 0
	}
	if abi == abiMS {
		return msOutgoingBytes(places)
	}
	return outgoingBytes(places)
}

// callArgs is the arguments a call passes, which for an indirect call is
// everything after the callee its first operand names.
func callArgs(in *ir.Inst) []*ir.Def {
	args := in.Args()
	if in.Op().Verb == ir.VCallInd {
		return args[1:]
	}
	return args
}

// typesOf is the types of a list of values, which is what the two
// classifiers take.
func typesOf(defs []*ir.Def) []ir.RegType {
	out := make([]ir.RegType, len(defs))
	for i, d := range defs {
		out[i] = d.Type()
	}
	return out
}

// stackParamOff is where a stack parameter at byte offset off in the
// caller's area ends up above RBP. Sixteen and not eight: the call
// pushed a return address and the prologue pushed RBP on top of it, and
// ptr.returnaddr reads the eightbyte between.
func stackParamOff(off int32) int32 { return 16 + off }

// alignUp rounds n up to the next multiple of a, a power of two.
func alignUp(n, a uint64) uint64 {
	if a == 0 {
		return n
	}
	return (n + a - 1) &^ (a - 1)
}

// An abiArg is one parameter or argument as the classifier sees it: its
// register type, and the aggregate it stands for when it carries byval.
type abiArg struct {
	t     ir.RegType
	byval ir.FType

	// sret is the aggregate this parameter returns through, set only on the
	// first. It is carried here rather than passed alongside because every
	// caller of classifySysV already builds these, and because what it
	// changes is one argument's place: a result §3.2.3 brings back in
	// registers needs no address from the caller, so the parameter is not
	// an argument at all.
	sret ir.FType

	// vararg says this argument sits in the variadic tail: past the last
	// parameter of a signature that declares one. SysV does not care —
	// §3.2.3 places a value by its type — but the Microsoft ABI sends a
	// float in the tail through both register files, and this is how it
	// knows which floats are in the tail.
	vararg bool
}

// scalarArgs is a list of plain values, with no byval among them.
func scalarArgs(types []ir.RegType) []abiArg {
	out := make([]abiArg, len(types))
	for i, t := range types {
		out[i] = abiArg{t: t}
	}
	return out
}

func classify(abi string, args []abiArg) ([]place, error) {
	if abi == abiMS {
		return classifyMS(args)
	}
	return classifySysV(args)
}

func classifyRet(abi string, types []ir.RegType) ([]place, error) {
	if abi == abiMS {
		return classifyMSRet(types)
	}
	return classifySysVRet(types)
}

// classifySysV places a list of arguments by §3.2.3. A fold and not a
// map: the registers of each file and the stack are one running state,
// so an aggregate that does not fit in what is left goes in memory and
// so does every one after it.
func classifySysV(args []abiArg) ([]place, error) {
	abi := abiSysV
	var ints, floats int
	var stackBytes uint64
	out := make([]place, len(args))

	// The one way an argument ends up in the outgoing area. SysV aligns
	// a stack argument to eight, or to its own alignment if stricter.
	toStack := func(size, align uint64) place {
		if align < 8 {
			align = 8
		}
		stackBytes = alignUp(stackBytes, align)
		p := place{kind: placeStack, off: int32(stackBytes), size: size}
		stackBytes += alignUp(size, 8)
		return p
	}

	for i, a := range args {
		if i == 0 && !a.sret.IsZero() {
			agg, inRegs, err := sretInRegs(a.sret)
			if err != nil {
				return nil, err
			}
			if inRegs {
				// The caller supplies no address, so this parameter takes
				// no register and no stack slot. It still has to be
				// somewhere the body can write through, and that is a slot
				// of the callee's own — an aggregate place with nothing
				// arriving in it.
				out[i] = place{kind: placeInt, byval: a.sret, size: agg.size}
				continue
			}
			// MEMORY: the hidden pointer is an ordinary first argument,
			// which is what the signature already says it is.
		}
		if a.byval.IsZero() {
			w, ok := widthOf(a.t)
			if !ok {
				return nil, fmt.Errorf("%s is not a value this package passes; only i32, i64, ptr, f32, and f64 are placed", a.t)
			}
			switch {
			case w.isFloat() && floats < numFloatArgs(abi):
				out[i] = place{kind: placeFloat, regs: []regSlot{{kind: placeFloat, i: floats, w: w}}}
				floats++
			case !w.isFloat() && ints < numIntArgs(abi):
				out[i] = place{kind: placeInt, regs: []regSlot{{kind: placeInt, i: ints, w: w}}}
				ints++
			default:
				// A float past the eighth takes a stack slot, the same
				// as an integer past the sixth, out of the same queue:
				// §3.2.3 gives the stack one sequence and not one per
				// class.
				//
				// An f128 takes two of them and wants sixteen-byte
				// alignment, because it is SSE and SSEUP: two eightbytes
				// that are one value, and the MOVAPS that reads it back
				// needs the address aligned.
				if w == wv128 {
					out[i] = toStack(16, 16)
					break
				}
				out[i] = toStack(8, 8)
			}
			out[i].scalarW = w
			continue
		}

		agg, err := classifyAggregate(a.byval)
		if err != nil {
			return nil, fmt.Errorf("byval %s: %w", a.byval, err)
		}

		// In registers if its classes fit in what is still free, and in
		// memory whole if not: §3.2.3 has no half-in-half-out case, and
		// splitting one would be a third convention neither side
		// implements.
		if !agg.inMemory() {
			needInts, needFloats := 0, 0
			for _, c := range agg.classes {
				if c == classSSE {
					needFloats++
				} else {
					needInts++
				}
			}
			if ints+needInts <= numIntArgs(abi) && floats+needFloats <= numFloatArgs(abi) {
				p := place{kind: placeInt, byval: a.byval, size: agg.size}
				for _, c := range agg.classes {
					if c == classSSE {
						p.regs = append(p.regs, regSlot{kind: placeFloat, i: floats, w: wf64})
						floats++
						continue
					}
					p.regs = append(p.regs, regSlot{kind: placeInt, i: ints, w: w64})
					ints++
				}
				if len(p.regs) > 0 {
					p.kind = p.regs[0].kind
				}
				out[i] = p
				continue
			}
		}

		p := toStack(agg.size, agg.align)
		p.byval = a.byval
		out[i] = p
	}
	return out, nil
}

// outgoingBytes is how large the caller's outgoing area has to be to
// hold these places.
func outgoingBytes(places []place) uint64 {
	var n uint64
	for _, p := range places {
		if p.kind != placeStack {
			continue
		}
		if end := uint64(p.off) + alignUp(p.size, 8); end > n {
			n = end
		}
	}
	return alignUp(n, 8)
}

// classifySysVRet places a call's results in the registers a return
// comes back in: two of each file, counted separately, so four values
// fit if they are the right four. No placeStack, deliberately — a value
// that does not fit comes back through memory, which is sret, and sret
// is a parameter-side mechanism rather than a wider return.
func classifySysVRet(types []ir.RegType) ([]place, error) {
	abi := abiSysV
	var ints, floats int
	out := make([]place, len(types))

	for i, t := range types {
		w, ok := widthOf(t)
		if !ok {
			return nil, fmt.Errorf("%s is not a value this package returns; only i32, i64, ptr, f32, and f64 are placed", t)
		}
		switch {
		case w.isFloat() && floats < numFloatRets(abi):
			out[i] = place{kind: placeFloat, regs: []regSlot{{kind: placeFloat, i: floats, w: w}}, scalarW: w}
			floats++
		case w.isFloat():
			return nil, fmt.Errorf("a float past the second comes back in memory, which is sret and is not written yet")
		case ints < numIntRets(abi):
			out[i] = place{kind: placeInt, regs: []regSlot{{kind: placeInt, i: ints, w: w}}, scalarW: w}
			ints++
		default:
			return nil, fmt.Errorf("an integer past the second comes back in memory, which is sret and is not written yet")
		}
	}
	return out, nil
}

// usesStack reports whether any of the places is in the outgoing area,
// which is what makes a function need an RBP to read its own from.
func usesStack(places []place) bool {
	for _, p := range places {
		if p.kind == placeStack {
			return true
		}
	}
	return false
}

// paramArgs is fn's parameters as the classifier sees them: the type,
// and the aggregate a byval attribute names.
// sretOf is the aggregate a signature's first parameter returns through, or
// the zero FType when it has none. §19.13 admits sret on at most one
// parameter and that parameter is the first, so this looks at the first and
// no further.
//
// The type and not merely the fact: §3.2.3 classifies a result exactly as it
// classifies an argument, so deciding whether one comes back in registers
// needs the type to classify.
func sretOf(sig *ir.Sig) ir.FType {
	if sig == nil || len(sig.Params()) == 0 {
		return ir.FType{}
	}
	for _, a := range sig.Params()[0].Attrs {
		if a.IsSRet() && a.Type() != nil {
			return a.Type().FType()
		}
	}
	return ir.FType{}
}

func sretParamType(fn *ir.Func) ir.FType { return sretOf(fn.Signature()) }

// callSRetType is the same question about a call's callee, answered from the
// signature at the call site: the argument is a pointer either way, and only
// the signature says which pointer it is.
func callSRetType(in *ir.Inst) ir.FType { return sretOf(calleeSig(in)) }

// sretRegs is sretInRegs for the named ABI, and sretSlots is
// sretRetSlots for it: what an aggregate small enough to return in
// registers occupies differs between the two conventions, and every
// caller of either has an ABI in hand.
func sretRegs(abi string, t ir.FType) (aggregate, bool, error) {
	if abi == abiMS {
		return msSretInRegs(t)
	}
	return sretInRegs(t)
}

func sretSlots(abi string, agg aggregate) []regSlot {
	if abi == abiMS {
		return msSretRetSlots(agg)
	}
	return sretRetSlots(agg)
}

// sretInRegs reports whether a result of type t comes back in registers, and
// in which classes. A MEMORY result is the caller's storage instead, whose
// address arrives in RDI and comes back in RAX.
func sretInRegs(t ir.FType) (aggregate, bool, error) {
	if t.IsZero() {
		return aggregate{}, false, nil
	}
	agg, err := classifyAggregate(t)
	if err != nil {
		return aggregate{}, false, fmt.Errorf("sret %s: %w", t, err)
	}
	if agg.inMemory() || len(agg.classes) == 0 {
		return agg, false, nil
	}
	return agg, true, nil
}

// sretRetSlots is where each eightbyte of a register-returned aggregate comes
// back: the integer classes into RAX and RDX in order, the SSE ones into XMM0
// and XMM1, counted separately the way §3.2.3 counts the argument files.
func sretRetSlots(agg aggregate) []regSlot {
	var ints, floats int
	out := make([]regSlot, 0, len(agg.classes))
	for _, c := range agg.classes {
		if c == classSSE {
			out = append(out, regSlot{kind: placeFloat, i: floats, w: wf64})
			floats++
			continue
		}
		out = append(out, regSlot{kind: placeInt, i: ints, w: w64})
		ints++
	}
	return out
}

func paramArgs(fn *ir.Func) []abiArg {
	ps := fn.Params()
	out := make([]abiArg, len(ps))
	for i, p := range ps {
		out[i] = abiArg{t: p.Type()}
	}
	// The attributes are on the signature, not on the values: a
	// parameter's Def is what the body reads, and byval is a fact about
	// the boundary it arrived across.
	if sig := fn.Signature(); sig != nil {
		sp := sig.Params()
		for i := range out {
			if i < len(sp) {
				out[i].byval = byvalOf(sp[i].Attrs)
			}
		}
		if len(out) > 0 {
			out[0].sret = sretOf(sig)
		}
	}
	return out
}

// byvalOf is the aggregate a parameter's byval attribute names, or the
// zero FType when it has none.
func byvalOf(attrs []ir.ParamAttr) ir.FType {
	for _, a := range attrs {
		if a.IsByVal() && a.Type() != nil {
			return a.Type().FType()
		}
	}
	return ir.FType{}
}

// callArgSpec is a call's arguments as the classifier sees them. The
// byval attributes come from the callee's signature, not the values:
// the argument is a pointer either way.
func callArgSpec(in *ir.Inst) []abiArg {
	args := callArgs(in)
	out := scalarArgs(typesOf(args))
	sig := calleeSig(in)
	if sig == nil {
		return out
	}
	ps := sig.Params()
	for i := range out {
		if i < len(ps) {
			out[i].byval = byvalOf(ps[i].Attrs)
			continue
		}
		out[i].vararg = sig.IsVariadic()
	}
	if len(out) > 0 {
		out[0].sret = sretOf(sig)
	}
	return out
}

// calleeSig is the signature a call is made against, which is the
// callee's for a direct call and the func typedef's for an indirect one.
func calleeSig(in *ir.Inst) *ir.Sig {
	if in.Op().Verb == ir.VCallInd {
		if t := in.NamedType(); t != nil {
			return t.Sig()
		}
		return nil
	}
	if c := in.Callee(); c != nil {
		return c.Signature()
	}
	return nil
}

// classifyParams copies fn's parameters out of the registers they
// arrived in and into vregs the allocator is free to place.
func classifyParams(fn *ir.Func, entry *mir.Block, vr *vregs, fr *frame) error {
	abi := fn.Module().Layout().ABI
	places, err := classify(abi, paramArgs(fn))
	if err != nil {
		return fmt.Errorf("lower: %s: %w", fn.Name(), err)
	}

	for i, p := range fn.Params() {
		pl := places[i]

		v, err := vr.define(p)
		if err != nil {
			return fmt.Errorf("lower: %s: parameter %q: %w", fn.Name(), p.Name(), err)
		}

		if pl.isAggregate() {
			if err := incomingAggregate(fn, entry, vr, fr, pl, v); err != nil {
				return fmt.Errorf("lower: %s: parameter %q: %w", fn.Name(), p.Name(), err)
			}
			continue
		}

		if pl.kind == placeStack {
			entry.Emit(mir.Instr{Op: argLoadOp{off: stackParamOff(pl.off), w: pl.w()}, Defs: []mir.VReg{v}})
			continue
		}

		slot := pl.regs[0]
		var incoming mir.VReg
		if slot.kind == placeFloat {
			incoming = vr.physicalXmm(floatArgReg(abi, slot.i), slot.w)
		} else {
			incoming = vr.physical(intArgReg(abi, slot.i), slot.w)
		}
		emitCopy(entry, v, incoming, slot.w)
	}
	return nil
}

// incomingAggregate gives a byval parameter the address the body reads
// it through. The two cases are opposite work: one the caller left in
// the incoming area is already a contiguous copy and the parameter is a
// lea of it, while one that arrived in registers has no address until
// this function stores the eightbytes into a slot to make one.
func incomingAggregate(fn *ir.Func, entry *mir.Block, vr *vregs, fr *frame, pl place, dst mir.VReg) error {
	abi := fn.Module().Layout().ABI

	if pl.indirect {
		// The caller passed the address of its own copy, so there is
		// nothing to lea and nothing to spill: the parameter is that
		// pointer, wherever it arrived.
		if pl.kind == placeStack {
			entry.Emit(mir.Instr{Op: argLoadOp{off: stackParamOff(pl.off), w: w64}, Defs: []mir.VReg{dst}})
			return nil
		}
		incoming := vr.physical(intArgReg(abi, pl.regs[0].i), w64)
		emitCopy(entry, dst, incoming, w64)
		return nil
	}

	if pl.kind == placeStack {
		entry.Emit(mir.Instr{Op: leaInOp{off: stackParamOff(pl.off)}, Defs: []mir.VReg{dst}})
		return nil
	}

	// Rounded to a whole eightbyte, because that is the unit the spills
	// below write: a three-byte struct arrives in one register and is
	// written back as eight, so a three-byte slot takes the five under it
	// with it. The alignment is at least eight for the same reason.
	size := alignUp(pl.size, 8)
	if size == 0 {
		size = 8
	}
	align := aggAlign(pl)
	if align < 8 {
		align = 8
	}
	off, err := fr.reserveBytes(size, align)
	if err != nil {
		return err
	}
	for k, slot := range pl.regs {
		var incoming mir.VReg
		if slot.kind == placeFloat {
			incoming = vr.physicalXmm(floatArgReg(abi, slot.i), slot.w)
		} else {
			incoming = vr.physical(intArgReg(abi, slot.i), slot.w)
		}
		entry.Emit(mir.Instr{
			Op:   spillOp{off: off + int32(k*8), w: eightbyteWidth(slot.kind)},
			Uses: []mir.VReg{incoming},
		})
	}
	entry.Emit(mir.Instr{Op: leaOp{off: off}, Defs: []mir.VReg{dst}})
	return nil
}

// eightbyteWidth is how one eightbyte of a byval aggregate is stored:
// always eight bytes, in whichever file its class named.
func eightbyteWidth(k placeKind) width {
	if k == placeFloat {
		return wf64
	}
	return w64
}

// aggAlign is the alignment a byval aggregate's storage needs, which is
// its type's, bounded by what the frame can guarantee.
func aggAlign(pl place) uint64 {
	_, align, err := sizeAlign(pl.byval)
	if err != nil || align == 0 {
		return 8
	}
	if align > maxAlign {
		return maxAlign
	}
	return align
}

// classifyBlockParams allocates a vreg for every block's own parameters.
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

// wideVector reports whether t occupies a whole vector register.
func wideVector(t ir.RegType) bool { return t == ir.TypeF128 || t == ir.TypeV128 }
