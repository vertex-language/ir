package arm64

import (
	"fmt"

	"github.com/vertex-language/arm64/reg"

	"github.com/vertex-language/ir"
	"github.com/vertex-language/ir/lower/mir"
)

// frame is one function's stack frame.
//
// The layout is AAPCS64's: the frame record — the saved frame pointer and link
// register — sits at the top, X29 points at it, and everything this function
// owns is below. Locals are addressed as a negative displacement from X29, the
// way they are from RBP on the other architecture.
type frame struct {
	// apple is Apple's arm64 variant, whose stack arguments are packed.
	// See classifyAAPCS.
	apple bool

	// next is how many bytes below X29 are spoken for. A running total
	// rather than a final size, because spilling adds to it.
	next uint64

	// outArgs is the outgoing argument area at the bottom of the frame.
	outArgs uint64

	slot map[*ir.Inst]int64 // ptr.alloc -> its offset from X29

	// overAlign is the alignment a slot asked for that is stricter than
	// the frame's own. Its address is rounded up to that where it is
	// taken, inside the padding reserved with it.
	overAlign map[*ir.Inst]uint64

	saveAt map[reg.X]int64 // callee-saved register -> its slot

	// saveAtVec is the same for the vector file. AAPCS64 preserves only
	// the low 64 bits of V8 through V15, which is a D register — and a D
	// register is the widest float this package has, so the promise the
	// ABI makes and the promise this package needs are the same one.
	saveAtVec map[reg.V]int64

	// saveBytes is how much of the frame the callee-saved area occupies.
	//
	// It sits directly below the frame record, above every local, and that
	// is not a preference: libunwind reads a saved register out of a fixed
	// displacement from X29 rather than out of anything the object file
	// says, so a frame that puts its saves anywhere else cannot be
	// described by a compact unwind encoding at all. See unwind.go.
	//
	// Which is awkward, because the size is not known until the register
	// allocator has run and every local already has an offset. So the
	// locals are planned as though the area were not there and moved down
	// by this much on the way out — that is what local() below is.
	saveBytes int64

	// force makes a prologue even when nothing needs storage.
	force bool

	// vaOffset is where this function's variadic arguments begin in the
	// caller's outgoing area: after whatever its named parameters put
	// there. Only meaningful when variadic is set.
	vaOffset int64
	variadic bool
}

// maxAlign is the strictest alignment a slot can ask for, and the alignment
// SP is required to hold at every instruction that uses it as a base.
const maxAlign = 16

// size is how much the prologue subtracts from SP after the frame record.
func (f *frame) size() uint64 {
	return alignUp(f.next+uint64(f.saveBytes)+f.outArgs, maxAlign)
}

// local is where an offset planned before the save area existed finally is.
//
// The sign is the whole test. Everything this function owns is below X29 and
// is therefore negative; an incoming stack argument is in the caller's frame,
// above the frame record, and is positive. Only the first moves.
func (f *frame) local(off int64) int64 {
	if off < 0 {
		return off - f.saveBytes
	}
	return off
}

// outgoing is the size of the outgoing argument area, which a dynamic
// allocation has to leave at the bottom of the frame.
func (f *frame) outgoing() int64 { return int64(f.outArgs) }

// needed reports whether this function has a prologue at all.
//
// A leaf that stores nothing does not: X30 still holds the return address, so
// there is nothing to save and nothing to restore.
func (f *frame) needed() bool { return f.force || f.size() > 0 }

func (f *frame) reserveOutArgs(n uint64) {
	if n > f.outArgs {
		f.outArgs = n
	}
}

// reserve takes eight more bytes of frame and returns their offset from X29.
func (f *frame) reserve() int64 {
	f.next += 8
	return -int64(f.next)
}

// reserveBytes takes size bytes at the given alignment and returns the offset
// of the lowest one from X29. Growing downwards, so the alignment is applied
// to the total rather than to the offset.
func (f *frame) reserveBytes(size, align uint64) int64 {
	if align == 0 {
		align = 8
	}
	if align > maxAlign {
		align = maxAlign
	}
	f.next = alignUp(f.next+size, align)
	return -int64(f.next)
}

// reserveSaves gives every callee-saved register a slot, in the one order the
// compact unwind encoding can name: X29-8 downwards, the integer file before
// the vector file, each in register order. unwindEncoding depends on it and
// so does every unwinder that reads what it writes.
func (f *frame) reserveSaves(regs []reg.X, vecs []reg.V) {
	if len(regs) == 0 && len(vecs) == 0 {
		return
	}
	f.saveAt = make(map[reg.X]int64, len(regs))
	f.saveAtVec = make(map[reg.V]int64, len(vecs))
	at := int64(0)
	for _, r := range regs {
		at -= 8
		f.saveAt[r] = at
	}
	for _, r := range vecs {
		at -= 8
		f.saveAtVec[r] = at
	}
	f.saveBytes = -at
}

// planFrame assigns every ptr.alloc in fn a slot and returns the frame they
// add up to.
func planFrame(fn *ir.Func, opts Options) (*frame, error) {
	fr := &frame{slot: map[*ir.Inst]int64{}, overAlign: map[*ir.Inst]uint64{}}

	fr.apple = opts.Variadic == VariadicDarwin
	if places, err := classifyAAPCS(paramArgs(fn), sretParamType(fn), fr.apple); err == nil {
		fr.force = usesStack(places)
		// A variadic function's list starts after whatever its named
		// parameters put on the stack, which under Apple's variant is
		// the only place a variadic argument ever is.
		if sig := fn.Signature(); sig != nil && sig.IsVariadic() {
			fr.variadic = true
			fr.vaOffset = int64(alignUp(stackEnd(places), 8))
			fr.force = true
		}
	}

	var off uint64
	for _, blk := range fn.Blocks() {
		// All, not Insts: an invoke is a terminator, and it is a call —
		// it needs the frame record saved and it needs room for whatever
		// arguments did not fit in registers. A frame planned without it
		// puts the outgoing area over the lowest local.
		for _, in := range blk.All() {
			switch in.Op().Verb {
			case ir.VTLSAddr:
				// The thread-local sequence calls a thunk, so it needs
				// the frame record saved for the same reason a call
				// does: X30 does not survive it.
				fr.force = true
				continue
			case ir.VCall, ir.VCallInd, ir.VInvoke, ir.VInvokeInd:
				// A call needs the frame record saved, because it will
				// overwrite X30 with its own return address.
				fr.force = true
				fr.reserveOutArgs(callStackBytes(in, opts))
				continue
			case ir.VResume:
				// _Unwind_Resume is a call like any other.
				fr.force = true
				continue
			case ir.VFrameAddr, ir.VReturnAddr:
				fr.force = true
				continue
			case ir.VAlloca, ir.VStackSave, ir.VStackRestore:
				// SP moves, so X29 has to exist to get it back and
				// to reach everything addressed from it.
				fr.force = true
				continue
			}
			if in.Op().Verb == ir.VAlloc && in.Zeroed() {
				// The memset that zeroes it is a call.
				fr.force = true
			}
			if _, ok := libcalls[in.Op().Verb]; ok {
				// §E is a call whether or not the module wrote one.
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
			if align > maxAlign {
				// More than the frame itself promises. The slot is given
				// the bytes the rounding can cost -- X29 is 16-aligned,
				// so reaching any stricter alignment moves the address up
				// by less than that alignment -- and the address is
				// rounded where it is taken (see frameOp).
				fr.overAlign[in] = align
				size += align - maxAlign
				align = maxAlign
			}
			off = alignUp(off+size, align)
			if off > 1<<31 {
				return nil, fmt.Errorf("lower: %s: the frame exceeds 2GB", fn.Name())
			}
			fr.slot[in] = -int64(off)
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

// callStackBytes is how much outgoing area one call needs.
func callStackBytes(in *ir.Inst, opts Options) uint64 {
	places, err := callPlaces(in, opts)
	if err != nil {
		return 0
	}
	return outgoingBytes(places)
}

// callArgs is the operands one call actually passes.
//
// An indirect call's first operand is the callee's address, which goes in no
// argument register: including it would shift every real argument one place
// right and reserve room the call never writes to.
func callArgs(in *ir.Inst) []*ir.Def {
	args := in.Args()
	if v := in.Op().Verb; (v == ir.VCallInd || v == ir.VInvokeInd) && len(args) > 0 {
		return args[1:]
	}
	return args
}

// callSig is the signature one call was written against, which is what says
// where its variadic tail begins.
func callSig(in *ir.Inst) *ir.Sig {
	switch in.Op().Verb {
	case ir.VCall, ir.VInvoke:
		if callee := in.Callee(); callee != nil {
			return callee.Signature()
		}
	case ir.VCallInd, ir.VInvokeInd:
		if t := in.NamedType(); t != nil {
			return t.Sig()
		}
	}
	return nil
}

// callPlaces is where one call's arguments go.
func callPlaces(in *ir.Inst, opts Options) ([]place, error) {
	args := callArgs(in)
	named := len(args)
	variadic := false
	if sig := callSig(in); sig != nil && sig.IsVariadic() {
		variadic, named = true, len(sig.Params())
	}
	return classifyCall(callArgSpec(in, args), named, variadic, opts.Variadic, callSRetType(in))
}

// classifyCall places a call's arguments, honouring the variadic cut.
//
// Under the base standard there is no cut for the caller: an argument past
// the last named one is placed exactly as a named one would be, in the next
// free register of its file or else on the stack (AAPCS64 §6.4.2). The work
// is the callee's -- va_start spills the argument registers to a save area --
// so a call to printf is an ordinary call.
//
// Under Apple's variant every argument past the last named one goes on the
// stack whatever its type, one eight-byte slot each, continuing from wherever
// the named arguments left the stack. The named ones are placed exactly as
// they would be in a non-variadic call.
func classifyCall(args []abiArg, named int, variadic bool, abi VariadicABI, sret ir.FType) ([]place, error) {
	if !variadic || abi != VariadicDarwin {
		return classifyAAPCS(args, sret, abi == VariadicDarwin)
	}

	head, err := classifyAAPCS(args[:named], sret, true)
	if err != nil {
		return nil, err
	}
	out := make([]place, 0, len(args))
	out = append(out, head...)

	// The tail's eightbyte slots start at the next eightbyte after the
	// named arguments, which Apple's packing may have left at any byte.
	off := int64(alignUp(stackEnd(head), 8))
	for _, a := range args[named:] {
		if !a.byval.IsZero() {
			// An aggregate in the variadic tail would have to be copied
			// into the list rather than placed, and va_arg would have to
			// read it back out. Neither is written.
			return nil, fmt.Errorf("a byval aggregate past the last named argument is not passed yet")
		}
		w, ok := widthOf(a.t)
		if !ok {
			return nil, fmt.Errorf("%s is not a value this package passes", a.t)
		}
		out = append(out, place{kind: placeStack, off: off, w: w})
		off += 8
	}
	return out, nil
}

// sretOf is the aggregate a signature's first parameter returns through, or
// the zero FType when it has none. §19.13 admits sret on at most one
// parameter and that parameter is the first, so this looks at the first and
// no further.
//
// The type and not merely the fact: §5.5 returns a result the same way §5.4
// passes an argument, so what comes back in registers and what comes back
// through the caller's storage is the same classification, and it needs the
// type to make it.
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
// signature at the call site rather than from the arguments: the argument is
// a pointer either way, and only the signature says which pointer it is.
func callSRetType(in *ir.Inst) ir.FType { return sretOf(callSig(in)) }

// sretInRegs reports whether a result of type t comes back in registers, and
// in which. A result that does not is the caller's storage, whose address
// arrives in X8.
func sretInRegs(t ir.FType) (aggregate, bool, error) {
	if t.IsZero() {
		return aggregate{}, false, nil
	}
	agg, err := classifyAggregate(t)
	if err != nil {
		return aggregate{}, false, fmt.Errorf("sret %s: %w", t, err)
	}
	switch agg.kind {
	case aggGPR, aggHFA:
		return agg, true, nil
	}
	return agg, false, nil
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

// stackParamOff is where a stack parameter at byte offset off in the caller's
// outgoing area ends up above this function's X29.
//
// Sixteen, and for the same reason it is sixteen on the other architecture:
// the frame record this function pushed is two eightbytes, and the caller's
// area begins just above it.
func stackParamOff(off int64) int64 { return 16 + off }

func alignUp(n, a uint64) uint64 {
	if a == 0 {
		return n
	}
	return (n + a - 1) &^ (a - 1)
}

// classifyAAPCS places a list of arguments by AAPCS64 §6.4.
//
// The two files are counted separately and the stack is one queue, which is
// the same shape SysV has. What differs is the count — eight of each file
// rather than six and eight — and that a stack argument is packed to its own
// size rather than given a whole eightbyte.
//
// apple selects Apple's variant, which differs in one respect that matters
// here: a scalar stack argument is packed at its own size and alignment -- a
// bool is one byte, an int four at the next four-byte boundary -- where the
// base standard gives every one a doubleword. Aggregates keep the base rule.
func classifyAAPCS(args []abiArg, sret ir.FType, apple bool) ([]place, error) {
	var ints, floats int
	var stackBytes uint64
	out := make([]place, len(args))

	toStack := func(size, align uint64) place {
		if align < 8 {
			align = 8
		}
		stackBytes = alignUp(stackBytes, align)
		p := place{kind: placeStack, off: int64(stackBytes)}
		stackBytes += alignUp(size, 8)
		return p
	}
	// A scalar under Apple's packing: its own size, its own alignment.
	toStackPacked := func(size uint64) place {
		stackBytes = alignUp(stackBytes, size)
		p := place{kind: placeStack, off: int64(stackBytes), size: size}
		stackBytes += size
		return p
	}

	for i, a := range args {
		// A pointer that is the indirect result register by
		// declaration. Nothing is re-derived: Swift's `@out` is X8
		// whatever the type's size, which is the whole difference
		// from sret below.
		if a.out {
			out[i] = place{kind: placeIndirect, w: w64}
			continue
		}
		if i == 0 && !sret.IsZero() {
			agg, inRegs, err := sretInRegs(sret)
			if err != nil {
				return nil, err
			}
			if inRegs {
				// §5.5 brings this result back in registers, so the caller
				// supplies no address at all. The parameter the front end
				// wrote still has to be somewhere the body can write
				// through, and that somewhere is a slot of the callee's
				// own: an aggregate place with no registers arriving.
				out[i] = place{kind: placeInt, byval: sret, size: agg.size, align: agg.align}
				continue
			}
			// §6.9's indirect result location register, which is not part
			// of the argument sequence: the first real argument is still
			// X0. ir guarantees this attribute is on the first parameter
			// and on no other.
			out[i] = place{kind: placeIndirect, w: w64}
			continue
		}

		// The self register, which is beside the sequence rather than
		// in it: the argument after this one is placed as though this
		// one were not there.
		if a.self {
			w, ok := widthOf(a.t)
			if !ok {
				return nil, fmt.Errorf("%s is not a value the self register carries", a.t)
			}
			out[i] = place{kind: placeSelf, w: w}
			continue
		}
		// The async context register, beside the sequence for the same
		// reason the self register is.
		if a.async {
			w, ok := widthOf(a.t)
			if !ok {
				return nil, fmt.Errorf("%s is not a value the async context register carries", a.t)
			}
			out[i] = place{kind: placeAsync, w: w}
			continue
		}

		if a.byval.IsZero() {
			w, ok := widthOf(a.t)
			if !ok {
				return nil, fmt.Errorf("%s is not a value this package passes; only i32, i64, ptr, f32, and f64 are placed", a.t)
			}
			switch {
			case w.isFloat() && floats < len(aapcsFloatArgs):
				out[i] = place{kind: placeFloat, i: floats, w: w}
				floats++
			case !w.isFloat() && ints < len(aapcsIntArgs):
				out[i] = place{kind: placeInt, i: ints, w: w}
				ints++
			case apple:
				size := uint64(8)
				switch {
				case a.narrow != 0 && w == w32:
					size = uint64(a.narrow)
				case w == w32 || w == wf32:
					size = 4
				}
				p := toStackPacked(size)
				p.w = w
				if a.narrow != 0 && w == w32 {
					p.narrow, p.nsigned = a.narrow, a.nsigned
				}
				out[i] = p
			default:
				p := toStack(8, 8)
				p.w = w
				out[i] = p
			}
			continue
		}

		agg, err := classifyAggregate(a.byval)
		if err != nil {
			return nil, fmt.Errorf("byval %s: %w", a.byval, err)
		}

		switch agg.kind {
		case aggEmpty:
			// No bytes, so no register and no slot. The parameter still
			// needs an address to be read through; the callee makes one.
			out[i] = place{kind: placeInt, i: -1, w: w64, byval: a.byval}
			continue

		case aggIndirect:
			// §5.4 replaced the aggregate with a pointer to the caller's
			// copy, and a pointer is an ordinary integer argument — which
			// is what the signature already carries, so nothing here has
			// bytes to move.
			if ints < len(aapcsIntArgs) {
				out[i] = place{kind: placeInt, i: ints, w: w64}
				ints++
				continue
			}
			p := toStack(8, 8)
			p.w = w64
			out[i] = p
			continue
		}

		// A homogeneous aggregate takes SIMD registers and any other takes
		// general-purpose ones, and in both cases all of them or none:
		// §5.4 has no half-in-half-out case, and splitting one would be a
		// third convention neither end implements.
		free, used := len(aapcsIntArgs)-ints, &ints
		regKind := placeInt
		if agg.kind == aggHFA {
			free, used = len(aapcsFloatArgs)-floats, &floats
			regKind = placeFloat
		}
		if agg.n <= free {
			p := place{kind: regKind, byval: a.byval, size: agg.size, align: agg.align}
			for k := 0; k < agg.n; k++ {
				p.regs = append(p.regs, regSlot{
					kind: regKind,
					i:    *used,
					w:    agg.w,
					off:  int64(uint64(k) * agg.step),
				})
				*used++
			}
			out[i] = p
			continue
		}

		// Out of registers of that file. §5.4 stops handing them out
		// rather than skipping to a later one, so the file is exhausted
		// for every argument after this too.
		*used = len(aapcsIntArgs)
		p := toStack(agg.size, agg.align)
		p.byval, p.size, p.align = a.byval, agg.size, agg.align
		out[i] = p
	}
	return out, nil
}

// An abiArg is one argument as the classifier sees it: its register type,
// and the aggregate it stands for when the signature says byval.
type abiArg struct {
	t     ir.RegType
	byval ir.FType
	self  bool
	// async is the async context pointer, which goes in X22 by
	// declaration rather than by size. See ir.SwiftAsync.
	async bool
	// out is the indirect-result pointer, which goes in X8 by
	// declaration rather than by size. See ir.SwiftIndirectResult.
	out bool
	// narrow is the source type's size for an i32 that is really a char,
	// short or bool, and nsigned its sign; zero otherwise. See ir.Narrow.
	narrow  uint8
	nsigned bool
}

// scalarArgs is a list of register types with no byval among them.
func scalarArgs(types []ir.RegType) []abiArg {
	out := make([]abiArg, len(types))
	for i, t := range types {
		out[i] = abiArg{t: t}
	}
	return out
}

// byvalOf is the aggregate a parameter's byval attribute names, or the zero
// FType when it has none.
// outOf reports whether a parameter carries the address of storage
// the result is written into, by declaration rather than by size.
func outOf(attrs []ir.ParamAttr) bool {
	for _, a := range attrs {
		if a.IsSwiftIndirectResult() {
			return true
		}
	}
	return false
}

// asyncOf reports whether a parameter carries an async context, and so
// travels in the async context register.
func asyncOf(attrs []ir.ParamAttr) bool {
	for _, a := range attrs {
		if a.IsSwiftAsync() {
			return true
		}
	}
	return false
}

// selfOf reports whether a parameter travels in the self register.
func selfOf(attrs []ir.ParamAttr) bool {
	for _, a := range attrs {
		if a.IsSwiftSelf() {
			return true
		}
	}
	return false
}

// narrowOf reads an ir.Narrow attribute.
func narrowOf(attrs []ir.ParamAttr) (uint8, bool) {
	for _, a := range attrs {
		if n, signed, ok := a.NarrowWidth(); ok {
			return uint8(n), signed
		}
	}
	return 0, false
}

func byvalOf(attrs []ir.ParamAttr) ir.FType {
	for _, a := range attrs {
		if a.IsByVal() && a.Type() != nil {
			return a.Type().FType()
		}
	}
	return ir.FType{}
}

// paramArgs is fn's parameters as the classifier sees them. byval is a fact
// about the signature rather than about the Def, which is a pointer either
// way.
func paramArgs(fn *ir.Func) []abiArg {
	out := scalarArgs(paramTypes(fn))
	sig := fn.Signature()
	if sig == nil {
		return out
	}
	ps := sig.Params()
	for i := range out {
		if i < len(ps) {
			out[i].byval = byvalOf(ps[i].Attrs)
			out[i].self = selfOf(ps[i].Attrs)
			out[i].async = asyncOf(ps[i].Attrs)
			out[i].out = outOf(ps[i].Attrs)
			out[i].narrow, out[i].nsigned = narrowOf(ps[i].Attrs)
		}
	}
	return out
}

// callArgSpec is a call's arguments as the classifier sees them, with the
// byval attributes read off the callee's signature.
func callArgSpec(in *ir.Inst, args []*ir.Def) []abiArg {
	return sigArgSpec(callSig(in), args)
}

// sigArgSpec pairs a call's argument values with the signature that says how
// they travel. The argument is a pointer whether or not it is byval, so only
// the signature can tell the two apart.
func sigArgSpec(sig *ir.Sig, args []*ir.Def) []abiArg {
	out := scalarArgs(typesOf(args))
	if sig == nil {
		return out
	}
	ps := sig.Params()
	for i := range out {
		if i < len(ps) {
			out[i].byval = byvalOf(ps[i].Attrs)
			out[i].self = selfOf(ps[i].Attrs)
			out[i].async = asyncOf(ps[i].Attrs)
			out[i].out = outOf(ps[i].Attrs)
			out[i].narrow, out[i].nsigned = narrowOf(ps[i].Attrs)
		}
	}
	return out
}

func usesStack(places []place) bool {
	for _, p := range places {
		if p.kind == placeStack {
			return true
		}
	}
	return false
}

// stackEnd is the first byte past the stack arguments, unrounded — where the
// next thing placed on the stack would go.
func stackEnd(places []place) uint64 {
	var n uint64
	for _, p := range places {
		if p.kind != placeStack {
			continue
		}
		if end := uint64(p.off) + stackBytesOf(p); end > n {
			n = end
		}
	}
	return n
}

// stackBytesOf is how much of the outgoing area one stack argument occupies.
// A scalar takes a doubleword whatever its width; an aggregate takes its own
// size, rounded up to keep the next argument aligned.
func stackBytesOf(p place) uint64 {
	if p.isAggregate() {
		return alignUp(p.size, 8)
	}
	if p.size != 0 {
		return p.size // packed, under Apple's variant
	}
	return 8
}

func outgoingBytes(places []place) uint64 {
	return alignUp(stackEnd(places), maxAlign)
}

// classifyParams copies fn's parameters out of the registers they arrived in
// and into vregs the allocator is free to place.
func classifyParams(fn *ir.Func, entry *mir.Block, vr *vregs, fr *frame) error {
	places, err := classifyAAPCS(paramArgs(fn), sretParamType(fn), fr.apple)
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
			incomingAggregate(entry, vr, fr, pl, v)
			continue
		}
		if pl.kind == placeStack {
			entry.Emit(mir.Instr{
				Op:   frameLoadOp{off: stackParamOff(pl.off), w: pl.w, narrow: pl.narrow, signed: pl.nsigned},
				Defs: []mir.VReg{v},
			})
			continue
		}
		var incoming mir.VReg
		switch pl.kind {
		case placeFloat:
			incoming = vr.physicalVec(aapcsFloatArgs[pl.i], pl.w)
		case placeIndirect:
			incoming = vr.physical(reg.X8, pl.w)
		case placeSelf:
			incoming = vr.physical(reg.X20, pl.w)
		case placeAsync:
			incoming = vr.physical(reg.X22, pl.w)
		default:
			incoming = vr.physical(aapcsIntArgs[pl.i], pl.w)
		}
		emitCopy(entry, v, incoming, pl.w)
	}
	return nil
}

// incomingAggregate gives a byval parameter the address its body reads it
// through. The two cases are opposite work: one the caller left in the
// incoming area is already a contiguous copy, so the parameter is its
// address, while one that arrived in registers has no address at all until
// this function stores them into a slot to make one.
func incomingAggregate(entry *mir.Block, vr *vregs, fr *frame, pl place, dst mir.VReg) {
	if pl.kind == placeStack {
		entry.Emit(mir.Instr{Op: frameOp{off: stackParamOff(pl.off)}, Defs: []mir.VReg{dst}})
		return
	}

	// The slot has to hold the registers, not merely the aggregate: a
	// three-byte struct arrives in one X register and is written back as a
	// full doubleword, so a slot of three bytes would take the five below it
	// with it. An empty aggregate still needs somewhere to point, so it gets
	// a slot too — nothing reads the bytes, but the address has to be real.
	size := alignUp(pl.size, 8)
	if size == 0 {
		size = 8
	}
	align := pl.align
	if align < 8 {
		align = 8
	}
	off := fr.reserveBytes(size, align)

	for _, slot := range pl.regs {
		var incoming mir.VReg
		if slot.kind == placeFloat {
			incoming = vr.physicalVec(aapcsFloatArgs[slot.i], slot.w)
		} else {
			incoming = vr.physical(aapcsIntArgs[slot.i], slot.w)
		}
		entry.Emit(mir.Instr{
			Op:   frameStoreOp{off: off + slot.off, w: slot.w},
			Uses: []mir.VReg{incoming},
		})
	}
	entry.Emit(mir.Instr{Op: frameOp{off: off}, Defs: []mir.VReg{dst}})
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

// errorResult is the index of a signature's swifterror result, or -1.
//
// It travels in X21 rather than in the return sequence, so it is left
// out of the ordinary placement at both ends: the results before and
// after it are placed as though it were not there. Swift's rule, and
// the only thing that makes it work is that both ends say so.
func errorResult(sig *ir.Sig) int {
	if sig == nil {
		return -1
	}
	for i, r := range sig.Rets() {
		for _, a := range r.Attrs {
			if a.IsSwiftError() {
				return i
			}
		}
	}
	return -1
}

// funcErrorResult is errorResult for a function's own signature.
func funcErrorResult(fn *ir.Func) int {
	if fn == nil {
		return -1
	}
	return errorResult(fn.Signature())
}
