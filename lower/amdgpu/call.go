package amdgpu

// The device calling convention. Every function this backend emits is
// compiled by this backend, so the convention is its own rather than
// LLVM's, and chosen for what O0 can do without an allocator that
// understands clobbers:
//
//   - Arguments in v0 upward, a dword each, a 64-bit value in an
//     even-aligned pair, an i1 as 0 or 1 in a dword; results the same
//     from v0. More than thirty-two dwords of either is refused.
//   - The return address in s[30:31], written by s_swappc_b64; s32 the
//     stack pointer, s33 the frame pointer, s[34:35] scratch for the
//     target address. The stack is the private segment: SP counts bytes
//     of it, per lane, the way scratch addresses do.
//   - The callee saves every register it touches and restores it before
//     returning — VGPRs to its frame, SGPRs to lanes of v58 — except the
//     ones its results come back in. A caller therefore keeps nothing
//     live in v0..v31 across a call and everything else where it was,
//     which is why the call instruction needs no clobber list: its
//     arguments and results are pinned vregs, and the allocator keeps
//     everything else out of those registers while they are live.
//   - The execution mask is the caller's, and is what it was on return.
//   - The work-item's place in the grid travels implicitly, the way
//     LLVM passes it: the three work-item ids packed ten bits each in
//     v31, the workgroup ids in s[36:38], the workgroup sizes in
//     s[39:41] and the workgroup counts in s[42:44]. A kernel that calls
//     fills them at entry, whatever it uses itself, and every device
//     function may read them.
//
// A function that calls, and every device function, allocates from a
// pool that leaves v0..v31 and s30..s45 alone, so a pinned argument
// never overlaps a value the allocator placed.
//
// A kernel's private segment is its own frame plus the deepest chain of
// frames beneath it, so device functions are lowered before the kernels
// that call them, callees before callers; recursion is refused, since
// it has no bound. An indirect call is assumed to reach the deepest
// device function in the module, and its target is read from the first
// active lane: a function pointer that differs across the wave is
// refused, since a waterfall loop is not written.

import (
	"fmt"

	"github.com/vertex-language/ir"
	"github.com/vertex-language/ir/lower/mir"
)

// The reserved registers.
const (
	retAddrSGPR   = 30 // s[30:31]
	spSGPR        = 32
	fpSGPR        = 33
	callTempSGPR  = 34 // s[34:35]
	sgprSaveVGPR  = 58 // lanes hold the callee's saved SGPRs
	maxArgDwords  = 32
	argVGPRsFirst = 32 // the singles a calling function may allocate from

	// The implicit registers.
	tidVGPR       = 31 // the work-item ids, packed
	wgIDSGPR      = 36 // s[36:38]
	wgSizeSGPR    = 39 // s[39:41]
	wgCountSGPR   = 42 // s[42:44]
	callPairsFrom = 46 // the SGPR pairs a calling function may allocate from
)

// A slot is where one argument or result travels.
type slot struct {
	phys int   // the VGPR number
	w    width // v32 or v64
	i1   bool  // an i1, as 0 or 1 in a dword
}

// argSlots assigns the parameters or results of a signature.
func argSlots(types []ir.RegType) ([]slot, error) {
	var out []slot
	n := 0
	for _, t := range types {
		w, ok := widthOf(t)
		if !ok {
			return nil, fmt.Errorf("%s crosses no call", t)
		}
		s := slot{w: w}
		switch w {
		case s64:
			s.w, s.i1 = v32, true
			fallthrough
		case v32:
			s.phys = n
			n++
		case v64:
			n = (n + 1) &^ 1
			s.phys = n
			n += 2
		}
		if n > maxArgDwords {
			return nil, fmt.Errorf("more than %d dwords of arguments; the rest would go on the stack, which is not laid out yet", maxArgDwords)
		}
		out = append(out, s)
	}
	return out, nil
}

func paramTypes(sig *ir.Sig) []ir.RegType {
	var out []ir.RegType
	for _, p := range sig.Params() {
		out = append(out, p.Type)
	}
	return out
}

func retTypes(sig *ir.Sig) []ir.RegType {
	var out []ir.RegType
	for _, r := range sig.Rets() {
		out = append(out, r.Type)
	}
	return out
}

// The call's MIR ops.
type (
	// callOp is s_swappc_b64 to a symbol, or to the address in the
	// last two Uses when sym is empty. Its other Uses are the pinned
	// argument vregs, its Defs the pinned result vregs.
	callOp struct {
		sym    string
		nargs  int
		scalar bool // the target is the SGPR pair in the last Use, not a VGPR pair
	}
	// callResultsOp defines the pinned result vregs after a call and
	// emits nothing: a def on the call itself would interfere with the
	// argument pinned to the same register, which a call consumes.
	callResultsOp struct{}
	// spInitOp is the kernel's s_mov_b32 s32, <its frame size>, filled
	// in at emit when the size is known.
	spInitOp struct{}
)

// entryDevice fills a device function's entry: parameters out of their
// pinned VGPRs, and the implicit registers pinned for §W to read.
func (x *fnState) entryDevice(mb *mir.Block) error {
	c := x.cursor(mb)
	x.pinImplicit()
	slots, err := argSlots(paramTypes(x.fn.Signature()))
	if err != nil {
		return err
	}
	for i, d := range x.fn.Params() {
		s := slots[i]
		in := x.vr.fresh(s.w)
		x.vr.pin(in, physOf(s.w, s.phys))
		v, err := x.vr.define(d)
		if err != nil {
			return err
		}
		if s.i1 {
			x.emit(c, "v_cmp_ne_u32", rs(v), rs(in), def(0), imm(0), use(0))
			continue
		}
		emitCopy(c, v, in, s.w)
	}
	return nil
}

// pinImplicit binds the implicit registers to vregs the §W verbs read.
func (x *fnState) pinImplicit() {
	x.tid[0] = x.vr.fresh(v32)
	x.vr.pin(x.tid[0], physOf(v32, tidVGPR))
	for axis := 0; axis < 3; axis++ {
		x.wgID[axis] = x.vr.fresh(s32)
		x.vr.pin(x.wgID[axis], physOf(s32, wgIDSGPR+axis))
		x.wgSize[axis] = x.vr.fresh(s32)
		x.vr.pin(x.wgSize[axis], physOf(s32, wgSizeSGPR+axis))
		x.wgCount[axis] = x.vr.fresh(s32)
		x.vr.pin(x.wgCount[axis], physOf(s32, wgCountSGPR+axis))
	}
	x.implicit = true
}

// fillImplicit is the kernel's side: the implicit registers from what
// the dispatcher handed it, before its first call.
func (x *fnState) fillImplicit(c *cursor) {
	tid := x.vr.fresh(v32)
	x.vr.pin(tid, physOf(v32, tidVGPR))
	if x.l.packedIDs() {
		emitCopy(c, tid, x.tid[0], v32)
	} else {
		// x | y << 10 | z << 20.
		t := x.vr.temp(v32)
		x.emit(c, "v_lshlrev_b32", rs(t), rs(x.tid[1]), def(0), imm(10), use(0))
		x.emit(c, "v_or_b32", rs(t), rs(x.tid[0], t), def(0), use(0), use(1))
		x.emit(c, "v_lshlrev_b32", rs(tid), rs(x.tid[2]), def(0), imm(20), use(0))
		x.emit(c, "v_or_b32", rs(tid), rs(t, tid), def(0), use(0), use(1))
	}
	for axis := 0; axis < 3; axis++ {
		id := x.vr.fresh(s32)
		x.vr.pin(id, physOf(s32, wgIDSGPR+axis))
		c.Emit(mir.Instr{Op: amdOp{mn: "s_mov_b32", ops: []opnd{def(0), use(0)}}, Defs: rs(id), Uses: rs(x.wgID[axis])})
		size := x.vr.fresh(s32)
		x.vr.pin(size, physOf(s32, wgSizeSGPR+axis))
		off := x.k.groupSize[axis]
		c.Emit(mir.Instr{Op: amdOp{mn: "s_load_dword", wait: true, ops: []opnd{def(0), {kind: oSMEM, i: 0, imm: off &^ 3}}}, Defs: rs(size), Uses: rs(x.kernarg)})
		if off%4 == 2 {
			c.Emit(mir.Instr{Op: amdOp{mn: "s_lshr_b32", ops: []opnd{def(0), use(0), imm(16)}}, Defs: rs(size), Uses: rs(size)})
		} else {
			c.Emit(mir.Instr{Op: amdOp{mn: "s_and_b32", ops: []opnd{def(0), use(0), imm(0xffff)}}, Defs: rs(size), Uses: rs(size)})
		}
		count := x.vr.fresh(s32)
		x.vr.pin(count, physOf(s32, wgCountSGPR+axis))
		c.Emit(mir.Instr{Op: amdOp{mn: "s_load_dword", wait: true, ops: []opnd{def(0), {kind: oSMEM, i: 0, imm: x.k.blockCount[axis]}}}, Defs: rs(count), Uses: rs(x.kernarg)})
	}
}

// ret is a device function's return: results into their pinned VGPRs,
// then the epilogue, which emit writes at the endpgmOp.
func (x *fnState) ret(c *cursor, in *ir.Inst) error {
	slots, err := argSlots(retTypes(x.fn.Signature()))
	if err != nil {
		return err
	}
	var pinned []mir.VReg
	for i, a := range in.Args() {
		v, err := x.vr.use(a)
		if err != nil {
			return err
		}
		s := slots[i]
		out := x.vr.fresh(s.w)
		x.vr.pin(out, physOf(s.w, s.phys))
		if s.i1 {
			x.emit(c, "v_cndmask_b32", rs(out), rs(v), def(0), imm(0), imm(1), use(0))
		} else {
			emitCopy(c, out, v, s.w)
		}
		pinned = append(pinned, out)
	}
	c.Emit(mir.Instr{Op: endpgmOp{}, Uses: pinned})
	return nil
}

// call selects a direct or indirect call.
func (x *fnState) call(c *cursor, in *ir.Inst) error {
	var (
		sig  *ir.Sig
		args []*ir.Def
		sym  string
		addr mir.VReg
	)
	if in.Op().Verb == ir.VCall {
		callee := in.Callee()
		f, ok := callee.(*ir.Func)
		if !ok {
			return fmt.Errorf("@%s is an import; a call leaves the module only through a symbol the loader resolves, which is not set up yet", callee.Name())
		}
		if f.Signature().CallConv() == ir.Kernel {
			return fmt.Errorf("@%s is a kernel; nothing on the device calls one", f.Name())
		}
		sig, args, sym = f.Signature(), in.Args(), f.Name()
	} else {
		t := in.NamedType()
		if t == nil || t.Sig() == nil {
			return fmt.Errorf("callind names no func type")
		}
		p, err := x.vr.use(in.Arg(0))
		if err != nil {
			return err
		}
		sig, args, addr = t.Sig(), in.Args()[1:], p
	}
	if sig.IsVariadic() {
		return fmt.Errorf("a variadic call has no device convention")
	}
	aslots, err := argSlots(paramTypes(sig))
	if err != nil {
		return err
	}
	rslots, err := argSlots(retTypes(sig))
	if err != nil {
		return err
	}
	if len(aslots) != len(args) {
		return fmt.Errorf("%d arguments for a signature of %d", len(args), len(aslots))
	}

	// The arguments into their registers.
	var uses []mir.VReg
	for i, a := range args {
		v, err := x.vr.use(a)
		if err != nil {
			return err
		}
		s := aslots[i]
		p := x.vr.fresh(s.w)
		x.vr.pin(p, physOf(s.w, s.phys))
		if s.i1 {
			x.emit(c, "v_cndmask_b32", rs(p), rs(v), def(0), imm(0), imm(1), use(0))
		} else {
			emitCopy(c, p, v, s.w)
		}
		uses = append(uses, p)
	}
	// The results into pinned vregs the call defines.
	var defs []mir.VReg
	for _, s := range rslots {
		p := x.vr.fresh(s.w)
		x.vr.pin(p, physOf(s.w, s.phys))
		defs = append(defs, p)
	}
	switch {
	case sym != "":
		c.Emit(mir.Instr{Op: callOp{sym: sym, nargs: len(args)}, Uses: uses})
	case x.uni.isUniform(in.Arg(0)):
		c.Emit(mir.Instr{Op: callOp{nargs: len(args)}, Uses: append(uses, addr)})
	default:
		x.waterfall(c, addr, uses)
	}
	c.Emit(mir.Instr{Op: callResultsOp{}, Defs: defs})
	for i, r := range in.Results() {
		d, err := x.vr.define(r)
		if err != nil {
			return err
		}
		if rslots[i].i1 {
			x.emit(c, "v_cmp_ne_u32", rs(d), rs(defs[i]), def(0), imm(0), use(0))
		} else {
			emitCopy(c, d, defs[i], rslots[i].w)
		}
	}
	return nil
}

// waterfall calls through a pointer that differs across the wave: each
// time round, the first active lane's pointer is the target, the lanes
// that hold it make the call under an execution mask narrowed to them,
// and drop out; the loop ends when every lane has called. The arguments
// sit in their registers throughout and each lane's results land in
// theirs, since a callee writes only the lanes it runs for. The
// function is structurized for this, so the branch below is a
// predicate and the mask nests inside the flow's own.
func (x *fnState) waterfall(c *cursor, addr mir.VReg, uses []mir.VReg) {
	head, done := c.open("waterfall"), c.open("called")
	c.Emit(mir.Instr{Op: branchOp{target: head.Label}})
	c.mf.Succ(c.blk, head.Label)
	c.blk = head
	pick, same, save, rest := x.vr.temp(s64), x.vr.temp(s64), x.vr.temp(s64), x.vr.temp(s64)
	x.emit(c, "v_readfirstlane_b32", rs(pick), rs(addr), defLo(0), useLo(0))
	x.emit(c, "v_readfirstlane_b32", rs(pick), rs(addr, pick), defHi(0), useHi(0))
	x.emit(c, "v_cmp_eq_u64", rs(same), rs(addr, pick), def(0), use(0), use(1))
	c.Emit(mir.Instr{Op: execSaveOp{}, Defs: rs(save)})
	c.Emit(mir.Instr{Op: execAndOp{}, Uses: rs(same)})
	c.Emit(mir.Instr{Op: callOp{nargs: len(uses), scalar: true}, Uses: append(append([]mir.VReg(nil), uses...), pick)})
	c.Emit(mir.Instr{Op: execRestoreOp{}, Uses: rs(save)})
	x.emit(c, "s_not_b64", rs(rest), rs(same), def(0), use(0))
	c.Emit(mir.Instr{Op: cbranchOp{then: head.Label, els: done.Label}, Uses: rs(rest)})
	c.mf.Succ(c.blk, head.Label)
	c.mf.Succ(c.blk, done.Label)
	c.blk = done
}

// —— the call graph ——

// callees is the device functions f calls directly.
func callees(f *ir.Func) []*ir.Func {
	var out []*ir.Func
	seen := map[*ir.Func]bool{}
	f.WalkInsts(func(in *ir.Inst) bool {
		if in.Op().Verb == ir.VCall {
			if g, ok := in.Callee().(*ir.Func); ok && !seen[g] {
				seen[g] = true
				out = append(out, g)
			}
		}
		return true
	})
	return out
}

// hasCallInd reports whether f calls through a pointer.
func hasCallInd(f *ir.Func) bool {
	found := false
	f.WalkInsts(func(in *ir.Inst) bool {
		if in.Op().Verb == ir.VCallInd {
			found = true
		}
		return !found
	})
	return found
}

// orderFuncs is the module's functions with every callee before its
// callers, kernels last; recursion is an error.
func orderFuncs(funcs []*ir.Func) ([]*ir.Func, error) {
	const (
		unseen = iota
		visiting
		done
	)
	state := map[*ir.Func]int{}
	var out []*ir.Func
	var visit func(f *ir.Func, path []string) error
	visit = func(f *ir.Func, path []string) error {
		switch state[f] {
		case done:
			return nil
		case visiting:
			return fmt.Errorf("lower: @%s is recursive through %v, and a device stack has no bound for it", f.Name(), path)
		}
		state[f] = visiting
		for _, g := range callees(f) {
			if err := visit(g, append(path, f.Name())); err != nil {
				return err
			}
		}
		state[f] = done
		out = append(out, f)
		return nil
	}
	for _, f := range funcs {
		if f.Signature().CallConv() != ir.Kernel {
			if err := visit(f, nil); err != nil {
				return nil, err
			}
		}
	}
	for _, f := range funcs {
		if f.Signature().CallConv() == ir.Kernel {
			if err := visit(f, nil); err != nil {
				return nil, err
			}
		}
	}
	return out, nil
}

// calleeNeed is the deepest frame chain beneath f, in bytes: each
// callee's own frame plus its own need, the largest of them; an indirect
// call counts as the deepest device function there is.
func (l *lowerer) calleeNeed(f *ir.Func) uint32 {
	var need uint32
	for _, g := range callees(f) {
		if n := l.frameNeed[g.Name()]; n > need {
			need = n
		}
	}
	if hasCallInd(f) {
		for _, n := range l.frameNeed {
			if n > need {
				need = n
			}
		}
	}
	return need
}
