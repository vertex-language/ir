package amdgpu

// §H and workgroup storage: where a pointer points decides the
// instruction, and the scope decides the cache bits around it.
//
// # Where a pointer points
//
// Every VIR pointer is a flat address, and the hardware routes a flat
// access to LDS when the address falls in the shared aperture — so a
// shared global's address is the aperture's high half over its LDS
// offset, and a flat_load through it is correct. It is also slow, and
// LDS atomics through flat are slower still, so an access whose pointer
// provably came from a shared global — a ptr.getaddr of one, through
// ptr.add and ptr.sub — is a ds_* instruction on the address's low
// dword, which is the LDS offset by construction. Everything else is
// flat.
//
// # Scopes
//
// The memory model is LLVM's for each generation, read off llc:
//
//   - GFX9 (gfx900, gfx908): one L2 per agent, so a release is the
//     s_waitcnt every memory instruction already has and an acquire at
//     device or system scope invalidates L1 (buffer_wbinvl1_vol). An
//     atomic load at those scopes bypasses L1 (glc).
//   - gfx90a: the same, but a system-scope release writes L2 back
//     (buffer_wbl2) and a system-scope acquire invalidates it
//     (buffer_invl2) before L1.
//   - gfx940: scope bits on the access itself — sc0 for the workgroup,
//     sc1 for the device, both for the system — and buffer_wbl2 and
//     buffer_inv with the scope's bits for release and acquire beyond
//     the workgroup.
//
// An rmw always uses the returning form (sc0 or glc), since a VIR atomic
// has a result; on gfx940 the system scope adds sc1.

import (
	"fmt"

	"github.com/vertex-language/amdgpu/feature"

	"github.com/vertex-language/ir"
	"github.com/vertex-language/ir/lower/mir"
)

// The scratch VGPRs a compare-and-swap packs its operands into: the
// instruction reads {new, expected} as one register tuple, which the
// allocator's single-register classes cannot hand out. v[60:63] is
// carved out of the top of the singles for it.
const casScratch = 60

// sharedPtr reports whether p provably addresses workgroup storage.
func sharedPtr(p *ir.Def) bool {
	for depth := 0; depth < 16; depth++ {
		in := p.Inst()
		if in == nil {
			return false
		}
		switch in.Op().Verb {
		case ir.VGetAddr:
			switch g := in.Symbol().(type) {
			case *ir.Global:
				return g.Domain() == ir.Shared
			case *ir.GlobalImport:
				return g.Domain() == ir.Shared
			}
			return false
		case ir.VAdd, ir.VSub:
			if in.Op().Type != ir.TypePtr {
				return false
			}
			p = in.Arg(0)
		default:
			return false
		}
	}
	return false
}

// getaddrShared is a shared global's flat address: the aperture's high
// half over the LDS offset.
func (x *fnState) getaddrShared(c *cursor, d mir.VReg, g ir.Symbol) error {
	off, ok := x.l.lds[g.Name()]
	if !ok {
		return fmt.Errorf("@%s has no LDS offset", g.Name())
	}
	ap := x.vr.temp(s64)
	c.Emit(mir.Instr{Op: amdOp{mn: "s_mov_b64", ops: []opnd{def(0), {kind: oSharedBase}}}, Defs: rs(ap)})
	x.emit(c, "v_mov_b32", rs(d), nil, defLo(0), imm(int64(off)))
	x.emit(c, "v_mov_b32", rs(d), rs(ap, d), defHi(0), useHi(0))
	return nil
}

// —— the model ——

type memModel uint8

const (
	modelGFX9 memModel = iota
	modelGFX90A
	modelGFX940
)

func (l *lowerer) model() memModel {
	switch l.opts.ASIC {
	case feature.GFX940, feature.GFX942:
		return modelGFX940
	case feature.GFX90A:
		return modelGFX90A
	}
	return modelGFX9
}

func scopeOf(s ir.Scope) ir.Scope {
	if s == ir.NoScope {
		return ir.System
	}
	return s
}

func isRelease(o ir.Ordering) bool { return o == ir.Release || o == ir.AcqRel || o == ir.SeqCst }
func isAcquire(o ir.Ordering) bool { return o == ir.Acquire || o == ir.AcqRel || o == ir.SeqCst }

// release is what precedes a releasing access: the write-back the scope
// needs, after every earlier memory instruction has completed, which
// the s_waitcnt after each of them guarantees.
func (x *fnState) release(c *cursor, s ir.Scope) {
	s = scopeOf(s)
	switch x.l.model() {
	case modelGFX940:
		if s == ir.Device || s == ir.System {
			x.cacheOp(c, "buffer_wbl2", s == ir.System, true)
		}
	case modelGFX90A:
		if s == ir.System {
			x.cacheOp(c, "buffer_wbl2", false, false)
		}
	}
}

// acquire is what follows an acquiring access, once it has completed:
// the invalidation the scope needs.
func (x *fnState) acquire(c *cursor, s ir.Scope) {
	s = scopeOf(s)
	if s == ir.Workgroup {
		return
	}
	switch x.l.model() {
	case modelGFX940:
		x.cacheOp(c, "buffer_inv", s == ir.System, true)
	case modelGFX90A:
		if s == ir.System {
			x.cacheOp(c, "buffer_invl2", false, false)
		}
		x.cacheOp(c, "buffer_wbinvl1_vol", false, false)
	default:
		x.cacheOp(c, "buffer_wbinvl1_vol", false, false)
	}
}

// cacheOp is one cache-control instruction, waited for like any other
// memory instruction. The gfx940 ones take scope bits.
func (x *fnState) cacheOp(c *cursor, mn string, sc0, sc1 bool) {
	var ops []opnd
	if mn == "buffer_wbl2" && x.l.model() == modelGFX940 || mn == "buffer_inv" {
		ops = []opnd{{kind: oCache, glc: sc0, sc1: sc1}}
	} else if mn == "buffer_wbl2" {
		ops = []opnd{{kind: oCache}}
	}
	c.Emit(mir.Instr{Op: amdOp{mn: mn, ops: ops, wait: true}})
}

// accessBits is the cache bits an atomic load or store carries at a scope.
func (x *fnState) accessBits(s ir.Scope, load bool) (glc, sc1 bool) {
	s = scopeOf(s)
	if x.l.model() == modelGFX940 {
		return s == ir.Workgroup || s == ir.System, s == ir.Device || s == ir.System
	}
	return load && s != ir.Workgroup, false
}

// —— the verbs ——

func (x *fnState) fence(c *cursor, in *ir.Inst) error {
	if in.SingleThread() {
		return nil
	}
	o := in.Orderings()[0]
	s := in.Scope()
	// Every memory instruction is followed by s_waitcnt vmcnt(0)
	// lgkmcnt(0), so the wait a fence needs has already happened; what
	// is left is the cache control beyond the workgroup.
	if isRelease(o) {
		x.release(c, s)
	}
	if isAcquire(o) {
		x.acquire(c, s)
	}
	return nil
}

func (x *fnState) atomic(c *cursor, in *ir.Inst) error {
	op := in.Op()
	t := op.Type
	ords := in.Orderings()
	o := ords[0]
	s := in.Scope()
	wide := t == ir.TypeI64 || t == ir.TypePtr || t == ir.TypeF64
	a, err := x.args(in)
	if err != nil {
		return err
	}
	if bits := narrowBits(op.Verb); bits != 0 {
		return x.narrow(c, in, a, bits, o, s)
	}

	switch op.Verb {
	case ir.VAtomicLoad:
		d, err := x.result(in)
		if err != nil {
			return err
		}
		p := a[0]
		if sharedPtr(in.Arg(0)) {
			mn := "ds_read_b32"
			if wide {
				mn = "ds_read_b64"
			}
			x.emitMem(c, mn, rs(d), rs(p), def(0), ds(0))
		} else {
			mn := "flat_load_dword"
			if wide {
				mn = "flat_load_dwordx2"
			}
			glc, sc1 := x.accessBits(s, true)
			x.emitMem(c, mn, rs(d), rs(p), def(0), opnd{kind: oFlat, i: 0, glc: glc, sc1: sc1})
		}
		if isAcquire(o) {
			x.acquire(c, s)
		}
		return nil

	case ir.VAtomicStore:
		v, p := a[0], a[1]
		if isRelease(o) {
			x.release(c, s)
		}
		if sharedPtr(in.Arg(1)) {
			mn := "ds_write_b32"
			if wide {
				mn = "ds_write_b64"
			}
			x.emitMem(c, mn, nil, rs(p, v), ds(0), use(1))
		} else {
			mn := "flat_store_dword"
			if wide {
				mn = "flat_store_dwordx2"
			}
			glc, sc1 := x.accessBits(s, false)
			x.emitMem(c, mn, nil, rs(p, v), opnd{kind: oFlat, i: 0, glc: glc, sc1: sc1}, use(1))
		}
		return nil
	}

	d, err := x.result(in)
	if err != nil {
		return err
	}
	if isRelease(o) {
		x.release(c, s)
	}
	if op.Verb == ir.VAtomicCas {
		if err := x.cas(c, in, d, a, wide); err != nil {
			return err
		}
	} else {
		if err := x.rmw(c, in, d, a, wide); err != nil {
			return err
		}
	}
	// A compare-and-swap's acquire is the stronger of its two orderings.
	acq := o
	if len(ords) > 1 && ords[1] > acq {
		acq = ords[1]
	}
	if isAcquire(acq) {
		x.acquire(c, s)
	}
	return nil
}

// rmw is one read-modify-write: a[0] the value, a[1] the pointer.
func (x *fnState) rmw(c *cursor, in *ir.Inst, d mir.VReg, a []mir.VReg, wide bool) error {
	t := in.Op().Type
	v, p := a[0], a[1]
	shared := sharedPtr(in.Arg(1))
	var mn string
	if shared {
		suffix := "u32"
		bits := "b32"
		signed := "i32"
		if wide {
			suffix, bits, signed = "u64", "b64", "i64"
		}
		switch in.Op().Verb {
		case ir.VAtomicRmwAdd:
			switch t {
			case ir.TypeF32:
				mn = "ds_add_rtn_f32"
			case ir.TypeF64:
				if x.l.model() == modelGFX9 {
					return fmt.Errorf("an f64 atomic add on LDS needs gfx90a")
				}
				mn = "ds_add_rtn_f64"
			default:
				mn = "ds_add_rtn_" + suffix
			}
		case ir.VAtomicRmwSub:
			mn = "ds_sub_rtn_" + suffix
		case ir.VAtomicRmwAnd:
			mn = "ds_and_rtn_" + bits
		case ir.VAtomicRmwOr:
			mn = "ds_or_rtn_" + bits
		case ir.VAtomicRmwXor:
			mn = "ds_xor_rtn_" + bits
		case ir.VAtomicRmwXchg:
			mn = "ds_wrxchg_rtn_" + bits
		case ir.VAtomicRmwSMin:
			mn = "ds_min_rtn_" + signed
		case ir.VAtomicRmwSMax:
			mn = "ds_max_rtn_" + signed
		case ir.VAtomicRmwUMin:
			mn = "ds_min_rtn_" + suffix
		case ir.VAtomicRmwUMax:
			mn = "ds_max_rtn_" + suffix
		}
		x.emitMem(c, mn, rs(d), rs(p, v), def(0), ds(0), use(1))
		return nil
	}
	x2 := ""
	if wide {
		x2 = "_x2"
	}
	switch in.Op().Verb {
	case ir.VAtomicRmwAdd:
		switch t {
		case ir.TypeF32:
			if x.l.model() != modelGFX940 {
				// No flat_atomic_add_f32 before gfx940: the sum goes in
				// by compare-and-swap, as LLVM spins it.
				x.floatAddLoop(c, d, p, v, false, in.Scope())
				return nil
			}
			mn = "flat_atomic_add_f32"
		case ir.TypeF64:
			if x.l.model() == modelGFX9 {
				x.floatAddLoop(c, d, p, v, true, in.Scope())
				return nil
			}
			mn = "flat_atomic_add_f64"
		default:
			mn = "flat_atomic_add" + x2
		}
	case ir.VAtomicRmwSub:
		mn = "flat_atomic_sub" + x2
	case ir.VAtomicRmwAnd:
		mn = "flat_atomic_and" + x2
	case ir.VAtomicRmwOr:
		mn = "flat_atomic_or" + x2
	case ir.VAtomicRmwXor:
		mn = "flat_atomic_xor" + x2
	case ir.VAtomicRmwXchg:
		mn = "flat_atomic_swap" + x2
	case ir.VAtomicRmwSMin:
		mn = "flat_atomic_smin" + x2
	case ir.VAtomicRmwSMax:
		mn = "flat_atomic_smax" + x2
	case ir.VAtomicRmwUMin:
		mn = "flat_atomic_umin" + x2
	case ir.VAtomicRmwUMax:
		mn = "flat_atomic_umax" + x2
	}
	x.emitMem(c, mn, rs(d), rs(p, v), def(0), x.rmwAddr(0, in.Scope()), use(1))
	return nil
}

// rmwAddr is a flat atomic's address operand with the returning bit and
// the scope's.
func (x *fnState) rmwAddr(i int, s ir.Scope) opnd {
	o := opnd{kind: oFlat, i: i, glc: true}
	if x.l.model() == modelGFX940 && scopeOf(s) == ir.System {
		o.sc1 = true
	}
	return o
}

// cas is compare-and-swap: a[0] expected, a[1] new, a[2] the pointer.
// The flat form reads {new, expected} as one tuple, packed into the
// scratch VGPRs; the DS form takes them as two operands, expected first.
func (x *fnState) cas(c *cursor, in *ir.Inst, d mir.VReg, a []mir.VReg, wide bool) error {
	exp, nw, p := a[0], a[1], a[2]
	if sharedPtr(in.Arg(2)) {
		mn := "ds_cmpst_rtn_b32"
		if wide {
			mn = "ds_cmpst_rtn_b64"
		}
		x.emitMem(c, mn, rs(d), rs(p, exp, nw), def(0), ds(0), use(1), use(2))
		return nil
	}
	if !wide {
		x.emit(c, "v_mov_b32", nil, rs(nw), fixedV(casScratch), use(0))
		x.emit(c, "v_mov_b32", nil, rs(exp), fixedV(casScratch+1), use(0))
		x.emitMem(c, "flat_atomic_cmpswap", rs(d), rs(p), def(0), x.rmwAddr(0, in.Scope()), fixedTuple(casScratch, 2))
		return nil
	}
	x.emit(c, "v_mov_b32", nil, rs(nw), fixedV(casScratch), useLo(0))
	x.emit(c, "v_mov_b32", nil, rs(nw), fixedV(casScratch+1), useHi(0))
	x.emit(c, "v_mov_b32", nil, rs(exp), fixedV(casScratch+2), useLo(0))
	x.emit(c, "v_mov_b32", nil, rs(exp), fixedV(casScratch+3), useHi(0))
	x.emitMem(c, "flat_atomic_cmpswap_x2", rs(d), rs(p), def(0), x.rmwAddr(0, in.Scope()), fixedTuple(casScratch, 4))
	return nil
}

func ds(i int) opnd                { return opnd{kind: oDS, i: i} }
func fixedV(n int) opnd            { return opnd{kind: oFixedV, imm: int64(n)} }
func fixedTuple(n, count int) opnd { return opnd{kind: oFixedV, imm: int64(n), i: count} }

// —— the narrow forms ——

// narrowBits is 8 or 16 for a narrow atomic verb, else 0.
func narrowBits(v ir.Verb) int {
	switch v {
	case ir.VAtomicULoad8, ir.VAtomicStore8, ir.VAtomicCas8,
		ir.VAtomicRmwAdd8, ir.VAtomicRmwSub8, ir.VAtomicRmwAnd8, ir.VAtomicRmwOr8, ir.VAtomicRmwXor8, ir.VAtomicRmwXchg8:
		return 8
	case ir.VAtomicULoad16, ir.VAtomicStore16, ir.VAtomicCas16,
		ir.VAtomicRmwAdd16, ir.VAtomicRmwSub16, ir.VAtomicRmwAnd16, ir.VAtomicRmwOr16, ir.VAtomicRmwXor16, ir.VAtomicRmwXchg16:
		return 16
	}
	return 0
}

// narrow is a sub-word atomic. A load and a store are the byte or
// halfword instruction, which the hardware does atomically; a
// read-modify-write or compare-and-swap is a loop on the containing
// dword: read it, compute the dword with the narrow field replaced, and
// compare-and-swap it in until the dword read back is the one the
// field was computed from. The loop leaves lane by lane, so the
// function is structurized for it.
func (x *fnState) narrow(c *cursor, in *ir.Inst, a []mir.VReg, bits int, o ir.Ordering, s ir.Scope) error {
	verb := in.Op().Verb
	load, store := "flat_load_ubyte", "flat_store_byte"
	if bits == 16 {
		load, store = "flat_load_ushort", "flat_store_short"
	}
	switch verb {
	case ir.VAtomicULoad8, ir.VAtomicULoad16:
		d, err := x.result(in)
		if err != nil {
			return err
		}
		if sharedPtr(in.Arg(0)) {
			load = pick(bits == 8, "ds_read_u8", "ds_read_u16")
			x.emitMem(c, load, rs(d), rs(a[0]), def(0), ds(0))
		} else {
			glc, sc1 := x.accessBits(s, true)
			x.emitMem(c, load, rs(d), rs(a[0]), def(0), opnd{kind: oFlat, glc: glc, sc1: sc1})
		}
		if isAcquire(o) {
			x.acquire(c, s)
		}
		return nil
	case ir.VAtomicStore8, ir.VAtomicStore16:
		if isRelease(o) {
			x.release(c, s)
		}
		if sharedPtr(in.Arg(1)) {
			store = pick(bits == 8, "ds_write_b8", "ds_write_b16")
			x.emitMem(c, store, nil, rs(a[1], a[0]), ds(0), use(1))
		} else {
			glc, sc1 := x.accessBits(s, false)
			x.emitMem(c, store, nil, rs(a[1], a[0]), opnd{kind: oFlat, glc: glc, sc1: sc1}, use(1))
		}
		return nil
	}

	d, err := x.result(in)
	if err != nil {
		return err
	}
	// The operands: the pointer last, a value before it; a
	// compare-and-swap's expected value first of all.
	p := a[len(a)-1]
	v := a[len(a)-2]
	var expect mir.VReg
	isCas := verb == ir.VAtomicCas8 || verb == ir.VAtomicCas16
	if isCas {
		expect = a[0]
	}
	shared := sharedPtr(in.Arg(len(a) - 1))
	t := func() mir.VReg { return x.vr.temp(v32) }

	// The dword's address, and the field's position in it.
	word := x.vr.temp(v64)
	x.emit(c, "v_and_b32", rs(word), rs(p), defLo(0), imm(-4), useLo(0))
	x.emit(c, "v_mov_b32", rs(word), rs(p, word), defHi(0), useHi(0))
	shift := t()
	x.emit(c, "v_and_b32", rs(shift), rs(p), def(0), imm(3), useLo(0))
	x.emit(c, "v_lshlrev_b32", rs(shift), rs(shift), def(0), imm(3), use(0))
	fieldMask := x.constV32(c, uint32(1<<bits-1))
	mask, notMask := t(), t()
	x.emit(c, "v_lshlrev_b32", rs(mask), rs(shift, fieldMask), def(0), use(0), use(1))
	x.emit(c, "v_not_b32", rs(notMask), rs(mask), def(0), use(0))

	if isRelease(o) {
		x.release(c, s)
	}
	// old is the dword last read; the loop keeps it.
	old := t()
	if shared {
		x.emitMem(c, "ds_read_b32", rs(old), rs(word), def(0), ds(0))
	} else {
		x.emitMem(c, "flat_load_dword", rs(old), rs(word), def(0), flat(0))
	}
	head, done := c.open("narrow"), c.open("narrowed")
	c.Emit(mir.Instr{Op: branchOp{target: head.Label}})
	c.mf.Succ(c.blk, head.Label)
	c.blk = head
	// The field as it is, and as it will be.
	field, nf := t(), t()
	x.emit(c, "v_lshrrev_b32", rs(field), rs(shift, old), def(0), use(0), use(1))
	x.emit(c, "v_and_b32", rs(field), rs(field, fieldMask), def(0), use(0), use(1))
	switch verb {
	case ir.VAtomicRmwAdd8, ir.VAtomicRmwAdd16:
		x.emit(c, "v_add_u32", rs(nf), rs(field, v), def(0), use(0), use(1))
	case ir.VAtomicRmwSub8, ir.VAtomicRmwSub16:
		x.emit(c, "v_sub_u32", rs(nf), rs(field, v), def(0), use(0), use(1))
	case ir.VAtomicRmwAnd8, ir.VAtomicRmwAnd16:
		x.emit(c, "v_and_b32", rs(nf), rs(field, v), def(0), use(0), use(1))
	case ir.VAtomicRmwOr8, ir.VAtomicRmwOr16:
		x.emit(c, "v_or_b32", rs(nf), rs(field, v), def(0), use(0), use(1))
	case ir.VAtomicRmwXor8, ir.VAtomicRmwXor16:
		x.emit(c, "v_xor_b32", rs(nf), rs(field, v), def(0), use(0), use(1))
	default: // xchg and cas: the new value is the one given
		emitCopy(c, nf, v, v32)
	}
	x.emit(c, "v_and_b32", rs(nf), rs(nf, fieldMask), def(0), use(0), use(1))
	// The dword with the field replaced, and the swap.
	nw, keep := t(), t()
	x.emit(c, "v_lshlrev_b32", rs(nw), rs(shift, nf), def(0), use(0), use(1))
	x.emit(c, "v_and_b32", rs(keep), rs(old, notMask), def(0), use(0), use(1))
	x.emit(c, "v_or_b32", rs(nw), rs(nw, keep), def(0), use(0), use(1))
	if isCas {
		// Only where the field is the expected value; elsewhere the
		// dword goes back as it was, and the field is the answer.
		same := x.vr.temp(s64)
		x.emit(c, "v_cmp_eq_u32", rs(same), rs(field, expect), def(0), use(0), use(1))
		x.emit(c, "v_cndmask_b32", rs(nw), rs(old, nw, same), def(0), use(0), use(1), use(2))
	}
	got := t()
	if shared {
		x.emitMem(c, "ds_cmpst_rtn_b32", rs(got), rs(word, old, nw), def(0), ds(0), use(1), use(2))
	} else {
		x.emit(c, "v_mov_b32", nil, rs(nw), fixedV(casScratch), use(0))
		x.emit(c, "v_mov_b32", nil, rs(old), fixedV(casScratch+1), use(0))
		x.emitMem(c, "flat_atomic_cmpswap", rs(got), rs(word), def(0), x.rmwAddr(0, s), fixedTuple(casScratch, 2))
	}
	// The dword changed under us: go round with what it is now.
	changed := x.vr.temp(s64)
	x.emit(c, "v_cmp_ne_u32", rs(changed), rs(got, old), def(0), use(0), use(1))
	emitCopy(c, old, got, v32)
	c.Emit(mir.Instr{Op: cbranchOp{then: head.Label, els: done.Label}, Uses: rs(changed)})
	c.mf.Succ(c.blk, head.Label)
	c.mf.Succ(c.blk, done.Label)
	c.blk = done
	if isAcquire(o) {
		x.acquire(c, s)
	}
	// The field as it was: from the dword the swap succeeded against,
	// which old now holds.
	x.emit(c, "v_lshrrev_b32", rs(d), rs(shift, old), def(0), use(0), use(1))
	x.emit(c, "v_and_b32", rs(d), rs(d, fieldMask), def(0), use(0), use(1))
	return nil
}

// floatAddLoop is a float atomic add on a generation with no instruction
// for it: read the value, add, and compare-and-swap the sum in until the
// value read back is the one the sum was computed from. The loop leaves
// lane by lane, so the function is structurized.
func (x *fnState) floatAddLoop(c *cursor, d, p, v mir.VReg, wide bool, s ir.Scope) {
	w := v32
	load, add := "flat_load_dword", "v_add_f32"
	if wide {
		w, load, add = v64, "flat_load_dwordx2", "v_add_f64"
	}
	old := x.vr.temp(w)
	x.emitMem(c, load, rs(old), rs(p), def(0), flat(0))
	head, done := c.open("fadd"), c.open("fadded")
	c.Emit(mir.Instr{Op: branchOp{target: head.Label}})
	c.mf.Succ(c.blk, head.Label)
	c.blk = head
	sum, got := x.vr.temp(w), x.vr.temp(w)
	x.emit(c, add, rs(sum), rs(old, v), def(0), use(0), use(1))
	if !wide {
		x.emit(c, "v_mov_b32", nil, rs(sum), fixedV(casScratch), use(0))
		x.emit(c, "v_mov_b32", nil, rs(old), fixedV(casScratch+1), use(0))
		x.emitMem(c, "flat_atomic_cmpswap", rs(got), rs(p), def(0), x.rmwAddr(0, s), fixedTuple(casScratch, 2))
	} else {
		x.emit(c, "v_mov_b32", nil, rs(sum), fixedV(casScratch), useLo(0))
		x.emit(c, "v_mov_b32", nil, rs(sum), fixedV(casScratch+1), useHi(0))
		x.emit(c, "v_mov_b32", nil, rs(old), fixedV(casScratch+2), useLo(0))
		x.emit(c, "v_mov_b32", nil, rs(old), fixedV(casScratch+3), useHi(0))
		x.emitMem(c, "flat_atomic_cmpswap_x2", rs(got), rs(p), def(0), x.rmwAddr(0, s), fixedTuple(casScratch, 4))
	}
	changed := x.vr.temp(s64)
	if wide {
		x.emit(c, "v_cmp_ne_u64", rs(changed), rs(got, old), def(0), use(0), use(1))
	} else {
		x.emit(c, "v_cmp_ne_u32", rs(changed), rs(got, old), def(0), use(0), use(1))
	}
	emitCopy(c, old, got, w)
	c.Emit(mir.Instr{Op: cbranchOp{then: head.Label, els: done.Label}, Uses: rs(changed)})
	c.mf.Succ(c.blk, head.Label)
	c.mf.Succ(c.blk, done.Label)
	c.blk = done
	emitCopy(c, d, old, w)
}
