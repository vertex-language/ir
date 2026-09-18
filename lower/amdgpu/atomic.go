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
			g, ok := in.Symbol().(*ir.Global)
			return ok && g.Domain() == ir.Shared
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
func (x *fnState) getaddrShared(c *cursor, d mir.VReg, g *ir.Global) error {
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
	switch op.Verb {
	case ir.VAtomicULoad8, ir.VAtomicULoad16, ir.VAtomicStore8, ir.VAtomicStore16,
		ir.VAtomicRmwAdd8, ir.VAtomicRmwSub8, ir.VAtomicRmwAnd8, ir.VAtomicRmwOr8, ir.VAtomicRmwXor8, ir.VAtomicRmwXchg8,
		ir.VAtomicRmwAdd16, ir.VAtomicRmwSub16, ir.VAtomicRmwAnd16, ir.VAtomicRmwOr16, ir.VAtomicRmwXor16, ir.VAtomicRmwXchg16,
		ir.VAtomicCas8, ir.VAtomicCas16:
		return fmt.Errorf("narrow atomics are a compare-and-swap loop on the containing word, which is not lowered yet")
	}
	wide := t == ir.TypeI64 || t == ir.TypePtr || t == ir.TypeF64
	a, err := x.args(in)
	if err != nil {
		return err
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
				return fmt.Errorf("an f32 atomic add through a flat pointer is flat_atomic_add_f32, which needs gfx940; before it LLVM spins a compare-and-swap, which is not lowered yet")
			}
			mn = "flat_atomic_add_f32"
		case ir.TypeF64:
			if x.l.model() == modelGFX9 {
				return fmt.Errorf("an f64 atomic add needs gfx90a")
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
