package amdgpu

// §E bulk memory as loops, and br_table as an if-chain: the rows that
// need control flow of their own. isel opens MIR blocks for them under
// the VIR block it is in, moves the cursor into the block where the VIR
// block continues, and leaves the loops to the same lowering every other
// branch gets — scalar when the whole function is uniform, predicated
// when it is structurized. A size that differs across lanes makes the
// function divergent, which needsStructure knows to look for.
//
// A constant count of at most sixty-four bytes is unrolled instead: a
// dword a time, then bytes.

import (
	"fmt"

	"github.com/vertex-language/ir"
	"github.com/vertex-language/ir/lower/globals"
	"github.com/vertex-language/ir/lower/mir"
)

const unrollBytes = 64

func (x *fnState) bulk(c *cursor, in *ir.Inst) error {
	a, err := x.args(in)
	if err != nil {
		return err
	}
	switch in.Op().Verb {
	case ir.VMemCpy:
		if n, ok := constOf(in.Arg(2)); ok && n <= unrollBytes {
			x.unrolledCopy(c, a[0], a[1], n)
			return nil
		}
		x.copyLoop(c, a[0], a[1], a[2], false)
	case ir.VMemMove:
		if n, ok := constOf(in.Arg(2)); ok && n <= unrollBytes {
			// Loaded whole before anything is stored: overlap is no
			// concern.
			x.unrolledMove(c, a[0], a[1], n)
			return nil
		}
		// dst above src copies backwards, else forwards; both loops
		// are laid out and the compare picks.
		dst, src, n := a[0], a[1], a[2]
		above := x.vr.temp(s64)
		x.emit(c, "v_cmp_gt_u64", rs(above), rs(dst, src), def(0), use(0), use(1))
		back, fwd, done := c.open("back"), c.open("fwd"), c.open("moved")
		c.Emit(mir.Instr{Op: cbranchOp{then: back.Label, els: fwd.Label}, Uses: rs(above)})
		c.mf.Succ(c.blk, back.Label)
		c.mf.Succ(c.blk, fwd.Label)
		c.blk = back
		x.copyLoop(c, dst, src, n, true)
		c.Emit(mir.Instr{Op: branchOp{target: done.Label}})
		c.mf.Succ(c.blk, done.Label)
		c.blk = fwd
		x.copyLoop(c, dst, src, n, false)
		c.Emit(mir.Instr{Op: branchOp{target: done.Label}})
		c.mf.Succ(c.blk, done.Label)
		c.blk = done
	case ir.VMemSet:
		if n, ok := constOf(in.Arg(2)); ok && n <= unrollBytes {
			x.unrolledSet(c, a[0], a[1], n)
			return nil
		}
		x.setLoop(c, a[0], a[1], a[2])
	case ir.VMemCmp:
		d, err := x.result(in)
		if err != nil {
			return err
		}
		x.cmpLoop(c, d, a[0], a[1], a[2])
	default:
		return fmt.Errorf("not lowered")
	}
	return nil
}

// constOf is a §A7 constant's value, if p is one.
func constOf(p *ir.Def) (uint64, bool) {
	in := p.Inst()
	if in == nil || in.Op().Verb != ir.VConst {
		return 0, false
	}
	lit, ok := in.Lit()
	if !ok || lit.Kind() != ir.ConstInt {
		return 0, false
	}
	v, err := globals.ConstInt(layout{}, lit)
	if err != nil || v < 0 {
		return 0, false
	}
	return uint64(v), true
}

// —— unrolled ——

func (x *fnState) unrolledCopy(c *cursor, dst, src mir.VReg, n uint64) {
	var at uint64
	for ; at+4 <= n; at += 4 {
		t := x.vr.temp(v32)
		x.emitMem(c, "flat_load_dword", rs(t), rs(src), def(0), opnd{kind: oFlat, i: 0, imm: int64(at)})
		x.emitMem(c, "flat_store_dword", nil, rs(dst, t), opnd{kind: oFlat, i: 0, imm: int64(at)}, use(1))
	}
	for ; at < n; at++ {
		t := x.vr.temp(v32)
		x.emitMem(c, "flat_load_ubyte", rs(t), rs(src), def(0), opnd{kind: oFlat, i: 0, imm: int64(at)})
		x.emitMem(c, "flat_store_byte", nil, rs(dst, t), opnd{kind: oFlat, i: 0, imm: int64(at)}, use(1))
	}
}

func (x *fnState) unrolledMove(c *cursor, dst, src mir.VReg, n uint64) {
	type piece struct {
		at   uint64
		dw   bool
		temp mir.VReg
	}
	var pieces []piece
	var at uint64
	for ; at+4 <= n; at += 4 {
		t := x.vr.temp(v32)
		x.emitMem(c, "flat_load_dword", rs(t), rs(src), def(0), opnd{kind: oFlat, i: 0, imm: int64(at)})
		pieces = append(pieces, piece{at, true, t})
	}
	for ; at < n; at++ {
		t := x.vr.temp(v32)
		x.emitMem(c, "flat_load_ubyte", rs(t), rs(src), def(0), opnd{kind: oFlat, i: 0, imm: int64(at)})
		pieces = append(pieces, piece{at, false, t})
	}
	for _, p := range pieces {
		mn := "flat_store_byte"
		if p.dw {
			mn = "flat_store_dword"
		}
		x.emitMem(c, mn, nil, rs(dst, p.temp), opnd{kind: oFlat, i: 0, imm: int64(p.at)}, use(1))
	}
}

func (x *fnState) unrolledSet(c *cursor, dst, val mir.VReg, n uint64) {
	var at uint64
	if n >= 4 {
		// The byte replicated across a dword: v_perm with a selector
		// that reads byte 0 four times.
		sel := x.constV32(c, 0)
		word := x.vr.temp(v32)
		x.emit(c, "v_perm_b32", rs(word), rs(val, sel), def(0), imm(0), use(0), use(1))
		for ; at+4 <= n; at += 4 {
			x.emitMem(c, "flat_store_dword", nil, rs(dst, word), opnd{kind: oFlat, i: 0, imm: int64(at)}, use(1))
		}
	}
	for ; at < n; at++ {
		x.emitMem(c, "flat_store_byte", nil, rs(dst, val), opnd{kind: oFlat, i: 0, imm: int64(at)}, use(1))
	}
}

// —— loops ——

// counter is a 64-bit loop index in a vreg written each iteration.
func (x *fnState) counter(c *cursor, init mir.VReg) mir.VReg {
	i := x.vr.temp(v64)
	emitCopy(c, i, init, v64)
	return i
}

// at is p + i, a fresh pair.
func (x *fnState) at(c *cursor, p, i mir.VReg) mir.VReg {
	t := x.vr.temp(v64)
	x.emit(c, "v_add_co_u32", rs(t), rs(p, i), defLo(0), vcc(), useLo(0), useLo(1))
	x.emit(c, "v_addc_co_u32", rs(t), rs(p, i, t), defHi(0), vcc(), useHi(0), useHi(1), vcc())
	return t
}

// step adds a signed constant to a 64-bit counter in place.
func (x *fnState) step(c *cursor, i mir.VReg, by int64) {
	if by >= 0 {
		x.emit(c, "v_add_co_u32", rs(i), rs(i), defLo(0), vcc(), imm(by), useLo(0))
		x.emit(c, "v_addc_co_u32", rs(i), rs(i), defHi(0), vcc(), imm(0), useHi(0), vcc())
		return
	}
	// A negative step is its two's complement added: the low dword as an
	// inline constant, all ones above it.
	x.emit(c, "v_add_co_u32", rs(i), rs(i), defLo(0), vcc(), imm(by), useLo(0))
	x.emit(c, "v_addc_co_u32", rs(i), rs(i), defHi(0), vcc(), imm(-1), useHi(0), vcc())
}

// loop opens a counted loop: head tests cond(i), body runs, and the
// cursor is left in the block after the loop. body is emitted by the
// caller between head and the back edge, through the returned cursor.
type loopBlocks struct {
	head, body, done *mir.Block
}

func (x *fnState) openLoop(c *cursor) loopBlocks {
	l := loopBlocks{head: c.open("head"), body: c.open("body"), done: c.open("done")}
	c.Emit(mir.Instr{Op: branchOp{target: l.head.Label}})
	c.mf.Succ(c.blk, l.head.Label)
	c.blk = l.head
	return l
}

// test ends the head: continue into the body on cond, else leave.
func (x *fnState) test(c *cursor, l loopBlocks, cond mir.VReg) {
	c.Emit(mir.Instr{Op: cbranchOp{then: l.body.Label, els: l.done.Label}, Uses: rs(cond)})
	c.mf.Succ(c.blk, l.body.Label)
	c.mf.Succ(c.blk, l.done.Label)
	c.blk = l.body
}

// closeLoop ends the body with the back edge and moves on.
func (x *fnState) closeLoop(c *cursor, l loopBlocks) {
	c.Emit(mir.Instr{Op: branchOp{target: l.head.Label}})
	c.mf.Succ(c.blk, l.head.Label)
	c.blk = l.done
}

// copyLoop copies n bytes, forward from zero or backward from n.
func (x *fnState) copyLoop(c *cursor, dst, src, n mir.VReg, backward bool) {
	zero := x.constV64(c, 0)
	i := x.counter(c, zero)
	if backward {
		i = x.counter(c, n)
	}
	l := x.openLoop(c)
	cond := x.vr.temp(s64)
	if backward {
		x.emit(c, "v_cmp_ne_u64", rs(cond), rs(i, zero), def(0), use(0), use(1))
	} else {
		x.emit(c, "v_cmp_lt_u64", rs(cond), rs(i, n), def(0), use(0), use(1))
	}
	x.test(c, l, cond)
	if backward {
		x.step(c, i, -1)
	}
	b := x.vr.temp(v32)
	x.emitMem(c, "flat_load_ubyte", rs(b), rs(x.at(c, src, i)), def(0), flat(0))
	x.emitMem(c, "flat_store_byte", nil, rs(x.at(c, dst, i), b), flat(0), use(1))
	if !backward {
		x.step(c, i, 1)
	}
	x.closeLoop(c, l)
}

func (x *fnState) setLoop(c *cursor, dst, val, n mir.VReg) {
	zero := x.constV64(c, 0)
	i := x.counter(c, zero)
	l := x.openLoop(c)
	cond := x.vr.temp(s64)
	x.emit(c, "v_cmp_lt_u64", rs(cond), rs(i, n), def(0), use(0), use(1))
	x.test(c, l, cond)
	x.emitMem(c, "flat_store_byte", nil, rs(x.at(c, dst, i), val), flat(0), use(1))
	x.step(c, i, 1)
	x.closeLoop(c, l)
}

// cmpLoop is memcmp: the first differing byte's difference, else zero.
func (x *fnState) cmpLoop(c *cursor, d, p, q, n mir.VReg) {
	zero := x.constV64(c, 0)
	i := x.counter(c, zero)
	r := x.vr.temp(v32)
	x.emit(c, "v_mov_b32", rs(r), nil, def(0), imm(0))
	l := x.openLoop(c)
	cond := x.vr.temp(s64)
	x.emit(c, "v_cmp_lt_u64", rs(cond), rs(i, n), def(0), use(0), use(1))
	x.test(c, l, cond)
	a, b := x.vr.temp(v32), x.vr.temp(v32)
	x.emitMem(c, "flat_load_ubyte", rs(a), rs(x.at(c, p, i)), def(0), flat(0))
	x.emitMem(c, "flat_load_ubyte", rs(b), rs(x.at(c, q, i)), def(0), flat(0))
	same := x.vr.temp(s64)
	x.emit(c, "v_cmp_eq_u32", rs(same), rs(a, b), def(0), use(0), use(1))
	next, differ := c.open("next"), c.open("differ")
	c.Emit(mir.Instr{Op: cbranchOp{then: next.Label, els: differ.Label}, Uses: rs(same)})
	c.mf.Succ(c.blk, next.Label)
	c.mf.Succ(c.blk, differ.Label)
	c.blk = differ
	x.emit(c, "v_sub_u32", rs(r), rs(a, b), def(0), use(0), use(1))
	c.Emit(mir.Instr{Op: branchOp{target: l.done.Label}})
	c.mf.Succ(c.blk, l.done.Label)
	c.blk = next
	x.step(c, i, 1)
	x.closeLoop(c, l)
	emitCopy(c, d, r, v32)
}

// —— br_table ——

// brTable is an if-chain: one compare and branch per case, the default
// at the end.
func (x *fnState) brTable(c *cursor, in *ir.Inst) error {
	ts := in.Targets()
	cases, dflt := ts[:len(ts)-1], ts[len(ts)-1]
	sel, err := x.vr.use(in.Arg(0))
	if err != nil {
		return err
	}
	for i, t := range cases {
		dest, err := x.edgeTarget(c, t, "case")
		if err != nil {
			return err
		}
		m := x.vr.temp(s64)
		if i <= 64 {
			x.emit(c, "v_cmp_eq_u32", rs(m), rs(sel), def(0), imm(int64(i)), use(0))
		} else {
			k := x.constV32(c, uint32(i))
			x.emit(c, "v_cmp_eq_u32", rs(m), rs(k, sel), def(0), use(0), use(1))
		}
		next := c.open("nocase")
		c.Emit(mir.Instr{Op: cbranchOp{then: dest, els: next.Label}, Uses: rs(m)})
		c.mf.Succ(c.blk, dest)
		c.mf.Succ(c.blk, next.Label)
		c.blk = next
	}
	dest, err := x.edgeTarget(c, dflt, "default")
	if err != nil {
		return err
	}
	c.Emit(mir.Instr{Op: branchOp{target: dest}})
	c.mf.Succ(c.blk, dest)
	return nil
}
