package amdgpu

// §W's wave verbs. A shuffle is ds_bpermute_b32, which reads another
// lane's value by byte address; up and down compute the source lane
// from the lane id and keep the lane's own value past the edge. The
// mask operand names the participating lanes, which on this hardware
// is every active lane: it narrows a ballot on a 32-wide wave, where
// it can, and is otherwise the wave's exec.

import (
	"fmt"

	"github.com/vertex-language/ir"
	"github.com/vertex-language/ir/lower/mir"
)

func (x *fnState) wave(c *cursor, in *ir.Inst) error {
	d, err := x.result(in)
	if err != nil {
		return err
	}
	a, err := x.args(in)
	if err != nil {
		return err
	}
	switch in.Op().Verb {
	case ir.VWaveReadFirstLane:
		s := x.vr.temp(s32)
		x.emit(c, "v_readfirstlane_b32", rs(s), rs(a[0]), def(0), use(0))
		x.emit(c, "v_mov_b32", rs(d), rs(s), def(0), use(0))
		return nil

	case ir.VWaveBallot:
		// The bits of c in the active lanes, into a pair.
		m := x.vr.temp(s64)
		x.emit(c, "s_and_b64", rs(m), rs(a[0]), def(0), use(0), opnd{kind: oExec})
		if !x.l.wave64() {
			x.emit(c, "s_and_b32", rs(m), rs(m, a[1]), defLo(0), useLo(0), use(1))
		}
		x.movPair(c, d, m)
		return nil

	case ir.VWaveAny, ir.VWaveAll:
		// A uniform i1: all ones or all zeros, from SCC.
		t := x.vr.temp(s64)
		if in.Op().Verb == ir.VWaveAny {
			x.emit(c, "s_and_b64", rs(t), rs(a[0]), def(0), use(0), opnd{kind: oExec})
			c.Emit(mir.Instr{Op: amdOp{mn: "s_cmp_lg_u64", ops: []opnd{use(0), imm(0)}}, Uses: rs(t)})
		} else {
			x.emit(c, "s_andn2_b64", rs(t), rs(a[0]), def(0), opnd{kind: oExec}, use(0))
			c.Emit(mir.Instr{Op: amdOp{mn: "s_cmp_eq_u64", ops: []opnd{use(0), imm(0)}}, Uses: rs(t)})
		}
		x.emit(c, "s_cselect_b64", rs(d), nil, def(0), imm(-1), imm(0))
		return nil

	case ir.VWaveShflIdx:
		x.bpermute(c, d, a[0], a[1])
		return nil
	case ir.VWaveShflXor:
		lane := x.laneID(c)
		src := x.vr.temp(v32)
		x.emit(c, "v_xor_b32", rs(src), rs(lane, a[1]), def(0), use(0), use(1))
		x.bpermute(c, d, a[0], src)
		return nil
	case ir.VWaveShflUp, ir.VWaveShflDown:
		lane := x.laneID(c)
		src, inRange := x.vr.temp(v32), x.vr.temp(s64)
		if in.Op().Verb == ir.VWaveShflUp {
			x.emit(c, "v_sub_u32", rs(src), rs(lane, a[1]), def(0), use(0), use(1))
			x.emit(c, "v_cmp_le_i32", rs(inRange), rs(src), def(0), imm(0), use(0))
		} else {
			width := int64(64)
			if !x.l.wave64() {
				width = 32
			}
			x.emit(c, "v_add_u32", rs(src), rs(lane, a[1]), def(0), use(0), use(1))
			x.emit(c, "v_cmp_gt_u32", rs(inRange), rs(src), def(0), imm(width), use(0))
		}
		got := x.vr.temp(v32)
		x.bpermute(c, got, a[0], src)
		x.emit(c, "v_cndmask_b32", rs(d), rs(a[0], got, inRange), def(0), use(0), use(1), use(2))
		return nil
	}
	return fmt.Errorf("not lowered")
}

// bpermute reads val from lane src: the address is the lane times four.
func (x *fnState) bpermute(c *cursor, d, val, src mir.VReg) {
	addr := x.vr.temp(v32)
	x.emit(c, "v_lshlrev_b32", rs(addr), rs(src), def(0), imm(2), use(0))
	x.emitMem(c, "ds_bpermute_b32", rs(d), rs(addr, val), def(0), opnd{kind: oDS32, i: 0}, use(1))
}

// laneID is the lane's index within the wave.
func (x *fnState) laneID(c *cursor) mir.VReg {
	t, d := x.vr.temp(v32), x.vr.temp(v32)
	x.emit(c, "v_mbcnt_lo_u32_b32", rs(t), nil, def(0), imm(-1), imm(0))
	x.emit(c, "v_mbcnt_hi_u32_b32", rs(d), rs(t), def(0), imm(-1), use(0))
	return d
}
