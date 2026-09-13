package amdgpu_test

import (
	"strings"
	"testing"

	"github.com/vertex-language/amdgpu/feature"

	"github.com/vertex-language/ir"
	lower "github.com/vertex-language/ir/lower/amdgpu"
	"github.com/vertex-language/ir/verify"
)

// scale is a[tid] *= 2, n times, when n > 0: a branch and a loop whose
// conditions every lane agrees on, since n is a kernel argument.
func scale() *ir.Module {
	m := ir.NewModule("scale", ir.AMDGCN)
	fn := m.Func("scale").Export().CallConv(ir.Kernel).NoUnwind()
	a := fn.ParamPtr("a", ir.NoAlias)
	n := fn.ParamI32("n")

	entry := fn.Entry()
	loop := fn.Block("loop")
	k := loop.ParamI32("k")
	done := fn.Block("done")

	tid := entry.I32.WorkitemID(ir.X)
	off := entry.I64.Shl(entry.I64.ZExtI32(tid), entry.I64.Const(2))
	p := entry.Ptr.Add(a, off)
	entry.BrIf(entry.I32.SLt(entry.I32.Const(0), n), loop.To(entry.I32.Const(0)), done.To())

	v := loop.F32.Load(p)
	loop.F32.Store(loop.F32.Mul(v, loop.F32.Const(2)), p)
	k1 := loop.I32.Add(k, loop.I32.Const(1))
	loop.BrIf(loop.I32.SLt(k1, n), loop.To(k1), done.To())

	done.Return()
	return m
}

func TestUniformBranchAndLoop(t *testing.T) {
	o := lowerObj(t, scale(), lower.Options{ASIC: feature.GFX942})
	// Branch targets are dword offsets from the following instruction:
	// the entry's s_cbranch_scc1 hops over the s_branch to done, the
	// loop's hops over s_endpgm to the back edge's copy of k.
	same(t, disassemble(t, o, feature.GFX942), `
s_load_dwordx2 s[32:33], s[0:1], 0x0
s_waitcnt vmcnt(0) lgkmcnt(0)
v_mov_b32_e32 v64, s32
v_mov_b32_e32 v65, s33
s_load_dword s8, s[0:1], 0x8
s_waitcnt vmcnt(0) lgkmcnt(0)
v_mov_b32_e32 v1, s8
v_and_b32_e32 v2, 0x3ff, v0
v_mov_b32_e32 v66, v2
v_mov_b32_e32 v67, 0
v_mov_b32_e32 v68, 2
v_mov_b32_e32 v69, 0
v_lshlrev_b64 v[70:71], v68, v[66:67]
v_add_co_u32_e32 v66, vcc, v64, v70
v_addc_co_u32_e32 v67, vcc, v65, v71, vcc
v_mov_b32_e32 v2, 0
v_cmp_lt_i32_e64 s[32:33], v2, v1
v_mov_b32_e32 v2, 0
s_and_b64 vcc, s[32:33], exec
s_cbranch_scc1 1
s_branch 15
flat_load_dword v3, v[66:67]
s_waitcnt vmcnt(0) lgkmcnt(0)
v_mov_b32_e32 v4, 2.0
v_mul_f32_e32 v5, v3, v4
flat_store_dword v[66:67], v5
s_waitcnt vmcnt(0) lgkmcnt(0)
v_mov_b32_e32 v3, 1
v_add_u32_e32 v4, v2, v3
v_cmp_lt_i32_e64 s[32:33], v4, v1
s_and_b64 vcc, s[32:33], exec
s_cbranch_scc1 1
s_endpgm
v_mov_b32_e32 v2, v4
s_branch 65518`)
}

// A branch whose condition depends on the lane is refused by name until
// the execution mask lowering exists.
func TestDivergentBranchRefused(t *testing.T) {
	m := ir.NewModule("div", ir.AMDGCN)
	fn := m.Func("k").Export().CallConv(ir.Kernel).NoUnwind()
	n := fn.ParamI32("n")
	entry := fn.Entry()
	yes := fn.Block("yes")
	done := fn.Block("done")
	tid := entry.I32.WorkitemID(ir.X)
	entry.BrIf(entry.I32.ULt(tid, n), yes.To(), done.To())
	yes.Trap()
	done.Return()
	if err := verify.Module(m); err != nil {
		t.Fatal(err)
	}
	_, err := lower.Lower(m, lower.Options{ASIC: feature.GFX942})
	if err == nil || !strings.Contains(err.Error(), "differs across the wave") {
		t.Fatalf("err = %v", err)
	}
}
