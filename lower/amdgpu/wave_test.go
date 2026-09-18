package amdgpu_test

import (
	"os"
	"strings"
	"testing"

	"github.com/vertex-language/amdgpu/feature"

	"github.com/vertex-language/ir"
	lower "github.com/vertex-language/ir/lower/amdgpu"
)

// lowers builds, lowers and disassembles a kernel and checks the listing
// holds each wanted fragment; the listing is logged under
// AMDGPU_LISTING.
func lowers(t *testing.T, m *ir.Module, asic feature.ASIC, wants ...string) []string {
	t.Helper()
	o := lowerObj(t, m, lower.Options{ASIC: asic})
	got := disassemble(t, o, asic)
	if got == nil {
		return nil
	}
	text := strings.Join(got, "\n")
	if os.Getenv("AMDGPU_LISTING") != "" {
		t.Log("\n" + text)
	}
	for _, w := range wants {
		if !strings.Contains(text, w) {
			t.Errorf("no %q in:\n%s", w, text)
		}
	}
	return got
}

// A wave reduction: each lane's value shuffled down and summed, the
// ballot of the lanes that are non-zero, whether any lane is, and the
// first lane's total.
func TestWaveVerbs(t *testing.T) {
	m := ir.NewModule("wv", ir.AMDGCN)
	fn := m.Func("k").Export().CallConv(ir.Kernel).NoUnwind()
	p := fn.ParamPtr("p")
	e := fn.Entry()
	tid := e.I32.WorkitemID(ir.X)
	v := e.I32.Load(e.Ptr.Add(p, e.I64.Shl(e.I64.ZExtI32(tid), e.I64.Const(2))))
	all := e.I32.Const(-1)
	s := e.I32.Add(v, e.I32.WaveShflDown(v, e.I32.Const(32), all))
	s = e.I32.Add(s, e.I32.WaveShflXor(s, e.I32.Const(16), all))
	s = e.I32.Add(s, e.I32.WaveShflUp(s, e.I32.Const(8), all))
	s = e.I32.Add(s, e.I32.WaveShflIdx(s, e.I32.Const(0), all))
	nz := e.I32.Ne(v, e.I32.Const(0))
	b := e.I64.WaveBallot(nz, all)
	any := e.I1.WaveAny(nz, all)
	total := e.I32.WaveReadFirstLane(s)
	e.I32.Store(e.I32.Add(e.I32.Add(total, e.I32.WrapI64(b)), e.I32.ZExtI1(any)), p)
	e.Return()
	lowers(t, m, feature.GFX942,
		"v_mbcnt_lo_u32_b32", "v_mbcnt_hi_u32_b32", "ds_bpermute_b32", "v_cndmask_b32",
		"s_and_b64 s", "s_cmp_lg_u64", "s_cselect_b64", "v_readfirstlane_b32")
}

// Bulk memory: a constant copy unrolled, a lane-sized copy looped under
// the execution mask, a set and a compare.
func TestBulk(t *testing.T) {
	m := ir.NewModule("bulk", ir.AMDGCN)
	fn := m.Func("k").Export().CallConv(ir.Kernel).NoUnwind()
	dst := fn.ParamPtr("dst", ir.NoAlias)
	src := fn.ParamPtr("src", ir.NoAlias)
	e := fn.Entry()
	tid := e.I32.WorkitemID(ir.X)
	off := e.I64.Shl(e.I64.ZExtI32(tid), e.I64.Const(4))
	d, s := e.Ptr.Add(dst, off), e.Ptr.Add(src, off)
	e.MemCpy(d, s, e.I64.Const(13))
	e.MemCpy(d, s, e.I64.ZExtI32(tid))
	e.MemSet(d, e.I32.Const(0xab), e.I64.Const(6))
	r := e.MemCmp(d, s, e.I64.Const(1000))
	e.I32.Store(r, dst)
	e.Return()
	lowers(t, m, feature.GFX942,
		"flat_load_dword", "flat_load_ubyte", "flat_store_byte", "v_perm_b32", "v_cmp_lt_u64",
		"s_cbranch_execz", "v_sub_u32")
}

// br_table on a lane's value: an if-chain.
func TestBrTable(t *testing.T) {
	m := ir.NewModule("bt", ir.AMDGCN)
	fn := m.Func("k").Export().CallConv(ir.Kernel).NoUnwind()
	p := fn.ParamPtr("p")
	e := fn.Entry()
	out := fn.Block("out")
	v := out.ParamI32("v")
	c0, c1, c2 := fn.Block("c0"), fn.Block("c1"), fn.Block("c2")
	tid := e.I32.WorkitemID(ir.X)
	e.BrTable(e.I32.And(tid, e.I32.Const(3)), []ir.BlockTarget{c0.To(), c1.To(), c2.To()}, out.To(e.I32.Const(-1)))
	c0.Br(out.To(c0.I32.Const(10)))
	c1.Br(out.To(c1.I32.Const(20)))
	c2.Br(out.To(c2.I32.Const(30)))
	out.I32.Store(v, p)
	out.Return()
	lowers(t, m, feature.GFX942, "v_cmp_eq_u32_e64 s", "s_cbranch_execz", "flat_store_dword")
}

// Inline assembly: a vector instruction on the operands' own registers,
// and a scalar one on a value read into an SGPR for it.
func TestAsm(t *testing.T) {
	m := ir.NewModule("asm", ir.AMDGCN)
	fn := m.Func("k").Export().CallConv(ir.Kernel).NoUnwind()
	p := fn.ParamPtr("p")
	e := fn.Entry()
	tid := e.I32.WorkitemID(ir.X)
	a := e.Asm("v_add_u32 %0, %1, %2").Out(ir.TypeI32, ir.CStr("v")).In(tid, ir.CStr("v")).In(e.I32.Const(5), ir.CStr("v")).Clobber("vcc").Emit().I32(0)
	b := e.Asm("s_lshl_b32 %0, %1, 2").Out(ir.TypeI32, ir.CStr("s")).In(a, ir.CStr("s")).Emit().I32(0)
	e.I32.Store(e.I32.Add(a, b), p)
	e.Return()
	lowers(t, m, feature.GFX942, "v_add_u32_e32 v", "v_readfirstlane_b32 s", "s_lshl_b32 s", ", 2")
}
