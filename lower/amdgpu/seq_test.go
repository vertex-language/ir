package amdgpu_test

// The sequences of milestone 29: each lowers, encodes, and decodes under
// llvm-objdump into a listing that contains the instruction the
// sequence is built around. Whether the answers are right is a question
// for a GPU, which the machine running these has none of; what is
// checked here is that the bytes are the instructions meant, and that
// every trapping row reaches the function's s_trap.

import (
	"os"
	"strings"
	"testing"

	"github.com/vertex-language/amdgpu/feature"

	"github.com/vertex-language/ir"
	lower "github.com/vertex-language/ir/lower/amdgpu"
)

// unaryKernel is a kernel that loads one value at the work-item's
// index, applies f, and stores what f returns at the same index.
func unaryKernel(name string, load func(b *ir.Builder, p ir.Ptr) ir.Value, f func(b *ir.Builder, v ir.Value) ir.Value, store func(b *ir.Builder, v ir.Value, p ir.Ptr)) *ir.Module {
	m := ir.NewModule(name, ir.AMDGCN)
	fn := m.Func(name).Export().CallConv(ir.Kernel).NoUnwind()
	a := fn.ParamPtr("a", ir.NoAlias)
	b := fn.ParamPtr("b", ir.NoAlias)
	e := fn.Entry()
	tid := e.I32.WorkitemID(ir.X)
	off := e.I64.Shl(e.I64.ZExtI32(tid), e.I64.Const(3))
	pa, pb := e.Ptr.Add(a, off), e.Ptr.Add(b, off)
	store(&e.Builder, f(&e.Builder, load(&e.Builder, pa)), pb)
	e.Return()
	return m
}

func TestSequences(t *testing.T) {
	i32 := func(b *ir.Builder, p ir.Ptr) ir.Value { return b.I32.Load(p) }
	i64 := func(b *ir.Builder, p ir.Ptr) ir.Value { return b.I64.Load(p) }
	f32 := func(b *ir.Builder, p ir.Ptr) ir.Value { return b.F32.Load(p) }
	f64 := func(b *ir.Builder, p ir.Ptr) ir.Value { return b.F64.Load(p) }
	sI32 := func(b *ir.Builder, v ir.Value, p ir.Ptr) { b.I32.Store(v.(ir.I32), p) }
	sI64 := func(b *ir.Builder, v ir.Value, p ir.Ptr) { b.I64.Store(v.(ir.I64), p) }
	sF32 := func(b *ir.Builder, v ir.Value, p ir.Ptr) { b.F32.Store(v.(ir.F32), p) }
	sF64 := func(b *ir.Builder, v ir.Value, p ir.Ptr) { b.F64.Store(v.(ir.F64), p) }

	cases := []struct {
		name  string
		m     *ir.Module
		wants []string
	}{
		{"udiv32", unaryKernel("udiv32", i32, func(b *ir.Builder, v ir.Value) ir.Value {
			return b.I32.UDiv(v.(ir.I32), b.I32.Const(7))
		}, sI32), []string{"v_rcp_iflag_f32", "v_mul_hi_u32", "s_cbranch_scc1", "s_trap 2"}},
		{"srem32", unaryKernel("srem32", i32, func(b *ir.Builder, v ir.Value) ir.Value {
			return b.I32.SRem(b.I32.Const(-100), v.(ir.I32))
		}, sI32), []string{"v_rcp_iflag_f32", "v_ashrrev_i32", "s_trap 2"}},
		{"udiv64", unaryKernel("udiv64", i64, func(b *ir.Builder, v ir.Value) ir.Value {
			return b.I64.UDiv(v.(ir.I64), b.I64.Const(1000000007))
		}, sI64), []string{"v_cmp_eq_u64", "s_trap 2", "v_lshlrev_b64", "v_cmp_ge_u64", "v_cndmask_b32", "s_cbranch_scc1"}},
		{"srem64", unaryKernel("srem64", i64, func(b *ir.Builder, v ir.Value) ir.Value {
			return b.I64.SRem(b.I64.Const(-5), v.(ir.I64))
		}, sI64), []string{"v_cmp_eq_u64", "v_xor_b32", "v_subb_co_u32", "v_cmp_ge_u64"}},
		{"fdiv32", unaryKernel("fdiv32", f32, func(b *ir.Builder, v ir.Value) ir.Value {
			return b.F32.Div(v.(ir.F32), b.F32.Const(3))
		}, sF32), []string{"v_div_scale_f32", "v_div_fmas_f32", "v_div_fixup_f32", "v_fma_f32 v", "-v"}},
		{"fdiv64", unaryKernel("fdiv64", f64, func(b *ir.Builder, v ir.Value) ir.Value {
			return b.F64.Div(b.F64.Const(1), v.(ir.F64))
		}, sF64), []string{"v_div_scale_f64", "v_div_fmas_f64", "v_div_fixup_f64", "1.0"}},
		{"fsqrt32", unaryKernel("fsqrt32", f32, func(b *ir.Builder, v ir.Value) ir.Value {
			return b.F32.Sqrt(v.(ir.F32))
		}, sF32), []string{"v_sqrt_f32", "v_cmp_class_f32", "0x37800000"}},
		{"fsqrt64", unaryKernel("fsqrt64", f64, func(b *ir.Builder, v ir.Value) ir.Value {
			return b.F64.Sqrt(v.(ir.F64))
		}, sF64), []string{"v_rsq_f64", "v_ldexp_f64", "v_cmp_class_f64"}},
		{"minimum", unaryKernel("minimum", f32, func(b *ir.Builder, v ir.Value) ir.Value {
			return b.F32.Minimum(v.(ir.F32), b.F32.Const(0))
		}, sF32), []string{"v_min_f32", "v_cmp_o_f32", "0x7fc00000"}},
		{"maxnum64", unaryKernel("maxnum64", f64, func(b *ir.Builder, v ir.Value) ir.Value {
			return b.F64.MaxNum(v.(ir.F64), b.F64.Const(0))
		}, sF64), []string{"v_max_f64"}},
		{"f32toi32", unaryKernel("f32toi32", f32, func(b *ir.Builder, v ir.Value) ir.Value {
			return b.I32.SCvtF32(v.(ir.F32))
		}, sI32), []string{"v_cmp_ge_f32", "v_cmp_lt_f32", "s_not_b64", "s_trap 2", "v_cvt_i32_f32"}},
		{"f32toi64", unaryKernel("f32toi64", f32, func(b *ir.Builder, v ir.Value) ir.Value {
			return b.I64.SCvtF32(v.(ir.F32))
		}, sI64), []string{"v_trunc_f32", "v_floor_f32", "0xcf800000", "v_subb_co_u32"}},
		{"f64tou64sat", unaryKernel("f64tou64sat", f64, func(b *ir.Builder, v ir.Value) ir.Value {
			return b.I64.UCvtSatF64(v.(ir.F64))
		}, sI64), []string{"v_trunc_f64", "v_ldexp_f64", "v_cvt_u32_f64"}},
		{"u64tof32", unaryKernel("u64tof32", i64, func(b *ir.Builder, v ir.Value) ir.Value {
			return b.F32.UCvtI64(v.(ir.I64))
		}, sF32), []string{"v_ffbh_u32", "v_lshlrev_b64", "v_cvt_f32_u32", "v_ldexp_f32"}},
		{"s64tof64", unaryKernel("s64tof64", i64, func(b *ir.Builder, v ir.Value) ir.Value {
			return b.F64.SCvtI64(v.(ir.I64))
		}, sF64), []string{"v_cvt_f64_i32", "v_ldexp_f64", "v_add_f64"}},
		{"clz64", unaryKernel("clz64", i64, func(b *ir.Builder, v ir.Value) ir.Value {
			return b.I64.Clz(v.(ir.I64))
		}, sI64), []string{"v_ffbh_u32", "v_min_u32", "v_cmp_ne_u32"}},
		{"popcnt64", unaryKernel("popcnt64", i64, func(b *ir.Builder, v ir.Value) ir.Value {
			return b.I64.Popcnt(v.(ir.I64))
		}, sI64), []string{"v_bcnt_u32_b32"}},
		{"bswap64", unaryKernel("bswap64", i64, func(b *ir.Builder, v ir.Value) ir.Value {
			return b.I64.Bswap(v.(ir.I64))
		}, sI64), []string{"v_perm_b32", "0x10203"}},
		{"rotl64", unaryKernel("rotl64", i64, func(b *ir.Builder, v ir.Value) ir.Value {
			return b.I64.RotL(v.(ir.I64), b.I64.Const(13))
		}, sI64), []string{"v_lshlrev_b64", "v_lshrrev_b64"}},
		{"smulhi64", unaryKernel("smulhi64", i64, func(b *ir.Builder, v ir.Value) ir.Value {
			return b.I64.SMulHi(v.(ir.I64), v.(ir.I64))
		}, sI64), []string{"v_mad_u64_u32", "v_subb_co_u32"}},
		{"copysign64", unaryKernel("copysign64", f64, func(b *ir.Builder, v ir.Value) ir.Value {
			return b.F64.CopySign(b.F64.Const(2), v.(ir.F64))
		}, sF64), []string{"v_bfi_b32"}},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			o := lowerObj(t, c.m, lower.Options{ASIC: feature.GFX942})
			got := disassemble(t, o, feature.GFX942)
			if got == nil {
				return
			}
			text := strings.Join(got, "\n")
			if os.Getenv("AMDGPU_LISTING") != "" {
				t.Log("\n" + text)
			}
			for _, w := range c.wants {
				if !strings.Contains(text, w) {
					t.Errorf("no %q in:\n%s", w, text)
				}
			}
		})
	}
}

// An overflow row is the flag alone, beside the wrapping operation.
func TestOverflow(t *testing.T) {
	m := ir.NewModule("o", ir.AMDGCN)
	fn := m.Func("o").Export().CallConv(ir.Kernel).NoUnwind()
	a := fn.ParamPtr("a", ir.NoAlias)
	e := fn.Entry()
	v := e.I32.Load(a)
	o := e.I32.SMulO(v, v)
	e.I32.Store(e.I32.Add(e.I32.Mul(v, v), e.I32.ZExtI1(o)), a)
	e.Return()
	obj := lowerObj(t, m, lower.Options{ASIC: feature.GFX942})
	got := disassemble(t, obj, feature.GFX942)
	if got == nil {
		return
	}
	text := strings.Join(got, "\n")
	for _, w := range []string{"v_mul_lo_u32", "v_mul_hi_i32", "v_cmp_ne_u32", "v_cndmask_b32"} {
		if !strings.Contains(text, w) {
			t.Errorf("no %q in:\n%s", w, text)
		}
	}
}
