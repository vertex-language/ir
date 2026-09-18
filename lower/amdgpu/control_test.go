package amdgpu_test

import (
	"os"
	"strings"
	"testing"

	"github.com/vertex-language/amdgpu/feature"

	"github.com/vertex-language/ir"
	lower "github.com/vertex-language/ir/lower/amdgpu"
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

// A branch whose condition depends on the lane: the bounds check every
// kernel has. The function is structurized — each block behind a flow
// that narrows exec to the lanes whose predicate is set — and the
// listing is pinned.
func TestDivergentBranch(t *testing.T) {
	m := ir.NewModule("div", ir.AMDGCN)
	fn := m.Func("k").Export().CallConv(ir.Kernel).NoUnwind()
	a := fn.ParamPtr("a", ir.NoAlias)
	n := fn.ParamI32("n")
	entry := fn.Entry()
	body := fn.Block("body")
	done := fn.Block("done")
	tid := entry.I32.WorkitemID(ir.X)
	entry.BrIf(entry.I32.ULt(tid, n), body.To(), done.To())
	off := body.I64.Shl(body.I64.ZExtI32(tid), body.I64.Const(2))
	p := body.Ptr.Add(a, off)
	body.F32.Store(body.F32.Mul(body.F32.Load(p), body.F32.Const(2)), p)
	body.Return()
	done.Return()
	o := lowerObj(t, m, lower.Options{ASIC: feature.GFX942})
	// s[36:37] and s[34:35] are body's and done's predicates, set from the
	// compare and its complement under exec; each block's flow saves
	// exec, narrows it to the predicate, consumes the predicate, and
	// skips the block when no lane is left. The restore at the next flow
	// serves both paths.
	same(t, disassemble(t, o, feature.GFX942), `
s_mov_b64 s[34:35], 0
s_mov_b64 s[36:37], 0
s_load_dwordx2 s[32:33], s[0:1], 0x0
s_waitcnt vmcnt(0) lgkmcnt(0)
v_mov_b32_e32 v64, s32
v_mov_b32_e32 v65, s33
s_load_dword s8, s[0:1], 0x8
s_waitcnt vmcnt(0) lgkmcnt(0)
v_mov_b32_e32 v1, s8
v_and_b32_e32 v2, 0x3ff, v0
v_cmp_lt_u32_e64 s[32:33], v2, v1
s_and_b64 vcc, s[32:33], exec
s_andn2_b64 s[36:37], s[36:37], exec
s_or_b64 s[36:37], s[36:37], vcc
s_andn2_b64 vcc, exec, s[32:33]
s_andn2_b64 s[34:35], s[34:35], exec
s_or_b64 s[34:35], s[34:35], vcc
s_mov_b64 s[32:33], exec
s_and_b64 exec, exec, s[36:37]
s_andn2_b64 s[36:37], s[36:37], exec
s_cbranch_execz 17
v_mov_b32_e32 v66, v2
v_mov_b32_e32 v67, 0
v_mov_b32_e32 v68, 2
v_mov_b32_e32 v69, 0
v_lshlrev_b64 v[70:71], v68, v[66:67]
v_add_co_u32_e32 v66, vcc, v64, v70
v_addc_co_u32_e32 v67, vcc, v65, v71, vcc
flat_load_dword v1, v[66:67]
s_waitcnt vmcnt(0) lgkmcnt(0)
v_mov_b32_e32 v2, 2.0
v_mul_f32_e32 v3, v1, v2
flat_store_dword v[66:67], v3
s_waitcnt vmcnt(0) lgkmcnt(0)
s_mov_b64 exec, s[32:33]
s_mov_b64 s[32:33], exec
s_and_b64 exec, exec, s[34:35]
s_andn2_b64 s[34:35], s[34:35], exec
s_cbranch_execz 0
s_mov_b64 exec, s[32:33]
s_endpgm`)
}

// A loop whose trip count is the lane's: lanes drop out one by one, and
// the loop runs until none is left.
func TestDivergentLoop(t *testing.T) {
	m := ir.NewModule("dl", ir.AMDGCN)
	fn := m.Func("k").Export().CallConv(ir.Kernel).NoUnwind()
	a := fn.ParamPtr("a", ir.NoAlias)
	entry := fn.Entry()
	loop := fn.Block("loop")
	k := loop.ParamI32("k")
	acc := loop.ParamF32("acc")
	done := fn.Block("done")
	r := done.ParamF32("r")
	tid := entry.I32.WorkitemID(ir.X)
	off := entry.I64.Shl(entry.I64.ZExtI32(tid), entry.I64.Const(2))
	p := entry.Ptr.Add(a, off)
	entry.Br(loop.To(entry.I32.Const(0), entry.F32.Const(0)))
	acc1 := loop.F32.Add(acc, loop.F32.Load(p))
	k1 := loop.I32.Add(k, loop.I32.Const(1))
	loop.BrIf(loop.I32.ULt(k1, tid), loop.To(k1, acc1), done.To(acc1))
	done.F32.Store(r, p)
	done.Return()
	o := lowerObj(t, m, lower.Options{ASIC: feature.GFX942})
	// The loop's pre-flow saves exec in s[40:41]; its header flow narrows
	// exec to the lanes whose loop predicate (s[36:37]) is set, and the
	// back edge's copy of k and acc sets it again for the lanes that
	// continue. s_cbranch_execz 23 is the exit, which restores s[40:41].
	same(t, disassemble(t, o, feature.GFX942), `
s_mov_b64 s[34:35], 0
s_load_dwordx2 s[32:33], s[0:1], 0x0
s_waitcnt vmcnt(0) lgkmcnt(0)
v_mov_b32_e32 v64, s32
v_mov_b32_e32 v65, s33
v_and_b32_e32 v3, 0x3ff, v0
v_mov_b32_e32 v66, v3
v_mov_b32_e32 v67, 0
v_mov_b32_e32 v68, 2
v_mov_b32_e32 v69, 0
v_lshlrev_b64 v[70:71], v68, v[66:67]
v_add_co_u32_e32 v66, vcc, v64, v70
v_addc_co_u32_e32 v67, vcc, v65, v71, vcc
v_mov_b32_e32 v4, 0
v_mov_b32_e32 v5, 0
s_or_b64 s[34:35], s[34:35], exec
s_mov_b64 s[36:37], 0
s_mov_b64 s[32:33], exec
s_and_b64 exec, exec, s[34:35]
s_andn2_b64 s[34:35], s[34:35], exec
s_cbranch_execz 3
v_mov_b32_e32 v1, v4
v_mov_b32_e32 v2, v5
s_or_b64 s[36:37], s[36:37], exec
s_mov_b64 exec, s[32:33]
s_mov_b64 s[34:35], 0
s_mov_b64 s[38:39], 0
s_mov_b64 s[40:41], exec
s_and_b64 exec, exec, s[36:37]
s_andn2_b64 s[36:37], s[36:37], exec
s_cbranch_execz 23
flat_load_dword v4, v[66:67]
s_waitcnt vmcnt(0) lgkmcnt(0)
v_add_f32_e32 v5, v2, v4
v_mov_b32_e32 v2, 1
v_add_u32_e32 v4, v1, v2
v_cmp_lt_u32_e64 s[32:33], v4, v3
s_and_b64 vcc, s[32:33], exec
s_andn2_b64 s[38:39], s[38:39], exec
s_or_b64 s[38:39], s[38:39], vcc
s_andn2_b64 vcc, exec, s[32:33]
s_andn2_b64 s[34:35], s[34:35], exec
s_or_b64 s[34:35], s[34:35], vcc
s_mov_b64 s[32:33], exec
s_and_b64 exec, exec, s[38:39]
s_andn2_b64 s[38:39], s[38:39], exec
s_cbranch_execz 3
v_mov_b32_e32 v1, v4
v_mov_b32_e32 v2, v5
s_or_b64 s[36:37], s[36:37], exec
s_mov_b64 exec, s[32:33]
s_branch 65510
s_mov_b64 exec, s[40:41]
s_mov_b64 s[32:33], 0
s_mov_b64 s[36:37], exec
s_and_b64 exec, exec, s[34:35]
s_andn2_b64 s[34:35], s[34:35], exec
s_cbranch_execz 2
v_mov_b32_e32 v1, v5
s_or_b64 s[32:33], s[32:33], exec
s_mov_b64 exec, s[36:37]
s_mov_b64 s[34:35], exec
s_and_b64 exec, exec, s[32:33]
s_andn2_b64 s[32:33], s[32:33], exec
s_cbranch_execz 3
flat_store_dword v[66:67], v1
s_waitcnt vmcnt(0) lgkmcnt(0)
s_mov_b64 exec, s[34:35]
s_endpgm`)
}

// A kernel calling a device helper: the helper is inlined, the kernel
// lowers, and the helper — internal, and now uncalled — is left alone.
func TestDeviceCallInlined(t *testing.T) {
	m := ir.NewModule("h", ir.AMDGCN)
	twice := m.Func("twice").Internal().ReturnsF32().NoUnwind()
	x := twice.ParamF32("x")
	te := twice.Entry()
	te.Return(te.F32.Mul(x, te.F32.Const(2)))

	fn := m.Func("k").Export().CallConv(ir.Kernel).NoUnwind()
	a := fn.ParamPtr("a", ir.NoAlias)
	entry := fn.Entry()
	tid := entry.I32.WorkitemID(ir.X)
	off := entry.I64.Shl(entry.I64.ZExtI32(tid), entry.I64.Const(2))
	p := entry.Ptr.Add(a, off)
	entry.F32.Store(entry.Call(twice, entry.F32.Load(p)).F32(0), p)
	entry.Return()

	o := lowerObj(t, m, lower.Options{ASIC: feature.GFX942})
	if kds := o.KernelDescriptors(); len(kds) != 1 || kds[0].Name != "k" {
		t.Fatalf("descriptors = %+v", kds)
	}
	got := disassemble(t, o, feature.GFX942)
	if got == nil {
		return
	}
	text := strings.Join(got, "\n")
	for _, want := range []string{"flat_load_dword", "v_mul_f32_e32", "flat_store_dword", "s_endpgm"} {
		if !strings.Contains(text, want) {
			t.Errorf("no %s in:\n%s", want, text)
		}
	}
}

// a && b, as a frontend spells it: two branches sharing an else, which
// is the shape a structurizer exists for. And a loop with a break in a
// nested if, whose exit edges come from two blocks.
func TestStructuredShapes(t *testing.T) {
	m := ir.NewModule("ss", ir.AMDGCN)
	fn := m.Func("k").Export().CallConv(ir.Kernel).NoUnwind()
	a := fn.ParamPtr("a", ir.NoAlias)
	n := fn.ParamI32("n")
	entry := fn.Entry()
	rhs := fn.Block("rhs")
	then := fn.Block("then")
	join := fn.Block("join")
	loop := fn.Block("loop")
	k := loop.ParamI32("k")
	sum := loop.ParamI32("sum")
	chk := fn.Block("chk")
	kv := chk.ParamI32("kv")
	latch := fn.Block("latch")
	done := fn.Block("done")
	total := done.ParamI32("total")

	tid := entry.I32.WorkitemID(ir.X)
	off := entry.I64.Shl(entry.I64.ZExtI32(tid), entry.I64.Const(2))
	p := entry.Ptr.Add(a, off)
	entry.BrIf(entry.I32.ULt(tid, n), rhs.To(), join.To())
	rhs.BrIf(rhs.I32.SLt(rhs.I32.Const(0), rhs.I32.Load(p)), then.To(), join.To())
	then.I32.Store(then.I32.Const(1), p)
	then.Br(join.To())
	join.Br(loop.To(join.I32.Const(0), join.I32.Const(0)))
	// for k < tid: v = a[k]; if v == 0 break; sum += v
	pk := loop.Ptr.Add(a, loop.I64.Shl(loop.I64.ZExtI32(k), loop.I64.Const(2)))
	loop.BrIf(loop.I32.ULt(k, tid), chk.To(loop.I32.Load(pk)), done.To(sum))
	chk.BrIf(chk.I32.Eq(kv, chk.I32.Const(0)), done.To(sum), latch.To())
	latch.Br(loop.To(latch.I32.Add(k, latch.I32.Const(1)), latch.I32.Add(sum, kv)))
	done.I32.Store(total, p)
	done.Return()

	o := lowerObj(t, m, lower.Options{ASIC: feature.GFX942})
	got := disassemble(t, o, feature.GFX942)
	if got == nil {
		return
	}
	if os.Getenv("AMDGPU_LISTING") != "" {
		t.Log("\n" + strings.Join(got, "\n"))
	}
	text := strings.Join(got, "\n")
	for _, w := range []string{"s_cbranch_execz", "s_and_b64 exec, exec", "s_endpgm"} {
		if !strings.Contains(text, w) {
			t.Errorf("no %q in:\n%s", w, text)
		}
	}
}
