package amdgpu_test

import (
	"strings"
	"testing"

	"github.com/vertex-language/amdgpu/feature"

	"github.com/vertex-language/ir"
	lower "github.com/vertex-language/ir/lower/amdgpu"
)

// The calling convention: a device function that calls another and
// keeps a value across the call, reached from a kernel through a
// pointer — the one call the inliner leaves — and an exported device
// function with a divergent branch and an alloc, reached from nothing.
func TestCalls(t *testing.T) {
	m := ir.NewModule("cc", ir.AMDGCN)

	// g(a i64, b i1) -> i64: b ? a*2 : a
	g := m.Func("g").Internal().ReturnsI64().NoUnwind()
	ga, gb := g.ParamI64("a"), g.ParamI1("b")
	ge := g.Entry()
	ge.Return(ge.I64.Select(gb, ge.I64.Mul(ga, ge.I64.Const(2)), ga))

	// f(x i32, y f32) -> i32: g(x, y > 0) + x, with x live across the call
	f := m.Func("f").Internal().ReturnsI32().NoUnwind()
	fx, fy := f.ParamI32("x"), f.ParamF32("y")
	fe := f.Entry()
	r := fe.Call(g, fe.I64.SExtI32(fx), fe.F32.Lt(fe.F32.Const(0), fy)).I64(0)
	fe.Return(fe.I32.Add(fe.I32.WrapI64(r), fx))
	fT := m.FuncType("f_t", ir.NewSig().Param(ir.TypeI32).Param(ir.TypeF32).Ret(ir.TypeI32))

	// h(p ptr) -> i32, exported: a lane-dependent branch and a frame.
	h := m.Func("h").Export().ReturnsI32().NoUnwind()
	hp := h.ParamPtr("p")
	he := h.Entry()
	hthen := h.Block("then")
	hout := h.Block("out")
	hv := hout.ParamI32("v")
	buf := he.Ptr.Alloc(16, 4)
	he.I32.Store(he.I32.Const(9), buf)
	he.BrIf(he.I32.SLt(he.I32.Load(hp), he.I32.Const(0)), hthen.To(), hout.To(he.I32.Load(buf)))
	hthen.Br(hout.To(hthen.I32.Const(-1)))
	hout.Return(hv)

	k := m.Func("k").Export().CallConv(ir.Kernel).NoUnwind()
	kp := k.ParamPtr("p")
	ke := k.Entry()
	tid := ke.I32.WorkitemID(ir.X)
	fp := ke.Ptr.GetAddr(f)
	res := ke.CallInd(fp, fT, tid, ke.F32.Const(1.5)).I32(0)
	ke.I32.Store(res, ke.Ptr.Add(kp, ke.I64.Shl(ke.I64.ZExtI32(tid), ke.I64.Const(2))))
	ke.Return()

	o := lowerObj(t, m, lower.Options{ASIC: feature.GFX942})
	for _, name := range []string{"f", "g", "h", "k"} {
		if _, ok := o.Symbol(name); !ok {
			t.Errorf("no symbol %s", name)
		}
	}
	kd := o.KernelDescriptors()[0]
	if kd.PrivateSegmentFixedSize == 0 {
		t.Errorf("the kernel's private segment is empty; its callees' frames are in it")
	}
	got := lowers(t, m, feature.GFX942,
		"s_swappc_b64 s[30:31], s[34:35]", "v_readfirstlane_b32 s34", "s_setpc_b64 s[30:31]",
		"s_mov_b32 s33, s32", "s_add_u32 s32, s32", "v_writelane_b32 v58", "v_readlane_b32 s30, v58",
		"scratch_store_dword off, v58, s32", "s_mov_b32 s32, s33", "s_mov_b32 s32,")
	if got == nil {
		return
	}
	// f keeps x across its call to g: x lives in a register g saves.
	text := strings.Join(got, "\n")
	if !strings.Contains(text, "s_getpc_b64 s[34:35]") {
		t.Errorf("no direct call sequence in:\n%s", text)
	}
}

// Recursion has no bound on the device and is refused by name.
func TestRecursionRefused(t *testing.T) {
	m := ir.NewModule("rec", ir.AMDGCN)
	f := m.Func("f").Export().ReturnsI32().NoUnwind()
	fx := f.ParamI32("x")
	fe := f.Entry()
	fe.Return(fe.Call(f, fx).I32(0))
	_, err := lower.Lower(m, lower.Options{ASIC: feature.GFX942})
	if err == nil || !strings.Contains(err.Error(), "recursive") {
		t.Fatalf("err = %v", err)
	}
}

// A function pointer that differs across the wave: the call is a
// waterfall loop, each pass calling the first active lane's target for
// the lanes that share it.
func TestWaterfall(t *testing.T) {
	m := ir.NewModule("wf", ir.AMDGCN)
	fT := m.FuncType("f_t", ir.NewSig().Param(ir.TypeI32).Ret(ir.TypeI32))
	dbl := m.Func("dbl").Internal().ReturnsI32().NoUnwind()
	dx := dbl.ParamI32("x")
	dbl.Entry().Return(dbl.Entry().I32.Mul(dx, dbl.Entry().I32.Const(2)))
	inc := m.Func("inc").Internal().ReturnsI32().NoUnwind()
	ix := inc.ParamI32("x")
	inc.Entry().Return(inc.Entry().I32.Add(ix, inc.Entry().I32.Const(1)))

	k := m.Func("k").Export().CallConv(ir.Kernel).NoUnwind()
	p := k.ParamPtr("p")
	e := k.Entry()
	tid := e.I32.WorkitemID(ir.X)
	odd := e.I32.Ne(e.I32.And(tid, e.I32.Const(1)), e.I32.Const(0))
	fp := e.Ptr.Select(odd, e.Ptr.GetAddr(dbl), e.Ptr.GetAddr(inc))
	r := e.CallInd(fp, fT, tid).I32(0)
	e.I32.Store(r, e.Ptr.Add(p, e.I64.Shl(e.I64.ZExtI32(tid), e.I64.Const(2))))
	e.Return()
	got := lowers(t, m, feature.GFX942,
		"v_readfirstlane_b32 s", "v_cmp_eq_u64_e64 s", "s_and_b64 exec, exec, s", "s_swappc_b64 s[30:31], s[", "s_not_b64 s")
	if got == nil {
		return
	}
	text := strings.Join(got, "\n")
	if strings.Count(text, "s_swappc_b64") != 1 {
		t.Errorf("the waterfall should hold one call:\n%s", text)
	}
}

// A table of function pointers in the data: the functions are emitted
// because the table names them, and the code object carries a dynamic
// relocation for each entry.
func TestFunctionTable(t *testing.T) {
	m := ir.NewModule("tbl", ir.AMDGCN)
	fT := m.FuncType("f_t", ir.NewSig().Param(ir.TypeI32).Ret(ir.TypeI32))
	dbl := m.Func("dbl").Internal().ReturnsI32().NoUnwind()
	dx := dbl.ParamI32("x")
	dbl.Entry().Return(dbl.Entry().I32.Mul(dx, dbl.Entry().I32.Const(2)))
	inc := m.Func("inc").Internal().ReturnsI32().NoUnwind()
	ix := inc.ParamI32("x")
	inc.Entry().Return(inc.Entry().I32.Add(ix, inc.Entry().I32.Const(1)))
	tbl := m.Global("tbl", ir.RW, ir.Array(2, ir.StorePtr.FType())).Align(8).
		Init(ir.List(ir.RelocInit(dbl), ir.RelocInit(inc)))

	k := m.Func("k").Export().CallConv(ir.Kernel).NoUnwind()
	p := k.ParamPtr("p")
	e := k.Entry()
	tid := e.I32.WorkitemID(ir.X)
	slot := e.Ptr.Add(e.Ptr.GetAddr(tbl), e.I64.Shl(e.I64.ZExtI32(e.I32.And(tid, e.I32.Const(1))), e.I64.Const(3)))
	r := e.CallInd(e.Ptr.Load(slot), fT, tid).I32(0)
	e.I32.Store(r, p)
	e.Return()

	o := lowerObj(t, m, lower.Options{ASIC: feature.GFX942})
	for _, name := range []string{"dbl", "inc", "tbl"} {
		if _, ok := o.Symbol(name); !ok {
			t.Errorf("no symbol %s", name)
		}
	}
	if refs := o.SectionNamed(".data").Refs(); len(refs) != 2 {
		t.Errorf("the table carries %d references, want 2", len(refs))
	}
	lowers(t, m, feature.GFX942, "s_swappc_b64")
}

// A device function reads its place in the grid from the implicit
// registers a calling kernel fills.
func TestImplicitIDs(t *testing.T) {
	m := ir.NewModule("imp", ir.AMDGCN)
	gid := m.Func("gid").Export().ReturnsI32().NoUnwind()
	ge := gid.Entry()
	ge.Return(ge.I32.Add(ge.I32.Mul(ge.I32.WorkgroupID(ir.Y), ge.I32.WorkgroupSize(ir.X)), ge.I32.WorkitemID(ir.Z)))
	fT := m.FuncType("gid_t", ir.NewSig().Ret(ir.TypeI32))

	k := m.Func("k").Export().CallConv(ir.Kernel).NoUnwind()
	p := k.ParamPtr("p")
	e := k.Entry()
	r := e.CallInd(e.Ptr.GetAddr(gid), fT).I32(0)
	e.I32.Store(e.I32.Add(r, e.I32.NumWorkgroups(ir.X)), p)
	e.Return()
	for _, c := range []struct {
		asic  feature.ASIC
		wants []string
	}{
		// gfx942 packs the ids in v0 and has no scratch SGPRs before the
		// workgroup ids; gfx900 packs them itself, after flat scratch init.
		{feature.GFX942, []string{"s_mov_b32 s36, s2", "v_mov_b32_e32 v31, v"}},
		{feature.GFX900, []string{"s_mov_b32 s36, s4", "v_or_b32_e32 v31, v"}},
	} {
		asic := c.asic
		o := lowerObj(t, m, lower.Options{ASIC: asic})
		kd := o.KernelDescriptors()[0]
		// Both extra workgroup ids and all three work-item ids are asked for.
		if kd.ComputePgmRsrc2&(3<<8) != 3<<8 || (kd.ComputePgmRsrc2>>11)&3 != 2 {
			t.Errorf("%s: rsrc2 = %#x", asic, kd.ComputePgmRsrc2)
		}
		if len(kd.Args) != 7 {
			t.Errorf("%s: %d kernel arguments, want the pointer and six hidden ones", asic, len(kd.Args))
		}
		lowers(t, m, asic, append(c.wants, "20, v31", "v_mov_b32_e32 v32, s37", "v_mov_b32_e32 v33, s39")...)
	}
}

// A struct by value: the aggregate lies in the kernarg buffer at its own
// offset, and the parameter is its address there.
func TestKernelByVal(t *testing.T) {
	m := ir.NewModule("kbv", ir.AMDGCN)
	pair := m.Struct("pair").Field("a", ir.StoreI32.FType()).Field("b", ir.StoreI64.FType())
	k := m.Func("k").Export().CallConv(ir.Kernel).NoUnwind()
	n := k.ParamI32("n")
	p := k.ParamPtr("p", ir.ByVal(pair))
	out := k.ParamPtr("out")
	e := k.Entry()
	b := e.I64.Load(e.Ptr.Add(p, e.I64.Const(8)))
	e.I64.Store(e.I64.Add(b, e.I64.SExtI32(e.I32.Add(n, e.I32.Load(p)))), out)
	e.Return()
	o := lowerObj(t, m, lower.Options{ASIC: feature.GFX942})
	kd := o.KernelDescriptors()[0]
	// n at 0, the pair at 8 (its alignment), out at 24.
	if len(kd.Args) < 3 || kd.Args[1].Offset != 8 || kd.Args[1].Size != 16 || kd.Args[2].Offset != 24 {
		t.Fatalf("kernel arguments: %+v", kd.Args)
	}
	lowers(t, m, feature.GFX942, "s_add_u32 s", "s_addc_u32 s", "s_load_dword s")
}
