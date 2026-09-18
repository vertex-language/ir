package amdgpu_test

import (
	"os"
	"strings"
	"testing"

	"github.com/vertex-language/amdgpu/feature"

	"github.com/vertex-language/ir"
	lower "github.com/vertex-language/ir/lower/amdgpu"
)

// Workgroup storage: each lane stores to the shared array, the workgroup
// barriers, and each lane reads its neighbour's slot. The shared global
// is an LDS offset, its address the aperture over that offset, and the
// accesses through it ds_write and ds_read.
func TestShared(t *testing.T) {
	m := ir.NewModule("lds", ir.AMDGCN)
	buf := m.Global("buf", ir.Shared, ir.Array(256, ir.StoreI32.FType())).Align(4)
	fn := m.Func("k").Export().CallConv(ir.Kernel).NoUnwind()
	a := fn.ParamPtr("a", ir.NoAlias)
	e := fn.Entry()
	tid := e.I32.WorkitemID(ir.X)
	off := e.I64.Shl(e.I64.ZExtI32(tid), e.I64.Const(2))
	p := e.Ptr.Add(a, off)
	s := e.Ptr.Add(e.Ptr.GetAddr(buf), off)
	e.I32.Store(e.I32.Load(p), s)
	e.Barrier()
	next := e.I32.And(e.I32.Add(tid, e.I32.Const(1)), e.I32.Const(255))
	sn := e.Ptr.Add(e.Ptr.GetAddr(buf), e.I64.Shl(e.I64.ZExtI32(next), e.I64.Const(2)))
	e.I32.Store(e.I32.Load(sn), p)
	e.Return()

	o := lowerObj(t, m, lower.Options{ASIC: feature.GFX942})
	if kd := o.KernelDescriptors()[0]; kd.GroupSegmentFixedSize != 1024 {
		t.Errorf("group segment = %d, want 1024", kd.GroupSegmentFixedSize)
	}
	got := disassemble(t, o, feature.GFX942)
	if got == nil {
		return
	}
	text := strings.Join(got, "\n")
	if os.Getenv("AMDGPU_LISTING") != "" {
		t.Log("\n" + text)
	}
	for _, w := range []string{"src_shared_base", "ds_write_b32", "s_barrier", "ds_read_b32", "flat_store_dword"} {
		if !strings.Contains(text, w) {
			t.Errorf("no %q in:\n%s", w, text)
		}
	}
}

// The memory model on each generation: an acq_rel device-scope add, a
// system-scope compare-and-swap, an acquire load and a release fence.
func TestAtomics(t *testing.T) {
	build := func() *ir.Module {
		m := ir.NewModule("at", ir.AMDGCN)
		fn := m.Func("k").Export().CallConv(ir.Kernel).NoUnwind()
		p := fn.ParamPtr("p")
		q := fn.ParamPtr("q")
		e := fn.Entry()
		old := e.I32.AtomicRmwAdd(e.I32.Const(1), p, ir.AcqRel, ir.DeviceScope)
		was := e.I64.AtomicCas(e.I64.Const(0), e.I64.ZExtI32(old), q, ir.SeqCst, ir.Monotonic)
		v := e.I32.AtomicLoad(p, ir.Acquire, ir.DeviceScope)
		e.Fence(ir.Release, ir.FenceDevice)
		e.I32.AtomicStore(e.I32.Add(v, e.I32.WrapI64(was)), p, ir.Monotonic, ir.WorkgroupScope)
		e.Return()
		return m
	}
	for _, c := range []struct {
		asic  feature.ASIC
		wants []string
	}{
		{feature.GFX942, []string{"buffer_wbl2 sc1", "flat_atomic_add v", "sc0", "buffer_inv sc1",
			"buffer_wbl2 sc0 sc1", "flat_atomic_cmpswap_x2 v", "v[60:63]", "buffer_inv sc0 sc1",
			"flat_load_dword v", "sc1", "flat_store_dword v"}},
		{feature.GFX90A, []string{"flat_atomic_add v", "glc", "buffer_wbinvl1_vol",
			"buffer_wbl2", "flat_atomic_cmpswap_x2 v", "buffer_invl2"}},
		{feature.GFX900, []string{"flat_atomic_add v", "glc", "buffer_wbinvl1_vol", "flat_atomic_cmpswap_x2 v"}},
	} {
		t.Run(c.asic.String(), func(t *testing.T) {
			o := lowerObj(t, build(), lower.Options{ASIC: c.asic})
			got := disassemble(t, o, c.asic)
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

// Atomics on workgroup storage are DS instructions.
func TestSharedAtomics(t *testing.T) {
	m := ir.NewModule("lat", ir.AMDGCN)
	cnt := m.Global("cnt", ir.Shared, ir.StoreI32.FType()).Align(4)
	fn := m.Func("k").Export().CallConv(ir.Kernel).NoUnwind()
	p := fn.ParamPtr("p")
	e := fn.Entry()
	s := e.Ptr.GetAddr(cnt)
	old := e.I32.AtomicRmwAdd(e.I32.Const(1), s, ir.Monotonic, ir.WorkgroupScope)
	was := e.I32.AtomicCas(old, e.I32.Const(7), s, ir.AcqRel, ir.Monotonic, ir.WorkgroupScope)
	e.F32.AtomicRmwAdd(e.F32.Const(1), s, ir.Monotonic, ir.WorkgroupScope)
	e.I32.Store(was, p)
	e.Return()
	o := lowerObj(t, m, lower.Options{ASIC: feature.GFX942})
	got := disassemble(t, o, feature.GFX942)
	if got == nil {
		return
	}
	text := strings.Join(got, "\n")
	if os.Getenv("AMDGPU_LISTING") != "" {
		t.Log("\n" + text)
	}
	for _, w := range []string{"ds_add_rtn_u32", "ds_cmpst_rtn_b32", "ds_add_rtn_f32"} {
		if !strings.Contains(text, w) {
			t.Errorf("no %q in:\n%s", w, text)
		}
	}
}

// Private memory: a ptr.alloc is a flat pointer in the private
// aperture, the descriptor asks for the segment, and before gfx940 a
// prologue builds FLAT_SCRATCH.
func TestAlloc(t *testing.T) {
	build := func() *ir.Module {
		m := ir.NewModule("pa", ir.AMDGCN)
		fn := m.Func("k").Export().CallConv(ir.Kernel).NoUnwind()
		p := fn.ParamPtr("p")
		e := fn.Entry()
		buf := e.Ptr.Alloc(64, 16)
		tid := e.I32.WorkitemID(ir.X)
		slot := e.Ptr.Add(buf, e.I64.Shl(e.I64.ZExtI32(e.I32.And(tid, e.I32.Const(15))), e.I64.Const(2)))
		e.I32.Store(tid, slot)
		e.I32.Store(e.I32.Load(e.Ptr.Add(buf, e.I64.Const(12))), p)
		e.Return()
		return m
	}
	for _, c := range []struct {
		asic  feature.ASIC
		wants []string
	}{
		{feature.GFX942, []string{"src_private_base", "flat_store_dword", "flat_load_dword"}},
		{feature.GFX900, []string{"s_add_u32 flat_scratch_lo, s2, s5", "s_addc_u32 flat_scratch_hi, s3, 0", "src_private_base"}},
	} {
		t.Run(c.asic.String(), func(t *testing.T) {
			o := lowerObj(t, build(), lower.Options{ASIC: c.asic})
			kd := o.KernelDescriptors()[0]
			if kd.PrivateSegmentFixedSize != 64 || kd.ComputePgmRsrc2&1 == 0 {
				t.Errorf("private segment %d, rsrc2 %#x", kd.PrivateSegmentFixedSize, kd.ComputePgmRsrc2)
			}
			got := disassemble(t, o, c.asic)
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

// Spilling: more values live than the file holds. The function is
// selected again with scratch, VGPRs spill to slots and SGPRs to lanes.
func TestSpill(t *testing.T) {
	build := func() *ir.Module {
		m := ir.NewModule("sp", ir.AMDGCN)
		fn := m.Func("k").Export().CallConv(ir.Kernel).NoUnwind()
		p := fn.ParamPtr("p")
		e := fn.Entry()
		// Seventy loads, all live until the sum: more than the fifty-eight
		// singles.
		var vs []ir.I32
		for i := 0; i < 70; i++ {
			vs = append(vs, e.I32.Load(e.Ptr.Add(p, e.I64.Const(int64(i*4)))))
		}
		sum := vs[0]
		for _, v := range vs[1:] {
			sum = e.I32.Add(sum, v)
		}
		e.I32.Store(sum, p)
		e.Return()
		return m
	}
	for _, asic := range []feature.ASIC{feature.GFX942, feature.GFX90A} {
		t.Run(asic.String(), func(t *testing.T) {
			o := lowerObj(t, build(), lower.Options{ASIC: asic})
			kd := o.KernelDescriptors()[0]
			if kd.PrivateSegmentFixedSize == 0 {
				t.Errorf("no private segment")
			}
			got := disassemble(t, o, asic)
			if got == nil {
				return
			}
			text := strings.Join(got, "\n")
			if os.Getenv("AMDGPU_LISTING") != "" {
				t.Log("\n" + text)
			}
			for _, w := range []string{"scratch_store_dword", "scratch_load_dword"} {
				if !strings.Contains(text, w) {
					t.Errorf("no %q in:\n%s", w, text)
				}
			}
		})
	}
}

// Scalar spills: forty lane masks live at once, more than the pairs
// hold, go to lanes of the reserved VGPR through v_writelane and come
// back through v_readlane.
func TestSpillMasks(t *testing.T) {
	m := ir.NewModule("sm", ir.AMDGCN)
	fn := m.Func("k").Export().CallConv(ir.Kernel).NoUnwind()
	p := fn.ParamPtr("p")
	e := fn.Entry()
	tid := e.I32.WorkitemID(ir.X)
	var ms []ir.I1
	for i := 0; i < 40; i++ {
		ms = append(ms, e.I32.ULt(tid, e.I32.Const(int64(i))))
	}
	all := ms[0]
	for _, mm := range ms[1:] {
		all = e.I1.And(all, mm)
	}
	e.I32.Store(e.I32.ZExtI1(all), p)
	e.Return()
	o := lowerObj(t, m, lower.Options{ASIC: feature.GFX942})
	got := disassemble(t, o, feature.GFX942)
	if got == nil {
		return
	}
	text := strings.Join(got, "\n")
	if os.Getenv("AMDGPU_LISTING") != "" {
		t.Log("\n" + text)
	}
	for _, w := range []string{"v_writelane_b32 v59", "v_readlane_b32 s"} {
		if !strings.Contains(text, w) {
			t.Errorf("no %q in:\n%s", w, text)
		}
	}
}

// Narrow atomics: a byte add is a compare-and-swap loop on the
// containing dword, a byte load and store are the byte instructions.
func TestNarrowAtomics(t *testing.T) {
	m := ir.NewModule("na", ir.AMDGCN)
	fn := m.Func("k").Export().CallConv(ir.Kernel).NoUnwind()
	p := fn.ParamPtr("p")
	e := fn.Entry()
	tid := e.I32.WorkitemID(ir.X)
	q := e.Ptr.Add(p, e.I64.ZExtI32(e.I32.And(tid, e.I32.Const(7))))
	old := e.I32.AtomicRmwAdd8(e.I32.Const(3), q, ir.Monotonic, ir.DeviceScope)
	was := e.I32.AtomicCas16(old, e.I32.Const(9), p, ir.AcqRel, ir.Monotonic)
	b := e.I32.AtomicULoad8(q, ir.Acquire, ir.DeviceScope)
	e.I32.AtomicStore8(e.I32.Add(e.I32.Add(old, was), b), q, ir.Release, ir.DeviceScope)
	e.Return()
	lowers(t, m, feature.GFX942,
		"v_and_b32_e32 v", "-4", "flat_atomic_cmpswap v", "v_cmp_ne_u32_e64 s", "s_cbranch_execz",
		"flat_load_ubyte", "flat_store_byte", "buffer_wbl2 sc1")
}

// A float atomic add on a generation with no instruction for it is a
// compare-and-swap loop.
func TestFloatAddLoop(t *testing.T) {
	m := ir.NewModule("fa", ir.AMDGCN)
	fn := m.Func("k").Export().CallConv(ir.Kernel).NoUnwind()
	p := fn.ParamPtr("p")
	e := fn.Entry()
	e.F32.AtomicRmwAdd(e.F32.Const(1), p, ir.Monotonic, ir.DeviceScope)
	e.Return()
	lowers(t, m, feature.GFX900, "v_add_f32_e32", "flat_atomic_cmpswap v", "v_cmp_ne_u32_e64", "s_cbranch_execz")
	lowers(t, m, feature.GFX942, "flat_atomic_add_f32")
}

// A frame past what a scratch offset reaches: the offset travels in a
// VGPR.
func TestLargeFrame(t *testing.T) {
	m := ir.NewModule("lf", ir.AMDGCN)
	fn := m.Func("k").Export().CallConv(ir.Kernel).NoUnwind()
	p := fn.ParamPtr("p")
	e := fn.Entry()
	big := e.Ptr.Alloc(8192, 16)
	e.I32.Store(e.I32.Const(1), e.Ptr.Add(big, e.I64.Const(8000)))
	// Enough live values to spill past the alloc.
	var vs []ir.I32
	for i := 0; i < 70; i++ {
		vs = append(vs, e.I32.Load(e.Ptr.Add(p, e.I64.Const(int64(i*4)))))
	}
	sum := vs[0]
	for _, v := range vs[1:] {
		sum = e.I32.Add(sum, v)
	}
	e.I32.Store(sum, p)
	e.Return()
	o := lowerObj(t, m, lower.Options{ASIC: feature.GFX942})
	if kd := o.KernelDescriptors()[0]; kd.PrivateSegmentFixedSize < 8192 {
		t.Errorf("private segment = %d", kd.PrivateSegmentFixedSize)
	}
	lowers(t, m, feature.GFX942, "v_mov_b32_e32 v57, 0x2", "scratch_store_dword v57, v")
}

// Dynamic workgroup storage: an unsized shared import lies past the
// module's own LDS, at an aligned offset the descriptor states as the
// group segment, and reads through ds_ instructions.
func TestDynamicShared(t *testing.T) {
	m := ir.NewModule("dyn", ir.AMDGCN)
	fixed := m.Global("fixed", ir.Shared, ir.Array(5, ir.StoreI32.FType())).Align(4)
	dyn := m.ImportGlobal("dyn", ir.Array(0, ir.StoreF32.FType())).Shared()
	k := m.Func("k").Export().CallConv(ir.Kernel).NoUnwind()
	p := k.ParamPtr("p")
	e := k.Entry()
	tid := e.I32.WorkitemID(ir.X)
	off := e.I64.Shl(e.I64.ZExtI32(tid), e.I64.Const(2))
	e.I32.Store(tid, e.Ptr.Add(e.Ptr.GetAddr(fixed), off))
	slot := e.Ptr.Add(e.Ptr.GetAddr(dyn), off)
	e.F32.Store(e.F32.Const(1.5), slot)
	e.Barrier()
	e.F32.Store(e.F32.Load(slot), e.Ptr.Add(p, off))
	e.Return()
	o := lowerObj(t, m, lower.Options{ASIC: feature.GFX942})
	kd := o.KernelDescriptors()[0]
	if kd.GroupSegmentFixedSize != 32 {
		t.Errorf("group segment = %d, want 32: five words rounded up for the dynamic storage after them", kd.GroupSegmentFixedSize)
	}
	lowers(t, m, feature.GFX942, "ds_write_b32", "ds_read_b32", "s_barrier")
}
