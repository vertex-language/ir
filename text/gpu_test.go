package text_test

// §W, kernel, shared, and the scoped forms, printed. The first test is the
// whole distance from a CPU module to a device one: one word on the func
// line and two verbs in the body.

import (
	"testing"

	"github.com/vertex-language/ir"
)

func TestPrintKernel(t *testing.T) {
	m := ir.NewModule("vecadd", ir.NVPTX64)

	fn := m.Func("vector_add").Export().CallConv(ir.Kernel).NoUnwind()
	a := fn.ParamPtr("a", ir.NoAlias)
	b := fn.ParamPtr("b", ir.NoAlias)
	c := fn.ParamPtr("c", ir.NoAlias)
	n := fn.ParamI32("n")

	entry := fn.Entry()
	body := fn.Block("body")
	done := fn.Block("done")

	gid := entry.I32.WorkgroupID(ir.X)
	gsz := entry.I32.WorkgroupSize(ir.X)
	base := entry.I32.Mul(gid, gsz)
	tid := entry.I32.WorkitemID(ir.X)
	i := entry.I32.Add(base, tid).Named("i")
	entry.BrIf(entry.I32.ULt(i, n), body.To(), done.To())

	off := body.I64.Shl(body.I64.ZExtI32(i), body.I64.Const(2))
	pa := body.Ptr.Add(a, off).Named("pa")
	pb := body.Ptr.Add(b, off).Named("pb")
	pc := body.Ptr.Add(c, off).Named("pc")
	body.F32.Store(body.F32.Add(body.F32.Load(pa), body.F32.Load(pb)), pc)
	body.Return()

	done.Return()

	same(t, format(t, m), `
module vecadd

use "nvptx64/cuda"

layout {
  abi        ptx,
  endian     little,
  ptrbits    64,
  stackalign 16,
  extfloat   none,
}

export func @vector_add kernel(%a ptr noalias, %b ptr noalias, %c ptr noalias, %n i32) nounwind {
@entry:
  %0 = i32.workgroup_id x
  %1 = i32.workgroup_size x
  %2 = i32.mul %0, %1
  %3 = i32.workitem_id x
  %i = i32.add %2, %3
  %4 = i32.ult %i, %n
  brif %4, @body, @done

@body:
  %5 = i64.zext_i32 %i
  %6 = i64.const 2
  %7 = i64.shl %5, %6
  %pa = ptr.add %a, %7
  %pb = ptr.add %b, %7
  %pc = ptr.add %c, %7
  %8 = f32.load %pa
  %9 = f32.load %pb
  %10 = f32.add %8, %9
  f32.store %10, %pc
  return

@done:
  return
}
`)
}

// A shared global, a barrier, the scoped atomics and fences, and the wave
// verbs — the rest of §W and §3.5–3.8 of the plan, in one function.
func TestPrintSharedAndScopes(t *testing.T) {
	m := ir.NewModule("reduce", ir.AMDGCN)

	tile := m.Global("tile", ir.Shared, ir.Array(256, ir.StoreF32.FType())).Align(16)
	count := m.Global("count", ir.RW, ir.StoreI32.FType())

	fn := m.Func("k").Export().CallConv(ir.Kernel)
	entry := fn.Entry()

	tid := entry.I32.WorkitemID(ir.X)
	lane := entry.I32.LaneID()
	ws := entry.I32.WaveSize()
	p := entry.Ptr.Add(entry.Ptr.GetAddr(tile), entry.I64.ZExtI32(tid))
	entry.F32.Store(entry.F32.Const(1), p)
	entry.Barrier()
	entry.Fence(ir.SeqCst, ir.FenceWorkgroup)
	entry.Fence(ir.Release, ir.FenceDevice)
	entry.Fence(ir.SeqCst, ir.FenceSystem)

	cp := entry.Ptr.GetAddr(count)
	entry.I32.AtomicRmwAdd(entry.I32.Const(1), cp, ir.Monotonic, ir.DeviceScope)
	entry.I32.AtomicRmwSMax(tid, cp, ir.AcqRel, ir.WorkgroupScope, ir.Volatile)
	entry.I32.AtomicRmwUMin(tid, cp, ir.Monotonic)
	entry.F32.AtomicRmwAdd(entry.F32.Const(2), cp, ir.Monotonic, ir.SystemScope)
	entry.I32.AtomicCas(tid, lane, cp, ir.SeqCst, ir.Acquire, ir.DeviceScope)
	entry.I32.AtomicLoad(cp, ir.Acquire, ir.DeviceScope)
	entry.I32.AtomicStore(ws, cp, ir.Release, ir.WorkgroupScope)

	full := entry.I32.Const(-1)
	x := entry.I32.WaveShflXor(tid, entry.I32.Const(16), full)
	entry.I32.WaveShflDown(x, entry.I32.Const(1), full)
	entry.I32.WaveShflUp(x, entry.I32.Const(1), full)
	entry.I32.WaveShflIdx(x, lane, full)
	entry.I32.WaveReadFirstLane(x)
	c := entry.I32.Eq(x, tid)
	entry.I64.WaveBallot(c, full)
	entry.I1.WaveAny(c, full)
	entry.I1.WaveAll(c, full)
	entry.F32.RsqrtApprox(entry.F32.Exp2Approx(entry.F32.Const(0.5)))
	entry.Return()

	same(t, format(t, m), `
module reduce

use "amdgcn/hsa"

layout {
  abi        hsa,
  endian     little,
  ptrbits    64,
  stackalign 16,
  extfloat   none,
}

global shared @tile [256]f32 align 16 = zeroed

global rw @count i32 = zeroed

export func @k kernel() {
@entry:
  %0 = i32.workitem_id x
  %1 = i32.lane_id
  %2 = i32.wave_size
  %3 = ptr.getaddr @tile
  %4 = i64.zext_i32 %0
  %5 = ptr.add %3, %4
  %6 = f32.const 1
  f32.store %6, %5
  barrier
  fence seq_cst workgroup
  fence release device
  fence seq_cst system
  %7 = ptr.getaddr @count
  %8 = i32.const 1
  %9 = i32.atomic_rmwadd %8, %7 monotonic device
  %10 = i32.atomic_rmwsmax %0, %7 acq_rel workgroup volatile
  %11 = i32.atomic_rmwumin %0, %7 monotonic
  %12 = f32.const 2
  %13 = f32.atomic_rmwadd %12, %7 monotonic system
  %14 = i32.atomic_cas %0, %1, %7 seq_cst acquire device
  %15 = i32.atomic_load %7 acquire device
  i32.atomic_store %2, %7 release workgroup
  %16 = i32.const -1
  %17 = i32.const 16
  %18 = i32.wave_shfl_xor %0, %17, %16
  %19 = i32.const 1
  %20 = i32.wave_shfl_down %18, %19, %16
  %21 = i32.const 1
  %22 = i32.wave_shfl_up %18, %21, %16
  %23 = i32.wave_shfl_idx %18, %1, %16
  %24 = i32.wave_readfirstlane %18
  %25 = i32.eq %18, %0
  %26 = i64.wave_ballot %25, %16
  %27 = i1.wave_any %25, %16
  %28 = i1.wave_all %25, %16
  %29 = f32.const 0.5
  %30 = f32.exp2_approx %29
  %31 = f32.rsqrt_approx %30
  return
}
`)
}
