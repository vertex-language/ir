package air_test

// Kernels built in VIR, lowered here, written by the air package as a
// .metallib, validated by Apple's air-validate, and run on the GPU, their
// answers checked against Go's reading of §0. The tests skip, saying why,
// where there is no Metal toolchain or GPU.

import (
	"context"
	"errors"
	"math"
	"strings"
	"testing"

	am "github.com/vertex-language/air"
	"github.com/vertex-language/air/check"
	"github.com/vertex-language/air/metallib"
	"github.com/vertex-language/ir"
	airlower "github.com/vertex-language/ir/lower/air"
	"github.com/vertex-language/ir/verify"
)

func skipIf(t *testing.T, err error) {
	t.Helper()
	if errors.Is(err, check.ErrNoToolchain) || errors.Is(err, check.ErrNoGPU) {
		t.Skip(err)
	}
	if err != nil {
		t.Fatal(err)
	}
}

// lower verifies m, lowers it, and writes its library.
func lower(t *testing.T, m *ir.Module, opts airlower.Options) []byte {
	t.Helper()
	if err := m.Err(); err != nil {
		t.Fatalf("building VIR: %v", err)
	}
	if err := verify.Module(m); err != nil {
		t.Fatalf("verify: %v", err)
	}
	am, err := airlower.Lower(m, opts)
	if err != nil {
		t.Fatal(err)
	}
	lib, err := metallib.Build(am)
	if err != nil {
		t.Fatal(err)
	}
	skipIf(t, check.Validate(context.Background(), lib))
	return lib
}

func run(t *testing.T, m *ir.Module, kernel string, g check.Grid, bufs ...check.Buffer) *check.Result {
	t.Helper()
	lib := lower(t, m, airlower.Options{})
	res, err := check.Run(context.Background(), lib, kernel, g, bufs...)
	skipIf(t, err)
	return res
}

func kernel(m *ir.Module, name string) *ir.Func {
	return m.Func(name).Export().CallConv(ir.Kernel).NoUnwind()
}

// gid is the global index along x: workgroup id times size, plus the
// work-item's own.
func gid(b *ir.Block) ir.I32 {
	return b.I32.Add(b.I32.Mul(b.I32.WorkgroupID(ir.X), b.I32.WorkgroupSize(ir.X)), b.I32.WorkitemID(ir.X))
}

// at is p + 4*i: the address of element i of an array of 4-byte values.
func at(b *ir.Block, p ir.Ptr, i ir.I32) ir.Ptr {
	return b.Ptr.Add(p, b.I64.Shl(b.I64.ZExtI32(i), b.I64.Const(2)))
}

func TestVectorAdd(t *testing.T) {
	m := ir.NewModule("vadd", ir.AIR64)
	f := kernel(m, "vector_add")
	a, bp, c := f.ParamPtr("a"), f.ParamPtr("b"), f.ParamPtr("c")
	n := f.ParamI32("n")
	entry, body, done := f.Entry(), f.Block("body"), f.Block("done")
	i := gid(entry)
	entry.BrIf(entry.I32.ULt(i, n), body.To(), done.To())
	body.F32.Store(body.F32.Add(body.F32.Load(at(body, a, i)), body.F32.Load(at(body, bp, i))), at(body, c, i))
	body.Br(done.To())
	done.Return()

	const N = 1000
	x, y := make([]float32, 1024), make([]float32, 1024)
	for k := range x {
		x[k], y[k] = float32(k), float32(k)*0.5
	}
	res := run(t, m, "vector_add", check.Grid1D(1024, 64), check.Floats(x...), check.Floats(y...),
		check.Out(4*1024), check.Uint32s(N))
	for k, v := range res.Floats(2) {
		want := x[k] + y[k]
		if k >= N {
			want = 0
		}
		if v != want {
			t.Fatalf("c[%d] = %v, want %v", k, v, want)
		}
	}
}

// TestLoopBlockParams is a loop carried in block parameters: phis in AIR,
// with the back edge a conditional branch's, through a trampoline.
func TestLoopBlockParams(t *testing.T) {
	m := ir.NewModule("loop", ir.AIR64)
	f := kernel(m, "triangle")
	out := f.ParamPtr("out")
	entry, loop, done := f.Entry(), f.Block("loop"), f.Block("done")
	i := loop.ParamI32("i")
	acc := loop.ParamI32("acc")
	res := done.ParamI32("res")
	g := gid(entry)
	entry.Br(loop.To(entry.I32.Const(0), entry.I32.Const(0)))
	acc2 := loop.I32.Add(acc, i)
	i2 := loop.I32.Add(i, loop.I32.Const(1))
	loop.BrIf(loop.I32.ULe(i2, g), loop.To(i2, acc2), done.To(acc2))
	done.I32.Store(res, at(done, out, gid(done)))
	done.Return()

	r := run(t, m, "triangle", check.Grid1D(256, 64), check.Out(4*256))
	for k, v := range r.Uint32s(0) {
		if want := uint32(k * (k + 1) / 2); v != want {
			t.Fatalf("out[%d] = %d, want %d", k, v, want)
		}
	}
}

// TestBrTable is a table whose cases carry arguments to one join block.
func TestBrTable(t *testing.T) {
	m := ir.NewModule("table", ir.AIR64)
	f := kernel(m, "classify")
	out := f.ParamPtr("out")
	entry, join := f.Entry(), f.Block("join")
	v := join.ParamI32("v")
	g := gid(entry)
	sel := entry.I32.URem(g, entry.I32.Const(5))
	c := func(n int64) ir.BlockTarget { return join.To(entry.I32.Const(n)) }
	entry.BrTable(sel, []ir.BlockTarget{c(10), c(20), c(30), c(40)}, c(99))
	join.I32.Store(v, at(join, out, g))
	join.Return()

	r := run(t, m, "classify", check.Grid1D(64, 32), check.Out(4*64))
	want := []uint32{10, 20, 30, 40, 99}
	for k, v := range r.Uint32s(0) {
		if v != want[k%5] {
			t.Fatalf("out[%d] = %d, want %d", k, v, want[k%5])
		}
	}
}

// TestSharedReduction sums through a shared global behind a barrier,
// then one float atomic per workgroup into the result.
func TestSharedReduction(t *testing.T) {
	m := ir.NewModule("reduce", ir.AIR64)
	tile := m.Global("tile", ir.Shared, ir.Array(256, ir.StoreF32.FType()))
	f := kernel(m, "reduce")
	in, out := f.ParamPtr("input"), f.ParamPtr("out")
	entry, loop, step, sync, done, first, exit := f.Entry(), f.Block("loop"), f.Block("step"), f.Block("sync"),
		f.Block("done"), f.Block("first"), f.Block("exit")
	s := loop.ParamI32("s")

	tid := entry.I32.WorkitemID(ir.X)
	base := entry.Ptr.GetAddr(tile)
	entry.F32.Store(entry.F32.Load(at(entry, in, gid(entry))), at(entry, base, tid))
	entry.Barrier()
	entry.Br(loop.To(entry.I32.Const(128)))

	// for s := 128; s > 0; s >>= 1 { if tid < s { tile[tid] += tile[tid+s] }; barrier }
	loop.BrIf(loop.I32.ULt(loop.I32.WorkitemID(ir.X), s), step.To(), sync.To())
	lt := step.I32.WorkitemID(ir.X)
	lb := step.Ptr.GetAddr(tile)
	sum := step.F32.Add(step.F32.Load(at(step, lb, lt)), step.F32.Load(at(step, lb, step.I32.Add(lt, s))))
	step.F32.Store(sum, at(step, lb, lt))
	step.Br(sync.To())
	sync.Barrier()
	next := sync.I32.UShr(s, sync.I32.Const(1))
	sync.BrIf(sync.I32.Eq(next, sync.I32.Const(0)), done.To(), loop.To(next))

	done.BrIf(done.I32.Eq(done.I32.WorkitemID(ir.X), done.I32.Const(0)), first.To(), exit.To())
	first.F32.AtomicRmwAdd(first.F32.Load(first.Ptr.GetAddr(tile)), out, ir.Monotonic, ir.DeviceScope)
	first.Br(exit.To())
	exit.Return()

	x := make([]float32, 4096)
	var want float32
	for k := range x {
		x[k] = float32(k % 9)
		want += x[k]
	}
	r := run(t, m, "reduce", check.Grid1D(4096, 256), check.Floats(x...), check.Floats(0))
	if got := r.Floats(1)[0]; got != want {
		t.Fatalf("sum = %v, want %v", got, want)
	}
}

func TestHistogram(t *testing.T) {
	m := ir.NewModule("hist", ir.AIR64)
	f := kernel(m, "histogram")
	in, bins := f.ParamPtr("input"), f.ParamPtr("bins")
	b := f.Entry()
	v := b.I32.Load(at(b, in, gid(b)))
	b.I32.AtomicRmwAdd(b.I32.Const(1), at(b, bins, b.I32.And(v, b.I32.Const(15))), ir.Monotonic, ir.DeviceScope)
	b.Return()

	x := make([]uint32, 1000)
	want := make([]uint32, 16)
	for k := range x {
		x[k] = uint32(k*k) % 16
		want[x[k]]++
	}
	r := run(t, m, "histogram", check.Grid1D(1000, 100), check.Uint32s(x...), check.Out(4*16))
	for k, v := range r.Uint32s(1) {
		if v != want[k] {
			t.Fatalf("bin %d = %d, want %d", k, v, want[k])
		}
	}
}

// TestWave is shuffles, a broadcast and a ballot, over 32-lane SIMD groups.
func TestWave(t *testing.T) {
	m := ir.NewModule("wave", ir.AIR64)
	f := kernel(m, "wave")
	out := f.ParamPtr("out")
	b := f.Entry()
	g := gid(b)
	lane := b.I32.LaneID()
	all := b.I32.Const(-1)
	x := b.I32.WaveShflXor(lane, b.I32.Const(31), all)
	first := b.I32.WaveReadFirstLane(b.I32.Add(g, b.I32.Const(1000)))
	even := b.I32.Eq(b.I32.And(lane, b.I32.Const(1)), b.I32.Const(0))
	ballot := b.I32.WrapI64(b.I64.WaveBallot(even, all))
	row := b.I64.Shl(b.I64.ZExtI32(g), b.I64.Const(4)) // 4 words per work-item
	p := b.Ptr.Add(out, row)
	b.I32.Store(x, p)
	b.I32.Store(first, b.Ptr.Add(p, b.I64.Const(4)))
	b.I32.Store(ballot, b.Ptr.Add(p, b.I64.Const(8)))
	b.I32.Store(b.I32.WaveSize(), b.Ptr.Add(p, b.I64.Const(12)))
	b.Return()

	r := run(t, m, "wave", check.Grid1D(64, 64), check.Out(16*64))
	o := r.Uint32s(0)
	for k := 0; k < 64; k++ {
		lane := uint32(k % 32)
		got := o[4*k : 4*k+4]
		want := []uint32{lane ^ 31, uint32(k/32*32 + 1000), 0x55555555, 32}
		for j := range want {
			if got[j] != want[j] {
				t.Fatalf("work-item %d word %d = %#x, want %#x", k, j, got[j], want[j])
			}
		}
	}
}

// TestArithmetic runs the rows where AIR and §0 disagree, each checked
// against what the backend promises: guarded division, saturating
// conversion, masked shifts, rotates, bswap, IEEE minimum, copysign.
func TestArithmetic(t *testing.T) {
	m := ir.NewModule("arith", ir.AIR64)
	f := kernel(m, "arith")
	out := f.ParamPtr("out")
	ints, floats := f.ParamPtr("ints"), f.ParamPtr("floats")
	b := f.Entry()
	w := func(k int, v ir.I32) { b.I32.Store(v, b.Ptr.Add(out, b.I64.Const(int64(4*k)))) }
	wf := func(k int, v ir.F32) { b.F32.Store(v, b.Ptr.Add(out, b.I64.Const(int64(4*k)))) }
	li := func(k int) ir.I32 { return b.I32.Load(b.Ptr.Add(ints, b.I64.Const(int64(4*k)))) }
	lf := func(k int) ir.F32 { return b.F32.Load(b.Ptr.Add(floats, b.I64.Const(int64(4*k)))) }
	zero, seven, min, neg1 := li(0), li(1), li(2), li(3)
	w(0, b.I32.SDiv(seven, zero)) // guarded: the dividend
	w(1, b.I32.SDiv(min, neg1))   // guarded: MIN
	w(2, b.I32.SRem(min, neg1))   // 0
	w(3, b.I32.UDiv(seven, zero))
	w(4, b.I32.Shl(seven, b.I32.Const(33))) // masked: 7 << 1
	w(5, b.I32.RotL(li(4), b.I32.Const(8)))
	w(6, b.I32.RotR(li(4), b.I32.Const(0)))
	w(7, b.I32.Bswap(li(4)))
	w(8, b.I32.Clz(li(5)))
	w(9, b.I32.Popcnt(li(4)))
	nan, big, negBig, half := lf(0), lf(1), lf(2), lf(3)
	w(10, b.I32.SCvtSatF32(nan))
	w(11, b.I32.SCvtSatF32(big))
	w(12, b.I32.SCvtSatF32(negBig))
	w(13, b.I32.UCvtSatF32(negBig))
	w(14, b.I32.SCvtSatF32(half))
	wf(15, b.F32.Minimum(nan, half))
	wf(16, b.F32.Minimum(b.F32.Const(0), b.F32.Neg(b.F32.Const(0))))
	wf(17, b.F32.Maximum(b.F32.Const(0), b.F32.Neg(b.F32.Const(0))))
	wf(18, b.F32.CopySign(half, negBig))
	wf(19, b.F32.Nearest(b.F32.Const(2.5)))
	w(20, b.I32.ZExtI1(b.I32.SAddO(li(6), b.I32.Const(1))))
	w(21, b.I32.ZExtI1(b.I32.UAddO(li(7), b.I32.Const(1))))
	b.Return()

	r := run(t, m, "arith", check.Grid1D(1, 1), check.Out(4*22),
		check.Uint32s(0, 7, 0x80000000, 0xffffffff, 0x11223344, 0x00010000, 0x7fffffff, 0xffffffff),
		check.Floats(float32(math.NaN()), 3e9, -3e9, 2.75))
	o := r.Uint32s(0)
	f32 := func(v float32) uint32 { return math.Float32bits(v) }
	want := map[int]uint32{
		0: 7, 1: 0x80000000, 2: 0, 3: 7, 4: 14, 5: 0x22334411, 6: 0x11223344, 7: 0x44332211,
		8: 15, 9: 10, 10: 0, 11: 0x7fffffff, 12: 0x80000000, 13: 0, 14: 2,
		16: f32(float32(math.Copysign(0, -1))), 17: 0, 18: f32(-2.75), 19: f32(2),
		20: 1, 21: 1,
	}
	for k, v := range want {
		if o[k] != v {
			t.Errorf("slot %d = %#x, want %#x", k, o[k], v)
		}
	}
	if !math.IsNaN(float64(math.Float32frombits(o[15]))) {
		t.Errorf("minimum(NaN, 2.75) = %v, want NaN", math.Float32frombits(o[15]))
	}
}

// TestConstantTable reads a read-only global: constant memory.
func TestConstantTable(t *testing.T) {
	m := ir.NewModule("table", ir.AIR64)
	tbl := m.Global("table", ir.RO, ir.Array(4, ir.StoreF32.FType())).Init(ir.List(
		ir.Lit(ir.Float(1.5)), ir.Lit(ir.Float(-2)), ir.Lit(ir.Float(0.25)), ir.Lit(ir.Float(8))))
	f := kernel(m, "lookup")
	out := f.ParamPtr("out")
	b := f.Entry()
	g := gid(b)
	e := b.F32.Load(at(b, b.Ptr.GetAddr(tbl), b.I32.And(g, b.I32.Const(3))))
	b.F32.Store(b.F32.Mul(e, b.F32.UCvtI32(g)), at(b, out, g))
	b.Return()

	r := run(t, m, "lookup", check.Grid1D(32, 32), check.Out(4*32))
	table := []float32{1.5, -2, 0.25, 8}
	for k, v := range r.Floats(0) {
		if want := table[k%4] * float32(k); v != want {
			t.Fatalf("out[%d] = %v, want %v", k, v, want)
		}
	}
}

// TestSpaceConflict is a pointer that could be device or threadgroup
// memory: refused by name.
func TestSpaceConflict(t *testing.T) {
	m := ir.NewModule("conflict", ir.AIR64)
	tile := m.Global("tile", ir.Shared, ir.StoreI32.FType())
	f := kernel(m, "k")
	p := f.ParamPtr("p")
	c := f.ParamI1("c")
	b := f.Entry()
	q := b.Ptr.Select(c, p, b.Ptr.GetAddr(tile))
	b.I32.Store(b.I32.Const(1), q)
	b.Return()
	_, err := airlower.Lower(m, airlower.Options{})
	if err == nil || !strings.Contains(err.Error(), "AIR has no generic pointer") {
		t.Fatalf("error %v; want the space conflict named", err)
	}
}

func TestNoF64(t *testing.T) {
	m := ir.NewModule("f64", ir.AIR64)
	f := kernel(m, "k")
	p := f.ParamPtr("p")
	b := f.Entry()
	b.F64.Store(b.F64.Const(1), p)
	b.Return()
	_, err := airlower.Lower(m, airlower.Options{})
	if err == nil || !strings.Contains(err.Error(), "no floating point wider than 32 bits") {
		t.Fatalf("error %v; want f64 refused", err)
	}
}

var _ = am.Float

// TestBindings is a kernel whose frontend names its bindings: the output
// is [[buffer(1)]], and the input [[buffer(0)]] in constant memory.
func TestBindings(t *testing.T) {
	m := ir.NewModule("bind", ir.AIR64)
	f := kernel(m, "scale").Meta(
		ir.Attached("binding", ir.MInt(1), ir.MInt(0)),
		ir.Attached("param_space", ir.MIdent("device"), ir.MIdent("constant")))
	out := f.ParamPtr("out")
	in := f.ParamPtr("in")
	b := f.Entry()
	g := gid(b)
	b.F32.Store(b.F32.Mul(b.F32.Load(at(b, in, g)), b.F32.Const(3)), at(b, out, g))
	b.Return()

	r := run(t, m, "scale", check.Grid1D(4, 4), check.Floats(1, 2, 3, 4), check.Out(16))
	for k, v := range r.Floats(1) {
		if want := float32(3 * (k + 1)); v != want {
			t.Fatalf("out[%d] = %v, want %v", k, v, want)
		}
	}
}

// TestPromotedPointer is a pointer kept in a local, as a C++ frontend
// lowers one: a frame slot holding the address of threadgroup memory,
// which promotion turns into a value whose space inference can see.
func TestPromotedPointer(t *testing.T) {
	m := ir.NewModule("slot", ir.AIR64)
	tile := m.Global("tile", ir.Shared, ir.Array(32, ir.StoreI32.FType()))
	f := kernel(m, "k")
	out := f.ParamPtr("out")
	b := f.Entry()
	slot := b.Ptr.Alloc(8, 8)
	b.Ptr.Store(b.Ptr.GetAddr(tile), slot)
	lid := b.I32.WorkitemID(ir.X)
	q := b.Ptr.Load(slot)
	b.I32.Store(b.I32.Mul(lid, b.I32.Const(10)), at(b, q, lid))
	b.Barrier()
	g := gid(b)
	b.I32.Store(b.I32.Load(at(b, b.Ptr.Load(slot), b.I32.Xor(lid, b.I32.Const(1)))), at(b, out, g))
	b.Return()

	r := run(t, m, "k", check.Grid1D(32, 32), check.Out(4*32))
	for k, v := range r.Uint32s(0) {
		if want := uint32(10 * (k ^ 1)); v != want {
			t.Fatalf("out[%d] = %v, want %v", k, v, want)
		}
	}
}

// TestBuiltinParams is a kernel whose frontend binds MSL's built-in
// arguments by name: the grid position as a uint and as a uint2, and the
// threadgroup's size, over a grid the threadgroups do not divide.
func TestBuiltinParams(t *testing.T) {
	m := ir.NewModule("builtins", ir.AIR64)
	pair := m.Struct("uint2").Field("x", ir.StoreI32.FType()).Field("y", ir.StoreI32.FType())
	f := kernel(m, "k").Meta(
		ir.Attached("binding", ir.MInt(0), ir.MIdent("-"), ir.MIdent("-"), ir.MIdent("-")),
		ir.Attached("param_builtin", ir.MIdent("-"),
			ir.MStr("thread_position_in_grid uint"),
			ir.MStr("thread_position_in_grid uint2"),
			ir.MStr("threads_per_threadgroup uint")))
	out := f.ParamPtr("out")
	gid := f.ParamI32("gid")
	pos := f.ParamPtr("pos", ir.ByVal(pair))
	tpt := f.ParamI32("tpt")
	b := f.Entry()
	x := b.I32.Load(pos)
	y := b.I32.Load(b.Ptr.Add(pos, b.I64.Const(4)))
	v := b.I32.Add(b.I32.Mul(tpt, b.I32.Const(1000)), b.I32.Add(b.I32.Mul(y, b.I32.Const(100)), x))
	b.I32.Store(v, at(b, out, gid))
	b.Return()

	r := run(t, m, "k", check.Grid1D(10, 4), check.Out(4*10))
	for k, v := range r.Uint32s(0) {
		t.Logf("out[%d] = %d", k, v)
		if v%1000 != uint32(k) {
			t.Fatalf("out[%d] = %d: the position is not %d", k, v, k)
		}
	}
}

// TestPartialGroup is VIR's workgroup size in the last group of a grid
// the groups do not divide: the size dispatched, as CUDA's blockDim is,
// so that the grid position computed from it is right there too.
func TestPartialGroup(t *testing.T) {
	m := ir.NewModule("partial", ir.AIR64)
	f := kernel(m, "k")
	out := f.ParamPtr("out")
	b := f.Entry()
	g := gid(b)
	b.I32.Store(b.I32.Add(b.I32.Mul(b.I32.WorkgroupSize(ir.X), b.I32.Const(1000)), g), at(b, out, g))
	b.Return()

	r := run(t, m, "k", check.Grid1D(10, 4), check.Out(4*10))
	for k, v := range r.Uint32s(0) {
		if want := uint32(4000 + k); v != want {
			t.Fatalf("out[%d] = %d, want %d", k, v, want)
		}
	}
}

// TestBulkMemory is memcpy between device and thread memory and memset of
// device memory: what a frontend makes of a struct copied and zeroed.
func TestBulkMemory(t *testing.T) {
	m := ir.NewModule("bulk", ir.AIR64)
	f := kernel(m, "k")
	out := f.ParamPtr("out")
	in := f.ParamPtr("in")
	b := f.Entry()
	g := gid(b)
	off := b.I64.Shl(b.I64.ZExtI32(g), b.I64.Const(4)) // 16 bytes a thread
	tmp := b.Ptr.Alloc(16, 4)
	b.MemCpy(tmp, b.Ptr.Add(in, off), b.I64.Const(16))
	b.I32.Store(b.I32.Add(b.I32.Load(tmp), b.I32.Const(1)), tmp)
	dst := b.Ptr.Add(out, off)
	b.MemSet(dst, b.I32.Const(0x1ff), b.I64.Const(16))
	b.MemCpy(dst, tmp, b.I64.Const(8))
	b.Return()

	words := make([]uint32, 4*8)
	for i := range words {
		words[i] = uint32(i * 3)
	}
	r := run(t, m, "k", check.Grid1D(8, 8), check.Out(16*8), check.Uint32s(words...))
	got := r.Uint32s(0)
	for k := 0; k < 8; k++ {
		want := []uint32{words[4*k] + 1, words[4*k+1], 0xffffffff, 0xffffffff}
		for j, w := range want {
			if got[4*k+j] != w {
				t.Fatalf("thread %d word %d = %#x, want %#x", k, j, got[4*k+j], w)
			}
		}
	}
}
