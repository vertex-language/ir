package ptx_test

import (
	"math"
	"testing"
	"unsafe"

	"github.com/vertex-language/ir"
)

// Milestone 18: §E as loops, br_table, and asm passed through. One thread
// does all of it, since the bulk verbs are about bytes and not lanes:
// memcpy a source into a depot slot, memmove it forward over itself,
// memset a tail, memcmp two ranges, and switch on the result.
func TestRunBulkTableAsm(t *testing.T) {
	m := ir.NewModule("bulk", ir.NVPTX64)
	fn := m.Func("k").Export().CallConv(ir.Kernel)
	src := fn.ParamPtr("src")
	out := fn.ParamPtr("out")
	cmpOut := fn.ParamPtr("cmp")

	entry := fn.Entry()
	only := fn.Block("only")
	exit := fn.Block("exit")
	c0 := fn.Block("c0")
	c1 := fn.Block("c1")
	c2 := fn.Block("c2")
	dflt := fn.Block("dflt")
	join := fn.Block("join")
	tag := join.ParamI32("tag")

	slot := entry.Ptr.Alloc(32, 16)
	entry.BrIf(entry.I32.Eq(entry.I32.WorkitemID(ir.X), entry.I32.Const(0)), only.To(), exit.To())

	b := only
	b.MemCpy(slot, src, b.I64.Const(16))                                            // slot[0:16] = src[0:16]
	b.MemMove(b.Ptr.Add(slot, b.I64.Const(4)), slot, b.I64.Const(12))               // slot[4:16] = slot[0:12], overlapping forward
	b.MemSet(b.Ptr.Add(slot, b.I64.Const(16)), b.I32.Const(0x1AB), b.I64.Const(16)) // low byte 0xAB
	b.MemCpy(out, slot, b.I64.Const(32))
	cmp := b.MemCmp(src, b.Ptr.Add(src, b.I64.Const(16)), b.I64.Const(16))
	b.I32.Store(cmp, cmpOut)
	// %laneid through inline asm, then a switch on cmp's sign: 0 → 0, <0 → 1, >0 → 2.
	lane := b.Asm("mov.u32 %0, %%laneid;").Out(ir.TypeI32, ir.CStr("=r")).Emit().I32(0)
	sel := b.I32.Select(b.I32.Eq(cmp, b.I32.Const(0)), b.I32.Const(0),
		b.I32.Select(b.I32.SLt(cmp, b.I32.Const(0)), b.I32.Const(1), b.I32.Const(2)))
	b.BrTable(b.I32.Add(sel, lane), []ir.BlockTarget{c0.To(), c1.To(), c2.To()}, dflt.To())
	c0.Br(join.To(c0.I32.Const(100)))
	c1.Br(join.To(c1.I32.Const(200)))
	c2.Br(join.To(c2.I32.Const(300)))
	dflt.Br(join.To(dflt.I32.Const(-1)))
	join.I32.Store(tag, join.Ptr.Add(cmpOut, join.I64.Const(4)))
	join.Return()
	exit.Return()

	c, mod := run(t, m)
	f, err := mod.function("k")
	if err != nil {
		t.Fatal(err)
	}
	in := make([]byte, 32)
	for i := range in {
		in[i] = byte(i + 1)
	}
	in[16+5] = 0xF0 // src[16:32] differs from src[0:16] from byte 0 on
	res := make([]byte, 32)
	cm := make([]int32, 2)
	_, ps := deviceSlice(t, c, in)
	bo, po := deviceSlice(t, c, res)
	bc, pc := deviceSlice(t, c, cm)
	kernel1D(t, c, f, 1, unsafe.Pointer(ps), unsafe.Pointer(po), unsafe.Pointer(pc))
	readBack(t, bo, res)
	readBack(t, bc, cm)

	want := make([]byte, 32)
	copy(want, in[:16])
	copy(want[4:16], append([]byte(nil), want[0:12]...))
	for i := 16; i < 32; i++ {
		want[i] = 0xAB
	}
	for i := range want {
		if res[i] != want[i] {
			t.Fatalf("out[%d] = %#x, want %#x (out = % x)", i, res[i], want[i], res)
		}
	}
	if wantCmp := int32(in[0]) - int32(in[16]); cm[0] != wantCmp {
		t.Fatalf("memcmp = %d, want %d", cm[0], wantCmp)
	}
	if cm[1] != 200 { // cmp < 0, lane 0 → case 1
		t.Fatalf("switch tag = %d, want 200", cm[1])
	}
}

// Milestone 19: the approximate six against Go's exact answers within
// the contract, f64 arithmetic, IEEE minimum/maximum, and the rounding
// four.
func TestRunFloat(t *testing.T) {
	m := ir.NewModule("flt", ir.NVPTX64)
	fn := m.Func("k").Export().CallConv(ir.Kernel)
	in := fn.ParamPtr("in")
	out := fn.ParamPtr("out")
	dout := fn.ParamPtr("dout")
	n := fn.ParamI32("n")
	entry := fn.Entry()
	body := fn.Block("body")
	exit := fn.Block("exit")
	i := globalIndex(entry)
	entry.BrIf(entry.I32.ULt(i, n), body.To(), exit.To())
	b := body
	x := b.F32.Load(elem(b, in, i, 4))
	fs := []ir.F32{
		b.F32.RcpApprox(x), b.F32.RsqrtApprox(x), b.F32.Exp2Approx(x), b.F32.Log2Approx(x),
		b.F32.SinApprox(x), b.F32.CosApprox(x),
		b.F32.Minimum(x, b.F32.Const(2)), b.F32.Maximum(x, b.F32.Const(2)),
		b.F32.MinNum(x, b.F32.Const(2)), b.F32.MaxNum(x, b.F32.Const(2)),
		b.F32.Ceil(x), b.F32.Floor(x), b.F32.Trunc(x), b.F32.Nearest(x),
		b.F32.CopySign(b.F32.Const(3), b.F32.Neg(x)), b.F32.Sqrt(x), b.F32.Div(b.F32.Const(1), x),
	}
	for j, v := range fs {
		b.F32.Store(v, elem(b, out, b.I32.Add(b.I32.Mul(i, b.I32.Const(int64(len(fs)))), b.I32.Const(int64(j))), 4))
	}
	d := b.F64.FCvtF32(x)
	ds := []ir.F64{
		b.F64.Mul(d, b.F64.Const(1.5)), b.F64.Div(d, b.F64.Const(7)), b.F64.FMA(d, d, b.F64.Const(-1)),
		b.F64.Sqrt(b.F64.Abs(d)), b.F64.SCvtI64(b.I64.SCvtSatF64(b.F64.Mul(d, b.F64.Const(1e6)))),
		b.F64.Nearest(d), b.F64.Minimum(d, b.F64.Const(0.25)),
	}
	for j, v := range ds {
		b.F64.Store(v, elem(b, dout, b.I32.Add(b.I32.Mul(i, b.I32.Const(int64(len(ds)))), b.I32.Const(int64(j))), 8))
	}
	b.Return()
	exit.Return()

	c, mod := run(t, m)
	kf, err := mod.function("k")
	if err != nil {
		t.Fatal(err)
	}
	inputs := []float32{0.5, 1, 2, 3.75, 0.1, 7.5, 100, 0.001, 1.5, 2.5, -2.5, 1e-3}
	nf, nd := 17, 7
	res, dres := make([]float32, len(inputs)*nf), make([]float64, len(inputs)*nd)
	_, pi := deviceSlice(t, c, inputs)
	bo, po := deviceSlice(t, c, res)
	bd, pd := deviceSlice(t, c, dres)
	nn := int32(len(inputs))
	kernel1D(t, c, kf, len(inputs), unsafe.Pointer(pi), unsafe.Pointer(po), unsafe.Pointer(pd), unsafe.Pointer(&nn))
	readBack(t, bo, res)
	readBack(t, bd, dres)

	close := func(got, want float32, rel float64) bool {
		if got == want || (got != got && want != want) {
			return true
		}
		return math.Abs(float64(got-want)) <= rel*math.Abs(float64(want))+1e-6
	}
	for k, x := range inputs {
		xd := float64(x)
		want := []struct {
			v   float32
			rel float64
		}{
			{float32(1 / xd), 1e-6}, {float32(1 / math.Sqrt(xd)), 1e-6}, {float32(math.Exp2(xd)), 1e-5}, {float32(math.Log2(xd)), 1e-5},
			{float32(math.Sin(xd)), 1e-4}, {float32(math.Cos(xd)), 1e-4},
			{float32(math.Min(xd, 2)), 0}, {float32(math.Max(xd, 2)), 0},
			{float32(math.Min(xd, 2)), 0}, {float32(math.Max(xd, 2)), 0},
			{float32(math.Ceil(xd)), 0}, {float32(math.Floor(xd)), 0}, {float32(math.Trunc(xd)), 0}, {float32(math.RoundToEven(xd)), 0},
			{float32(math.Copysign(3, -xd)), 0}, {float32(math.Sqrt(xd)), 0}, {1 / x, 0},
		}
		for j, w := range want {
			if got := res[k*nf+j]; !close(got, w.v, w.rel) {
				t.Errorf("input %v: f32 value %d = %v, want %v", x, j, got, w.v)
			}
		}
		dwant := []float64{
			xd * 1.5, xd / 7, math.FMA(xd, xd, -1), math.Sqrt(math.Abs(xd)),
			float64(int64(xd * 1e6)), math.RoundToEven(xd), math.Min(xd, 0.25),
		}
		for j, w := range dwant {
			if got := dres[k*nd+j]; got != w {
				t.Errorf("input %v: f64 value %d = %v, want %v", x, j, got, w)
			}
		}
	}
}

// Milestone 20: the wave verbs. A butterfly reduction across a warp with
// wave_shfl_xor, a ballot of which lanes hold an odd value, any/all, and
// the first lane's value read by every lane.
func TestRunWave(t *testing.T) {
	m := ir.NewModule("wave", ir.NVPTX64)
	fn := m.Func("k").Export().CallConv(ir.Kernel)
	in := fn.ParamPtr("in")
	out := fn.ParamPtr("out")
	entry := fn.Entry()
	b := entry
	tid := b.I32.WorkitemID(ir.X)
	v := b.I32.Load(elem(b, in, tid, 4))
	full := b.I32.Const(-1)
	s := v
	for _, d := range []int64{16, 8, 4, 2, 1} {
		s = b.I32.Add(s, b.I32.WaveShflXor(s, b.I32.Const(d), full))
	}
	odd := b.I32.Ne(b.I32.And(v, b.I32.Const(1)), b.I32.Const(0))
	ballot := b.I64.WaveBallot(odd, full)
	any := b.I1.WaveAny(b.I32.Eq(v, b.I32.Const(7)), full)
	all := b.I1.WaveAll(b.I32.SLt(v, b.I32.Const(1000)), full)
	first := b.I32.WaveReadFirstLane(v)
	up := b.I32.WaveShflUp(v, b.I32.Const(1), full)
	down := b.I32.WaveShflDown(v, b.I32.Const(1), full)
	idx := b.I32.WaveShflIdx(v, b.I32.Const(5), full)
	lane := b.I32.LaneID()
	ws := b.I32.WaveSize()
	vals := []ir.I64{
		b.I64.ZExtI32(s), ballot, b.I64.ZExtI1(any), b.I64.ZExtI1(all), b.I64.ZExtI32(first),
		b.I64.ZExtI32(up), b.I64.ZExtI32(down), b.I64.ZExtI32(idx), b.I64.ZExtI32(lane), b.I64.ZExtI32(ws),
	}
	for j, x := range vals {
		b.I64.Store(x, elem(b, out, b.I32.Add(b.I32.Mul(tid, b.I32.Const(int64(len(vals)))), b.I32.Const(int64(j))), 8))
	}
	b.Return()

	c, mod := run(t, m)
	kf, err := mod.function("k")
	if err != nil {
		t.Fatal(err)
	}
	data := make([]int32, 32)
	for i := range data {
		data[i] = int32(i*3 + 1)
	}
	data[7] = 7
	const nv = 10
	res := make([]int64, 32*nv)
	_, pi := deviceSlice(t, c, data)
	bo, po := deviceSlice(t, c, res)
	if err := c.launch(kf, [3]uint32{1, 1, 1}, [3]uint32{32, 1, 1}, 0, unsafe.Pointer(pi), unsafe.Pointer(po)); err != nil {
		t.Fatal(err)
	}
	readBack(t, bo, res)
	var sum int32
	var wantBallot int64
	for i, v := range data {
		sum += v
		if v&1 != 0 {
			wantBallot |= 1 << i
		}
	}
	for i := 0; i < 32; i++ {
		up := data[i]
		if i > 0 {
			up = data[i-1]
		}
		down := data[i]
		if i < 31 {
			down = data[i+1]
		}
		want := []int64{int64(sum), wantBallot, 1, 1, int64(data[0]), int64(up), int64(down), int64(data[5]), int64(i), 32}
		for j, w := range want {
			if got := res[i*nv+j]; got != w {
				t.Errorf("lane %d: value %d = %d, want %d", i, j, got, w)
			}
		}
	}
}
