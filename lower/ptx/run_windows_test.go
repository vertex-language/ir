package ptx_test

// Kernels run on the device, one per milestone. Each reference is
// computed a different way in Go.

import (
	"math"
	"math/bits"
	"testing"
	"unsafe"

	"github.com/vertex-language/ir"
)

// globalIndex is blockIdx.x*blockDim.x + threadIdx.x.
func globalIndex(b *ir.Block) ir.I32 {
	return b.I32.Add(b.I32.Mul(b.I32.WorkgroupID(ir.X), b.I32.WorkgroupSize(ir.X)), b.I32.WorkitemID(ir.X))
}

// elem is p + i*size, for a 32-bit index.
func elem(b *ir.Block, p ir.Ptr, i ir.I32, size int64) ir.Ptr {
	return b.Ptr.Add(p, b.I64.Mul(b.I64.ZExtI32(i), b.I64.Const(size)))
}

// kernel1D runs f over n work-items in blocks of 256.
func kernel1D(t *testing.T, c *cuda, f uintptr, n int, args ...unsafe.Pointer) {
	t.Helper()
	if err := c.launch(f, [3]uint32{uint32((n + 255) / 256), 1, 1}, [3]uint32{256, 1, 1}, 0, args...); err != nil {
		t.Fatal(err)
	}
}

// Milestone 12: a loop carried through block parameters, and integer
// division. out[i] = sum over k<i of (k / 3 + k % 7), one thread per i.
func TestRunLoopAndDivide(t *testing.T) {
	m := ir.NewModule("loop", ir.NVPTX64)
	fn := m.Func("k").Export().CallConv(ir.Kernel)
	out := fn.ParamPtr("out")
	n := fn.ParamI32("n")

	entry := fn.Entry()
	loop := fn.Block("loop")
	k := loop.ParamI32("k")
	acc := loop.ParamI32("acc")
	body := fn.Block("body")
	done := fn.Block("done")
	exit := fn.Block("exit")

	i := globalIndex(entry)
	entry.BrIf(entry.I32.ULt(i, n), loop.To(entry.I32.Const(0), entry.I32.Const(0)), exit.To())

	loop.BrIf(loop.I32.SLt(k, i), body.To(), done.To())

	q := body.I32.SDiv(k, body.I32.Const(3))
	r := body.I32.URem(k, body.I32.Const(7))
	body.Br(loop.To(body.I32.Add(k, body.I32.Const(1)), body.I32.Add(acc, body.I32.Add(q, r))))

	done.I32.Store(acc, elem(done, out, i, 4))
	done.Return()
	exit.Return()

	c, mod := run(t, m)
	f, err := mod.function("k")
	if err != nil {
		t.Fatal(err)
	}
	const count = 777
	res := make([]int32, count)
	bo, po := deviceSlice(t, c, res)
	nn := int32(count)
	kernel1D(t, c, f, count, unsafe.Pointer(po), unsafe.Pointer(&nn))
	readBack(t, bo, res)
	for i := range res {
		var want int32
		for k := int32(0); k < int32(i); k++ {
			want += k/3 + k%7
		}
		if res[i] != want {
			t.Fatalf("out[%d] = %d, want %d", i, res[i], want)
		}
	}
}

// Milestone 13: sub-width memory, the local depot, and globals. Each
// thread reads a byte and a halfword, sign- and zero-extends them, keeps
// a running value in a ptr.alloc slot across a store and a reload, adds
// a constant from an ro global and counts in an rw one.
func TestRunMemoryAndGlobals(t *testing.T) {
	m := ir.NewModule("mem", ir.NVPTX64)
	table := m.Global("table", ir.RO, ir.Array(4, ir.StoreI32.FType())).
		Init(ir.List(ir.Lit(ir.Int(10)), ir.Lit(ir.Int(20)), ir.Lit(ir.Int(30)), ir.Lit(ir.Int(40))))
	counter := m.Global("counter", ir.RW, ir.StoreI32.FType()).Export()

	fn := m.Func("k").Export().CallConv(ir.Kernel)
	bytesIn := fn.ParamPtr("b")
	halves := fn.ParamPtr("h")
	out := fn.ParamPtr("out")
	n := fn.ParamI32("n")

	entry := fn.Entry()
	body := fn.Block("body")
	exit := fn.Block("exit")
	slot := entry.Ptr.Alloc(8, 8)
	i := globalIndex(entry)
	entry.BrIf(entry.I32.ULt(i, n), body.To(), exit.To())

	sb := body.I32.SLoad8(elem(body, bytesIn, i, 1))
	ub := body.I32.ULoad8(elem(body, bytesIn, i, 1))
	sh := body.I64.SLoad16(elem(body, halves, i, 2))
	uh := body.I64.ULoad16(elem(body, halves, i, 2))
	sum := body.I64.Add(body.I64.SExtI32(body.I32.Add(sb, ub)), body.I64.Add(sh, uh))
	body.I64.Store(sum, slot)
	back := body.I64.Load(slot)
	idx := body.I32.And(i, body.I32.Const(3))
	tv := body.I32.Load(elem(body, body.Ptr.GetAddr(table), idx, 4))
	total := body.I64.Add(back, body.I64.SExtI32(tv))
	body.I64.Store(total, elem(body, out, i, 8))
	body.I32.AtomicRmwAdd(body.I32.Const(1), body.Ptr.GetAddr(counter), ir.Monotonic, ir.DeviceScope)
	body.I32.Store16(body.I32.Const(-2), elem(body, halves, i, 2)) // overwrite, checked on the host
	body.Return()
	exit.Return()

	c, mod := run(t, m)
	f, err := mod.function("k")
	if err != nil {
		t.Fatal(err)
	}
	const count = 300
	bs, hs, res := make([]int8, count), make([]int16, count), make([]int64, count)
	for i := range bs {
		bs[i] = int8(i*7 - 100)
		hs[i] = int16(i*131 - 20000)
	}
	_, pb := deviceSlice(t, c, bs)
	bh, ph := deviceSlice(t, c, hs)
	bo, po := deviceSlice(t, c, res)
	nn := int32(count)
	kernel1D(t, c, f, count, unsafe.Pointer(pb), unsafe.Pointer(ph), unsafe.Pointer(po), unsafe.Pointer(&nn))
	readBack(t, bo, res)
	tab := []int64{10, 20, 30, 40}
	for i := range res {
		want := int64(int32(bs[i])+int32(uint8(bs[i]))) + int64(hs[i]) + int64(uint16(hs[i])) + tab[i&3]
		if res[i] != want {
			t.Fatalf("out[%d] = %d, want %d", i, res[i], want)
		}
	}
	readBack(t, bh, hs)
	for i, h := range hs {
		if h != -2 {
			t.Fatalf("halves[%d] = %d after store16, want -2", i, h)
		}
	}
}

// Milestone 14: shared memory and barrier. A workgroup reduction: each
// block sums 256 floats through a tree in a shared tile and thread 0
// writes the block's sum.
func TestRunSharedReduction(t *testing.T) {
	m := ir.NewModule("reduce", ir.NVPTX64)
	tile := m.Global("tile", ir.Shared, ir.Array(256, ir.StoreF32.FType())).Align(16)

	fn := m.Func("k").Export().CallConv(ir.Kernel)
	in := fn.ParamPtr("in")
	out := fn.ParamPtr("out")

	entry := fn.Entry()
	loop := fn.Block("loop")
	stride := loop.ParamI32("stride")
	add := fn.Block("add")
	next := fn.Block("next")
	done := fn.Block("done")
	write := fn.Block("write")
	exit := fn.Block("exit")

	tid := entry.I32.WorkitemID(ir.X)
	i := globalIndex(entry)
	mine := elem(entry, entry.Ptr.GetAddr(tile), tid, 4)
	entry.F32.Store(entry.F32.Load(elem(entry, in, i, 4)), mine)
	entry.Barrier()
	entry.Br(loop.To(entry.I32.Const(128)))

	// for (stride = 128; stride > 0; stride >>= 1) { if (tid < stride) tile[tid] += tile[tid+stride]; barrier }
	loop.BrIf(loop.I32.ULt(loop.I32.Const(0), stride), add.To(), done.To())
	add.BrIf(add.I32.ULt(tid, stride), next.To(), next.To())
	// The add is predicated by a branch to the same block either way;
	// the real work is in next, guarded again, to exercise an edge that
	// carries nothing on both arms.
	partner := next.I32.Add(tid, stride)
	take := next.I32.ULt(tid, stride)
	a := next.F32.Load(elem(next, next.Ptr.GetAddr(tile), tid, 4))
	b := next.F32.Load(elem(next, next.Ptr.GetAddr(tile), next.I32.Select(take, partner, tid), 4))
	sum := next.F32.Select(take, next.F32.Add(a, b), a)
	next.Barrier()
	next.F32.Store(sum, elem(next, next.Ptr.GetAddr(tile), tid, 4))
	next.Barrier()
	next.Br(loop.To(next.I32.UShr(stride, next.I32.Const(1))))

	done.BrIf(done.I32.Eq(tid, done.I32.Const(0)), write.To(), exit.To())
	write.F32.Store(write.F32.Load(write.Ptr.GetAddr(tile)), elem(write, out, write.I32.WorkgroupID(ir.X), 4))
	write.Return()
	exit.Return()

	c, mod := run(t, m)
	f, err := mod.function("k")
	if err != nil {
		t.Fatal(err)
	}
	const blocks = 7
	data := make([]float32, blocks*256)
	for i := range data {
		data[i] = float32(i%13) - 6
	}
	sums := make([]float32, blocks)
	_, pin := deviceSlice(t, c, data)
	bo, po := deviceSlice(t, c, sums)
	kernel1D(t, c, f, blocks*256, unsafe.Pointer(pin), unsafe.Pointer(po))
	readBack(t, bo, sums)
	for b := 0; b < blocks; b++ {
		var want float32
		for i := 0; i < 256; i++ {
			want += data[b*256+i]
		}
		if sums[b] != want {
			t.Fatalf("block %d sum = %v, want %v", b, sums[b], want)
		}
	}
}

// Milestone 15: atomics. A histogram of bytes into 256 device-scope
// counters, a running maximum, and a compare-and-swap that claims a
// slot once.
func TestRunAtomics(t *testing.T) {
	m := ir.NewModule("atom", ir.NVPTX64)
	fn := m.Func("k").Export().CallConv(ir.Kernel)
	in := fn.ParamPtr("in")
	hist := fn.ParamPtr("hist")
	maxp := fn.ParamPtr("max")
	claim := fn.ParamPtr("claim")
	n := fn.ParamI32("n")

	entry := fn.Entry()
	body := fn.Block("body")
	exit := fn.Block("exit")
	i := globalIndex(entry)
	entry.BrIf(entry.I32.ULt(i, n), body.To(), exit.To())

	v := body.I32.ULoad8(elem(body, in, i, 1))
	body.I32.AtomicRmwAdd(body.I32.Const(1), elem(body, hist, v, 4), ir.Monotonic, ir.DeviceScope)
	body.I32.AtomicRmwUMax(body.I32.Mul(v, i), maxp, ir.AcqRel, ir.DeviceScope)
	// claim = i if claim was still -1; exactly one thread wins.
	body.I32.AtomicCas(body.I32.Const(-1), i, claim, ir.SeqCst, ir.Monotonic, ir.DeviceScope)
	body.Return()
	exit.Return()

	c, mod := run(t, m)
	f, err := mod.function("k")
	if err != nil {
		t.Fatal(err)
	}
	const count = 4096
	data := make([]uint8, count)
	for i := range data {
		data[i] = uint8(i*37 + i/7)
	}
	hs := make([]uint32, 256)
	mx := []uint32{0}
	cl := []int32{-1}
	_, pin := deviceSlice(t, c, data)
	bh, ph := deviceSlice(t, c, hs)
	bm, pm := deviceSlice(t, c, mx)
	bc, pc := deviceSlice(t, c, cl)
	nn := int32(count)
	kernel1D(t, c, f, count, unsafe.Pointer(pin), unsafe.Pointer(ph), unsafe.Pointer(pm), unsafe.Pointer(pc), unsafe.Pointer(&nn))
	readBack(t, bh, hs)
	readBack(t, bm, mx)
	readBack(t, bc, cl)
	wantH := make([]uint32, 256)
	var wantM uint32
	for i, v := range data {
		wantH[v]++
		if p := uint32(v) * uint32(i); p > wantM {
			wantM = p
		}
	}
	for v := range hs {
		if hs[v] != wantH[v] {
			t.Fatalf("hist[%d] = %d, want %d", v, hs[v], wantH[v])
		}
	}
	if mx[0] != wantM {
		t.Fatalf("max = %d, want %d", mx[0], wantM)
	}
	if cl[0] < 0 || cl[0] >= count {
		t.Fatalf("claim = %d, want a thread index", cl[0])
	}
}

// Milestone 16: device functions. A .func with two results called
// directly, and again through a pointer with a func typedef.
func TestRunCalls(t *testing.T) {
	m := ir.NewModule("calls", ir.NVPTX64)

	// divmod(a, b) -> (a/b, a%b)
	dm := m.Func("divmod").Internal()
	da := dm.ParamI32("a")
	db := dm.ParamI32("b")
	dm.ReturnsI32().ReturnsI32()
	dm.Entry().Return(dm.Entry().I32.UDiv(da, db), dm.Entry().I32.URem(da, db))

	// scale(x, f) -> x*f, called through a pointer
	sc := m.Func("scale")
	sx := sc.ParamF32("x")
	sf := sc.ParamF32("f")
	sc.ReturnsF32()
	sc.Entry().Return(sc.Entry().F32.Mul(sx, sf))
	scaleT := m.FuncType("scale_t", ir.NewSig().Param(ir.TypeF32).Param(ir.TypeF32).Ret(ir.TypeF32))

	fn := m.Func("k").Export().CallConv(ir.Kernel)
	out := fn.ParamPtr("out")
	fout := fn.ParamPtr("fout")
	n := fn.ParamI32("n")
	entry := fn.Entry()
	body := fn.Block("body")
	exit := fn.Block("exit")
	i := globalIndex(entry)
	entry.BrIf(entry.I32.ULt(i, n), body.To(), exit.To())

	r := body.Call(dm, body.I32.Add(i, body.I32.Const(1000)), body.I32.Const(7))
	body.I32.Store(body.I32.Add(body.I32.Mul(r.I32(0), body.I32.Const(100)), r.I32(1)), elem(body, out, i, 4))
	fp := body.Ptr.GetAddr(sc)
	x := body.F32.SCvtI32(i)
	s := body.CallInd(fp, scaleT, x, body.F32.Const(0.5))
	body.F32.Store(s.F32(0), elem(body, fout, i, 4))
	body.Return()
	exit.Return()

	c, mod := run(t, m)
	f, err := mod.function("k")
	if err != nil {
		t.Fatal(err)
	}
	const count = 500
	res, fres := make([]int32, count), make([]float32, count)
	bo, po := deviceSlice(t, c, res)
	bf, pf := deviceSlice(t, c, fres)
	nn := int32(count)
	kernel1D(t, c, f, count, unsafe.Pointer(po), unsafe.Pointer(pf), unsafe.Pointer(&nn))
	readBack(t, bo, res)
	readBack(t, bf, fres)
	for i := range res {
		a := uint32(i + 1000)
		if want := int32(a/7*100 + a%7); res[i] != want {
			t.Fatalf("out[%d] = %d, want %d", i, res[i], want)
		}
		if want := float32(i) * 0.5; fres[i] != want {
			t.Fatalf("fout[%d] = %v, want %v", i, fres[i], want)
		}
	}
}

// Milestone 17: the scalar ALU — conversions, overflow predicates,
// shifts and rotates, bit counting, selects and compares — one kernel
// computing sixteen answers per thread, checked against Go.
func TestRunScalarALU(t *testing.T) {
	m := ir.NewModule("alu", ir.NVPTX64)
	fn := m.Func("k").Export().CallConv(ir.Kernel)
	in := fn.ParamPtr("in")
	out := fn.ParamPtr("out")
	n := fn.ParamI32("n")
	entry := fn.Entry()
	body := fn.Block("body")
	exit := fn.Block("exit")
	i := globalIndex(entry)
	entry.BrIf(entry.I32.ULt(i, n), body.To(), exit.To())

	b := body
	a := b.I32.Load(elem(b, in, i, 4))
	a64 := b.I64.SExtI32(a)
	f := b.F32.SCvtI32(a)
	d := b.F64.FCvtF32(f)
	var vals []ir.I64
	push := func(v ir.I64) { vals = append(vals, v) }
	push(b.I64.ZExtI32(b.I32.RotL(a, b.I32.Const(5))))
	push(b.I64.ZExtI32(b.I32.RotR(a, b.I32.Const(70)))) // 70 mod 32
	push(b.I64.RotL(a64, b.I64.Const(13)))
	push(b.I64.ZExtI32(b.I32.Shl(a, b.I32.Const(33)))) // 33 mod 32 = 1
	push(b.I64.ZExtI32(b.I32.SShr(a, b.I32.Const(3))))
	push(b.I64.ZExtI32(b.I32.UShr(a, b.I32.Const(3))))
	push(b.I64.ZExtI32(b.I32.Clz(a)))
	push(b.I64.ZExtI32(b.I32.Ctz(a)))
	push(b.I64.Popcnt(a64))
	push(b.I64.ZExtI32(b.I32.Bswap(a)))
	push(b.I64.Bswap(a64))
	push(b.I64.ZExtI1(b.I32.SAddO(a, a)))
	push(b.I64.ZExtI1(b.I32.SMulO(a, b.I32.Const(3))))
	push(b.I64.ZExtI1(b.I32.UAddO(a, b.I32.Const(-1))))
	push(b.I64.ZExtI32(b.I32.SMulHi(a, b.I32.Const(0x10000))))
	push(b.I64.ZExtI32(b.I32.Select(b.I32.SLt(a, b.I32.Const(0)), b.I32.Neg(a), a)))
	push(b.I64.SCvtSatF32(b.F32.Mul(f, b.F32.Const(1e30))))
	push(b.I64.ZExtI32(b.I32.UCvtSatF64(d)))
	push(b.I64.SCvtF64(b.F64.Floor(b.F64.Div(d, b.F64.Const(3)))))
	push(b.I64.BitcastF64(b.F64.Sqrt(b.F64.Abs(d))))
	push(b.I64.ZExtI32(b.I32.BitcastF32(b.F32.FMA(f, f, b.F32.Const(1)))))
	push(b.I64.ZExtI1(b.I1.Xor(b.F32.Lt(f, b.F32.Const(100)), b.F32.Uno(f, f))))
	push(b.I64.ZExtI32(b.I32.WrapI64(b.I64.Mul(a64, a64))))
	push(b.I64.UMulHi(a64, b.I64.Const(0x123456789)))
	for j, v := range vals {
		b.I64.Store(v, elem(b, out, b.I32.Add(b.I32.Mul(i, b.I32.Const(int64(len(vals)))), b.I32.Const(int64(j))), 8))
	}
	b.Return()
	exit.Return()

	c, mod := run(t, m)
	kf, err := mod.function("k")
	if err != nil {
		t.Fatal(err)
	}
	inputs := []int32{0, 1, -1, 7, -7, 12345, -12345, 0x7fffffff, -0x80000000, 0x12345678, -0x12345678, 1 << 20, 3, 100, 99, 1000000}
	const nv = 24
	res := make([]int64, len(inputs)*nv)
	_, pi := deviceSlice(t, c, inputs)
	bo, po := deviceSlice(t, c, res)
	nn := int32(len(inputs))
	kernel1D(t, c, kf, len(inputs), unsafe.Pointer(pi), unsafe.Pointer(po), unsafe.Pointer(&nn))
	readBack(t, bo, res)

	for k, a := range inputs {
		u := uint32(a)
		a64 := int64(a)
		f := float32(a)
		d := float64(f)
		want := []int64{
			int64(bits.RotateLeft32(u, 5)),
			int64(bits.RotateLeft32(u, -(70 % 32))),
			int64(bits.RotateLeft64(uint64(a64), 13)),
			int64(u << 1),
			int64(uint32(a >> 3)),
			int64(u >> 3),
			int64(bits.LeadingZeros32(u)),
			int64(bits.TrailingZeros32(u)),
			int64(bits.OnesCount64(uint64(a64))),
			int64(bits.ReverseBytes32(u)),
			int64(bits.ReverseBytes64(uint64(a64))),
			b2i(int64(a)+int64(a) != int64(int32(a+a))),
			b2i(int64(a)*3 != int64(int32(a*3))),
			b2i(uint64(u)+0xffffffff > 0xffffffff),
			int64(uint32((int64(a) * 0x10000) >> 32)),
			int64(uint32(abs32(a))),
			satS64(float64(f * 1e30)),
			int64(satU32(d)),
			int64(math.Floor(d / 3)),
			int64(math.Float64bits(math.Sqrt(math.Abs(d)))),
			int64(math.Float32bits(float32(math.FMA(float64(f), float64(f), 1)))),
			b2i(f < 100),
			int64(uint32(a64 * a64)),
			int64(umulhi64(uint64(a64), 0x123456789)),
		}
		for j, w := range want {
			if got := res[k*nv+j]; got != w {
				t.Errorf("input %d (%d): value %d = %#x, want %#x", k, a, j, got, w)
			}
		}
	}
}

func b2i(b bool) int64 {
	if b {
		return 1
	}
	return 0
}

func abs32(a int32) int32 {
	if a < 0 {
		return -a
	}
	return a
}

func satS64(v float64) int64 {
	switch {
	case v != v:
		return 0
	case v >= 9223372036854775807:
		return math.MaxInt64
	case v <= -9223372036854775808:
		return math.MinInt64
	}
	return int64(v)
}

func satU32(v float64) uint32 {
	switch {
	case v != v, v <= 0:
		return 0
	case v >= 4294967295:
		return math.MaxUint32
	}
	return uint32(v)
}

func umulhi64(a, b uint64) uint64 {
	hi, _ := bits.Mul64(a, b)
	return hi
}
