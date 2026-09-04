package i386_test

// §G's indirect call, §G2's computed control flow, and §D3's dynamic frame.

import (
	"fmt"
	"testing"

	"github.com/vertex-language/ir"
)

func TestRunIndirectCall(t *testing.T) {
	m := ir.NewModule("t", ir.I386Linux)
	ft := m.FuncType("binop", ir.NewSig().Param(ir.TypeI32).Param(ir.TypeI32).Ret(ir.TypeI32))

	fn := m.Func("apply2").Export()
	p := fn.ParamPtr("p")
	a := fn.ParamI32("a")
	b := fn.ParamI32("b")
	fn.ReturnsI32()
	entry := fn.Entry()
	r := entry.CallInd(p, ft, a, b).Value(0).(ir.I32)
	// b live across the call, so the callee's address and a live value
	// cannot share a register.
	entry.Return(entry.I32.Add(r, b))

	wantOK(t, m, `
static int add(int x, int y) { return x + y; }
static int mul(int x, int y) { return x * y; }
int apply2(int (*)(int,int), int, int);
static void body(void) {
    chk32("indirect add", (unsigned)apply2(add, 10, 4), 18u);
    chk32("indirect mul", (unsigned)apply2(mul, 10, 4), 44u);
}
`)
}

func TestRunBrTable(t *testing.T) {
	m := ir.NewModule("t", ir.I386Linux)
	fn := m.Func("pick").Export()
	sel := fn.ParamI32("sel")
	fn.ReturnsI32()
	entry := fn.Entry()

	out := fn.Block("out")
	r := out.ParamI32("r")
	cases := make([]ir.BlockTarget, 5)
	for i := range cases {
		b := fn.Block(fmt.Sprintf("case%d", i))
		b.Br(out.To(b.I32.Const(int64(i+1) * 11)))
		cases[i] = b.To()
	}
	dflt := fn.Block("dflt")
	dflt.Br(out.To(dflt.I32.Const(-1)))

	entry.BrTable(sel, cases, dflt.To())
	out.Return(r)

	wantOK(t, m, `
int pick(int);
static void body(void) {
    for (int i = 0; i < 5; i++) chk32("pick", (unsigned)pick(i), (unsigned)((i + 1) * 11));
    chk32("past end", (unsigned)pick(5), 0xffffffffu);
    chk32("negative", (unsigned)pick(-1), 0xffffffffu);
    chk32("far",      (unsigned)pick(1000), 0xffffffffu);
    chk32("int min",  (unsigned)pick(-2147483647 - 1), 0xffffffffu);
}
`)
}

func TestRunBrInd(t *testing.T) {
	m := ir.NewModule("t", ir.I386Linux)
	fn := m.Func("jgoto").Export()
	which := fn.ParamI32("which")
	fn.ReturnsI32()
	entry := fn.Entry()

	one := fn.Block("one")
	two := fn.Block("two")
	out := fn.Block("out")
	r := out.ParamI32("r")

	slots := entry.Ptr.Alloc(8, 4)
	entry.Ptr.Store(entry.Ptr.BlockAddr(one), slots)
	entry.Ptr.Store(entry.Ptr.BlockAddr(two),
		entry.Ptr.Add(slots, entry.I64.Const(4)))
	target := entry.Ptr.Load(entry.Ptr.Add(slots,
		entry.I64.Mul(entry.I64.SExtI32(which), entry.I64.Const(4))))
	entry.BrInd(target, one, two)

	one.Br(out.To(one.I32.Const(111)))
	two.Br(out.To(two.I32.Const(222)))
	out.Return(r)

	wantOK(t, m, `
int jgoto(int);
static void body(void) {
    chk32("goto one", (unsigned)jgoto(0), 111u);
    chk32("goto two", (unsigned)jgoto(1), 222u);
}
`)
}

func TestRunDynamicFrame(t *testing.T) {
	m := ir.NewModule("t", ir.I386Linux)
	sink := m.ImportFunc("sink", ir.NewSig().Param(ir.TypeI32).Ret(ir.TypeI32))

	fn := m.Func("dyn").Export()
	n := fn.ParamI32("n")
	fn.ReturnsI32()
	entry := fn.Entry()

	buf := entry.Ptr.Alloca(entry.I64.Mul(entry.I64.SExtI32(n), entry.I64.Const(4)), 4)

	loop := fn.Block("loop")
	body := fn.Block("body")
	out := fn.Block("out")
	i := loop.ParamI32("i")

	entry.Br(loop.To(entry.I32.Const(0)))
	loop.BrIf(loop.I32.SLt(i, n), body.To(), out.To())
	slot := body.Ptr.Add(buf, body.I64.Mul(body.I64.SExtI32(i), body.I64.Const(4)))
	body.I32.Store(body.I32.Mul(i, body.I32.Const(3)), slot)
	body.Br(loop.To(body.I32.Add(i, body.I32.Const(1))))

	// A call after the allocation, whose outgoing area is at the bottom of
	// a frame the allocation moved.
	out.Call(sink, out.I32.Const(0))
	total := out.I32.Const(0)
	for k := 0; k < 4; k++ {
		p := out.Ptr.Add(buf, out.I64.Const(int64(k)*4))
		total = out.I32.Add(total, out.I32.Load(p))
	}
	out.Return(total)

	wantOK(t, m, `
int sink(int x) { return x; }
int dyn(int);
static void body(void) {
    /* 0*3 + 1*3 + 2*3 + 3*3 */
    chk32("alloca", (unsigned)dyn(64), 18u);
}
`)
}

// stacksave and stackrestore around a loop of allocations, which is what
// keeps the frame from growing without bound.
func TestRunStackSaveRestore(t *testing.T) {
	m := ir.NewModule("t", ir.I386Linux)
	fn := m.Func("churn").Export()
	n := fn.ParamI32("n")
	fn.ReturnsI32()
	entry := fn.Entry()

	loop := fn.Block("loop")
	body := fn.Block("body")
	out := fn.Block("out")
	i := loop.ParamI32("i")
	acc := loop.ParamI32("acc")

	entry.Br(loop.To(entry.I32.Const(0), entry.I32.Const(0)))
	loop.BrIf(loop.I32.SLt(i, n), body.To(), out.To())

	mark := body.Ptr.StackSave()
	buf := body.Ptr.Alloca(body.I64.Const(1024), 4)
	body.I32.Store(body.I32.Add(acc, i), buf)
	next := body.I32.Load(buf)
	body.Ptr.StackRestore(mark)
	body.Br(loop.To(body.I32.Add(i, body.I32.Const(1)), next))

	out.Return(acc)

	wantOK(t, m, `
int churn(int);
static void body(void) {
    /* Without the restore this walks off the stack long before it ends. */
    chk32("churn", (unsigned)churn(20000), (unsigned)(19999 * 20000 / 2));
}
`)
}

// §D3's frameaddr and returnaddr, level zero.
func TestRunFrameAddr(t *testing.T) {
	m := ir.NewModule("t", ir.I386Linux)
	fn := m.Func("whereami").Export()
	fn.ReturnsI32()
	entry := fn.Entry()
	// The return address, compared against what the caller knows.
	entry.Return(entry.I32.WrapI64(entry.I64.FromPtr(entry.Ptr.ReturnAddr())))

	fp := m.Func("myframe").Export()
	fp.ReturnsI32()
	e2 := fp.Entry()
	e2.Return(e2.I32.WrapI64(e2.I64.FromPtr(e2.Ptr.FrameAddr())))

	wantOK(t, m, `
int whereami(void), myframe(void);
static void body(void) {
    unsigned ra = (unsigned)whereami();
    /* The return address is inside this function, which is inside the
       kernel image: a loose but real check that it is an address at all. */
    if (ra < 0x100000u || ra > 0x200000u) {
        print("returnaddr looks wrong: "); printx32(ra); print("\n"); 
    }
    unsigned f1 = (unsigned)myframe();
    if (f1 < 0x100000u) { print("frameaddr looks wrong\n"); }
    chk32("both answered", (ra >= 0x100000u && f1 >= 0x100000u), 1u);
}
`)
}

// §D3's two zeroed allocations, which are the memset §E supplies.
//
// The check is a sum over storage that was dirtied first: the same frame slot
// is written non-zero on one call and read back on the next, so a zeroed
// alloc that emitted nothing would answer with the previous call's bytes
// rather than with zero.
func TestRunZeroedAllocations(t *testing.T) {
	m := ir.NewModule("t", ir.I386Linux)

	// dirty fills its slot with a pattern and returns nothing useful; sum
	// takes a zeroed slot of the same size at the same depth and adds it
	// up, which is zero only if the zeroing happened.
	dirty := m.Func("dirty").Export()
	dirty.ReturnsI32()
	de := dirty.Entry()
	dbuf := de.Ptr.Alloc(64, 4)
	for k := 0; k < 16; k++ {
		de.I32.Store(de.I32.Const(int64(0x5a5a0000+k)), de.Ptr.Add(dbuf, de.I64.Const(int64(k)*4)))
	}
	de.Return(de.I32.Load(dbuf))

	sum := m.Func("zsum").Export()
	sum.ReturnsI32()
	se := sum.Entry()
	zbuf := se.Ptr.Alloc(64, 4, ir.Zeroed)
	total := se.I32.Const(0)
	for k := 0; k < 16; k++ {
		total = se.I32.Add(total, se.I32.Load(se.Ptr.Add(zbuf, se.I64.Const(int64(k)*4))))
	}
	se.Return(total)

	// The dynamic form, whose count is a value rather than a constant.
	dyn := m.Func("zdyn").Export()
	n := dyn.ParamI32("n")
	dyn.ReturnsI32()
	ye := dyn.Entry()
	ybuf := ye.Ptr.Alloca(ye.I64.SExtI32(n), 4, ir.Zeroed)
	dtotal := ye.I32.Const(0)
	for k := 0; k < 16; k++ {
		dtotal = ye.I32.Add(dtotal, ye.I32.Load(ye.Ptr.Add(ybuf, ye.I64.Const(int64(k)*4))))
	}
	ye.Return(dtotal)

	wantOK(t, m, `
int dirty(void), zsum(void), zdyn(int);
static void body(void) {
    dirty();
    chk32("zeroed alloc", (unsigned)zsum(), 0u);
    dirty();
    chk32("zeroed alloca", (unsigned)zdyn(64), 0u);
}
`)
}
