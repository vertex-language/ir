package ir

import "testing"

// The device passes (FoldConstants, ReducePowersOfTwo, MergeBlocks,
// UnrollConstantLoops) are checked by running a function before and after
// on many arguments with run, a reference interpreter of the integer
// subset they produce, and by the shape of what is left.

// run interprets f, whose parameters and result are i32 or i64, on args.
func run(t *testing.T, f *Func, args ...int64) int64 {
	t.Helper()
	val := map[*Def]int64{}
	for i, p := range f.params {
		val[p] = norm(args[i], width(p.typ))
	}
	get := func(d *Def) int64 {
		v, ok := val[d]
		if !ok {
			t.Fatalf("run: %%%d read before it was set", d.id)
		}
		return v
	}
	b := f.blocks[0]
	for steps := 0; steps < 100000; steps++ {
		for _, in := range b.insts {
			var v int64
			switch in.op.Verb {
			case VConst:
				v = in.im.lit.i
			case VSelect:
				if get(in.args[0]) != 0 {
					v = get(in.args[1])
				} else {
					v = get(in.args[2])
				}
			case VSDiv, VSRem, VUDiv, VURem:
				w := width(in.op.Type)
				a, c := get(in.args[0]), get(in.args[1])
				mask := ^uint64(0) >> uint(64-w)
				switch in.op.Verb {
				case VSDiv:
					v = a / c
				case VSRem:
					v = a % c
				case VUDiv:
					v = int64((uint64(a) & mask) / (uint64(c) & mask))
				case VURem:
					v = int64((uint64(a) & mask) % (uint64(c) & mask))
				}
				v = norm(v, w)
			default:
				// Everything else this subset has is what foldInst
				// evaluates: give it constants and ask it.
				c := &Inst{op: in.op, results: in.results}
				for _, a := range in.args {
					k := &Inst{op: Op{a.typ, VConst}, im: &imm{lit: Int(get(a)), hasLit: true}}
					kd := &Def{typ: a.typ, inst: k}
					k.results = []*Def{kd}
					c.args = append(c.args, kd)
				}
				r, ok := foldInst(c)
				if !ok {
					t.Fatalf("run: cannot evaluate %s", in.op)
				}
				v = r
			}
			val[in.results[0]] = v
		}
		term := b.term
		switch term.op.Verb {
		case VReturn:
			return get(term.args[0])
		case VBr, VBrIf:
			to := term.im.targets[0]
			if term.op.Verb == VBrIf && get(term.args[0]) == 0 {
				to = term.im.targets[1]
			}
			next := map[*Def]int64{}
			for i, a := range to.args {
				next[to.blk.params[i]] = get(a)
			}
			for d, v := range next {
				val[d] = v
			}
			b = to.blk
		default:
			t.Fatalf("run: %s", term.op)
		}
	}
	t.Fatalf("run: too many steps")
	return 0
}

var samples = []int64{0, 1, 2, 3, 7, 8, 9, 31, 32, 33, 100, 255, 256, 1000, 65535, 1 << 31, 1<<31 - 1, 1 << 40,
	-1, -2, -3, -7, -8, -9, -31, -32, -33, -100, -1000, -(1 << 31), -(1 << 40), 1<<63 - 1, -1 << 63}

func TestReducePowersOfTwo(t *testing.T) {
	type mk func(b *Builder, x Value, c int64) Value
	ops := map[string]mk{
		"i64.sdiv": func(b *Builder, x Value, c int64) Value { return b.I64.SDiv(x.(I64), b.I64.Const(c)) },
		"i64.srem": func(b *Builder, x Value, c int64) Value { return b.I64.SRem(x.(I64), b.I64.Const(c)) },
		"i64.udiv": func(b *Builder, x Value, c int64) Value { return b.I64.UDiv(x.(I64), b.I64.Const(c)) },
		"i64.urem": func(b *Builder, x Value, c int64) Value { return b.I64.URem(x.(I64), b.I64.Const(c)) },
		"i64.mul":  func(b *Builder, x Value, c int64) Value { return b.I64.Mul(b.I64.Const(c), x.(I64)) },
		"i32.sdiv": func(b *Builder, x Value, c int64) Value { return b.I32.SDiv(x.(I32), b.I32.Const(c)) },
		// A non-negative operand: a zero-extension, as a GPU index is.
		"zext sdiv": func(b *Builder, x Value, c int64) Value {
			return b.I64.SDiv(b.I64.ZExtI32(b.I32.WrapI64(x.(I64))), b.I64.Const(c))
		},
		"zext srem": func(b *Builder, x Value, c int64) Value {
			return b.I64.SRem(b.I64.ZExtI32(b.I32.WrapI64(x.(I64))), b.I64.Const(c))
		},
	}
	for name, op := range ops {
		for _, c := range []int64{2, 4, 32, 256, 1 << 20} {
			build := func() (*Module, *Func) {
				m := NewModule("t", AArch64Linux)
				fn := m.Func("f").Export()
				var x Value
				if name == "i32.sdiv" {
					x = fn.ParamI32("x")
					fn.ReturnsI32()
				} else {
					x = fn.ParamI64("x")
					fn.ReturnsI64()
				}
				e := fn.Entry()
				e.Return(op(&e.Builder, x, c))
				if err := m.Err(); err != nil {
					t.Fatal(err)
				}
				return m, fn
			}
			_, before := build()
			_, after := build()
			if n := after.ReducePowersOfTwo(); n != 1 {
				t.Errorf("%s by %d: rewrote %d, want 1", name, c, n)
			}
			for _, in := range after.blocks[0].insts {
				switch in.op.Verb {
				case VSDiv, VSRem, VUDiv, VURem, VMul:
					t.Errorf("%s by %d: a %s is left", name, c, in.op)
				}
			}
			for _, x := range samples {
				if name == "i32.sdiv" {
					x = norm(x, 32)
				}
				if want, got := run(t, before, x), run(t, after, x); want != got {
					t.Errorf("%s by %d of %d: %d, want %d", name, c, x, got, want)
				}
			}
		}
	}
}

func TestFoldConstantsShiftArms(t *testing.T) {
	// `x << 4` as a frontend lowers it: arms for a negative amount and
	// for one of the width or more, chosen by comparing two constants.
	m := NewModule("t", AArch64Linux)
	fn := m.Func("f").Export()
	x := fn.ParamI32("x")
	fn.ReturnsI32()
	e := fn.Entry()
	big, neg, ok, join := fn.Block("big"), fn.Block("neg"), fn.Block("ok"), fn.Block("join")
	r := join.ParamI32("r")
	amt := e.I32.Const(4)
	e.BrIf(e.I32.SLe(e.I32.Const(32), amt), big.To(), neg.To())
	big.Br(join.To(big.I32.Const(0)))
	neg.BrIf(neg.I32.SLe(amt, neg.I32.Const(-1)), join.To(neg.I32.SShr(x, neg.I32.Const(31))), ok.To())
	ok.Br(join.To(ok.I32.Shl(x, ok.I32.And(amt, ok.I32.Const(31)))))
	join.Return(r)
	if err := m.Err(); err != nil {
		t.Fatal(err)
	}
	if fn.FoldConstants() == 0 {
		t.Fatalf("nothing folded")
	}
	fn.NumberValues()
	fn.MergeBlocks()
	if len(fn.blocks) != 1 {
		t.Errorf("%d blocks left, want 1", len(fn.blocks))
	}
	for _, v := range samples {
		v = norm(v, 32)
		if got, want := run(t, fn, v), norm(v<<4, 32); got != want {
			t.Errorf("%d << 4: %d, want %d", v, got, want)
		}
	}
}

func TestMergeBlocksChain(t *testing.T) {
	// The chain an inliner leaves: blocks that each do a little and
	// branch on, one of them only forwarding its parameter.
	m := NewModule("t", AArch64Linux)
	fn := m.Func("f").Export()
	x := fn.ParamI64("x")
	fn.ReturnsI64()
	e := fn.Entry()
	a, fwd, c := fn.Block("a"), fn.Block("fwd"), fn.Block("c")
	p := fwd.ParamI64("p")
	q := c.ParamI64("q")
	e.Br(a.To())
	a.Br(fwd.To(a.I64.Add(x, a.I64.Const(1))))
	fwd.Br(c.To(p))
	c.Return(c.I64.Mul(q, c.I64.Const(3)))
	if err := m.Err(); err != nil {
		t.Fatal(err)
	}
	fn.MergeBlocks()
	if len(fn.blocks) != 1 {
		t.Errorf("%d blocks left, want 1", len(fn.blocks))
	}
	for _, v := range samples {
		if got, want := run(t, fn, v), (v+1)*3; got != want {
			t.Errorf("f(%d) = %d, want %d", v, got, want)
		}
	}
}

func TestUnrollConstantLoops(t *testing.T) {
	// s = x; for i in 0..<n step st { s = s*3 + i }, written as a frontend
	// does: a header testing i against n, a body stepping it.
	for _, c := range []struct{ start, n, step int64 }{{0, 4, 1}, {0, 8, 2}, {3, 7, 1}, {5, 5, 1}, {0, 16, 1}} {
		build := func() *Func {
			m := NewModule("t", AArch64Linux)
			fn := m.Func("f").Export()
			x := fn.ParamI64("x")
			fn.ReturnsI64()
			e := fn.Entry()
			head, body, exit := fn.Block("head"), fn.Block("body"), fn.Block("exit")
			i, s := head.ParamI64("i"), head.ParamI64("s")
			e.Br(head.To(e.I64.Const(c.start), x))
			head.BrIf(head.I64.SLt(i, head.I64.Const(c.n)), body.To(), exit.To())
			body.Br(head.To(body.I64.Add(i, body.I64.Const(c.step)), body.I64.Add(body.I64.Mul(s, body.I64.Const(3)), i)))
			exit.Return(exit.I64.Add(s, i))
			if err := m.Err(); err != nil {
				t.Fatal(err)
			}
			return fn
		}
		before, after := build(), build()
		if n := after.UnrollConstantLoops(); n != 1 {
			t.Errorf("%+v: unrolled %d loops, want 1", c, n)
		}
		after.MergeBlocks()
		for _, b := range after.blocks {
			if t2 := b.term; t2 != nil && t2.op.Verb == VBrIf {
				t.Errorf("%+v: a branch is left in @%s", c, b.label)
			}
		}
		for _, v := range samples {
			if want, got := run(t, before, v), run(t, after, v); want != got {
				t.Errorf("%+v: f(%d) = %d, want %d", c, v, got, want)
			}
		}
	}
}

// TestUnrollLeavesUnknownTrips: a loop to a bound the function is given
// is not unrolled.
func TestUnrollLeavesUnknownTrips(t *testing.T) {
	m := NewModule("t", AArch64Linux)
	fn := m.Func("f").Export()
	n := fn.ParamI64("n")
	fn.ReturnsI64()
	e := fn.Entry()
	head, body, exit := fn.Block("head"), fn.Block("body"), fn.Block("exit")
	i := head.ParamI64("i")
	e.Br(head.To(e.I64.Const(0)))
	head.BrIf(head.I64.SLt(i, n), body.To(), exit.To())
	body.Br(head.To(body.I64.Add(i, body.I64.Const(1))))
	exit.Return(i)
	if err := m.Err(); err != nil {
		t.Fatal(err)
	}
	if got := fn.UnrollConstantLoops(); got != 0 {
		t.Errorf("unrolled %d, want 0", got)
	}
	if got := run(t, fn, 5); got != 5 {
		t.Errorf("f(5) = %d", got)
	}
}
