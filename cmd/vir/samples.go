package main

import "github.com/vertex-language/ir"

// A sample is one module this tool can build, print, and verify.
//
// Built rather than read. There is no .vir parser — the README's own
// reasoning is that reading text back in would make ir/text a second
// front door for constructing a module, with its own invariants to
// defend — so a tool that operates on modules operates on ones it builds
// by calling the same public API you would. That is the whole difference
// between this and a compiler driver: a driver's input is a file, and
// this one's is a name.
type sample struct {
	name  string
	about string
	build func() *ir.Module
}

// samples is every module `vir` knows how to build, in the order `vir
// list` prints them: sound ones first, then the one that is not.
var samples = []sample{
	{"add", "two i32 parameters and their sum — the smallest whole function", buildAdd},
	{"max", "a two-way branch whose arms rejoin at a block parameter", buildMax},
	{"sum", "a loop: 1+2+...+n, carried in block parameters", buildSum},
	{"asm", "the three forms of assembly: module-level, inline, and a naked body", buildAsm},
	{"unsound", "a value used where its definition does not dominate (§19.1)", buildUnsound},
}

// lookup finds a sample by name.
func lookup(name string) (sample, bool) {
	for _, s := range samples {
		if s.name == name {
			return s, true
		}
	}
	return sample{}, false
}

func buildAdd() *ir.Module {
	m := ir.NewModule("add", ir.X86_64Linux)

	fn := m.Func("add").Export().NoUnwind()
	a := fn.ParamI32("a")
	b := fn.ParamI32("b")
	fn.ReturnsI32()

	entry := fn.Entry()
	entry.Return(entry.I32.Add(a, b))

	return m
}

// buildMax is the shape a block parameter exists for: two arms computing
// different values, and one successor that takes whichever arrived rather
// than a phi naming where each came from.
func buildMax() *ir.Module {
	m := ir.NewModule("max", ir.X86_64Linux)

	fn := m.Func("max").Export().NoUnwind()
	a := fn.ParamI32("a")
	b := fn.ParamI32("b")
	fn.ReturnsI32()

	entry := fn.Entry()
	join := fn.Block("join")
	r := join.ParamI32("r")

	entry.BrIf(entry.I32.SLt(a, b), join.To(b), join.To(a))
	join.Return(r)

	return m
}

// buildSum carries its induction variable and its accumulator around the
// back edge as block parameters, which is what this IR has instead of
// mutable locals.
func buildSum() *ir.Module {
	m := ir.NewModule("sum", ir.X86_64Linux)

	fn := m.Func("sum").Export().NoUnwind()
	n := fn.ParamI32("n")
	fn.ReturnsI32()

	entry := fn.Entry()
	loop := fn.Block("loop")
	i := loop.ParamI32("i")
	acc := loop.ParamI32("acc")
	body := fn.Block("body")
	exit := fn.Block("exit")

	entry.Br(loop.To(entry.I32.Const(1), entry.I32.Const(0)))
	loop.BrIf(loop.I32.SLe(i, n), body.To(), exit.To())
	body.Br(loop.To(body.I32.Add(i, body.I32.Const(1)), body.I32.Add(acc, i)))
	exit.Return(acc)

	return m
}

// buildUnsound is a module the grammar admits and §19 does not: x is
// defined in @then, which only one of @join's two predecessors goes
// through, so on the other path the return reads a value that was never
// computed.
//
// The builder cannot catch this — it is a fact about two blocks and the
// paths between them, and the branch that makes @join reachable without
// @then is written before @then's own body — which is exactly the kind of
// rule ir/verify exists for, and why `vir verify` has something to print.
// buildAsm is every place assembly appears in this IR, in one module:
// a module-level block (§3b), an inline template with operands (§8b),
// and a naked function whose whole body is text (§7). The first and last
// have no operands and no allocation; the middle one is the reason the
// other two are the easy cases.
func buildAsm() *ir.Module {
	m := ir.NewModule("asm", ir.X86_64Linux)

	m.Asm(".pushsection .init_array,\"aw\"\n.quad ctor\n.popsection")

	sys := m.Func("write1").Export().NoUnwind()
	fd := sys.ParamI64("fd")
	buf := sys.ParamPtr("buf")
	n := sys.ParamI64("n")
	sys.ReturnsI64()
	e := sys.Entry()
	r := e.Asm("syscall").
		Volatile().
		Out(ir.TypeI64, ir.CStr("=a")).
		In(e.I64.Const(1), ir.CStr("a")).
		In(fd, ir.CStr("D")).
		In(buf, ir.CStr("S")).
		In(n, ir.CStr("d")).
		Clobber("rcx", "r11", "memory").
		Emit()
	e.Return(r.I64(0))

	start := m.Func("_start").Export().NoReturn()
	start.AsmBody("xorl %ebp, %ebp\n\tmovq %rsp, %rdi\n\tcall __libc_start_main")

	return m
}

func buildUnsound() *ir.Module {
	m := ir.NewModule("unsound", ir.X86_64Linux)

	fn := m.Func("f").Export().NoUnwind()
	a := fn.ParamI32("a")
	fn.ReturnsI32()

	entry := fn.Entry()
	then := fn.Block("then")
	join := fn.Block("join")

	entry.BrIf(entry.I32.SLt(a, a), then.To(), join.To())
	x := then.I32.Add(a, a)
	then.Br(join.To())
	join.Return(x)

	return m
}
