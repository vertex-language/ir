package ptx

// Terminators and edges.
//
// A block parameter is a register like any other value, and an edge into
// the block is a move into each. The moves on one edge are a parallel
// assignment — a back edge that swaps two parameters must not clobber
// the second while reading the first — and the CPU backends break the
// cycles. Here every argument goes through a fresh temporary first, which
// is the same answer with no cycle to find: PTX registers cost nothing
// and ptxas coalesces the ones that turn out to be one value.
//
// A conditional branch has two edges and PTX's bra has one target, so an
// edge that carries arguments is a trampoline: a label of its own holding
// the moves and the final bra. An edge that carries none is a bare bra,
// which is every edge a frontend emits for an if.

import (
	"fmt"

	"github.com/vertex-language/ir"
	"github.com/vertex-language/ptx"
)

func (x *fn) term(in *ir.Inst) error {
	ts := in.Targets()
	switch in.Op().Verb {
	case ir.VBr:
		x.edge(ts[0])
		x.b.Bra(x.labels[ts[0].Block()])

	case ir.VBrIf:
		c := x.arg(in, 0)
		then, els := ts[0], ts[1]
		if then.Bare() || len(then.Args()) == 0 {
			x.b.Bra(x.labels[then.Block()]).If(c)
			x.edge(els)
			x.b.Bra(x.labels[els.Block()])
			return nil
		}
		tramp := x.body.Label("$L_then")
		x.b.Bra(tramp).If(c)
		x.edge(els)
		x.b.Bra(x.labels[els.Block()])
		x.b.Bind(tramp)
		x.edge(then)
		x.b.Bra(x.labels[then.Block()])

	case ir.VBrTable:
		return x.brTable(in)

	case ir.VReturn:
		return x.ret(in)

	case ir.VTrap:
		x.b.Trap()

	case ir.VBrInd:
		return fmt.Errorf("brind has no PTX equivalent; brx.idx wants an index, not an address")
	case ir.VInvoke, ir.VInvokeInd, ir.VResume:
		return fmt.Errorf("there is no unwinding on the device")
	case ir.VTailCall, ir.VTailCallInd:
		return fmt.Errorf("PTX has no tail call; a call and a return would not keep the frame guarantee the verb makes")
	case ir.VAsmGoto:
		return fmt.Errorf("asm goto is not lowered; PTX has no label operand to substitute")
	default:
		return fmt.Errorf("not lowered")
	}
	return nil
}

// edge emits the moves an edge's arguments make into the target's
// parameters, through temporaries.
func (x *fn) edge(t ir.BlockTarget) {
	args := t.Args()
	if len(args) == 0 {
		return
	}
	params := t.Block().Params()
	tmps := make([]ptx.Reg, len(args))
	for i, a := range args {
		tmps[i] = x.temp(regType(a.Type()))
		x.mov(tmps[i], x.v(a), a.Type())
	}
	for i, p := range params {
		x.mov(x.v(p), tmps[i], p.Type())
	}
}

// mov is a register-to-register move of a VIR type.
func (x *fn) mov(d, s ptx.Reg, t ir.RegType) {
	switch t {
	case ir.TypeI1:
		x.b.Mov(ptx.Pred, d, s)
	case ir.TypeF32:
		x.b.Mov(ptx.F32, d, s)
	case ir.TypeF64:
		x.b.Mov(ptx.F64, d, s)
	default:
		x.b.Mov(bitsT(t), d, s)
	}
}

// brTable is brx.idx over a .branchtargets list, behind the range check
// §G2 puts on the frontend and this backend repeats — brx.idx with an
// index past the table is undefined, and §0 has no undefined.
func (x *fn) brTable(in *ir.Inst) error {
	if err := x.l.needSM(in.Op(), 30, "brx.idx"); err != nil {
		return err
	}
	ts := in.Targets()
	cases, dflt := ts[:len(ts)-1], ts[len(ts)-1]
	sel := x.arg(in, 0)

	// Each edge is a label: the block's own where it carries nothing, a
	// trampoline where it carries arguments.
	labelFor := func(t ir.BlockTarget, name string) (*ptx.Label, bool) {
		if len(t.Args()) == 0 {
			return x.labels[t.Block()], false
		}
		return x.body.Label(name), true
	}
	dl, dtramp := labelFor(dflt, "$L_default")
	p := x.temp(ptx.Pred)
	x.b.Setp(ptx.U32, ptx.Hs, p, sel, ptx.Imm(int64(len(cases))))
	x.b.Bra(dl).If(p)

	labels := make([]*ptx.Label, len(cases))
	tramps := make([]bool, len(cases))
	for i, c := range cases {
		labels[i], tramps[i] = labelFor(c, "$L_case")
	}
	if len(cases) > 0 {
		x.b.BrxIdx(sel, labels)
	} else {
		x.b.Bra(dl)
	}
	for i, c := range cases {
		if !tramps[i] {
			continue
		}
		x.b.Bind(labels[i])
		x.edge(c)
		x.b.Bra(x.labels[c.Block()])
	}
	if dtramp {
		x.b.Bind(dl)
		x.edge(dflt)
		x.b.Bra(x.labels[dflt.Block()])
	}
	return nil
}

// ret stores the results into the .param result a .func declares — one
// of the result's type, or a byte array at retShape's offsets — and
// returns; a kernel has none and returns.
func (x *fn) ret(in *ir.Inst) error {
	if x.pf != nil && len(in.Args()) > 0 {
		sh := retShapeOf(x.f.Signature())
		p := x.pf.Ret[0]
		for i, d := range in.Args() {
			var off int64
			if !sh.single {
				off = sh.offs[i]
			}
			v := x.v(d)
			if d.Type() == ir.TypeI1 {
				v = x.temp(ptx.B32)
				x.b.Selp(ptx.B32, v, ptx.Imm(1), ptx.Imm(0), x.v(d))
			}
			x.b.St(paramType(d.Type()), ptx.At(p, off), v, ptx.ParamSpace)
		}
	}
	x.b.Ret()
	return nil
}
