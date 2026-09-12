package ptx

// §E. There is no libc on the device, so the four bulk verbs are loops
// emitted inline, a byte at a time. Slow, and correct for every length
// and alignment; a wider copy is the day something measures one.
//
// memmove decides direction once: a backward copy when the destination
// is above the source, which is the one case a forward byte copy gets
// wrong.

import (
	"fmt"

	"github.com/vertex-language/ir"
	"github.com/vertex-language/ptx"
)

func (x *fn) bulk(in *ir.Inst) error {
	switch in.Op().Verb {
	case ir.VMemCpy:
		x.copyLoop(x.arg(in, 0), x.arg(in, 1), x.arg(in, 2), false)
	case ir.VMemMove:
		dst, src, n := x.arg(in, 0), x.arg(in, 1), x.arg(in, 2)
		back := x.body.Label("$L_back")
		done := x.body.Label("$L_done")
		p := x.temp(ptx.Pred)
		x.b.Setp(ptx.U64, ptx.Hi, p, dst, src)
		x.b.Bra(back).If(p)
		x.copyLoop(dst, src, n, false)
		x.b.Bra(done)
		x.b.Bind(back)
		x.copyLoop(dst, src, n, true)
		x.b.Bind(done)
	case ir.VMemSet:
		x.setLoop(x.arg(in, 0), x.arg(in, 1), x.arg(in, 2))
	case ir.VMemCmp:
		x.cmpLoop(x.res(in), x.arg(in, 0), x.arg(in, 1), x.arg(in, 2))
	default:
		return fmt.Errorf("not lowered")
	}
	return nil
}

// copyLoop copies n bytes from src to dst, forward or backward.
func (x *fn) copyLoop(dst, src, n ptx.Reg, backward bool) {
	head, done := x.body.Label("$L_cpy"), x.body.Label("$L_cpyend")
	i, p := x.temp(ptx.B64), x.temp(ptx.Pred)
	s, d, v := x.temp(ptx.B64), x.temp(ptx.B64), x.temp(ptx.B32)
	if backward {
		x.b.Mov(ptx.B64, i, n)
		x.b.Bind(head)
		x.b.Setp(ptx.U64, ptx.Eq, p, i, ptx.Imm(0))
		x.b.Bra(done).If(p)
		x.b.Sub(ptx.U64, i, i, ptx.Imm(1))
	} else {
		x.b.Mov(ptx.B64, i, ptx.Imm(0))
		x.b.Bind(head)
		x.b.Setp(ptx.U64, ptx.Hs, p, i, n)
		x.b.Bra(done).If(p)
	}
	x.b.Add(ptx.U64, s, src, i)
	x.b.Add(ptx.U64, d, dst, i)
	x.b.Ld(ptx.U8, v, ptx.At(s))
	x.b.St(ptx.U8, ptx.At(d), v)
	if !backward {
		x.b.Add(ptx.U64, i, i, ptx.Imm(1))
	}
	x.b.Bra(head)
	x.b.Bind(done)
}

// setLoop writes the low byte of val n times from dst.
func (x *fn) setLoop(dst, val, n ptx.Reg) {
	head, done := x.body.Label("$L_set"), x.body.Label("$L_setend")
	i, p, d := x.temp(ptx.B64), x.temp(ptx.Pred), x.temp(ptx.B64)
	x.b.Mov(ptx.B64, i, ptx.Imm(0))
	x.b.Bind(head)
	x.b.Setp(ptx.U64, ptx.Hs, p, i, n)
	x.b.Bra(done).If(p)
	x.b.Add(ptx.U64, d, dst, i)
	x.b.St(ptx.U8, ptx.At(d), val)
	x.b.Add(ptx.U64, i, i, ptx.Imm(1))
	x.b.Bra(head)
	x.b.Bind(done)
}

// cmpLoop is memcmp: zero, or the difference of the first differing
// bytes as unsigned values.
func (x *fn) cmpLoop(r, a, b, n ptx.Reg) {
	head, diff, done := x.body.Label("$L_cmp"), x.body.Label("$L_cmpdiff"), x.body.Label("$L_cmpend")
	i, p := x.temp(ptx.B64), x.temp(ptx.Pred)
	pa, pb := x.temp(ptx.B64), x.temp(ptx.B64)
	va, vb := x.temp(ptx.B32), x.temp(ptx.B32)
	x.b.Mov(ptx.B32, r, ptx.Imm(0))
	x.b.Mov(ptx.B64, i, ptx.Imm(0))
	x.b.Bind(head)
	x.b.Setp(ptx.U64, ptx.Hs, p, i, n)
	x.b.Bra(done).If(p)
	x.b.Add(ptx.U64, pa, a, i)
	x.b.Add(ptx.U64, pb, b, i)
	x.b.Ld(ptx.U8, va, ptx.At(pa))
	x.b.Ld(ptx.U8, vb, ptx.At(pb))
	x.b.Setp(ptx.U32, ptx.Ne, p, va, vb)
	x.b.Bra(diff).If(p)
	x.b.Add(ptx.U64, i, i, ptx.Imm(1))
	x.b.Bra(head)
	x.b.Bind(diff)
	x.b.Sub(ptx.S32, r, va, vb)
	x.b.Bind(done)
}
