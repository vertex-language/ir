package ptx

// Calls. PTX passes arguments and results through .param variables
// declared in a scope around the call, which is what a device function's
// signature declares them as too: st.param each argument, call, ld.param
// each result. This is the ABI — nvcc's, and the one ptxas requires for
// an indirect call, which is the same sequence against a .callprototype,
// the func typedef's spelling here. An i1 crosses as a .b32 holding 0 or
// 1; more than one result crosses as a byte array (retShape).

import (
	"fmt"

	"github.com/vertex-language/ir"
	"github.com/vertex-language/ptx"
)

func (x *fn) call(in *ir.Inst) error {
	var (
		callee ptx.Operand
		sig    *ir.Sig
		proto  *ptx.Proto
		args   []*ir.Def
	)
	if in.Op().Verb == ir.VCall {
		c := in.Callee()
		pf, ok := x.l.funcs[c]
		if !ok {
			return fmt.Errorf("@%s is a kernel; nothing on the device calls one", c.Name())
		}
		callee, sig, args = pf, c.Signature(), in.Args()
	} else {
		t := in.NamedType()
		p, err := x.proto(t)
		if err != nil {
			return err
		}
		callee, sig, proto, args = x.arg(in, 0), t.Sig(), p, in.Args()[1:]
	}

	rets := in.Results()
	sh := retShapeOf(sig)
	outer := x.b
	x.b.Scope(func(inner *ptx.Body) {
		x.b = inner
		var pargs, prets []ptx.Operand
		for i, a := range args {
			p := inner.Local(ptx.Var{Space: ptx.ParamSpace, Type: paramType(a.Type()), Name: "_a" + itoa(i)})
			pargs = append(pargs, p)
			v := x.v(a)
			if a.Type() == ir.TypeI1 {
				v = x.temp(ptx.B32)
				inner.Selp(ptx.B32, v, ptx.Imm(1), ptx.Imm(0), x.v(a))
			}
			inner.St(paramType(a.Type()), ptx.At(p), v, ptx.ParamSpace)
		}
		var rp *ptx.Var
		switch {
		case sh.single:
			rp = inner.Local(ptx.Var{Space: ptx.ParamSpace, Type: sh.typ, Name: "_cr"})
		case len(rets) > 1:
			rp = inner.Local(ptx.Var{Space: ptx.ParamSpace, Type: ptx.B8, Align: sh.align, Len: sh.size, Name: "_cr"})
		}
		if rp != nil {
			prets = []ptx.Operand{rp}
		}
		inner.CallVars(callee, pargs, prets, proto)
		for i, d := range rets {
			var off int64
			if !sh.single {
				off = sh.offs[i]
			}
			if d.Type() == ir.TypeI1 {
				r := x.temp(ptx.B32)
				inner.Ld(ptx.B32, r, ptx.At(rp, off), ptx.ParamSpace)
				inner.Setp(ptx.B32, ptx.Ne, x.v(d), r, ptx.Imm(0))
				continue
			}
			inner.Ld(paramType(d.Type()), x.v(d), ptx.At(rp, off), ptx.ParamSpace)
		}
	})
	x.b = outer
	return nil
}
