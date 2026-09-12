package ptx

// Calls. PTX passes arguments and results through .param variables
// declared in a scope around the call, which is what a device function's
// signature declares them as too: st.param each argument, call, ld.param
// each result. An indirect call is the same sequence against a
// .callprototype, which is the func typedef's spelling here.

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
	outer := x.b
	x.b.Scope(func(inner *ptx.Body) {
		x.b = inner
		var pargs, prets []ptx.Operand
		for i, a := range args {
			p := inner.Local(ptx.Var{Space: ptx.ParamSpace, Type: paramType(a.Type()), Name: "_a" + itoa(i)})
			pargs = append(pargs, p)
			if a.Type() == ir.TypeI1 {
				r := x.temp(ptx.B32)
				inner.Selp(ptx.B32, r, ptx.Imm(1), ptx.Imm(0), x.v(a))
				inner.St(ptx.B32, ptx.At(p), r, ptx.ParamSpace)
				continue
			}
			inner.St(paramType(a.Type()), ptx.At(p), x.v(a), ptx.ParamSpace)
		}
		for i, r := range sig.Rets() {
			p := inner.Local(ptx.Var{Space: ptx.ParamSpace, Type: paramType(r.Type), Name: "_r" + itoa(i)})
			prets = append(prets, p)
		}
		inner.CallVars(callee, pargs, prets, proto)
		for i, d := range rets {
			if d.Type() == ir.TypeI1 {
				r := x.temp(ptx.B32)
				inner.Ld(ptx.B32, r, ptx.At(prets[i].(*ptx.Var)), ptx.ParamSpace)
				inner.Setp(ptx.B32, ptx.Ne, x.v(d), r, ptx.Imm(0))
				continue
			}
			inner.Ld(paramType(d.Type()), x.v(d), ptx.At(prets[i].(*ptx.Var)), ptx.ParamSpace)
		}
	})
	x.b = outer
	return nil
}

// proto is the .callprototype of a func typedef.
func (x *fn) proto(t *ir.Type) (*ptx.Proto, error) {
	if t == nil {
		return nil, fmt.Errorf("callind names no type")
	}
	return x.l.proto(t)
}
