package ir_test

import (
	"strings"
	"testing"

	"github.com/vertex-language/ir"
	"github.com/vertex-language/ir/text"
	"github.com/vertex-language/ir/verify"
)

// after legalizing, no i128 may be left anywhere in the module.
func noI128(t *testing.T, m *ir.Module) {
	t.Helper()
	for _, f := range m.Funcs() {
		f.WalkInsts(func(in *ir.Inst) bool {
			if in.Op().Type == ir.TypeI128 {
				t.Errorf("%s: i128 op survived: %s", f.Name(), in.Op())
			}
			for _, r := range in.Results() {
				if r.Type() == ir.TypeI128 {
					t.Errorf("%s: i128 result survived", f.Name())
				}
			}
			for _, a := range in.Args() {
				if a != nil && a.Type() == ir.TypeI128 {
					t.Errorf("%s: i128 operand survived", f.Name())
				}
			}
			return true
		})
	}
}

func build(t *testing.T, body func(*ir.Func)) *ir.Module {
	t.Helper()
	m := ir.NewModule("t", ir.AArch64MacOS)
	f := m.Func("t").Export()
	body(f)
	if err := m.Err(); err != nil {
		t.Fatalf("build: %v", err)
	}
	return m
}

func TestLegalizeI128(t *testing.T) {
	cases := map[string]func(*ir.Func){
		"add": func(f *ir.Func) {
			pa, pb := f.ParamI64("a"), f.ParamI64("b")
			f.ReturnsI64()
			e := f.Entry()
			a, b := e.I128.SExtI64(pa), e.I128.SExtI64(pb)
			e.Return(e.I128.Lo(e.I128.Add(a, b)))
		},
		"shifts": func(f *ir.Func) {
			pa, pn := f.ParamI64("a"), f.ParamI64("n")
			f.ReturnsI64()
			e := f.Entry()
			a := e.I128.SExtI64(pa)
			n := e.I128.ZExtI64(pn)
			v := e.I128.Or(e.I128.Shl(a, n), e.I128.SShr(a, n))
			e.Return(e.I128.Hi(e.I128.UShr(v, n)))
		},
		"compare": func(f *ir.Func) {
			pa, pb := f.ParamI64("a"), f.ParamI64("b")
			f.ReturnsI64()
			e := f.Entry()
			a, b := e.I128.SExtI64(pa), e.I128.SExtI64(pb)
			c := e.I128.SLt(a, b)
			e.Return(e.I64.Select(c, e.I128.Lo(a), e.I128.Lo(b)))
		},
		"divide": func(f *ir.Func) {
			pa, pb := f.ParamI64("a"), f.ParamI64("b")
			f.ReturnsI64()
			e := f.Entry()
			a, b := e.I128.SExtI64(pa), e.I128.SExtI64(pb)
			e.Return(e.I128.Lo(e.I128.Add(e.I128.SDiv(a, b), e.I128.URem(a, b))))
		},
		"memory": func(f *ir.Func) {
			p := f.ParamPtr("p")
			f.ReturnsI64()
			e := f.Entry()
			v := e.I128.Load(p)
			e.I128.Store(e.I128.Mul(v, e.I128.Const(3)), p)
			e.Return(e.I128.Lo(v))
		},
	}
	for name, body := range cases {
		t.Run(name, func(t *testing.T) {
			m := build(t, body)
			if !m.LegalizeI128() {
				t.Fatal("pass reported no change")
			}
			if err := m.Err(); err != nil {
				t.Fatalf("after legalize: %v", err)
			}
			noI128(t, m)
			if err := verify.Module(m); err != nil {
				var sb strings.Builder
				text.Print(&sb, m)
				t.Fatalf("verify: %v\n%s", err, sb.String())
			}
		})
	}
}
