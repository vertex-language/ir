package ir_test

import (
	"strings"
	"testing"

	"github.com/vertex-language/ir"
	"github.com/vertex-language/ir/text"
	"github.com/vertex-language/ir/verify"
)

// A v128 value through a parameter, the verbs, memory and a result: the
// shape every intrinsic lowers into.
func TestVectorNamespace(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Windows)
	fn := m.Func("f").Export()
	p := fn.ParamPtr("p")
	q := fn.ParamPtr("q")
	fn.ReturnsI32()

	e := fn.Entry()
	v := e.V128()
	a := v.Load(p)
	b := v.Load(q, ir.Align(1))
	sum := v.I32x4Add(a, b)
	sum = v.I16x8AddSatU(sum, v.I8x16Splat(e.I32.Const(3)))
	sum = v.AndNot(v.I8x16Eq(sum, v.Zero()), sum)
	sum = v.I32x4Shuffle(v.I16x8Shl(sum, e.I32.Const(2)), 0x1b)
	v.Store(sum, p, ir.Align(1))
	e.Return(v.I8x16Bitmask(sum))

	if err := m.Err(); err != nil {
		t.Fatalf("Err = %v, want nil", err)
	}
	if err := verify.Module(m); err != nil {
		t.Fatalf("verify: %v", err)
	}

	var buf strings.Builder
	if err := text.Print(&buf, m); err != nil {
		t.Fatalf("print: %v", err)
	}
	got := buf.String()
	for _, want := range []string{
		"v128.load", "v128.i32x4_add", "v128.i16x8_add_sat_u",
		"v128.andnot", "v128.i8x16_eq", "v128.const", "v128.i32x4_shuffle",
		"v128.store", "v128.i8x16_bitmask",
	} {
		if !strings.Contains(got, want) {
			t.Errorf("no %s in:\n%s", want, got)
		}
	}
}

// The layout block gates the namespace, the same way it gates f80 and f128:
// a target with no vector register file gets ErrLayout rather than an
// emulation of sixteen bytes in two general registers.
func TestVectorNeedsTheLayoutToSaySo(t *testing.T) {
	m := ir.NewModule("t", ir.I386Linux)
	fn := m.Func("f").Export()
	fn.ReturnsI32()
	e := fn.Entry()
	e.Return(e.V128().I8x16Bitmask(e.V128().Zero()))

	if err := m.Err(); err == nil {
		t.Fatal("Err = nil, want ErrLayout")
	} else if !strings.Contains(err.Error(), "vector") {
		t.Errorf("Err = %v, want it to name the vector register file", err)
	}
}
