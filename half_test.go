package ir

import (
	"errors"
	"math"
	"testing"
)

// Every half pattern decodes and re-encodes to itself, NaNs aside.
func TestHalfBitsRoundTrip(t *testing.T) {
	for _, ht := range []RegType{TypeF16, TypeBF16} {
		for i := 0; i < 1<<16; i++ {
			h := uint16(i)
			v := HalfValue(ht, h)
			if math.IsNaN(v) {
				continue
			}
			if got := HalfBits(ht, v); got != h {
				t.Fatalf("%s %#x: value %v re-encodes as %#x", ht, h, v, got)
			}
		}
	}
}

// Ties go to even, once: the f64 just above a tie must not be rounded
// down to the tie by a pass through f32 and then to even.
func TestHalfBitsRoundsOnce(t *testing.T) {
	for _, tc := range []struct {
		t    RegType
		v    float64
		want uint16
	}{
		{TypeF16, 1 + 1.0/2048, 0x3c00},                      // tie, to even
		{TypeF16, 1 + 3.0/2048, 0x3c02},                      // tie, to even
		{TypeF16, 1 + 1.0/2048 + math.Ldexp(1, -40), 0x3c01}, // just above: up
		{TypeF16, 65520, 0x7c00},                             // rounds past the largest
		{TypeF16, 65519, 0x7bff},                             // does not
		{TypeF16, math.Ldexp(1, -25), 0x0000},                // half the smallest subnormal: to even
		{TypeF16, math.Ldexp(1, -25) * 1.0000001, 0x0001},    // above it: up
		{TypeBF16, 1 + 1.0/256, 0x3f80},                      // tie, to even
		{TypeBF16, 1 + 1.0/256 + math.Ldexp(1, -45), 0x3f81}, // just above: up
		{TypeBF16, -2, 0xc000},
		{TypeBF16, math.Inf(1), 0x7f80},
	} {
		if got := HalfBits(tc.t, tc.v); got != tc.want {
			t.Errorf("%s %v: got %#x want %#x", tc.t, tc.v, got, tc.want)
		}
	}
}

// A layout that does not list a half float refuses its namespace.
func TestHalfLayoutGate(t *testing.T) {
	l := X86_64Linux.Layout()
	l.HalfFloat = []RegType{TypeBF16}
	m := NewModuleLayout("t", "x86_64/linux", l)
	f := m.Func("f")
	e := f.Entry()
	e.BF16().Const(1)
	if err := m.Err(); err != nil {
		t.Fatalf("bf16 is listed: %v", err)
	}
	e.F16()
	if err := m.Err(); !errors.Is(err, ErrLayout) {
		t.Fatalf("f16 is not listed; got %v", err)
	}
}

// After LegalizeHalf no half is left anywhere, signatures included, and a
// second run finds nothing to do.
func TestLegalizeHalfLeavesNone(t *testing.T) {
	m := NewModule("t", AArch64MacOS)
	imp := m.ImportFunc("ext", NewSig().Param(TypeF16).Ret(TypeBF16))
	f := m.Func("f").Export()
	x := f.ParamF16("x")
	p := f.ParamPtr("p")
	f.ReturnsBF16()
	e := f.Entry()
	y := e.F16().FMA(x, x, e.F16().Const(0.5))
	e.F16().Store(y, p)
	r := e.Call(imp, y).BF16(0)
	e.Return(e.BF16().Add(r, e.BF16().FCvtF16(y)))
	if err := m.Err(); err != nil {
		t.Fatal(err)
	}
	if !m.LegalizeHalf() {
		t.Fatal("no change")
	}
	if err := m.Err(); err != nil {
		t.Fatal(err)
	}
	if f.usesHalf() {
		t.Fatal("a half is left in the body")
	}
	for _, s := range []*Sig{f.sig, imp.sig} {
		for _, q := range s.params {
			if q.Type.IsHalfFloat() {
				t.Fatalf("a half parameter is left")
			}
		}
		for _, q := range s.rets {
			if q.Type.IsHalfFloat() {
				t.Fatalf("a half result is left")
			}
		}
	}
	if m.LegalizeHalf() {
		t.Fatal("a second run changed something")
	}
}
