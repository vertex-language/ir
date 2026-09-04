package amd64_test

import (
	"bytes"
	"testing"

	"github.com/vertex-language/amd64/feature"
	"github.com/vertex-language/ir"
	amd64lower "github.com/vertex-language/ir/lower/amd64"
)

func TestLowerFloatExt(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	fn := m.Func("f").Export()
	x := fn.ParamF64("x")
	y := fn.ParamF64("y")
	z := fn.ParamF64("z")
	fn.ReturnsF64()

	entry := fn.Entry()
	// FMA(x, y, z) -> x * y + z -> dest is z
	fma := entry.F64.FMA(x, y, z)

	// Floor(fma)
	fl := entry.F64.Floor(fma)

	// MinNum(fl, y)
	mn := entry.F64.MinNum(fl, y)

	entry.Return(mn)

	opts := amd64lower.Options{}

	opts.Features = feature.NewSet(feature.V1).Add(feature.SSE41, feature.FMA)

	obj, err := amd64lower.Lower(m, opts)
	if err != nil {
		t.Fatalf("Lower failed: %v", err)
	}

	tb := obj.Sections()[0].Bytes()
	if !bytes.Contains(tb, []byte{0xc4}) { // VEX prefix
		t.Errorf("expected VEX prefix for fma")
	}
	if !bytes.Contains(tb, []byte{0x66, 0x0f, 0x3a, 0x0b}) { // roundsd
		t.Errorf("expected roundsd instruction")
	}
}
