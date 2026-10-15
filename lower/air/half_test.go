package air_test

import (
	"context"
	"encoding/binary"
	"math"
	"testing"

	"github.com/vertex-language/air/check"
	"github.com/vertex-language/air/metallib"
	"github.com/vertex-language/ir"
	airlower "github.com/vertex-language/ir/lower/air"
)

// widenKernel is c[i] = float32(a[i]), a holding f16 bits, legalized by
// legalize: what a quantized weight's scale is read with.
func widenKernel(t *testing.T, legalize func(*ir.Module) bool) []byte {
	t.Helper()
	m := ir.NewModule("widen", ir.AIR64)
	f := kernel(m, "widen")
	a, c := f.ParamPtr("a"), f.ParamPtr("c")
	n := f.ParamI32("n")
	entry, body, done := f.Entry(), f.Block("body"), f.Block("done")
	i := gid(entry)
	entry.BrIf(entry.I32.ULt(i, n), body.To(), done.To())
	bits := body.I32.ULoad16(body.Ptr.Add(a, body.I64.Shl(body.I64.ZExtI32(i), body.I64.Const(1))))
	body.F32.Store(body.F32.FCvtF16(body.F16().BitcastI32(bits)), at(body, c, i))
	body.Br(done.To())
	done.Return()
	if err := m.Err(); err != nil {
		t.Fatal(err)
	}
	legalize(m)
	am, err := airlower.Lower(m, airlower.Options{})
	if err != nil {
		t.Fatal(err)
	}
	lib, err := metallib.Build(am)
	if err != nil {
		t.Fatal(err)
	}
	skipIf(t, check.Validate(context.Background(), lib))
	return lib
}

// TestHalfNativeWiden: an f16 widened by the GPU's own conversion
// (LegalizeHalfNativeWiden) is the f32 the legalized arithmetic makes,
// for zeros, subnormals, normals, the largest, infinities and NaNs.
func TestHalfNativeWiden(t *testing.T) {
	var halves []uint16
	for _, h := range []uint16{0x0000, 0x8000, 0x0001, 0x8001, 0x03ff, 0x0400, 0x3c00, 0xbc00, 0xc100, 0x3555,
		0x7bff, 0xfbff, 0x7c00, 0xfc00, 0x7e00, 0x7c01, 0x2e66, 0x1234, 0x5678, 0x9abc} {
		halves = append(halves, h)
	}
	for h := uint16(0); len(halves) < 1024; h += 64 {
		halves = append(halves, h)
	}
	in := make([]byte, 2*len(halves))
	for k, h := range halves {
		binary.LittleEndian.PutUint16(in[2*k:], h)
	}
	g := check.Grid1D(len(halves), 64)
	native, err := check.Run(context.Background(), widenKernel(t, (*ir.Module).LegalizeHalfNativeWiden), "widen", g,
		check.Buffer(in), check.Out(4*len(halves)), check.Uint32s(uint32(len(halves))))
	skipIf(t, err)
	soft, err := check.Run(context.Background(), widenKernel(t, (*ir.Module).LegalizeHalf), "widen", g,
		check.Buffer(in), check.Out(4*len(halves)), check.Uint32s(uint32(len(halves))))
	skipIf(t, err)
	got, want := native.Floats(1), soft.Floats(1)
	for k := range halves {
		a, b := got[k], want[k]
		if math.IsNaN(float64(a)) && math.IsNaN(float64(b)) {
			continue
		}
		if math.Float32bits(a) != math.Float32bits(b) {
			t.Errorf("f16 %#04x: native %v (%#08x), legalized %v (%#08x)", halves[k], a, math.Float32bits(a), b, math.Float32bits(b))
		}
	}
}
