package amd64_test

// f128, whose arithmetic is compiler-rt: §0 says a namespace the layout
// block admits is usable whether or not silicon implements it.

import (
	"bytes"
	"testing"

	"github.com/vertex-language/ir"
)

// f128.add is a call to compiler-rt and nothing else. The two operands
// arrive in XMM0 and XMM1, which is where __addtf3 wants them, and the
// answer comes back in XMM0, which is where the function returns it — so
// every copy coalesces away.
//
// The prologue is not optional: a call is a call, and SysV promises the
// callee a 16-byte aligned RSP.
func TestLowerF128Add(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	fn := m.Func("addq").Export()
	a := fn.ParamF128("a")
	b := fn.ParamF128("b")
	fn.ReturnsF128()
	entry := fn.Entry()
	entry.Return(entry.F128().Add(a, b))

	tb, raw := lowerText(t, m)

	want := []byte{
		0x55,             // push rbp
		0x48, 0x8b, 0xec, // mov rbp, rsp
		0xe8, 0x00, 0x00, 0x00, 0x00, // call __addtf3
		0xc9, // leave
		0xc3, // ret
	}
	if !bytes.Equal(tb, want) {
		t.Errorf(".text bytes = % x, want % x", tb, want)
	}
	objdumpHasRelocs(t, "addq", raw, "R_X86_64_PLT32", "__addtf3")
}

// A comparison is the call and then a condition on the integer it
// returns. compiler-rt's ordering functions return a value whose sign is
// the answer, so lt is the call and setl.
//
// The test is against the whole 32-bit result and not its low byte.
// These return -1, 0 and 1 in practice, but the ABI promises only the
// sign, and a negative answer whose low byte happened to be zero would
// read as equal.
func TestLowerF128Compare(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	fn := m.Func("lt").Export()
	a := fn.ParamF128("a")
	b := fn.ParamF128("b")
	fn.ReturnsI1()
	entry := fn.Entry()
	entry.Return(entry.F128().Lt(a, b))

	tb, raw := lowerText(t, m)

	want := []byte{
		0x55,             // push rbp
		0x48, 0x8b, 0xec, // mov rbp, rsp
		0xe8, 0x00, 0x00, 0x00, 0x00, // call __lttf2
		0x81, 0xf8, 0x00, 0x00, 0x00, 0x00, // cmp eax, 0
		0x0f, 0x9c, 0xc0, // setl al
		0x0f, 0xb6, 0xc0, // movzx eax, al
		0xc9, // leave
		0xc3, // ret
	}
	if !bytes.Equal(tb, want) {
		t.Errorf(".text bytes = % x, want % x", tb, want)
	}
	objdumpHasRelocs(t, "lt", raw, "__lttf2")
}

// A conversion into f128 is named for its source and typed by its
// destination, which is why the table is keyed by the whole opcode:
// f128.fcvt_f64 is a widening and f64.fcvt_f128 is a narrowing, and
// neither half of either says so alone.
func TestLowerF128Conversions(t *testing.T) {
	for _, tc := range []struct {
		name  string
		sym   string
		build func(b *ir.Block, fn *ir.Func)
		ret   func(fn *ir.Func)
	}{
		{"f64 to f128", "__extenddftf2",
			func(b *ir.Block, fn *ir.Func) { b.Return(b.F128().FCvtF64(b.F64.Const(1))) },
			func(fn *ir.Func) { fn.ReturnsF128() }},
		{"f128 to f64", "__trunctfdf2",
			func(b *ir.Block, fn *ir.Func) { b.Return(b.F64.FCvtF128(b.F128().Const(1))) },
			func(fn *ir.Func) { fn.ReturnsF64() }},
		{"i32 to f128", "__floatsitf",
			func(b *ir.Block, fn *ir.Func) { b.Return(b.F128().SCvtI32(b.I32.Const(1))) },
			func(fn *ir.Func) { fn.ReturnsF128() }},
		{"u64 to f128", "__floatunditf",
			func(b *ir.Block, fn *ir.Func) { b.Return(b.F128().UCvtI64(b.I64.Const(1))) },
			func(fn *ir.Func) { fn.ReturnsF128() }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			m := ir.NewModule("t", ir.X86_64Linux)
			fn := m.Func("f").Export()
			tc.ret(fn)
			entry := fn.Entry()
			tc.build(entry, fn)

			_, raw := lowerText(t, m)
			objdumpHasRelocs(t, "f", raw, tc.sym)
		})
	}
}

// §C2's conversions out of f128 into an integer: compiler-rt's fix rounds
// toward zero and is undefined out of range, where §C2 traps and its
// saturating forms clamp — so each is the range check iselFloatToInt builds
// around the hardware instruction, built around a call instead, with the
// comparisons themselves calls too.
//
// What the disassembly has to show is all three: the unordered test that
// catches a NaN, the ordering test the two bounds share, and the conversion.
func TestLowerF128ToInteger(t *testing.T) {
	for _, tc := range []struct {
		name string
		ret  func(*ir.Func)
		emit func(*ir.Block, ir.F128) ir.Value
		sym  string
	}{
		{"trapping i32", func(f *ir.Func) { f.ReturnsI32() },
			func(b *ir.Block, a ir.F128) ir.Value { return b.I32.SCvtF128(a) }, "__fixtfsi"},
		{"trapping i64", func(f *ir.Func) { f.ReturnsI64() },
			func(b *ir.Block, a ir.F128) ir.Value { return b.I64.SCvtF128(a) }, "__fixtfdi"},
		{"unsigned i32", func(f *ir.Func) { f.ReturnsI32() },
			func(b *ir.Block, a ir.F128) ir.Value { return b.I32.UCvtF128(a) }, "__fixunstfsi"},
		{"saturating i32", func(f *ir.Func) { f.ReturnsI32() },
			func(b *ir.Block, a ir.F128) ir.Value { return b.I32.SCvtSatF128(a) }, "__fixtfsi"},
		{"saturating u64", func(f *ir.Func) { f.ReturnsI64() },
			func(b *ir.Block, a ir.F128) ir.Value { return b.I64.UCvtSatF128(a) }, "__fixunstfdi"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			m := ir.NewModule("t", ir.X86_64Linux)
			fn := m.Func("f").Export()
			a := fn.ParamF128("a")
			tc.ret(fn)
			e := fn.Entry()
			e.Return(tc.emit(e, a))

			_, raw := lowerText(t, m)
			objdumpHasRelocs(t, "f", raw, tc.sym, "__unordtf2", "__lttf2")
		})
	}
}

// The sign verbs are the one part of the namespace that needs no call.
// A float's sign is a bit and it is in the same place whatever the
// format, and ANDPD and XORPD operate on all sixteen bytes of a
// register — which is what makes them the right instructions here
// unchanged.
//
// No prologue, because nothing calls.
func TestLowerF128Neg(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	fn := m.Func("negq").Export()
	a := fn.ParamF128("a")
	fn.ReturnsF128()
	entry := fn.Entry()
	entry.Return(entry.F128().Neg(a))

	tb, raw := lowerText(t, m)

	// Two moves around the one instruction that does the work. XORPD is
	// two-address, so the result has to start in the destination, and
	// regalloc will not put the result in the operand's register even
	// though the operand dies here — a destination interferes with its
	// own operands, deliberately, because which operand a destination
	// may alias is per-opcode knowledge that mir does not carry.
	want := []byte{
		0x0f, 0x28, 0x15, 0x00, 0x00, 0x00, 0x00, // movaps xmm2, [rip + the mask]
		0x0f, 0x28, 0xc8, // movaps xmm1, xmm0
		0x66, 0x0f, 0x57, 0xca, // xorpd xmm1, xmm2
		0x0f, 0x28, 0xc1, // movaps xmm0, xmm1
		0xc3, // ret
	}
	if !bytes.Equal(tb, want) {
		t.Errorf(".text bytes = % x, want % x", tb, want)
	}

	// The mask is the sign bit alone: sixteen bytes of which only the
	// top one is set, little-endian.
	ro := section(t, raw, ".rodata")
	wantMask := []byte{
		0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0x80,
	}
	if !bytes.Equal(ro, wantMask) {
		t.Errorf(".rodata = % x, want % x", ro, wantMask)
	}
}

// A literal is widened here rather than at run time. ir.Const carries a
// float64, so the value is exactly the double the module wrote — and
// every double is exactly representable in binary128, so the widening
// is a re-bias and a shift that never rounds. A constant that cost a
// call would not be a constant.
func TestLowerF128Const(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	fn := m.Func("one").Export()
	fn.ReturnsF128()
	entry := fn.Entry()
	entry.Return(entry.F128().Const(1.5))

	tb, raw := lowerText(t, m)

	want := []byte{
		0x0f, 0x28, 0x05, 0x00, 0x00, 0x00, 0x00, // movaps xmm0, [rip + one.f128.0]
		0xc3, // ret
	}
	if !bytes.Equal(tb, want) {
		t.Errorf(".text bytes = % x, want % x", tb, want)
	}

	// 1.5 is sign 0, exponent 16383, and the top significand bit set:
	// 0x3fff800000000000 in the high eightbyte and nothing in the low.
	ro := section(t, raw, ".rodata")
	wantBits := []byte{
		0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0x80, 0xff, 0x3f,
	}
	if !bytes.Equal(ro, wantBits) {
		t.Errorf(".rodata = % x, want % x", ro, wantBits)
	}
	objdumpHas(t, "one", raw, "movaps")
}
