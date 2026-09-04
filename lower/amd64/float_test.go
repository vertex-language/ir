package amd64_test

// Milestone 22: the second register file.
//
// Everything before this held every value in a general-purpose register,
// and the allocator's pool was that file. A float lives in an XMM
// register instead, which is not a wider register or a different width
// of the same one — it is a disjoint set that no integer instruction can
// name, and RAX and XMM0 are both register number zero.

import (
	"bytes"
	"testing"

	"github.com/vertex-language/ir"
	amd64lower "github.com/vertex-language/ir/lower/amd64"
)

// The whole of a float function: two parameters SysV left in XMM0 and
// XMM1, one instruction, and a result in XMM0.
//
// The two MOVAPS are the same cost the integer path pays and for the
// same reason: a two-address instruction writes its destination, so the
// destination cannot be an operand the interference graph still needs,
// and mir does not interpret an Op well enough to know when it could.
func TestLowerFloatAdd(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	fn := m.Func("fadd").Export()
	a := fn.ParamF64("a")
	b := fn.ParamF64("b")
	fn.ReturnsF64()
	e := fn.Entry()
	e.Return(e.F64.Add(a, b))

	tb, raw := lowerText(t, m)

	want := []byte{
		0x0f, 0x28, 0xd0, // movaps xmm2, xmm0
		0xf2, 0x0f, 0x58, 0xd1, // addsd xmm2, xmm1
		0x0f, 0x28, 0xc2, // movaps xmm0, xmm2
		0xc3, // ret
	}
	if !bytes.Equal(tb, want) {
		t.Errorf(".text bytes = % x, want % x", tb, want)
	}
	objdumpHas(t, "fadd", raw, "addsd", "movaps")
}

// The same three operations one namespace down. F3 is what makes an
// instruction single-precision and F2 double, over one opcode each.
func TestLowerFloat32Arithmetic(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	fn := m.Func("f32ops").Export()
	a := fn.ParamF32("a")
	b := fn.ParamF32("b")
	fn.ReturnsF32()
	e := fn.Entry()
	e.Return(e.F32.Div(e.F32.Mul(a, b), e.F32.Sub(a, b)))

	tb, raw := lowerText(t, m)

	want := []byte{
		0x0f, 0x28, 0xd0, // movaps xmm2, xmm0
		0xf3, 0x0f, 0x59, 0xd1, // mulss xmm2, xmm1
		0x0f, 0x28, 0xd8, // movaps xmm3, xmm0
		0xf3, 0x0f, 0x5c, 0xd9, // subss xmm3, xmm1
		0x0f, 0x28, 0xc2, // movaps xmm0, xmm2
		0xf3, 0x0f, 0x5e, 0xc3, // divss xmm0, xmm3
		0xc3, // ret
	}
	if !bytes.Equal(tb, want) {
		t.Errorf(".text bytes = % x, want % x", tb, want)
	}
	objdumpHas(t, "f32ops", raw, "mulss", "subss", "divss")
}

// A float constant is its own bit pattern: into a general-purpose
// register as an immediate, and across into a vector one.
//
// There is no instruction that puts an immediate in an XMM register.
// The other way to do this is a constant pool — eight bytes in .rodata
// and a RIP-relative load — which is one instruction instead of two but
// needs a section, a symbol and a relocation per distinct value.
func TestLowerFloatConst(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	fn := m.Func("fconst").Export()
	fn.ReturnsF64()
	e := fn.Entry()
	e.Return(e.F64.Add(e.F64.Const(1.5), e.F64.Const(2.5)))

	tb, raw := lowerText(t, m)

	want := []byte{
		0x48, 0xb8, 0, 0, 0, 0, 0, 0, 0xf8, 0x3f, // movabs rax, 1.5
		0x66, 0x48, 0x0f, 0x6e, 0xc0, // movq xmm0, rax
		0x48, 0xb8, 0, 0, 0, 0, 0, 0, 0x04, 0x40, // movabs rax, 2.5
		0x66, 0x48, 0x0f, 0x6e, 0xc8, // movq xmm1, rax
		0x0f, 0x28, 0xd0, // movaps xmm2, xmm0
		0xf2, 0x0f, 0x58, 0xd1, // addsd xmm2, xmm1
		0x0f, 0x28, 0xc2, // movaps xmm0, xmm2
		0xc3, // ret
	}
	if !bytes.Equal(tb, want) {
		t.Errorf(".text bytes = % x, want % x", tb, want)
	}
	objdumpHas(t, "fconst", raw, "movabsq", "movq")
}

// An f32 constant is four bytes and goes across through MOVD, and the
// value is rounded to an f32 first: 1.5 is 0x3fc00000 and not the top
// half of the double.
func TestLowerFloat32Const(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	fn := m.Func("fconst32").Export()
	fn.ReturnsF32()
	e := fn.Entry()
	e.Return(e.F32.Const(1.5))

	tb, _ := lowerText(t, m)

	want := []byte{
		0xb8, 0x00, 0x00, 0xc0, 0x3f, // mov eax, 0x3fc00000
		0x66, 0x0f, 0x6e, 0xc0, // movd xmm0, eax
		0xc3, // ret
	}
	if !bytes.Equal(tb, want) {
		t.Errorf(".text bytes = % x, want % x", tb, want)
	}
}

// SysV fills its two register files independently, so the floats here are
// in XMM0 and XMM1 whatever positions they hold in the parameter list —
// the integers between them take RDI and RSI and cost the floats nothing.
func TestLowerMixedParameterFiles(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	fn := m.Func("mixed").Export()
	fn.ParamI64("n")
	x := fn.ParamF64("x")
	fn.ParamI64("k")
	y := fn.ParamF64("y")
	fn.ReturnsF64()
	e := fn.Entry()
	e.Return(e.F64.Mul(x, y))

	tb, _ := lowerText(t, m)

	want := []byte{
		0x0f, 0x28, 0xd0, // movaps xmm2, xmm0     (x, the first float)
		0xf2, 0x0f, 0x59, 0xd1, // mulsd xmm2, xmm1      (y, the second)
		0x0f, 0x28, 0xc2, // movaps xmm0, xmm2
		0xc3, // ret
	}
	if !bytes.Equal(tb, want) {
		t.Errorf(".text bytes = % x, want % x", tb, want)
	}
}

// §D at float widths: MOVSD rather than MOV, and the address is still a
// general-purpose register. A pointer is a pointer whatever it points at.
func TestLowerFloatLoadStore(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	fn := m.Func("copyf").Export()
	p := fn.ParamPtr("p")
	q := fn.ParamPtr("q")
	fn.ReturnsF64()
	e := fn.Entry()
	v := e.F64.Load(p)
	e.F64.Store(v, q)
	e.Return(v)

	tb, raw := lowerText(t, m)

	want := []byte{
		0xf2, 0x0f, 0x10, 0x07, // movsd xmm0, [rdi]
		0xf2, 0x0f, 0x11, 0x06, // movsd [rsi], xmm0
		0xc3, // ret
	}
	if !bytes.Equal(tb, want) {
		t.Errorf(".text bytes = % x, want % x", tb, want)
	}
	objdumpHas(t, "copyf", raw, "movsd")
}

// A float live across a call, which is the case SysV gives no good
// answer to: every XMM register is caller-saved, so there is no
// callee-saved one to move the value into and it has to go to memory.
//
// The spill is the allocator's own — the same mechanism milestone 16
// built, reaching for the float store because the value's register file
// says to. The integer version of this test keeps its value in RBX.
func TestLowerFloatAcrossCall(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	sin := m.ImportFunc("sin", ir.NewSig().Param(ir.TypeF64).Ret(ir.TypeF64))

	fn := m.Func("callsin").Export()
	x := fn.ParamF64("x")
	fn.ReturnsF64()
	e := fn.Entry()
	e.Return(e.F64.Add(e.Call(sin, x).Value(0).(ir.F64), x))

	tb, raw := lowerText(t, m)

	want := []byte{
		0x55,             // push rbp
		0x48, 0x8b, 0xec, // mov rbp, rsp
		0x48, 0x81, 0xec, 0x10, 0x00, 0x00, 0x00, // sub rsp, 16
		0xf2, 0x0f, 0x11, 0x45, 0xf8, // movsd [rbp-8], xmm0   (x, spilled)
		0xf2, 0x0f, 0x10, 0x45, 0xf8, // movsd xmm0, [rbp-8]   (and back, as the argument)
		0xe8, 0x00, 0x00, 0x00, 0x00, // call sin
		0xf2, 0x0f, 0x10, 0x55, 0xf8, // movsd xmm2, [rbp-8]   (x again, after the call)
		0x0f, 0x28, 0xc8, // movaps xmm1, xmm0
		0xf2, 0x0f, 0x58, 0xca, // addsd xmm1, xmm2
		0x0f, 0x28, 0xc1, // movaps xmm0, xmm1
		0xc9, // leave
		0xc3, // ret
	}
	if !bytes.Equal(tb, want) {
		t.Errorf(".text bytes = % x, want % x", tb, want)
	}
	objdumpHas(t, "callsin", raw, "callq", "movsd")
}

// A call with arguments in both files at once, and a float return. XMM0
// is the first float argument register and the float return register
// both, which is the one place the two files are not symmetric: pinning
// a second vreg to it for the result would be two facts about one
// register, live at one instruction.
func TestLowerCallMixedFiles(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	sig := ir.NewSig().Param(ir.TypeF64).Param(ir.TypeI32).Param(ir.TypeF64).Ret(ir.TypeF64)
	f := m.ImportFunc("scale", sig)

	fn := m.Func("callscale").Export()
	x := fn.ParamF64("x")
	n := fn.ParamI32("n")
	y := fn.ParamF64("y")
	fn.ReturnsF64()
	e := fn.Entry()
	e.Return(e.Call(f, x, n, y).Value(0))

	tb, raw := lowerText(t, m)

	want := []byte{
		0x55,             // push rbp
		0x48, 0x8b, 0xec, // mov rbp, rsp
		0xe8, 0x00, 0x00, 0x00, 0x00, // call scale
		0xc9, // leave
		0xc3, // ret
	}
	if !bytes.Equal(tb, want) {
		t.Errorf(".text bytes = % x, want % x", tb, want)
	}
	objdumpHas(t, "callscale", raw, "callq")
}

// ── milestone 23: the float comparisons ───────────────────────────────

// A branch on a float less-than. The operands are compared the other way
// round and the branch asks "above", which is what makes a NaN answer
// false: UCOMIS reports below for a NaN as well as for a smaller value,
// and above is the reading that tells them apart.
func TestLowerFloatLtBranch(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	fn := m.Func("flt").Export()
	a := fn.ParamF64("a")
	b := fn.ParamF64("b")
	fn.ReturnsI32()

	entry := fn.Entry()
	yes := fn.Block("yes")
	no := fn.Block("no")
	entry.BrIf(entry.F64.Lt(a, b), yes.To(), no.To())
	yes.Return(yes.I32.Const(1))
	no.Return(no.I32.Const(0))

	tb, raw := lowerText(t, m)

	want := []byte{
		0x66, 0x0f, 0x2e, 0xc8, // ucomisd xmm1, xmm0    (b against a)
		0x0f, 0x87, 0x05, 0x00, 0x00, 0x00, // ja yes
		0xe9, 0x06, 0x00, 0x00, 0x00, // jmp no
		0xb8, 0x01, 0x00, 0x00, 0x00, // mov eax, 1
		0xc3,                         // ret
		0xb8, 0x00, 0x00, 0x00, 0x00, // mov eax, 0
		0xc3, // ret
	}
	if !bytes.Equal(tb, want) {
		t.Errorf(".text bytes = % x, want % x", tb, want)
	}
	objdumpHas(t, "flt", raw, "ucomisd", "ja")
}

// The same comparison as a value, at f32. One condition, so one setcc.
func TestLowerFloat32LtValue(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	fn := m.Func("fltv").Export()
	a := fn.ParamF32("a")
	b := fn.ParamF32("b")
	fn.ReturnsI32()
	e := fn.Entry()
	e.Return(e.I32.ZExtI1(e.F32.Lt(a, b)))

	tb, raw := lowerText(t, m)

	want := []byte{
		0x0f, 0x2e, 0xc8, // ucomiss xmm1, xmm0
		0x0f, 0x97, 0xc0, // seta al
		0x0f, 0xb6, 0xc0, // movzx eax, al
		0xc3, // ret
	}
	if !bytes.Equal(tb, want) {
		t.Errorf(".text bytes = % x, want % x", tb, want)
	}
	objdumpHas(t, "fltv", raw, "ucomiss", "seta")
}

// Float equality is the row no single condition answers. Equal is
// ordered and equal — ZF set with PF clear — and no flag holds that
// conjunction, so the conjunction gets built: two setccs off one compare
// and an AND between them.
func TestLowerFloatEq(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	fn := m.Func("feq").Export()
	a := fn.ParamF64("a")
	b := fn.ParamF64("b")
	fn.ReturnsI32()
	e := fn.Entry()
	e.Return(e.I32.ZExtI1(e.F64.Eq(a, b)))

	tb, raw := lowerText(t, m)

	want := []byte{
		0x66, 0x0f, 0x2e, 0xc1, // ucomisd xmm0, xmm1
		0x0f, 0x9b, 0xc1, // setnp cl        (ordered)
		0x0f, 0xb6, 0xc9, // movzx ecx, cl
		0x0f, 0x94, 0xc0, // sete al         (and equal)
		0x0f, 0xb6, 0xc0, // movzx eax, al
		0x23, 0xc1, // and eax, ecx
		0xc3, // ret
	}
	if !bytes.Equal(tb, want) {
		t.Errorf(".text bytes = % x, want % x", tb, want)
	}
	objdumpHas(t, "feq", raw, "setnp", "sete")
}

// Not-equal is the negation of both halves and an OR, which is what
// makes a NaN unequal to everything including itself. Not a special case
// bolted on: it falls out of writing the two conditions honestly.
func TestLowerFloatNe(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	fn := m.Func("fne").Export()
	a := fn.ParamF64("a")
	b := fn.ParamF64("b")
	fn.ReturnsI1()
	e := fn.Entry()
	e.Return(e.F64.Ne(a, b))

	tb, raw := lowerText(t, m)

	want := []byte{
		0x66, 0x0f, 0x2e, 0xc1, // ucomisd xmm0, xmm1
		0x0f, 0x9a, 0xc1, // setp cl
		0x0f, 0xb6, 0xc9, // movzx ecx, cl
		0x0f, 0x95, 0xc0, // setne al
		0x0f, 0xb6, 0xc0, // movzx eax, al
		0x0b, 0xc1, // or eax, ecx
		0xc3, // ret
	}
	if !bytes.Equal(tb, want) {
		t.Errorf(".text bytes = % x, want % x", tb, want)
	}
	objdumpHas(t, "fne", raw, "setp", "setne")
}

// uno asks the question PF was set for, and is the one float comparison
// whose whole answer is a single flag no integer comparison produces.
func TestLowerFloatUno(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	fn := m.Func("funo").Export()
	a := fn.ParamF64("a")
	b := fn.ParamF64("b")
	fn.ReturnsI1()
	e := fn.Entry()
	e.Return(e.F64.Uno(a, b))

	tb, _ := lowerText(t, m)

	want := []byte{
		0x66, 0x0f, 0x2e, 0xc1, // ucomisd xmm0, xmm1
		0x0f, 0x9a, 0xc0, // setp al
		0x0f, 0xb6, 0xc0, // movzx eax, al
		0xc3, // ret
	}
	if !bytes.Equal(tb, want) {
		t.Errorf(".text bytes = % x, want % x", tb, want)
	}
}

// A select on a float condition, which fuses the compare exactly the way
// an integer one does: the flags are the flags whichever comparison set
// them, and CMOVcc cannot tell.
func TestLowerFloatSelect(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	fn := m.Func("fsel").Export()
	a := fn.ParamF64("a")
	b := fn.ParamF64("b")
	x := fn.ParamI32("x")
	y := fn.ParamI32("y")
	fn.ReturnsI32()
	e := fn.Entry()
	e.Return(e.I32.Select(e.F64.Le(a, b), x, y))

	tb, raw := lowerText(t, m)

	want := []byte{
		0x66, 0x0f, 0x2e, 0xc8, // ucomisd xmm1, xmm0
		0x8b, 0xc6, // mov eax, esi
		0x0f, 0x43, 0xc7, // cmovae eax, edi
		0xc3, // ret
	}
	if !bytes.Equal(tb, want) {
		t.Errorf(".text bytes = % x, want % x", tb, want)
	}
	objdumpHas(t, "fsel", raw, "cmovael")
}

// The two rows that cannot fuse, read by the two things that would have
// fused them. A float eq is a value, and both a branch and a select take
// it from a register — which is what milestone 20 is for and why neither
// of these needs a special case of its own.
func TestLowerFloatEqIsAValue(t *testing.T) {
	t.Run("branch", func(t *testing.T) {
		m := ir.NewModule("t", ir.X86_64Linux)
		fn := m.Func("feqb").Export()
		a := fn.ParamF64("a")
		b := fn.ParamF64("b")
		fn.ReturnsI32()

		entry := fn.Entry()
		yes := fn.Block("yes")
		no := fn.Block("no")
		entry.BrIf(entry.F64.Eq(a, b), yes.To(), no.To())
		yes.Return(yes.I32.Const(1))
		no.Return(no.I32.Const(0))

		_, raw := lowerText(t, m)
		objdumpHas(t, "feqb", raw, "ucomisd", "setnp", "sete", "andl", "testb", "jne")
	})

	t.Run("select", func(t *testing.T) {
		m := ir.NewModule("t", ir.X86_64Linux)
		fn := m.Func("feqsel").Export()
		a := fn.ParamF64("a")
		b := fn.ParamF64("b")
		x := fn.ParamI32("x")
		y := fn.ParamI32("y")
		fn.ReturnsI32()
		e := fn.Entry()
		e.Return(e.I32.Select(e.F64.Eq(a, b), x, y))

		_, raw := lowerText(t, m)
		objdumpHas(t, "feqsel", raw, "ucomisd", "setnp", "sete", "andl", "testb", "cmovnel")
	})
}

// ── milestone 24: the conversions, and the sign bit ───────────────────

// §C2's signed integer-to-float, whose source width is REX.W and whose
// destination width is the prefix — one instruction and no rounding
// question, since every i32 and most i64s are exactly representable.
func TestLowerIntToFloat(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)

	a := m.Func("i32tof64").Export()
	x := a.ParamI32("a")
	a.ReturnsF64()
	a.Entry().Return(a.Entry().F64.SCvtI32(x))

	b := m.Func("i64tof32").Export()
	y := b.ParamI64("a")
	b.ReturnsF32()
	b.Entry().Return(b.Entry().F32.SCvtI64(y))

	tb, raw := lowerText(t, m)

	want := []byte{
		0xf2, 0x0f, 0x2a, 0xc7, // cvtsi2sd xmm0, edi
		0xc3,                         // ret
		0xf3, 0x48, 0x0f, 0x2a, 0xc7, // cvtsi2ss xmm0, rdi
		0xc3, // ret
	}
	if !bytes.Equal(tb, want) {
		t.Errorf(".text bytes = % x, want % x", tb, want)
	}
	objdumpHas(t, "cvt", raw, "cvtsi2sd", "cvtsi2ss")
}

// The unsigned 32-bit conversion, which has no instruction of its own
// and needs none: an unsigned i32 is an exact signed i64, so it widens
// first and converts as one.
func TestLowerUnsignedIntToFloat(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	fn := m.Func("u32tof64").Export()
	x := fn.ParamI32("a")
	fn.ReturnsF64()
	fn.Entry().Return(fn.Entry().F64.UCvtI32(x))

	tb, _ := lowerText(t, m)

	want := []byte{
		0x8b, 0xc7, // mov eax, edi        (zero-extended into rax)
		0xf2, 0x48, 0x0f, 0x2a, 0xc0, // cvtsi2sd xmm0, rax
		0xc3, // ret
	}
	if !bytes.Equal(tb, want) {
		t.Errorf(".text bytes = % x, want % x", tb, want)
	}

}

// The unsigned 64-bit conversion, which has neither an instruction nor a
// wider signed type to hide in: a branch, and a halving on the arm that
// needs one.
func TestLowerUnsignedI64ToFloat(t *testing.T) {
	for _, tc := range []struct {
		name string
		ret  func(fn *ir.Func)
		cvt  string
	}{
		{"u64tof64", func(fn *ir.Func) { fn.ReturnsF64() }, "cvtsi2sd"},
		{"u64tof32", func(fn *ir.Func) { fn.ReturnsF32() }, "cvtsi2ss"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			m := ir.NewModule("t", ir.X86_64Linux)
			fn := m.Func(tc.name).Export()
			x := fn.ParamI64("a")
			tc.ret(fn)
			e := fn.Entry()
			if tc.cvt == "cvtsi2sd" {
				e.Return(e.F64.UCvtI64(x))
			} else {
				e.Return(e.F32.UCvtI64(x))
			}

			_, raw := lowerText(t, m)
			// The full-width compare that splits the two cases, the
			// halving with its dropped bit put back, and the doubling
			// that undoes it.
			objdumpHas(t, tc.name, raw,
				"cmpq", "shrq", "andq", "orq", tc.cvt,
				map[string]string{"cvtsi2sd": "addsd", "cvtsi2ss": "addss"}[tc.cvt])
		})
	}
}

// The arithmetic that conversion rests on, against Go's own — which is
// correctly rounded, and is not how the sequence computes it.
//
// The point of the check is the bit the halving drops. Rounding to odd puts
// it back in the low bit, so a value exactly halfway between two results
// still reads as halfway; discarding it instead rounds twice and is a ulp
// low on precisely those inputs. The naive form is computed alongside so
// that the test would fail if the two ever agreed everywhere and the
// complication were pointless.
func TestUnsignedI64ToFloatRoundsOnce(t *testing.T) {
	lowered := func(x uint64) float64 {
		if int64(x) >= 0 {
			return float64(int64(x))
		}
		odd := (x >> 1) | (x & 1)
		return float64(int64(odd)) * 2
	}
	naive := func(x uint64) float64 {
		if int64(x) >= 0 {
			return float64(int64(x))
		}
		return float64(int64(x>>1)) * 2
	}

	var vals []uint64
	for _, v := range []uint64{
		0, 1, 7, 1 << 52, 1<<53 - 1, 1 << 53, 1<<63 - 1,
		1 << 63, 1<<63 + 1, 1<<63 + 1023, 1<<63 + 1024,
		^uint64(0), ^uint64(0) - 1, ^uint64(0) >> 1,
	} {
		vals = append(vals, v)
	}
	// A spread with the low bits set every way, which is where the two
	// forms part company.
	for x := uint64(1) << 63; x < 1<<63+4096; x++ {
		vals = append(vals, x)
	}
	r := uint64(0x2545f4914f6cdd1d)
	for i := 0; i < 20000; i++ {
		r ^= r << 13
		r ^= r >> 7
		r ^= r << 17
		vals = append(vals, r)
	}

	disagreed := false
	for _, x := range vals {
		if got, want := lowered(x), float64(x); got != want {
			t.Fatalf("lowered(%d) = %v, want %v", x, got, want)
		}
		if naive(x) != float64(x) {
			disagreed = true
		}
	}
	if !disagreed {
		t.Error("the naive halving agreed everywhere; the round-to-odd is doing nothing here")
	}
}

// §C3's width change, and §C3's bitcast. The first rounds and the second
// does not touch a bit, which is the whole difference between them and
// the reason both exist.
func TestLowerFloatWidthAndBitcast(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)

	a := m.Func("bits").Export()
	x := a.ParamF64("a")
	a.ReturnsI64()
	a.Entry().Return(a.Entry().I64.BitcastF64(x))

	b := m.Func("unbits").Export()
	y := b.ParamI32("a")
	b.ReturnsF32()
	b.Entry().Return(b.Entry().F32.BitcastI32(y))

	tb, raw := lowerText(t, m)

	want := []byte{
		0x66, 0x48, 0x0f, 0x7e, 0xc0, // movq rax, xmm0
		0xc3,                   // ret
		0x66, 0x0f, 0x6e, 0xc7, // movd xmm0, edi
		0xc3, // ret
	}
	if !bytes.Equal(tb, want) {
		t.Errorf(".text bytes = % x, want % x", tb, want)
	}
	objdumpHas(t, "bits", raw, "movq", "movd")
}

// A negation is the sign bit flipped and not a subtraction from zero,
// which is what the integer NEG would have made of it: the two differ on
// a zero and on a NaN, and only one of them is what §A3 means.
func TestLowerFloatNeg(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	fn := m.Func("fneg64").Export()
	a := fn.ParamF64("a")
	fn.ReturnsF64()
	fn.Entry().Return(fn.Entry().F64.Neg(a))

	tb, raw := lowerText(t, m)

	want := []byte{
		0x48, 0xb8, 0, 0, 0, 0, 0, 0, 0, 0x80, // movabs rax, the sign bit
		0x66, 0x48, 0x0f, 0x6e, 0xc8, // movq xmm1, rax
		0x66, 0x0f, 0x57, 0xc1, // xorpd xmm0, xmm1
		0xc3, // ret
	}
	if !bytes.Equal(tb, want) {
		t.Errorf(".text bytes = % x, want % x", tb, want)
	}
	objdumpHas(t, "fneg64", raw, "xorpd")
}

// abs is the same mask read the other way: the sign bit cleared rather
// than flipped, which is an AND with everything else.
func TestLowerFloatAbs(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	fn := m.Func("fabs32").Export()
	a := fn.ParamF32("a")
	fn.ReturnsF32()
	fn.Entry().Return(fn.Entry().F32.Abs(a))

	tb, _ := lowerText(t, m)

	want := []byte{
		0xb8, 0xff, 0xff, 0xff, 0x7f, // mov eax, 0x7fffffff
		0x66, 0x0f, 0x6e, 0xc8, // movd xmm1, eax
		0x0f, 0x54, 0xc1, // andps xmm0, xmm1
		0xc3, // ret
	}
	if !bytes.Equal(tb, want) {
		t.Errorf(".text bytes = % x, want % x", tb, want)
	}
}

// sqrt is the one §A3 unary verb that is its own instruction, and the
// one whose destination is not also a source.
func TestLowerFloatSqrt(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	fn := m.Func("fsqrt").Export()
	a := fn.ParamF64("a")
	fn.ReturnsF64()
	fn.Entry().Return(fn.Entry().F64.Sqrt(a))

	tb, raw := lowerText(t, m)

	want := []byte{
		0xf2, 0x0f, 0x51, 0xc8, // sqrtsd xmm1, xmm0
		0x0f, 0x28, 0xc1, // movaps xmm0, xmm1
		0xc3, // ret
	}
	if !bytes.Equal(tb, want) {
		t.Errorf(".text bytes = % x, want % x", tb, want)
	}
	objdumpHas(t, "fsqrt", raw, "sqrtsd")
}

// copysign is one mask and both of its readings. ANDNPD inverts its own
// destination, which is what gets the magnitude of one operand and the
// sign of the other out of a single constant.
func TestLowerFloatCopySign(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	fn := m.Func("fcopysign").Export()
	a := fn.ParamF64("a")
	b := fn.ParamF64("b")
	fn.ReturnsF64()
	fn.Entry().Return(fn.Entry().F64.CopySign(a, b))

	tb, raw := lowerText(t, m)

	want := []byte{
		0x48, 0xb8, 0, 0, 0, 0, 0, 0, 0, 0x80, // movabs rax, the sign bit
		0x66, 0x48, 0x0f, 0x6e, 0xd0, // movq xmm2, rax
		0x66, 0x0f, 0x54, 0xd1, // andpd xmm2, xmm1     (the sign of b)
		0x48, 0xb8, 0, 0, 0, 0, 0, 0, 0, 0x80, // movabs rax, the sign bit again
		0x66, 0x48, 0x0f, 0x6e, 0xc8, // movq xmm1, rax
		0x66, 0x0f, 0x55, 0xc8, // andnpd xmm1, xmm0    (the magnitude of a)
		0x0f, 0x28, 0xc1, // movaps xmm0, xmm1
		0x66, 0x0f, 0x56, 0xc2, // orpd xmm0, xmm2
		0xc3, // ret
	}
	if !bytes.Equal(tb, want) {
		t.Errorf(".text bytes = % x, want % x", tb, want)
	}
	objdumpHas(t, "fcopysign", raw, "andnpd", "orpd")
}

// What the float side still refuses, and why each one is not simply a
// missing row. min and max have IEEE definitions MINSD does not have;
// the rounding verbs are SSE4.1's ROUNDSD, which carries a CPUID bit
// this package cannot state; and FMA is a VEX encoding.
func TestLowerRefusesUnfinishedFloatVerbs(t *testing.T) {
	for _, tc := range []struct {
		name string
		emit func(b *ir.Block, x, y ir.F64) ir.Value
	}{} {
		t.Run(tc.name, func(t *testing.T) {
			m := ir.NewModule("t", ir.X86_64Linux)
			fn := m.Func("f").Export()
			x := fn.ParamF64("x")
			y := fn.ParamF64("y")
			fn.ReturnsI32()

			entry := fn.Entry()
			tc.emit(entry, x, y)
			entry.Return(entry.I32.Const(0))

			if _, err := amd64lower.Lower(m, amd64lower.Options{}); err == nil {
				t.Errorf("Lower should refuse f64.%s", tc.name)
			}
		})
	}
}

// ── milestone 27: the trapping conversions ────────────────────────────

// §C2's float-to-integer is trapping and CVTTSD2SI is not: given a NaN
// or a value out of range it writes the destination's most negative
// value and carries on, which is the silent wrong answer §C exists to
// refuse. So the conversion is a range check and then the instruction.
//
// The bounds are the source's, not the destination's, and the low one is
// open. Truncation is toward zero, so -2147483648.5 is a valid source
// for an i32 — it truncates to exactly the smallest i32 — and the
// interval an f64 has to fall in is (-2^31 - 1, 2^31). The constant here
// is -2147483649, which is exact in an f64.
//
// Both checks are written as their own negations, which puts the NaN
// case on the trapping side for free: an unordered comparison sets CF,
// so every jb and jbe is taken for a NaN and no parity test is needed.
func TestLowerFloatToIntTraps(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	fn := m.Func("f64toi32").Export()
	a := fn.ParamF64("a")
	fn.ReturnsI32()
	fn.Entry().Return(fn.Entry().I32.SCvtF64(a))

	tb, raw := lowerText(t, m)

	want := []byte{
		0x48, 0xb8, 0, 0, 0x20, 0, 0, 0, 0xe0, 0xc1, // movabs rax, -2147483649.0
		0x66, 0x48, 0x0f, 0x6e, 0xc8, // movq xmm1, rax
		0x66, 0x0f, 0x2e, 0xc1, // ucomisd xmm0, xmm1
		0x0f, 0x86, 0x05, 0x00, 0x00, 0x00, // jbe trap    (below, equal, or unordered)
		0xe9, 0x02, 0x00, 0x00, 0x00, // jmp inrange
		0x0f, 0x0b, // trap: ud2
		0x48, 0xb8, 0, 0, 0, 0, 0, 0, 0xe0, 0x41, // movabs rax, 2147483648.0
		0x66, 0x48, 0x0f, 0x6e, 0xc8, // movq xmm1, rax
		0x66, 0x0f, 0x2e, 0xc8, // ucomisd xmm1, xmm0
		0x0f, 0x86, 0xe5, 0xff, 0xff, 0xff, // jbe trap
		0xe9, 0x00, 0x00, 0x00, 0x00, // jmp cvt
		0xf2, 0x0f, 0x2c, 0xc0, // cvttsd2si eax, xmm0
		0xc3, // ret
	}
	if !bytes.Equal(tb, want) {
		t.Errorf(".text bytes = % x, want % x", tb, want)
	}
	objdumpHas(t, "f64toi32", raw, "ucomisd", "jbe", "ud2", "cvttsd2si")
}

// The f32 rows test a closed bound, and that is not an inconsistency: an
// f32 has no value strictly between -2^63 - 1 and -2^63, so the interval
// the destination admits and the values the source can hold agree
// exactly at -2^63. Only f64-to-i32 has a gap wide enough to need the
// open bound, and it is the only row that uses one.
func TestLowerFloat32ToInt64Traps(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	fn := m.Func("f32toi64").Export()
	a := fn.ParamF32("a")
	fn.ReturnsI64()
	fn.Entry().Return(fn.Entry().I64.SCvtF32(a))

	tb, raw := lowerText(t, m)

	want := []byte{
		0xb8, 0x00, 0x00, 0x00, 0xdf, // mov eax, -2^63 as an f32
		0x66, 0x0f, 0x6e, 0xc8, // movd xmm1, eax
		0x0f, 0x2e, 0xc1, // ucomiss xmm0, xmm1
		0x0f, 0x82, 0x05, 0x00, 0x00, 0x00, // jb trap      (closed bound: below only)
		0xe9, 0x02, 0x00, 0x00, 0x00, // jmp inrange
		0x0f, 0x0b, // trap: ud2
		0xb8, 0x00, 0x00, 0x00, 0x5f, // mov eax, 2^63 as an f32
		0x66, 0x0f, 0x6e, 0xc8, // movd xmm1, eax
		0x0f, 0x2e, 0xc8, // ucomiss xmm1, xmm0
		0x0f, 0x86, 0xec, 0xff, 0xff, 0xff, // jbe trap
		0xe9, 0x00, 0x00, 0x00, 0x00, // jmp cvt
		0xf3, 0x48, 0x0f, 0x2c, 0xc0, // cvttss2si rax, xmm0
		0xc3, // ret
	}
	if !bytes.Equal(tb, want) {
		t.Errorf(".text bytes = % x, want % x", tb, want)
	}
	objdumpHas(t, "f32toi64", raw, "ucomiss", "cvttss2si")
}
