package amd64_test

import (
	"bytes"
	"testing"

	"github.com/vertex-language/ir"
	amd64lower "github.com/vertex-language/ir/lower/amd64"
)

// Tests i64 add round trip.
func TestLowerI64RoundTrip(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	fn := m.Func("add64").Export()
	a := fn.ParamI64("a")
	b := fn.ParamI64("b")
	fn.ReturnsI64()

	entry := fn.Entry()
	entry.Return(entry.I64.Add(a, b))

	tb, raw := lowerText(t, m)

	want := []byte{
		0x48, 0x8b, 0xc7, // mov rax, rdi
		0x48, 0x03, 0xc6, // add rax, rsi
		0xc3, // ret
	}
	if !bytes.Equal(tb, want) {
		t.Errorf(".text bytes = % x, want % x", tb, want)
	}

	objdumpHas(t, "add64", raw, "movq", "addq", "retq")
}

// Tests ptr addition lowering.
func TestLowerPtrAdd(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	fn := m.Func("advance").Export()
	p := fn.ParamPtr("p")
	off := fn.ParamI64("off")
	fn.ReturnsPtr()

	entry := fn.Entry()
	entry.Return(entry.Ptr.Add(p, off))

	tb, raw := lowerText(t, m)

	want := []byte{
		0x48, 0x8b, 0xc7, // mov rax, rdi
		0x48, 0x03, 0xc6, // add rax, rsi
		0xc3, // ret
	}
	if !bytes.Equal(tb, want) {
		t.Errorf(".text bytes = % x, want % x", tb, want)
	}

	objdumpHas(t, "advance", raw, "movq", "addq", "retq")
}

// Tests loading a large i64 constant.
func TestLowerI64Const(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	fn := m.Func("big").Export()
	fn.ReturnsI64()
	fn.Entry().Return(fn.Entry().I64.Const(0x1122334455667788))

	tb, raw := lowerText(t, m)

	want := []byte{
		0x48, 0xb8, 0x88, 0x77, 0x66, 0x55, 0x44, 0x33, 0x22, 0x11, // movabs rax, 0x1122334455667788
		0xc3, // ret
	}
	if !bytes.Equal(tb, want) {
		t.Errorf(".text bytes = % x, want % x", tb, want)
	}

	objdumpHas(t, "big", raw, "movabsq", "retq")
}

// Tests ptr.const which is always null.
func TestLowerPtrConstIsNull(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	fn := m.Func("nullptr").Export()
	fn.ReturnsPtr()
	fn.Entry().Return(fn.Entry().Ptr.Const())

	tb, _ := lowerText(t, m)

	want := []byte{
		0x48, 0xb8, 0, 0, 0, 0, 0, 0, 0, 0, // movabs rax, 0
		0xc3, // ret
	}
	if !bytes.Equal(tb, want) {
		t.Errorf(".text bytes = % x, want % x", tb, want)
	}
}

// Helper for testing fused compares.
func wideCompareText(opcode byte) []byte {
	return []byte{
		0x48, 0x3b, 0xfe, // cmp rdi, rsi
		0x0f, opcode, 0x05, 0x00, 0x00, 0x00, // jcc +5 (to "f.yes")
		0xe9, 0x04, 0x00, 0x00, 0x00, // jmp +4 (to "f.no")
		0x48, 0x8b, 0xc7, // yes: mov rax, rdi
		0xc3,             // ret
		0x48, 0x8b, 0xc6, // no: mov rax, rsi
		0xc3, // ret
	}
}

// Tests i64 compare conditions.
func TestLowerI64CompareConditions(t *testing.T) {
	for _, tc := range []struct {
		verb     string
		cmp      func(b *ir.Block, a, c ir.I64) ir.I1
		opcode   byte
		mnemonic string
	}{
		{"eq", func(b *ir.Block, a, c ir.I64) ir.I1 { return b.I64.Eq(a, c) }, 0x84, "je"},
		{"ne", func(b *ir.Block, a, c ir.I64) ir.I1 { return b.I64.Ne(a, c) }, 0x85, "jne"},
		{"slt", func(b *ir.Block, a, c ir.I64) ir.I1 { return b.I64.SLt(a, c) }, 0x8c, "jl"},
		{"sle", func(b *ir.Block, a, c ir.I64) ir.I1 { return b.I64.SLe(a, c) }, 0x8e, "jle"},
		{"ult", func(b *ir.Block, a, c ir.I64) ir.I1 { return b.I64.ULt(a, c) }, 0x82, "jb"},
		{"ule", func(b *ir.Block, a, c ir.I64) ir.I1 { return b.I64.ULe(a, c) }, 0x86, "jbe"},
	} {
		t.Run(tc.verb, func(t *testing.T) {
			m := ir.NewModule("t", ir.X86_64Linux)
			fn := m.Func("f").Export()
			a := fn.ParamI64("a")
			b := fn.ParamI64("b")
			fn.ReturnsI64()

			entry := fn.Entry()
			yes := fn.Block("yes")
			no := fn.Block("no")

			entry.BrIf(tc.cmp(entry, a, b), yes.To(), no.To())
			yes.Return(a)
			no.Return(b)

			tb, raw := lowerText(t, m)
			if want := wideCompareText(tc.opcode); !bytes.Equal(tb, want) {
				t.Errorf(".text bytes = % x, want % x", tb, want)
			}
			objdumpHas(t, "f", raw, "cmpq", tc.mnemonic, "retq")
		})
	}
}

// Tests ptr compare conditions.
func TestLowerPtrCompareConditions(t *testing.T) {
	for _, tc := range []struct {
		verb     string
		cmp      func(b *ir.Block, a, c ir.Ptr) ir.I1
		opcode   byte
		mnemonic string
	}{
		{"eq", func(b *ir.Block, a, c ir.Ptr) ir.I1 { return b.Ptr.Eq(a, c) }, 0x84, "je"},
		{"ne", func(b *ir.Block, a, c ir.Ptr) ir.I1 { return b.Ptr.Ne(a, c) }, 0x85, "jne"},
		{"lt", func(b *ir.Block, a, c ir.Ptr) ir.I1 { return b.Ptr.Lt(a, c) }, 0x82, "jb"},
		{"le", func(b *ir.Block, a, c ir.Ptr) ir.I1 { return b.Ptr.Le(a, c) }, 0x86, "jbe"},
	} {
		t.Run(tc.verb, func(t *testing.T) {
			m := ir.NewModule("t", ir.X86_64Linux)
			fn := m.Func("f").Export()
			a := fn.ParamPtr("a")
			b := fn.ParamPtr("b")
			fn.ReturnsPtr()

			entry := fn.Entry()
			yes := fn.Block("yes")
			no := fn.Block("no")

			entry.BrIf(tc.cmp(entry, a, b), yes.To(), no.To())
			yes.Return(a)
			no.Return(b)

			tb, raw := lowerText(t, m)
			if want := wideCompareText(tc.opcode); !bytes.Equal(tb, want) {
				t.Errorf(".text bytes = % x, want % x", tb, want)
			}
			objdumpHas(t, "f", raw, "cmpq", tc.mnemonic, "retq")
		})
	}
}

// Tests rejection of modules not meant for this target.
//
// x86_64/windows is not among them any more: its layout declares the
// Microsoft ABI, which this package now implements, and the only thing left
// that makes a module foreign is a pointer width or an endianness that is not
// this machine's.
func TestLowerRejectsForeignLayout(t *testing.T) {
	for _, tc := range []struct {
		name   string
		target ir.Target
	}{
		{"i386", ir.I386Linux},
		{"aarch64", ir.AArch64Linux},
	} {
		t.Run(tc.name, func(t *testing.T) {
			m := ir.NewModule("t", tc.target)
			fn := m.Func("add").Export()
			a := fn.ParamI32("a")
			b := fn.ParamI32("b")
			fn.ReturnsI32()
			fn.Entry().Return(fn.Entry().I32.Add(a, b))

			if _, err := amd64lower.Lower(m, amd64lower.Options{}); err == nil {
				t.Errorf("Lower should refuse a %s module", tc.name)
			}
		})
	}
}

// Tests binary operations at both widths.
func TestLowerBinaryOps(t *testing.T) {
	for _, tc := range []struct {
		name   string
		i32    func(b *ir.Block, x, y ir.I32) ir.I32
		i64    func(b *ir.Block, x, y ir.I64) ir.I64
		op32   []byte
		op64   []byte
		mnem32 string
	}{
		{
			"add",
			func(b *ir.Block, x, y ir.I32) ir.I32 { return b.I32.Add(x, y) },
			func(b *ir.Block, x, y ir.I64) ir.I64 { return b.I64.Add(x, y) },
			[]byte{0x03, 0xc6}, []byte{0x48, 0x03, 0xc6}, "addl",
		},
		{
			"sub",
			func(b *ir.Block, x, y ir.I32) ir.I32 { return b.I32.Sub(x, y) },
			func(b *ir.Block, x, y ir.I64) ir.I64 { return b.I64.Sub(x, y) },
			[]byte{0x2b, 0xc6}, []byte{0x48, 0x2b, 0xc6}, "subl",
		},
		{
			"mul",
			func(b *ir.Block, x, y ir.I32) ir.I32 { return b.I32.Mul(x, y) },
			func(b *ir.Block, x, y ir.I64) ir.I64 { return b.I64.Mul(x, y) },
			[]byte{0x0f, 0xaf, 0xc6}, []byte{0x48, 0x0f, 0xaf, 0xc6}, "imull",
		},
		{
			"and",
			func(b *ir.Block, x, y ir.I32) ir.I32 { return b.I32.And(x, y) },
			func(b *ir.Block, x, y ir.I64) ir.I64 { return b.I64.And(x, y) },
			[]byte{0x23, 0xc6}, []byte{0x48, 0x23, 0xc6}, "andl",
		},
		{
			"or",
			func(b *ir.Block, x, y ir.I32) ir.I32 { return b.I32.Or(x, y) },
			func(b *ir.Block, x, y ir.I64) ir.I64 { return b.I64.Or(x, y) },
			[]byte{0x0b, 0xc6}, []byte{0x48, 0x0b, 0xc6}, "orl",
		},
		{
			"xor",
			func(b *ir.Block, x, y ir.I32) ir.I32 { return b.I32.Xor(x, y) },
			func(b *ir.Block, x, y ir.I64) ir.I64 { return b.I64.Xor(x, y) },
			[]byte{0x33, 0xc6}, []byte{0x48, 0x33, 0xc6}, "xorl",
		},
	} {
		t.Run(tc.name+"/i32", func(t *testing.T) {
			m := ir.NewModule("t", ir.X86_64Linux)
			fn := m.Func("f").Export()
			a := fn.ParamI32("a")
			b := fn.ParamI32("b")
			fn.ReturnsI32()
			fn.Entry().Return(tc.i32(fn.Entry(), a, b))

			tb, raw := lowerText(t, m)
			want := append([]byte{0x8b, 0xc7}, tc.op32...) // mov eax, edi
			want = append(want, 0xc3)
			if !bytes.Equal(tb, want) {
				t.Errorf(".text bytes = % x, want % x", tb, want)
			}
			objdumpHas(t, "f", raw, tc.mnem32)
		})

		t.Run(tc.name+"/i64", func(t *testing.T) {
			m := ir.NewModule("t", ir.X86_64Linux)
			fn := m.Func("f").Export()
			a := fn.ParamI64("a")
			b := fn.ParamI64("b")
			fn.ReturnsI64()
			fn.Entry().Return(tc.i64(fn.Entry(), a, b))

			tb, _ := lowerText(t, m)
			want := append([]byte{0x48, 0x8b, 0xc7}, tc.op64...) // mov rax, rdi
			want = append(want, 0xc3)
			if !bytes.Equal(tb, want) {
				t.Errorf(".text bytes = % x, want % x", tb, want)
			}
		})
	}
}

// Tests unary ops.
func TestLowerUnaryOps(t *testing.T) {
	for _, tc := range []struct {
		name string
		un   func(b *ir.Block, x ir.I32) ir.I32
		op   []byte
		mnem string
	}{
		{"neg", func(b *ir.Block, x ir.I32) ir.I32 { return b.I32.Neg(x) }, []byte{0xf7, 0xd8}, "negl"},
		{"not", func(b *ir.Block, x ir.I32) ir.I32 { return b.I32.Not(x) }, []byte{0xf7, 0xd0}, "notl"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			m := ir.NewModule("t", ir.X86_64Linux)
			fn := m.Func("f").Export()
			a := fn.ParamI32("a")
			fn.ReturnsI32()
			fn.Entry().Return(tc.un(fn.Entry(), a))

			tb, raw := lowerText(t, m)
			want := append([]byte{0x8b, 0xc7}, tc.op...) // mov eax, edi
			want = append(want, 0xc3)
			if !bytes.Equal(tb, want) {
				t.Errorf(".text bytes = % x, want % x", tb, want)
			}
			objdumpHas(t, "f", raw, tc.mnem)
		})
	}
}

// Tests pointer subtraction lowering.
func TestLowerPtrSub(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	fn := m.Func("back").Export()
	p := fn.ParamPtr("p")
	off := fn.ParamI64("off")
	fn.ReturnsPtr()
	fn.Entry().Return(fn.Entry().Ptr.Sub(p, off))

	tb, _ := lowerText(t, m)
	want := []byte{
		0x48, 0x8b, 0xc7, // mov rax, rdi
		0x48, 0x2b, 0xc6, // sub rax, rsi
		0xc3,
	}
	if !bytes.Equal(tb, want) {
		t.Errorf(".text bytes = % x, want % x", tb, want)
	}
}

// Tests integer divisions for i32.
func TestLowerDivisions(t *testing.T) {
	for _, tc := range []struct {
		name string
		bin  func(b *ir.Block, x, y ir.I32) ir.I32
		want []byte
		mnem string
	}{
		{
			"sdiv",
			func(b *ir.Block, x, y ir.I32) ir.I32 { return b.I32.SDiv(x, y) },
			[]byte{
				0x8b, 0xc7, // mov eax, edi
				0x99,       // cdq
				0xf7, 0xfe, // idiv esi
				0xc3, // ret
			},
			"idivl",
		},
		{
			"udiv",
			func(b *ir.Block, x, y ir.I32) ir.I32 { return b.I32.UDiv(x, y) },
			[]byte{
				0x8b, 0xc7, // mov eax, edi
				0x33, 0xd2, // xor edx, edx
				0xf7, 0xf6, // div esi
				0xc3,
			},
			"divl",
		},
		{
			"srem",
			func(b *ir.Block, x, y ir.I32) ir.I32 { return b.I32.SRem(x, y) },
			[]byte{
				0x8b, 0xc7, // mov eax, edi
				0x99,       // cdq
				0xf7, 0xfe, // idiv esi
				0x8b, 0xc2, // mov eax, edx
				0xc3,
			},
			"idivl",
		},
		{
			"urem",
			func(b *ir.Block, x, y ir.I32) ir.I32 { return b.I32.URem(x, y) },
			[]byte{
				0x8b, 0xc7, // mov eax, edi
				0x33, 0xd2, // xor edx, edx
				0xf7, 0xf6, // div esi
				0x8b, 0xc2, // mov eax, edx
				0xc3,
			},
			"divl",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			m := ir.NewModule("t", ir.X86_64Linux)
			fn := m.Func("f").Export()
			a := fn.ParamI32("a")
			b := fn.ParamI32("b")
			fn.ReturnsI32()
			fn.Entry().Return(tc.bin(fn.Entry(), a, b))

			tb, raw := lowerText(t, m)
			if !bytes.Equal(tb, tc.want) {
				t.Errorf(".text bytes = % x, want % x", tb, tc.want)
			}
			objdumpHas(t, "f", raw, tc.mnem)
		})
	}
}

// Tests integer divisions for i64.
func TestLowerDivision64(t *testing.T) {
	for _, tc := range []struct {
		name string
		bin  func(b *ir.Block, x, y ir.I64) ir.I64
		want []byte
	}{
		{
			"sdiv",
			func(b *ir.Block, x, y ir.I64) ir.I64 { return b.I64.SDiv(x, y) },
			[]byte{
				0x48, 0x8b, 0xc7, // mov rax, rdi
				0x48, 0x99, // cqo
				0x48, 0xf7, 0xfe, // idiv rsi
				0xc3,
			},
		},
		{
			"udiv",
			func(b *ir.Block, x, y ir.I64) ir.I64 { return b.I64.UDiv(x, y) },
			[]byte{
				0x48, 0x8b, 0xc7, // mov rax, rdi
				0x33, 0xd2, // xor edx, edx
				0x48, 0xf7, 0xf6, // div rsi
				0xc3,
			},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			m := ir.NewModule("t", ir.X86_64Linux)
			fn := m.Func("f").Export()
			a := fn.ParamI64("a")
			b := fn.ParamI64("b")
			fn.ReturnsI64()
			fn.Entry().Return(tc.bin(fn.Entry(), a, b))

			tb, _ := lowerText(t, m)
			if !bytes.Equal(tb, tc.want) {
				t.Errorf(".text bytes = % x, want % x", tb, tc.want)
			}
		})
	}
}

// Tests integer shifts.
func TestLowerShifts(t *testing.T) {
	for _, tc := range []struct {
		name string
		bin  func(b *ir.Block, x, y ir.I32) ir.I32
		op   byte
		mnem string
	}{
		{"shl", func(b *ir.Block, x, y ir.I32) ir.I32 { return b.I32.Shl(x, y) }, 0xe0, "shll"},
		{"sshr", func(b *ir.Block, x, y ir.I32) ir.I32 { return b.I32.SShr(x, y) }, 0xf8, "sarl"},
		{"ushr", func(b *ir.Block, x, y ir.I32) ir.I32 { return b.I32.UShr(x, y) }, 0xe8, "shrl"},
		{"rotl", func(b *ir.Block, x, y ir.I32) ir.I32 { return b.I32.RotL(x, y) }, 0xc0, "roll"},
		{"rotr", func(b *ir.Block, x, y ir.I32) ir.I32 { return b.I32.RotR(x, y) }, 0xc8, "rorl"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			m := ir.NewModule("t", ir.X86_64Linux)
			fn := m.Func("f").Export()
			a := fn.ParamI32("a")
			b := fn.ParamI32("b")
			fn.ReturnsI32()
			fn.Entry().Return(tc.bin(fn.Entry(), a, b))

			tb, raw := lowerText(t, m)
			want := []byte{
				0x8b, 0xce, // mov ecx, esi
				0x8b, 0xc7, // mov eax, edi
				0xd3, tc.op, // shift eax by cl
				0xc3,
			}
			if !bytes.Equal(tb, want) {
				t.Errorf(".text bytes = % x, want % x", tb, want)
			}
			objdumpHas(t, "f", raw, tc.mnem)
		})
	}
}

// Tests lowering of a select acting as a max function.
func TestLowerSelect(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	fn := m.Func("mx").Export()
	a := fn.ParamI32("a")
	b := fn.ParamI32("b")
	fn.ReturnsI32()

	entry := fn.Entry()
	entry.Return(entry.I32.Select(entry.I32.SLt(a, b), b, a))

	tb, raw := lowerText(t, m)

	want := []byte{
		0x3b, 0xfe, // cmp edi, esi
		0x8b, 0xc7, // mov eax, edi
		0x0f, 0x4c, 0xc6, // cmovl eax, esi
		0xc3, // ret
	}
	if !bytes.Equal(tb, want) {
		t.Errorf(".text bytes = % x, want % x", tb, want)
	}

	objdumpHas(t, "mx", raw, "cmpl", "cmovl", "retq")
}

// Tests 64-bit unsigned select.
func TestLowerSelect64Unsigned(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	fn := m.Func("mx64").Export()
	a := fn.ParamI64("a")
	b := fn.ParamI64("b")
	fn.ReturnsI64()

	entry := fn.Entry()
	entry.Return(entry.I64.Select(entry.I64.ULt(a, b), b, a))

	tb, raw := lowerText(t, m)

	want := []byte{
		0x48, 0x3b, 0xfe, // cmp rdi, rsi
		0x48, 0x8b, 0xc7, // mov rax, rdi
		0x48, 0x0f, 0x42, 0xc6, // cmovb rax, rsi
		0xc3, // ret
	}
	if !bytes.Equal(tb, want) {
		t.Errorf(".text bytes = % x, want % x", tb, want)
	}

	objdumpHas(t, "mx64", raw, "cmpq", "cmovb", "retq")
}

// Tests select driven by an i1 value rather than a compare.
func TestLowerSelectOnValue(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	fn := m.Func("sel").Export()
	c := fn.ParamI1("c")
	a := fn.ParamI32("a")
	b := fn.ParamI32("b")
	fn.ReturnsI32()

	entry := fn.Entry()
	entry.Return(entry.I32.Select(c, a, b))

	tb, raw := lowerText(t, m)

	want := []byte{
		0x40, 0x84, 0xff, // test dil, dil
		0x8b, 0xc2, // mov eax, edx
		0x0f, 0x45, 0xc6, // cmovne eax, esi
		0xc3, // ret
	}
	if !bytes.Equal(tb, want) {
		t.Errorf(".text bytes = % x, want % x", tb, want)
	}
	objdumpHas(t, "sel", raw, "testb", "cmovnel")
}
