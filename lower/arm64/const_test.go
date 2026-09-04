package arm64_test

// §A7's constants, which are where a fixed instruction width shows most: no
// A64 instruction takes a literal wider than sixteen bits, so anything wider
// is a sequence and the sequence is chosen per value.

import (
	"testing"

	"github.com/vertex-language/ir"
)

// One MOVZ, which is every constant that fits in a halfword.
func TestLowerConstNarrow(t *testing.T) {
	m := ir.NewModule("t", ir.AArch64Linux)
	fn := m.Func("k").Export()
	fn.ReturnsI64()
	entry := fn.Entry()
	entry.Return(entry.I64.Const(0x1234))

	got, raw := lowerWords(t, m)
	equalWords(t, got, []uint32{
		0xd2824680, // mov x0, #4660
		0xd65f03c0, // ret
	})
	objdumpHas(t, "k", raw, "mov")
}

// One MOVN, because every halfword but the lowest is already ones. Starting
// from MOVZ would have cost four instructions for a value C writes as -1.
func TestLowerConstMinusOne(t *testing.T) {
	m := ir.NewModule("t", ir.AArch64Linux)
	fn := m.Func("k").Export()
	fn.ReturnsI64()
	entry := fn.Entry()
	entry.Return(entry.I64.Const(-1))

	got, raw := lowerWords(t, m)
	equalWords(t, got, []uint32{
		0x92800000, // mov x0, #-1
		0xd65f03c0, // ret
	})
	objdumpHas(t, "k", raw, "mov")
}

// A MOVZ and three MOVKs: four halfwords, none of them free.
func TestLowerConstWide(t *testing.T) {
	m := ir.NewModule("t", ir.AArch64Linux)
	fn := m.Func("k").Export()
	fn.ReturnsI64()
	entry := fn.Entry()
	entry.Return(entry.I64.Const(0x1234_5678_9abc_def0))

	got, raw := lowerWords(t, m)
	equalWords(t, got, []uint32{
		0xd29bde00, // mov  x0, #57072
		0xf2b35780, // movk x0, #39612, lsl #16
		0xf2cacf00, // movk x0, #22136, lsl #32
		0xf2e24680, // movk x0, #4660,  lsl #48
		0xd65f03c0, // ret
	})
	objdumpHas(t, "k", raw, "movk")
}

// Every one of these runs, and computes what C says the same literal is.
// A wide constant was refused outright before the sequence was written; a
// wrong one would be a wrong answer, which is what this checks.
func TestRunConstants(t *testing.T) {
	m := ir.NewModule("t", ir.AArch64Linux)

	vals := []int64{
		0, 1, -1, 0xffff, -0x10000,
		0x1234_5678_9abc_def0, -0x1234_5678_9abc_def0,
		0x7fff_ffff_ffff_ffff, -0x8000_0000_0000_0000,
		0xffff_0000_ffff_0000 - 1<<63 - 1<<63, // sign-extended
		0x0000_ffff_0000_0000,
	}

	src := "#include <stdio.h>\nstatic int fail = 0;\n"
	for i, v := range vals {
		name := "_k" + itoa(i)
		fn := m.Func(name).Export()
		fn.ReturnsI64()
		entry := fn.Entry()
		entry.Return(entry.I64.Const(v))
		src += "long " + name[1:] + "(void);\n"
	}
	src += "int main(void) {\n"
	for i, v := range vals {
		src += "  { long got = k" + itoa(i) + "(); long want = " + i64lit(v) + ";\n"
		src += "    if (got != want) { printf(\"k" + itoa(i) + ": got %ld want %ld\\n\", got, want); fail = 1; } }\n"
	}
	src += "  printf(\"%s\\n\", fail ? \"MISMATCH\" : \"ok\");\n  return 0;\n}\n"

	if got := runNative(t, m, src); got != "ok\n" {
		t.Errorf("printed %q, want %q", got, "ok\n")
	}
}

func itoa(i int) string {
	if i < 10 {
		return string(rune('0' + i))
	}
	return itoa(i/10) + string(rune('0'+i%10))
}

// i64lit writes v as C source. INT64_MIN has no literal form — the token
// 9223372036854775808 does not fit a long — so it is written as the
// expression C itself writes it as.
func i64lit(v int64) string {
	if v == -1<<63 {
		return "(-9223372036854775807L - 1)"
	}
	neg := ""
	u := uint64(v)
	if v < 0 {
		neg, u = "-", uint64(-v)
	}
	return neg + "0x" + hex(u) + "L"
}

func hex(u uint64) string {
	const digits = "0123456789abcdef"
	if u < 16 {
		return string(digits[u])
	}
	return hex(u/16) + string(digits[u%16])
}
