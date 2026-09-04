package amd64_test

import (
	"bytes"
	"errors"
	"testing"

	"github.com/vertex-language/ir"
	"github.com/vertex-language/ir/lower/regalloc"
)

// Tests register spills to stack.
func TestLowerSpills(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	fn := m.Func("wide").Export()
	a := fn.ParamI32("a")
	b := fn.ParamI32("b")
	fn.ReturnsI32()

	entry := fn.Entry()
	var live []ir.I32
	for i := 0; i < 13; i++ {
		live = append(live, entry.I32.Add(a, b))
	}
	sum := live[0]
	for _, v := range live[1:] {
		sum = entry.I32.Add(sum, v)
	}
	entry.Return(sum)

	tb, raw := lowerText(t, m)

	if !bytes.HasPrefix(tb, []byte{0x55, 0x48, 0x8b, 0xec}) {
		t.Errorf("no prologue; a function that spills has a frame\n% x", tb)
	}
	if !bytes.HasSuffix(tb, []byte{0xc9, 0xc3}) {
		t.Errorf("no leave before the ret\n% x", tb)
	}

	stores := bytes.Count(tb, []byte{0x48, 0x89, 0x7d}) // mov [rbp+d8], rdi..
	if stores == 0 {
		stores = bytes.Count(tb, []byte{0x4c, 0x89, 0x7d}) // ..or an extended one
	}
	if stores == 0 {
		t.Errorf("no spill store found\n% x", tb)
	}

	objdumpHas(t, "wide", raw, "pushq", "movq", "addl", "leave", "retq")
}

// Tests that pin conflicts are reported correctly instead of spilled.
func TestLowerPinConflictIsNotSpilled(t *testing.T) {
	if errors.Is(regalloc.ErrPinConflict, regalloc.ErrOutOfRegisters) {
		t.Error("ErrPinConflict and ErrOutOfRegisters are the same error; they are different failures")
	}
}

// Tests reuse of dead registers.
func TestLowerReusesDeadRegisters(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	fn := m.Func("chain").Export()
	a := fn.ParamI32("a")
	b := fn.ParamI32("b")
	fn.ReturnsI32()

	entry := fn.Entry()
	sum := entry.I32.Add(a, b)
	for i := 0; i < 11; i++ {
		sum = entry.I32.Add(sum, b)
	}
	entry.Return(sum)

	tb, raw := lowerText(t, m)

	want := []byte{
		0x8b, 0xc7, // mov eax, edi
		0x03, 0xc6, // add eax, esi   (the first sum, in EAX)
	}
	for i := 0; i < 11; i++ {
		if i%2 == 0 {
			want = append(want, 0x8b, 0xc8, 0x03, 0xce) // mov ecx, eax ; add ecx, esi
			continue
		}
		want = append(want, 0x8b, 0xc1, 0x03, 0xc6) // mov eax, ecx ; add eax, esi
	}
	want = append(want,
		0x8b, 0xc1, // mov eax, ecx   (the last sum is in ECX)
		0xc3)
	if !bytes.Equal(tb, want) {
		t.Errorf(".text bytes = % x, want % x", tb, want)
	}

	objdumpHas(t, "chain", raw, "addl", "retq")
}

// Tests that loop-carried values remain live through the loop body.
func TestLowerKeepsLoopCarriedValuesLive(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	fn := m.Func("countdown").Export()
	n := fn.ParamI32("n")
	fn.ReturnsI32()

	entry := fn.Entry()
	loop := fn.Block("loop")
	i := loop.ParamI32("i")
	body := fn.Block("body")
	exit := fn.Block("exit")

	seed := entry.I32.Add(n, n)
	entry.Br(loop.To(entry.I32.Const(0)))
	loop.BrIf(loop.I32.SLt(i, seed), body.To(), exit.To())
	body.Br(loop.To(body.I32.Add(i, body.I32.Const(1))))
	exit.Return(i)

	tb, _ := lowerText(t, m)

	cmp := bytes.Index(tb, []byte{0x3b})
	if cmp < 0 {
		t.Fatalf("no compare found in % x", tb)
	}
	seedReg := tb[cmp+1] & 7 // the modrm r/m field: cmp r32, rm32

	if bytes.Contains(tb[cmp:], []byte{0xb8 | seedReg, 0x01, 0x00, 0x00, 0x00}) {
		t.Errorf("the body's constant landed in seed's register (b8+%d)\n% x", seedReg, tb)
	}
}

// Tests register pinning hazards.
func TestLowerFixedRegisterHazards(t *testing.T) {
	t.Run("divisor already in rdx", func(t *testing.T) {
		m := ir.NewModule("t", ir.X86_64Linux)
		fn := m.Func("g").Export()
		a := fn.ParamI32("a")
		fn.ParamI32("pad")
		c := fn.ParamI32("c") // the third argument arrives in EDX
		fn.ReturnsI32()
		fn.Entry().Return(fn.Entry().I32.SDiv(a, c))

		tb, _ := lowerText(t, m)
		want := []byte{
			0x8b, 0xca, // mov ecx, edx
			0x8b, 0xc7, // mov eax, edi
			0x99,       // cdq
			0xf7, 0xf9, // idiv ecx
			0xc3,
		}
		if !bytes.Equal(tb, want) {
			t.Errorf(".text bytes = % x, want % x", tb, want)
		}
	})

	t.Run("count already in rcx", func(t *testing.T) {
		m := ir.NewModule("t", ir.X86_64Linux)
		fn := m.Func("h").Export()
		x := fn.ParamI32("x")
		fn.ParamI32("p1")
		fn.ParamI32("p2")
		amt := fn.ParamI32("amt") // the fourth argument arrives in ECX
		fn.ReturnsI32()
		fn.Entry().Return(fn.Entry().I32.Shl(x, amt))

		tb, _ := lowerText(t, m)
		want := []byte{
			0x8b, 0xc7, // mov eax, edi
			0xd3, 0xe0, // shl eax, cl
			0xc3,
		}
		if !bytes.Equal(tb, want) {
			t.Errorf(".text bytes = % x, want % x", tb, want)
		}
	})
}
