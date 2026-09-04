package amd64_test

// Milestone 14: calls.
//
// A call is where every mechanism the milestones before it built has to
// hold at once — the frame, liveness, the interference graph, pinned
// vregs, coalescing — so these tests read the whole function rather than
// one instruction.

import (
	"bytes"
	"fmt"
	"testing"

	elfcore "github.com/vertex-language/elf"

	"github.com/vertex-language/ir"
	amd64lower "github.com/vertex-language/ir/lower/amd64"
)

// dbl is an i32 -> i32 import, which is the smallest callee worth having.
func dblSig() *ir.Sig { return ir.NewSig().Param(ir.TypeI32).Ret(ir.TypeI32) }

// The simplest call: one argument, already in the register SysV wants it
// in, and a result taken straight from RAX.
//
// A function that calls anything has a prologue whether or not it needs
// storage: SysV wants RSP 16-byte aligned at a call, entry leaves it
// eight off that, and the push of RBP is what puts it back. There is no
// subtraction here because nothing asked for a byte of frame — no
// allocation, no spill, and no callee-saved register used.
func TestLowerCall(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	dbl := m.ImportFunc("dbl", dblSig())

	fn := m.Func("callit").Export()
	a := fn.ParamI32("a")
	fn.ReturnsI32()
	entry := fn.Entry()
	entry.Return(entry.Call(dbl, a).Value(0))

	tb, raw := lowerText(t, m)

	want := []byte{
		0x55,             // push rbp
		0x48, 0x8b, 0xec, // mov rbp, rsp
		0xe8, 0x00, 0x00, 0x00, 0x00, // call dbl
		0xc9, // leave
		0xc3, // ret
	}
	if !bytes.Equal(tb, want) {
		t.Errorf(".text bytes = % x, want % x", tb, want)
	}

	// Not one move. The argument is already in EDI and the result is
	// already in EAX, and both copies coalesced to nothing.
	objdumpHas(t, "callit", raw, "callq", "leave", "retq")
}

// The case a call exists to make hard: a value the caller still needs
// after the callee has run.
//
// Every caller-saved register is a destination of the call instruction,
// so anything live across it interferes with all nine and the allocator
// cannot leave it in one. The pool's callee-saved tail is where it goes
// instead — RBX here — and the prologue is what makes that legal, by
// saving RBX into the frame before the function touches it. Sixteen
// bytes of frame, which is one slot rounded up to the stack alignment:
// the save area is sized to the registers actually used, which is only
// knowable once allocation has run.
func TestLowerCallAcrossLiveValue(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	dbl := m.ImportFunc("dbl", dblSig())

	fn := m.Func("across").Export()
	a := fn.ParamI32("a")
	fn.ReturnsI32()
	entry := fn.Entry()
	r := entry.Call(dbl, a).Value(0).(ir.I32)
	entry.Return(entry.I32.Add(r, a)) // a is read after the call

	tb, raw := lowerText(t, m)

	want := []byte{
		0x55,             // push rbp
		0x48, 0x8b, 0xec, // mov rbp, rsp
		0x48, 0x81, 0xec, 0x10, 0x00, 0x00, 0x00, // sub rsp, 16
		0x48, 0x89, 0x5d, 0xf8, // mov [rbp-8], rbx   (save what we are about to use)
		0x8b, 0xdf, // mov ebx, edi       (a, somewhere the call cannot reach)
		0x8b, 0xfb, // mov edi, ebx       (and back out, as the argument)
		0xe8, 0x00, 0x00, 0x00, 0x00, // call dbl
		0x8b, 0xc8, // mov ecx, eax       (the result)
		0x03, 0xcb, // add ecx, ebx       (plus a, which survived)
		0x8b, 0xc1, // mov eax, ecx
		0x48, 0x8b, 0x5d, 0xf8, // mov rbx, [rbp-8]   (restore)
		0xc9, // leave
		0xc3, // ret
	}
	if !bytes.Equal(tb, want) {
		t.Errorf(".text bytes = % x, want % x", tb, want)
	}

	objdumpHas(t, "across", raw, "callq", "addl", "leave", "retq")
}

// A call to a symbol this module imports is a relocation like any other
// reference to one. RefPC32 and not RefPLT32: this object does not go
// through the procedure linkage table, which is a linkage decision and
// belongs in Options the day Options carries one.
func TestLowerCallRelocation(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	dbl := m.ImportFunc("dbl", dblSig())

	fn := m.Func("callit").Export()
	a := fn.ParamI32("a")
	fn.ReturnsI32()
	entry := fn.Entry()
	entry.Return(entry.Call(dbl, a).Value(0))

	f := lowerFile(t, m)

	if s := symbol(t, f, "dbl"); !s.Undefined() {
		t.Errorf("dbl names section %d; an imported callee is defined elsewhere", s.Shndx)
	}

	relocs, err := f.Section(".text").Relocs()
	if err != nil {
		t.Fatalf(".text Relocs: %v", err)
	}
	if len(relocs) != 1 {
		t.Fatalf("found %d relocations in .text, want 1", len(relocs))
	}
	r := relocs[0]
	if r.Sym == nil || r.Sym.Name != "dbl" {
		t.Errorf("relocation names %v, want dbl", r.Sym)
	}
	// PLT32, which is what a call is: it is what clang emits for one, and
	// the linker relaxes it to a direct branch where the target turns out
	// to be local. The Mach-O container has no equivalent slack — there the
	// same kind is X86_64_RELOC_BRANCH, and it is the only thing that tells
	// the linker to route a call to an import through a stub.
	if elfcore.RelocX86_64(r.Type) != elfcore.R_X86_64_PLT32 {
		t.Errorf("relocation type = %v, want R_X86_64_PLT32", elfcore.RelocX86_64(r.Type))
	}
	if r.Addend != -4 {
		t.Errorf("relocation addend = %d, want -4 — the displacement is from the end of the call", r.Addend)
	}
}

// A call that returns nothing still clobbers RAX, and simply never reads
// the destination that says so.
func TestLowerCallVoid(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	side := m.ImportFunc("side", ir.NewSig())

	fn := m.Func("v").Export()
	fn.ReturnsI32()
	entry := fn.Entry()
	entry.Call(side)
	entry.Return(entry.I32.Const(0))

	tb, _ := lowerText(t, m)

	want := []byte{
		0x55,             // push rbp
		0x48, 0x8b, 0xec, // mov rbp, rsp
		0xe8, 0x00, 0x00, 0x00, 0x00, // call side
		0xb8, 0x00, 0x00, 0x00, 0x00, // mov eax, 0
		0xc9, // leave
		0xc3, // ret
	}
	if !bytes.Equal(tb, want) {
		t.Errorf(".text bytes = % x, want % x", tb, want)
	}
}

// Six arguments is the whole of what SysV puts in registers, and each
// lands in its own. The seventh needs a stack slot, which is refused.
func TestLowerCallSixArguments(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	sig := ir.NewSig()
	for i := 0; i < 6; i++ {
		sig.Param(ir.TypeI32)
	}
	sig.Ret(ir.TypeI32)
	six := m.ImportFunc("six", sig)

	fn := m.Func("f").Export()
	var p [6]ir.I32
	for i := range p {
		p[i] = fn.ParamI32("p")
	}
	fn.ReturnsI32()
	entry := fn.Entry()
	entry.Return(entry.Call(six, p[0], p[1], p[2], p[3], p[4], p[5]).Value(0))

	tb, _ := lowerText(t, m)

	// Every argument is already in the register its position names, so
	// the six copies are all coalesced away and the call stands alone.
	want := []byte{
		0x55,             // push rbp
		0x48, 0x8b, 0xec, // mov rbp, rsp
		0xe8, 0x00, 0x00, 0x00, 0x00, // call six
		0xc9, // leave
		0xc3, // ret
	}
	if !bytes.Equal(tb, want) {
		t.Errorf(".text bytes = % x, want % x", tb, want)
	}
}

func TestLowerRejectsUnsupportedCalls(t *testing.T) {
	// Three of a kind is where the registers run out. Two integers come
	// back in RAX and RDX and two floats in XMM0 and XMM1; a third of
	// either goes through memory the caller allocated, which is sret and
	// is a mechanism this package does not have.
	t.Run("three integer results", func(t *testing.T) {
		m := ir.NewModule("t", ir.X86_64Linux)
		g := m.ImportFunc("g", ir.NewSig().Ret(ir.TypeI32).Ret(ir.TypeI32).Ret(ir.TypeI32))

		fn := m.Func("f").Export()
		fn.ReturnsI32()
		entry := fn.Entry()
		entry.Return(entry.Call(g).Value(0))

		if _, err := amd64lower.Lower(m, amd64lower.Options{}); err == nil {
			t.Error("Lower should refuse a three-integer result; the third comes back through memory")
		}
	})

	t.Run("three float results", func(t *testing.T) {
		m := ir.NewModule("t", ir.X86_64Linux)
		h := m.ImportFunc("h", ir.NewSig().Ret(ir.TypeF64).Ret(ir.TypeF64).Ret(ir.TypeF64))

		fn := m.Func("f").Export()
		fn.ReturnsF64()
		entry := fn.Entry()
		entry.Return(entry.Call(h).Value(0))

		if _, err := amd64lower.Lower(m, amd64lower.Options{}); err == nil {
			t.Error("Lower should refuse a three-float result; the third comes back through memory")
		}
	})

	// The same limit read from the other side: it is the return that
	// cannot place them, not only the call.
	t.Run("returning three integers", func(t *testing.T) {
		m := ir.NewModule("t", ir.X86_64Linux)
		fn := m.Func("f").Export()
		a := fn.ParamI32("a")
		fn.ReturnsI32().ReturnsI32().ReturnsI32()
		entry := fn.Entry()
		entry.Return(a, a, a)

		if _, err := amd64lower.Lower(m, amd64lower.Options{}); err == nil {
			t.Error("Lower should refuse a three-integer return")
		}
	})
}

// ── milestone 32: the float side of the stack ─────────────────────────

// Ten float arguments: eight in XMM0 through XMM7 and two in the
// outgoing area, which is what an integer past the sixth has done since
// milestone 18. The two files run out at different counts; what happens
// past the count is the same thing.
func TestLowerCallTenFloatArguments(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	sig := ir.NewSig()
	for i := 0; i < 10; i++ {
		sig = sig.Param(ir.TypeF64)
	}
	g := m.ImportFunc("g", sig)

	fn := m.Func("ten").Export()
	x := fn.ParamF64("x")
	entry := fn.Entry()
	entry.Call(g, x, x, x, x, x, x, x, x, x, x)
	entry.Return()

	tb, raw := lowerText(t, m)

	// x is parked in XMM7 first, because XMM0 — where it arrived — is
	// also the first argument register and is about to be written.
	want := []byte{
		0x55,             // push rbp
		0x48, 0x8b, 0xec, // mov rbp, rsp
		0x48, 0x81, 0xec, 0x10, 0x00, 0x00, 0x00, // sub rsp, 16
		0x0f, 0x28, 0xf8, // movaps xmm7, xmm0
		0xf2, 0x0f, 0x11, 0x3c, 0x24, // movsd [rsp], xmm7
		0xf2, 0x0f, 0x11, 0x7c, 0x24, 0x08, // movsd [rsp+8], xmm7
		0x0f, 0x28, 0xc7, // movaps xmm0, xmm7
		0x0f, 0x28, 0xcf, // movaps xmm1, xmm7
		0x0f, 0x28, 0xd7, // movaps xmm2, xmm7
		0x0f, 0x28, 0xdf, // movaps xmm3, xmm7
		0x0f, 0x28, 0xe7, // movaps xmm4, xmm7
		0x0f, 0x28, 0xef, // movaps xmm5, xmm7
		0x0f, 0x28, 0xf7, // movaps xmm6, xmm7
		0xe8, 0x00, 0x00, 0x00, 0x00, // call g
		0xc9, // leave
		0xc3, // ret
	}
	if !bytes.Equal(tb, want) {
		t.Errorf(".text bytes = % x, want % x", tb, want)
	}
	objdumpHas(t, "ten", raw, "movsd", "callq")
}

// Both files overflowing at once. There is one queue of stack slots, not
// one per class: §3.2.3 gives the stack a single sequence, so the
// seventh integer takes slot zero and the ninth float takes slot one,
// in the order the arguments are written.
func TestLowerCallMixedStackArguments(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	sig := ir.NewSig()
	for i := 0; i < 7; i++ {
		sig = sig.Param(ir.TypeI32)
	}
	for i := 0; i < 9; i++ {
		sig = sig.Param(ir.TypeF64)
	}
	g := m.ImportFunc("g", sig)

	fn := m.Func("mix").Export()
	a := fn.ParamI32("a")
	x := fn.ParamF64("x")
	entry := fn.Entry()
	entry.Call(g, a, a, a, a, a, a, a, x, x, x, x, x, x, x, x, x)
	entry.Return()

	tb, raw := lowerText(t, m)

	want := []byte{
		0x55,             // push rbp
		0x48, 0x8b, 0xec, // mov rbp, rsp
		0x48, 0x81, 0xec, 0x10, 0x00, 0x00, 0x00, // sub rsp, 16
		0x44, 0x8b, 0xcf, // mov r9d, edi
		0x0f, 0x28, 0xf8, // movaps xmm7, xmm0
		0x44, 0x89, 0x0c, 0x24, // mov [rsp], r9d          (the seventh integer)
		0xf2, 0x0f, 0x11, 0x7c, 0x24, 0x08, // movsd [rsp+8], xmm7  (the ninth float)
		0x41, 0x8b, 0xf9, // mov edi, r9d
		0x41, 0x8b, 0xf1, // mov esi, r9d
		0x41, 0x8b, 0xd1, // mov edx, r9d
		0x41, 0x8b, 0xc9, // mov ecx, r9d
		0x45, 0x8b, 0xc1, // mov r8d, r9d
		0x0f, 0x28, 0xc7, // movaps xmm0, xmm7
		0x0f, 0x28, 0xcf, // movaps xmm1, xmm7
		0x0f, 0x28, 0xd7, // movaps xmm2, xmm7
		0x0f, 0x28, 0xdf, // movaps xmm3, xmm7
		0x0f, 0x28, 0xe7, // movaps xmm4, xmm7
		0x0f, 0x28, 0xef, // movaps xmm5, xmm7
		0x0f, 0x28, 0xf7, // movaps xmm6, xmm7
		0xe8, 0x00, 0x00, 0x00, 0x00, // call g
		0xc9, // leave
		0xc3, // ret
	}
	if !bytes.Equal(tb, want) {
		t.Errorf(".text bytes = % x, want % x", tb, want)
	}
	objdumpHas(t, "mix", raw, "callq")
}

// The incoming side: a float parameter past the eighth is read from
// above RBP, where the caller left it.
func TestLowerNinthFloatParameter(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	fn := m.Func("nf").Export()
	var last ir.F64
	for i := 0; i < 10; i++ {
		last = fn.ParamF64(fmt.Sprintf("x%d", i))
	}
	fn.ReturnsF64()
	entry := fn.Entry()
	entry.Return(last)

	tb, raw := lowerText(t, m)

	// Two loads, and the first is dead: every stack parameter is loaded
	// whether or not the body reads it, and there is no pass here to
	// notice. The second is the one asked for, at RBP+24 — the tenth
	// parameter is the second stack slot, and the slots start at RBP+16.
	want := []byte{
		0x55,             // push rbp
		0x48, 0x8b, 0xec, // mov rbp, rsp
		0xf2, 0x0f, 0x10, 0x45, 0x10, // movsd xmm0, [rbp+16]
		0xf2, 0x0f, 0x10, 0x45, 0x18, // movsd xmm0, [rbp+24]
		0xc9, // leave
		0xc3, // ret
	}
	if !bytes.Equal(tb, want) {
		t.Errorf(".text bytes = % x, want % x", tb, want)
	}
	objdumpHas(t, "nf", raw, "movsd", "retq")
}

// ── milestone 30: the second return register, and the variadic count ──

// Two integer results: RAX and RDX, which is what §3.2.3 gives an
// INTEGER-class return of two eightbytes.
//
// Both are read here, because a result nothing reads would not prove the
// copy out of RDX happened.
func TestLowerCallTwoIntResults(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	pair := m.ImportFunc("pair", ir.NewSig().Ret(ir.TypeI32).Ret(ir.TypeI32))

	fn := m.Func("sumpair").Export()
	fn.ReturnsI32()
	entry := fn.Entry()
	r := entry.Call(pair)
	entry.Return(entry.I32.Add(r.Value(0).(ir.I32), r.Value(1).(ir.I32)))

	tb, raw := lowerText(t, m)

	want := []byte{
		0x55,             // push rbp
		0x48, 0x8b, 0xec, // mov rbp, rsp
		0xe8, 0x00, 0x00, 0x00, 0x00, // call pair
		0x8b, 0xc8, // mov ecx, eax   (the first result out of RAX)
		0x03, 0xca, // add ecx, edx   (the second, still in RDX)
		0x8b, 0xc1, // mov eax, ecx
		0xc9, // leave
		0xc3, // ret
	}
	if !bytes.Equal(tb, want) {
		t.Errorf(".text bytes = % x, want % x", tb, want)
	}
	objdumpHas(t, "sumpair", raw, "callq", "addl", "retq")
}

// Two float results: the other register file's pair, XMM0 and XMM1.
func TestLowerCallTwoFloatResults(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	two := m.ImportFunc("twof", ir.NewSig().Ret(ir.TypeF64).Ret(ir.TypeF64))

	fn := m.Func("addf").Export()
	fn.ReturnsF64()
	entry := fn.Entry()
	r := entry.Call(two)
	entry.Return(entry.F64.Add(r.Value(0).(ir.F64), r.Value(1).(ir.F64)))

	tb, raw := lowerText(t, m)

	want := []byte{
		0x55,             // push rbp
		0x48, 0x8b, 0xec, // mov rbp, rsp
		0xe8, 0x00, 0x00, 0x00, 0x00, // call twof
		0x0f, 0x28, 0xd0, // movaps xmm2, xmm0
		0xf2, 0x0f, 0x58, 0xd1, // addsd xmm2, xmm1
		0x0f, 0x28, 0xc2, // movaps xmm0, xmm2
		0xc9, // leave
		0xc3, // ret
	}
	if !bytes.Equal(tb, want) {
		t.Errorf(".text bytes = % x, want % x", tb, want)
	}
	objdumpHas(t, "addf", raw, "callq", "addsd", "retq")
}

// The case callSite exists for: RDX is the third argument register and
// the second return register at once. Two vregs pinned to it would be
// regalloc.ErrPinConflict; one vreg, written by the call and read after
// it, is the truth.
func TestLowerCallRDXIsBothArgAndResult(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	g := m.ImportFunc("g", ir.NewSig().
		Param(ir.TypeI32).Param(ir.TypeI32).Param(ir.TypeI32).
		Ret(ir.TypeI32).Ret(ir.TypeI32))

	fn := m.Func("three").Export()
	a := fn.ParamI32("a")
	fn.ReturnsI32()
	entry := fn.Entry()
	r := entry.Call(g, a, a, a)
	entry.Return(entry.I32.Add(r.Value(0).(ir.I32), r.Value(1).(ir.I32)))

	tb, raw := lowerText(t, m)

	want := []byte{
		0x55,             // push rbp
		0x48, 0x8b, 0xec, // mov rbp, rsp
		0x8b, 0xd7, // mov edx, edi
		0x8b, 0xfa, // mov edi, edx
		0x8b, 0xf2, // mov esi, edx
		0xe8, 0x00, 0x00, 0x00, 0x00, // call g
		0x8b, 0xc8, // mov ecx, eax
		0x03, 0xca, // add ecx, edx   (RDX again, now the second result)
		0x8b, 0xc1, // mov eax, ecx
		0xc9, // leave
		0xc3, // ret
	}
	if !bytes.Equal(tb, want) {
		t.Errorf(".text bytes = % x, want % x", tb, want)
	}
	objdumpHas(t, "three", raw, "callq", "addl")
}

// A function returning two values of its own. The two copies are a
// parallel assignment into RAX and RDX, and here they do not overlap.
func TestLowerReturnTwoValues(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	fn := m.Func("swap").Export()
	a := fn.ParamI32("a")
	b := fn.ParamI32("b")
	fn.ReturnsI32().ReturnsI32()
	entry := fn.Entry()
	entry.Return(b, a)

	tb, raw := lowerText(t, m)

	// No prologue: nothing calls, nothing allocates, nothing spills.
	want := []byte{
		0x8b, 0xc6, // mov eax, esi   (b)
		0x8b, 0xd7, // mov edx, edi   (a)
		0xc3, // ret
	}
	if !bytes.Equal(tb, want) {
		t.Errorf(".text bytes = % x, want % x", tb, want)
	}
	objdumpHas(t, "swap", raw, "movl", "retq")
}

// The two files at once: an i32 in RAX and an f64 in XMM0. The float
// costs nothing — the first float parameter and the first float return
// are the same register, so the copy coalesces away.
func TestLowerReturnMixedFiles(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	fn := m.Func("both").Export()
	a := fn.ParamI32("a")
	x := fn.ParamF64("x")
	fn.ReturnsI32().ReturnsF64()
	entry := fn.Entry()
	entry.Return(a, x)

	tb, raw := lowerText(t, m)

	want := []byte{
		0x8b, 0xc7, // mov eax, edi
		0xc3, // ret
	}
	if !bytes.Equal(tb, want) {
		t.Errorf(".text bytes = % x, want % x", tb, want)
	}
	objdumpHas(t, "both", raw, "movl", "retq")
}

// A return that permutes the return registers: the values arrive from a
// call in RAX and RDX and go back out the other way round.
//
// This is the claim the pinned copies make good on. Emitted as a
// sequence they would lose one of the two values; the allocator sees two
// registers it has to keep apart and breaks the cycle through a third,
// which is the same machinery a permuted block edge uses.
func TestLowerReturnPermutesReturnRegisters(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	pair := m.ImportFunc("pair", ir.NewSig().Ret(ir.TypeI32).Ret(ir.TypeI32))

	fn := m.Func("flip").Export()
	fn.ReturnsI32().ReturnsI32()
	entry := fn.Entry()
	r := entry.Call(pair)
	entry.Return(r.Value(1), r.Value(0))

	tb, raw := lowerText(t, m)

	want := []byte{
		0x55,             // push rbp
		0x48, 0x8b, 0xec, // mov rbp, rsp
		0xe8, 0x00, 0x00, 0x00, 0x00, // call pair
		0x8b, 0xc8, // mov ecx, eax   (the first result out of the way)
		0x8b, 0xc2, // mov eax, edx
		0x8b, 0xd1, // mov edx, ecx
		0xc9, // leave
		0xc3, // ret
	}
	if !bytes.Equal(tb, want) {
		t.Errorf(".text bytes = % x, want % x", tb, want)
	}
	objdumpHas(t, "flip", raw, "callq", "retq")
}

// A variadic call with no float arguments still sets AL, to zero. The
// callee reads it either way, so leaving it alone is leaving the callee
// to read whatever happened to be in RAX.
func TestLowerVariadicCallNoFloats(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	printf := m.ImportFunc("printf", ir.NewSig().Param(ir.TypePtr).Variadic().Ret(ir.TypeI32))

	fn := m.Func("say").Export()
	p := fn.ParamPtr("p")
	fn.ReturnsI32()
	entry := fn.Entry()
	entry.Return(entry.Call(printf, p).Value(0))

	tb, raw := lowerText(t, m)

	want := []byte{
		0x55,             // push rbp
		0x48, 0x8b, 0xec, // mov rbp, rsp
		0xb8, 0x00, 0x00, 0x00, 0x00, // mov eax, 0   (no vector registers used)
		0xe8, 0x00, 0x00, 0x00, 0x00, // call printf
		0xc9, // leave
		0xc3, // ret
	}
	if !bytes.Equal(tb, want) {
		t.Errorf(".text bytes = % x, want % x", tb, want)
	}
	objdumpHas(t, "say", raw, "movl", "callq")
}

// And with floats, the count of the vector registers the call actually
// used — which is how much of its register save area the callee writes.
func TestLowerVariadicCallCountsFloats(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	printf := m.ImportFunc("printf", ir.NewSig().Param(ir.TypePtr).Variadic().Ret(ir.TypeI32))

	fn := m.Func("say2").Export()
	p := fn.ParamPtr("p")
	x := fn.ParamF64("x")
	y := fn.ParamF64("y")
	fn.ReturnsI32()
	entry := fn.Entry()
	entry.Return(entry.Call(printf, p, x, y).Value(0))

	tb, raw := lowerText(t, m)

	want := []byte{
		0x55,             // push rbp
		0x48, 0x8b, 0xec, // mov rbp, rsp
		0xb8, 0x02, 0x00, 0x00, 0x00, // mov eax, 2   (XMM0 and XMM1)
		0xe8, 0x00, 0x00, 0x00, 0x00, // call printf
		0xc9, // leave
		0xc3, // ret
	}
	if !bytes.Equal(tb, want) {
		t.Errorf(".text bytes = % x, want % x", tb, want)
	}
	objdumpHas(t, "say2", raw, "callq")
}

// An indirect call is a call: it needs AL for a variadic callee and it
// needs the frame the direct form needs.
//
// The address goes to RBX rather than a scratch register because every
// caller-saved register is a destination of the call, so the one value
// that has to survive until the call reads it cannot be in any of them.
func TestLowerVariadicCallInd(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	st := m.FuncType("pf", ir.NewSig().Param(ir.TypePtr).Variadic().Ret(ir.TypeI32))

	fn := m.Func("viacall").Export()
	f := fn.ParamPtr("f")
	p := fn.ParamPtr("p")
	fn.ReturnsI32()
	entry := fn.Entry()
	entry.Return(entry.CallInd(f, st, p).Value(0))

	tb, raw := lowerText(t, m)

	want := []byte{
		0x55,             // push rbp
		0x48, 0x8b, 0xec, // mov rbp, rsp
		0x48, 0x81, 0xec, 0x10, 0x00, 0x00, 0x00, // sub rsp, 16
		0x48, 0x89, 0x5d, 0xf8, // mov [rbp-8], rbx
		0x48, 0x8b, 0xdf, // mov rbx, rdi   (the callee address)
		0x48, 0x8b, 0xfe, // mov rdi, rsi   (p)
		0xb8, 0x00, 0x00, 0x00, 0x00, // mov eax, 0
		0xff, 0xd3, // call rbx
		0x48, 0x8b, 0x5d, 0xf8, // mov rbx, [rbp-8]
		0xc9, // leave
		0xc3, // ret
	}
	if !bytes.Equal(tb, want) {
		t.Errorf(".text bytes = % x, want % x", tb, want)
	}
	objdumpHas(t, "viacall", raw, "callq")
}

// An indirect call whose arguments run past the sixth. The two that go
// on the stack need an outgoing area, and the frame only reserves one
// for a call it counted — which, until the verb was added to planFrame,
// an indirect call was not. Without it the argument at 0x8(%rsp) and the
// saved RBX at -0x8(%rbp) are the same eight bytes, and the call writes
// over the register the epilogue is about to restore from.
func TestLowerCallIndStackArguments(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	sig := ir.NewSig()
	for i := 0; i < 8; i++ {
		sig = sig.Param(ir.TypeI32)
	}
	st := m.FuncType("eight", sig.Ret(ir.TypeI32))

	fn := m.Func("viaeight").Export()
	f := fn.ParamPtr("f")
	a := fn.ParamI32("a")
	fn.ReturnsI32()
	entry := fn.Entry()
	entry.Return(entry.CallInd(f, st, a, a, a, a, a, a, a, a).Value(0))

	tb, raw := lowerText(t, m)

	// 32 bytes, not 16: one slot for RBX and two for the outgoing
	// arguments, rounded up to the stack alignment.
	want := []byte{
		0x55,             // push rbp
		0x48, 0x8b, 0xec, // mov rbp, rsp
		0x48, 0x81, 0xec, 0x20, 0x00, 0x00, 0x00, // sub rsp, 32
		0x48, 0x89, 0x5d, 0xf8, // mov [rbp-8], rbx
		0x44, 0x8b, 0xce, // mov r9d, esi   (a)
		0x48, 0x8b, 0xdf, // mov rbx, rdi   (the callee address)
		0x44, 0x89, 0x0c, 0x24, // mov [rsp], r9d
		0x44, 0x89, 0x4c, 0x24, 0x08, // mov [rsp+8], r9d
		0x41, 0x8b, 0xf9, // mov edi, r9d
		0x41, 0x8b, 0xf1, // mov esi, r9d
		0x41, 0x8b, 0xd1, // mov edx, r9d
		0x41, 0x8b, 0xc9, // mov ecx, r9d
		0x45, 0x8b, 0xc1, // mov r8d, r9d
		0xff, 0xd3, // call rbx
		0x48, 0x8b, 0x5d, 0xf8, // mov rbx, [rbp-8]
		0xc9, // leave
		0xc3, // ret
	}
	if !bytes.Equal(tb, want) {
		t.Errorf(".text bytes = % x, want % x", tb, want)
	}
	objdumpHas(t, "viaeight", raw, "callq")
}

// ── milestone 18: the seventh argument ────────────────────────────────

// Eight arguments: six in registers and two in the outgoing area at the
// bottom of the frame, which is where the callee will look for them once
// its own call has pushed a return address on top.
//
// The two stores come first, before any of the six copies. The value is
// living in RDI — it is this function's own first parameter — and the
// first copy would overwrite it, so the allocator moved it to R9 and the
// stores read it there. Ordered the other way it would have had to
// survive all six copies, which is a callee-saved register and a save
// slot to hold it in.
func TestLowerEightArgumentCall(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	sig := ir.NewSig()
	for i := 0; i < 8; i++ {
		sig.Param(ir.TypeI64)
	}
	sig.Ret(ir.TypeI64)
	eight := m.ImportFunc("eight", sig)

	fn := m.Func("callit").Export()
	a := fn.ParamI64("a")
	fn.ReturnsI64()
	entry := fn.Entry()
	entry.Return(entry.Call(eight, a, a, a, a, a, a, a, a).Value(0))

	tb, raw := lowerText(t, m)

	want := []byte{
		0x55,             // push rbp
		0x48, 0x8b, 0xec, // mov rbp, rsp
		0x48, 0x81, 0xec, 0x10, 0x00, 0x00, 0x00, // sub rsp, 16   (the outgoing area)
		0x4c, 0x8b, 0xcf, // mov r9, rdi
		0x4c, 0x89, 0x0c, 0x24, // mov [rsp], r9        (argument 7)
		0x4c, 0x89, 0x4c, 0x24, 0x08, // mov [rsp+8], r9      (argument 8)
		0x49, 0x8b, 0xf9, // mov rdi, r9
		0x49, 0x8b, 0xf1, // mov rsi, r9
		0x49, 0x8b, 0xd1, // mov rdx, r9
		0x49, 0x8b, 0xc9, // mov rcx, r9
		0x4d, 0x8b, 0xc1, // mov r8, r9
		0xe8, 0x00, 0x00, 0x00, 0x00, // call eight
		0xc9, // leave
		0xc3, // ret
	}
	if !bytes.Equal(tb, want) {
		t.Errorf(".text bytes = % x, want % x", tb, want)
	}
	objdumpHas(t, "callit", raw, "movq\t%r9, (%rsp)", "movq\t%r9, 0x8(%rsp)")
}

// The other end of the same convention: a function with eight parameters
// reads the last two out of its caller's outgoing area, which is above
// RBP — past the frame pointer the prologue pushed and the return
// address the call pushed under it.
//
// The prologue is here for that reason alone. This function stores
// nothing and calls nothing; what it needs an RBP for is to have
// somewhere to read its own arguments from.
func TestLowerStackParameters(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	fn := m.Func("last").Export()
	var p [8]ir.I64
	for i := range p {
		p[i] = fn.ParamI64("p")
	}
	fn.ReturnsI64()
	entry := fn.Entry()
	entry.Return(entry.I64.Add(p[6], p[7]))

	tb, raw := lowerText(t, m)

	want := []byte{
		0x55,             // push rbp
		0x48, 0x8b, 0xec, // mov rbp, rsp
		0x48, 0x8b, 0x45, 0x10, // mov rax, [rbp+16]   (parameter 7)
		0x48, 0x8b, 0x4d, 0x18, // mov rcx, [rbp+24]   (parameter 8)
		0x48, 0x8b, 0xd0, // mov rdx, rax
		0x48, 0x03, 0xd1, // add rdx, rcx
		0x48, 0x8b, 0xc2, // mov rax, rdx
		0xc9, // leave
		0xc3, // ret
	}
	if !bytes.Equal(tb, want) {
		t.Errorf(".text bytes = % x, want % x", tb, want)
	}
	objdumpHas(t, "last", raw, "movq\t0x10(%rbp), %rax", "movq\t0x18(%rbp), %rcx")
}

// An i32 stack argument occupies an eightbyte of which it fills four.
// SysV leaves the upper half unspecified, and a four-byte access is what
// an i32 vreg has to give and to take: a register's upper half is not
// something it holds.
func TestLowerStackParametersI32(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	fn := m.Func("last32").Export()
	var p [8]ir.I32
	for i := range p {
		p[i] = fn.ParamI32("p")
	}
	fn.ReturnsI32()
	entry := fn.Entry()
	entry.Return(entry.I32.Add(p[6], p[7]))

	tb, _ := lowerText(t, m)

	want := []byte{
		0x55,             // push rbp
		0x48, 0x8b, 0xec, // mov rbp, rsp
		0x8b, 0x45, 0x10, // mov eax, [rbp+16]
		0x8b, 0x4d, 0x18, // mov ecx, [rbp+24]
		0x8b, 0xd0, // mov edx, eax
		0x03, 0xd1, // add edx, ecx
		0x8b, 0xc2, // mov eax, edx
		0xc9, // leave
		0xc3, // ret
	}
	if !bytes.Equal(tb, want) {
		t.Errorf(".text bytes = % x, want % x", tb, want)
	}
}

// The frame with both ends in it: an allocation and a callee-saved
// register growing down from RBP, and an outgoing argument area at the
// bottom, growing up from RSP.
//
// Eight bytes of allocation, eight of saved RBX, and eight of outgoing
// argument is twenty-four, rounded to thirty-two by the stack alignment.
// The rounding lands between the two ends and inside neither: the slots
// are at RBP-8 and RBP-16, the argument is at RSP, and RSP is RBP-32.
func TestLowerFrameWithOutgoingArgs(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	sig := ir.NewSig()
	for i := 0; i < 7; i++ {
		sig.Param(ir.TypeI32)
	}
	seven := m.ImportFunc("seven", sig)

	fn := m.Func("both").Export()
	a := fn.ParamI32("a")
	fn.ReturnsPtr()
	entry := fn.Entry()
	slot := entry.Ptr.Alloc(8, 8)
	entry.Call(seven, a, a, a, a, a, a, a)
	entry.Return(slot)

	tb, _ := lowerText(t, m)

	want := []byte{
		0x55,             // push rbp
		0x48, 0x8b, 0xec, // mov rbp, rsp
		0x48, 0x81, 0xec, 0x20, 0x00, 0x00, 0x00, // sub rsp, 32
		0x48, 0x89, 0x5d, 0xf0, // mov [rbp-16], rbx    (save)
		0x44, 0x8b, 0xcf, // mov r9d, edi
		0x48, 0x8d, 0x5d, 0xf8, // lea rbx, [rbp-8]     (the allocation)
		0x44, 0x89, 0x0c, 0x24, // mov [rsp], r9d       (argument 7)
		0x41, 0x8b, 0xf9, // mov edi, r9d
		0x41, 0x8b, 0xf1, // mov esi, r9d
		0x41, 0x8b, 0xd1, // mov edx, r9d
		0x41, 0x8b, 0xc9, // mov ecx, r9d
		0x45, 0x8b, 0xc1, // mov r8d, r9d
		0xe8, 0x00, 0x00, 0x00, 0x00, // call seven
		0x48, 0x8b, 0xc3, // mov rax, rbx
		0x48, 0x8b, 0x5d, 0xf0, // mov rbx, [rbp-16]    (restore)
		0xc9, // leave
		0xc3, // ret
	}
	if !bytes.Equal(tb, want) {
		t.Errorf(".text bytes = % x, want % x", tb, want)
	}
}
