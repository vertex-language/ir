package amd64_test

// Milestone 25: §H.
//
// x86-64's memory model is what makes most of this section cheap. It
// reorders a store followed by a load and nothing else, so every atomic
// load is an ordinary load, every atomic store below sequential
// consistency is an ordinary store, and the orderings §H carries mostly
// do not reach an instruction at all.

import (
	"bytes"
	"testing"

	"github.com/vertex-language/ir"
)

// An atomic load is a load, at every ordering. Not a liberty this
// package takes: on this architecture every load is an acquire load, and
// a sequentially consistent load needs nothing extra because the store
// side is where sequential consistency is paid for.
func TestLowerAtomicLoad(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)

	a := m.Func("aload").Export()
	p := a.ParamPtr("p")
	a.ReturnsI64()
	a.Entry().Return(a.Entry().I64.AtomicLoad(p, ir.SeqCst))

	b := m.Func("aload8").Export()
	q := b.ParamPtr("p")
	b.ReturnsI32()
	b.Entry().Return(b.Entry().I32.AtomicULoad8(q, ir.Acquire))

	tb, raw := lowerText(t, m)

	want := []byte{
		0x48, 0x8b, 0x07, // mov rax, [rdi]
		0xc3,             // ret
		0x0f, 0xb6, 0x07, // movzx eax, byte [rdi]   — §H's narrow loads are unsigned
		0xc3, // ret
	}
	if !bytes.Equal(tb, want) {
		t.Errorf(".text bytes = % x, want % x", tb, want)
	}
	objdumpHas(t, "aload", raw, "movq", "movzbl")
}

// A release store is a store; a sequentially consistent one is an
// exchange, because that ordering also has to keep loads after it from
// moving ahead, and XCHG is the full barrier that does it in one
// instruction. The value it reads back is dropped.
func TestLowerAtomicStore(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)

	a := m.Func("astore").Export()
	p := a.ParamPtr("p")
	v := a.ParamI32("v")
	a.ReturnsI32()
	ea := a.Entry()
	ea.I32.AtomicStore(v, p, ir.Release)
	ea.Return(ea.I32.Const(0))

	b := m.Func("astoreseq").Export()
	q := b.ParamPtr("p")
	w := b.ParamI64("v")
	b.ReturnsI32()
	eb := b.Entry()
	eb.I64.AtomicStore(w, q, ir.SeqCst)
	eb.Return(eb.I32.Const(0))

	tb, raw := lowerText(t, m)

	want := []byte{
		0x89, 0x37, // mov [rdi], esi
		0xb8, 0x00, 0x00, 0x00, 0x00, // mov eax, 0
		0xc3,             // ret
		0x48, 0x8b, 0xc6, // mov rax, rsi
		0x48, 0x87, 0x07, // xchg [rdi], rax
		0xb8, 0x00, 0x00, 0x00, 0x00, // mov eax, 0
		0xc3, // ret
	}
	if !bytes.Equal(tb, want) {
		t.Errorf(".text bytes = % x, want % x", tb, want)
	}
	objdumpHas(t, "astore", raw, "xchgq")
}

// Fetch-and-add is LOCK XADD, which is exactly that instruction: the sum
// goes to memory and memory's old value comes back in the register.
// Fetch-and-sub is the same instruction over a negated operand, since
// there is no XSUB and none is wanted.
func TestLowerAtomicFetchAdd(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)

	a := m.Func("afetchadd").Export()
	p := a.ParamPtr("p")
	v := a.ParamI64("v")
	a.ReturnsI64()
	a.Entry().Return(a.Entry().I64.AtomicRmwAdd(v, p, ir.SeqCst))

	b := m.Func("afetchsub").Export()
	q := b.ParamPtr("p")
	w := b.ParamI32("v")
	b.ReturnsI32()
	b.Entry().Return(b.Entry().I32.AtomicRmwSub(w, q, ir.SeqCst))

	tb, raw := lowerText(t, m)

	want := []byte{
		0x48, 0x8b, 0xc6, // mov rax, rsi
		0xf0, 0x48, 0x0f, 0xc1, 0x07, // lock xadd [rdi], rax
		0xc3,       // ret
		0xf7, 0xde, // neg esi
		0x8b, 0xc6, // mov eax, esi
		0xf0, 0x0f, 0xc1, 0x07, // lock xadd [rdi], eax
		0xc3, // ret
	}
	if !bytes.Equal(tb, want) {
		t.Errorf(".text bytes = % x, want % x", tb, want)
	}
	objdumpHas(t, "afetchadd", raw, "xaddq", "negl")
}

// An atomic exchange needs no prefix. XCHG with a memory operand is the
// one instruction on this architecture that is locked whether the prefix
// is written or not, so writing it would be a byte saying what the
// opcode already says.
func TestLowerAtomicExchange(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	fn := m.Func("aswap").Export()
	p := fn.ParamPtr("p")
	v := fn.ParamI32("v")
	fn.ReturnsI32()
	fn.Entry().Return(fn.Entry().I32.AtomicRmwXchg(v, p, ir.SeqCst))

	tb, _ := lowerText(t, m)

	want := []byte{
		0x8b, 0xc6, // mov eax, esi
		0x87, 0x07, // xchg [rdi], eax
		0xc3, // ret
	}
	if !bytes.Equal(tb, want) {
		t.Errorf(".text bytes = % x, want % x", tb, want)
	}
}

// A compare-and-swap is one instruction and no test. LOCK CMPXCHG
// compares against the accumulator and leaves what it read there, which
// is exactly what §H's cas returns — success is an ordinary equality
// against what was expected, which the caller writes for themselves.
func TestLowerAtomicCas(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	fn := m.Func("acas").Export()
	p := fn.ParamPtr("p")
	expect := fn.ParamI64("expect")
	fresh := fn.ParamI64("new")
	fn.ReturnsI64()
	fn.Entry().Return(fn.Entry().I64.AtomicCas(expect, fresh, p, ir.SeqCst, ir.Monotonic))

	tb, raw := lowerText(t, m)

	want := []byte{
		0x48, 0x8b, 0xc6, // mov rax, rsi          (the expected value)
		0xf0, 0x48, 0x0f, 0xb1, 0x17, // lock cmpxchg [rdi], rdx
		0xc3, // ret                    (the value read is already in RAX)
	}
	if !bytes.Equal(tb, want) {
		t.Errorf(".text bytes = % x, want % x", tb, want)
	}
	objdumpHas(t, "acas", raw, "cmpxchgq")
}

// A narrow atomic is a narrow access in a whole register: one halfword
// of memory, and an i32 value, which is the same split §D2 has.
func TestLowerNarrowAtomicRmw(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	fn := m.Func("aadd16").Export()
	p := fn.ParamPtr("p")
	v := fn.ParamI32("v")
	fn.ReturnsI32()
	fn.Entry().Return(fn.Entry().I32.AtomicRmwAdd16(v, p, ir.SeqCst))

	tb, _ := lowerText(t, m)

	want := []byte{
		0x8b, 0xc6, // mov eax, esi
		0xf0, 0x66, 0x0f, 0xc1, 0x07, // lock xadd word [rdi], ax
		0xc3, // ret
	}
	if !bytes.Equal(tb, want) {
		t.Errorf(".text bytes = % x, want % x", tb, want)
	}
}

// Only the sequentially consistent fence is an instruction. The others
// order accesses this architecture does not reorder, so what they need
// is a barrier against the compiler — and this package has no pass that
// would move a memory access across anything, so honouring them is
// emitting nothing.
func TestLowerFence(t *testing.T) {
	for _, tc := range []struct {
		name  string
		emit  func(b *ir.Block)
		fence bool
	}{
		{"seq_cst", func(b *ir.Block) { b.Fence(ir.SeqCst) }, true},
		{"acquire", func(b *ir.Block) { b.Fence(ir.Acquire) }, false},
		{"release", func(b *ir.Block) { b.Fence(ir.Release) }, false},
		{"single thread", func(b *ir.Block) { b.Fence(ir.SeqCst, ir.SingleThread) }, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			m := ir.NewModule("t", ir.X86_64Linux)
			fn := m.Func("f").Export()
			fn.ReturnsI32()
			e := fn.Entry()
			tc.emit(e)
			e.Return(e.I32.Const(0))

			tb, _ := lowerText(t, m)

			want := []byte{0xb8, 0x00, 0x00, 0x00, 0x00, 0xc3}
			if tc.fence {
				want = append([]byte{0x0f, 0xae, 0xf0}, want...)
			}
			if !bytes.Equal(tb, want) {
				t.Errorf(".text bytes = % x, want % x", tb, want)
			}
		})
	}
}

// The three §H verbs with no instruction of their own. LOCK AND writes
// memory and returns nothing, and every rmw verb returns the old value,
// so an atomic and is a load, an and, and a compare-and-swap that
// retries.
//
// The read value stays in RAX across the loop, which is what makes the
// retry free of a reload: CMPXCHG leaves what it found there on failure,
// so the next round's operand is already the value that was there. The
// initial load is the only ordinary read of the location.
func TestLowerAtomicLoopRmw(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	fn := m.Func("aand").Export()
	p := fn.ParamPtr("p")
	v := fn.ParamI64("v")
	fn.ReturnsI64()
	fn.Entry().Return(fn.Entry().I64.AtomicRmwAnd(v, p, ir.SeqCst))

	tb, raw := lowerText(t, m)

	want := []byte{
		0x48, 0x8b, 0x07, // mov rax, [rdi]
		0xe9, 0x00, 0x00, 0x00, 0x00, // jmp retry
		0x48, 0x8b, 0xc8, // retry: mov rcx, rax
		0x48, 0x23, 0xce, // and rcx, rsi
		0xf0, 0x48, 0x0f, 0xb1, 0x0f, // lock cmpxchg [rdi], rcx
		0x0f, 0x85, 0xef, 0xff, 0xff, 0xff, // jne retry
		0xe9, 0x00, 0x00, 0x00, 0x00, // jmp cont
		0xc3, // cont: ret
	}
	if !bytes.Equal(tb, want) {
		t.Errorf(".text bytes = % x, want % x", tb, want)
	}
	objdumpHas(t, "aand", raw, "cmpxchgq", "jne")
}

// The narrow form, and the whole point of splitting the block: the
// instructions after the loop go into a block of their own, and the
// value the loop produced is an ordinary value there.
func TestLowerAtomicLoopContinues(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	fn := m.Func("axor").Export()
	p := fn.ParamPtr("p")
	v := fn.ParamI32("v")
	fn.ReturnsI32()
	e := fn.Entry()
	old := e.I32.AtomicRmwXor(v, p, ir.SeqCst)
	e.Return(e.I32.Add(old, v))

	tb, raw := lowerText(t, m)

	want := []byte{
		0x8b, 0x07, // mov eax, [rdi]
		0xe9, 0x00, 0x00, 0x00, 0x00, // jmp retry
		0x8b, 0xc8, // retry: mov ecx, eax
		0x33, 0xce, // xor ecx, esi
		0xf0, 0x0f, 0xb1, 0x0f, // lock cmpxchg [rdi], ecx
		0x0f, 0x85, 0xf2, 0xff, 0xff, 0xff, // jne retry
		0xe9, 0x00, 0x00, 0x00, 0x00, // jmp cont
		0x8b, 0xc8, // cont: mov ecx, eax
		0x03, 0xce, // add ecx, esi
		0x8b, 0xc1, // mov eax, ecx
		0xc3, // ret
	}
	if !bytes.Equal(tb, want) {
		t.Errorf(".text bytes = % x, want % x", tb, want)
	}
	objdumpHas(t, "axor", raw, "cmpxchgl", "addl")
}

// The eight-bit loop, whose initial read is the zero-extending load and
// whose compare-and-swap touches one byte.
func TestLowerNarrowAtomicLoopRmw(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	fn := m.Func("aor8").Export()
	p := fn.ParamPtr("p")
	v := fn.ParamI32("v")
	fn.ReturnsI32()
	fn.Entry().Return(fn.Entry().I32.AtomicRmwOr8(v, p, ir.SeqCst))

	_, raw := lowerText(t, m)
	objdumpHas(t, "aor8", raw, "movzbl", "orl", "cmpxchgb")
}
