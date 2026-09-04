package amd64_test

import (
	"bytes"
	"testing"

	"github.com/vertex-language/ir"
)

// Tests loads of varying widths.
func TestLowerLoad(t *testing.T) {
	for _, tc := range []struct {
		name string
		load func(b *ir.Block, p ir.Ptr) ir.Value
		ret  func(fn *ir.Func) *ir.Func
		want []byte
		mnem string
	}{
		{
			"i32",
			func(b *ir.Block, p ir.Ptr) ir.Value { return b.I32.Load(p) },
			func(fn *ir.Func) *ir.Func { return fn.ReturnsI32() },
			[]byte{0x8b, 0x07, 0xc3}, // mov eax, [rdi] ; ret
			"movl",
		},
		{
			"i64",
			func(b *ir.Block, p ir.Ptr) ir.Value { return b.I64.Load(p) },
			func(fn *ir.Func) *ir.Func { return fn.ReturnsI64() },
			[]byte{0x48, 0x8b, 0x07, 0xc3}, // mov rax, [rdi] ; ret
			"movq",
		},
		{
			"ptr",
			func(b *ir.Block, p ir.Ptr) ir.Value { return b.Ptr.Load(p) },
			func(fn *ir.Func) *ir.Func { return fn.ReturnsPtr() },
			[]byte{0x48, 0x8b, 0x07, 0xc3}, // mov rax, [rdi] ; ret
			"movq",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			m := ir.NewModule("t", ir.X86_64Linux)
			fn := m.Func("deref").Export()
			p := fn.ParamPtr("p")
			tc.ret(fn)

			entry := fn.Entry()
			entry.Return(tc.load(entry, p))

			tb, raw := lowerText(t, m)
			if !bytes.Equal(tb, tc.want) {
				t.Errorf(".text bytes = % x, want % x", tb, tc.want)
			}
			objdumpHas(t, "deref", raw, tc.mnem, "retq")
		})
	}
}

// Tests stores of i32.
func TestLowerStore(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	fn := m.Func("put").Export()
	p := fn.ParamPtr("p")
	v := fn.ParamI32("v")
	fn.ReturnsI32()

	entry := fn.Entry()
	entry.I32.Store(v, p)
	entry.Return(v)

	tb, raw := lowerText(t, m)

	want := []byte{
		0x89, 0x37, // mov [rdi], esi
		0x8b, 0xc6, // mov eax, esi   (the return value)
		0xc3, // ret
	}
	if !bytes.Equal(tb, want) {
		t.Errorf(".text bytes = % x, want % x", tb, want)
	}

	objdumpHas(t, "put", raw, "movl", "retq")
}

// Tests a load and store roundtrip with pointer offset.
func TestLowerLoadStoreRoundTrip(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	fn := m.Func("copy8").Export()
	dst := fn.ParamPtr("dst")
	src := fn.ParamPtr("src")
	off := fn.ParamI64("off")
	fn.ReturnsI64()

	entry := fn.Entry()
	at := entry.Ptr.Add(src, off)
	v := entry.I64.Load(at, ir.Volatile)
	entry.I64.Store(v, dst, ir.Align(4))
	entry.Return(v)

	tb, raw := lowerText(t, m)

	want := []byte{
		0x48, 0x8b, 0xc6, // mov rax, rsi    (at = src)
		0x48, 0x03, 0xc2, // add rax, rdx    (at += off)
		0x48, 0x8b, 0x08, // mov rcx, [rax]  (the volatile load)
		0x48, 0x89, 0x0f, // mov [rdi], rcx  (the store, align 4)
		0x48, 0x8b, 0xc1, // mov rax, rcx    (the return value)
		0xc3, // ret
	}
	if !bytes.Equal(tb, want) {
		t.Errorf(".text bytes = % x, want % x", tb, want)
	}

	objdumpHas(t, "copy8", raw, "movq", "addq", "retq")
}

// Tests sub-width extending loads.
func TestLowerExtendingLoads(t *testing.T) {
	for _, tc := range []struct {
		name           string
		i32            func(b *ir.Block, p ir.Ptr) ir.I32 // nil where §D2 has no i32 form
		i64            func(b *ir.Block, p ir.Ptr) ir.I64
		want32, want64 []byte
		mnem           string
	}{
		{
			name:   "load8s",
			i32:    func(b *ir.Block, p ir.Ptr) ir.I32 { return b.I32.SLoad8(p) },
			i64:    func(b *ir.Block, p ir.Ptr) ir.I64 { return b.I64.SLoad8(p) },
			want32: []byte{0x0f, 0xbe, 0x07, 0xc3},       // movsx eax, byte [rdi]
			want64: []byte{0x48, 0x0f, 0xbe, 0x07, 0xc3}, // movsx rax, byte [rdi]
			mnem:   "movsb",
		},
		{
			name:   "load8u",
			i32:    func(b *ir.Block, p ir.Ptr) ir.I32 { return b.I32.ULoad8(p) },
			i64:    func(b *ir.Block, p ir.Ptr) ir.I64 { return b.I64.ULoad8(p) },
			want32: []byte{0x0f, 0xb6, 0x07, 0xc3},       // movzx eax, byte [rdi]
			want64: []byte{0x48, 0x0f, 0xb6, 0x07, 0xc3}, // movzx rax, byte [rdi]
			mnem:   "movzb",
		},
		{
			name:   "load16s",
			i32:    func(b *ir.Block, p ir.Ptr) ir.I32 { return b.I32.SLoad16(p) },
			i64:    func(b *ir.Block, p ir.Ptr) ir.I64 { return b.I64.SLoad16(p) },
			want32: []byte{0x48, 0x0f, 0xbf, 0x07, 0xc3}, // movsx rax, word [rdi]
			want64: []byte{0x48, 0x0f, 0xbf, 0x07, 0xc3},
			mnem:   "movsw",
		},
		{
			name:   "load16u",
			i32:    func(b *ir.Block, p ir.Ptr) ir.I32 { return b.I32.ULoad16(p) },
			i64:    func(b *ir.Block, p ir.Ptr) ir.I64 { return b.I64.ULoad16(p) },
			want32: []byte{0x0f, 0xb7, 0x07, 0xc3},       // movzx eax, word [rdi]
			want64: []byte{0x48, 0x0f, 0xb7, 0x07, 0xc3}, // movzx rax, word [rdi]
			mnem:   "movzw",
		},
		{
			name:   "load32s",
			i64:    func(b *ir.Block, p ir.Ptr) ir.I64 { return b.I64.SLoad32(p) },
			want64: []byte{0x48, 0x63, 0x07, 0xc3}, // movsxd rax, dword [rdi]
			mnem:   "movslq",
		},
		{
			name:   "load32u",
			i64:    func(b *ir.Block, p ir.Ptr) ir.I64 { return b.I64.ULoad32(p) },
			want64: []byte{0x8b, 0x07, 0xc3}, // mov eax, [rdi]
			mnem:   "movl",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if tc.i32 != nil {
				m := ir.NewModule("t", ir.X86_64Linux)
				fn := m.Func("f").Export()
				p := fn.ParamPtr("p")
				fn.ReturnsI32()
				fn.Entry().Return(tc.i32(fn.Entry(), p))

				tb, raw := lowerText(t, m)
				if !bytes.Equal(tb, tc.want32) {
					t.Errorf("i32: .text bytes = % x, want % x", tb, tc.want32)
				}
				objdumpHas(t, "f", raw, tc.mnem)
			}

			m := ir.NewModule("t", ir.X86_64Linux)
			fn := m.Func("f").Export()
			p := fn.ParamPtr("p")
			fn.ReturnsI64()
			fn.Entry().Return(tc.i64(fn.Entry(), p))

			tb, raw := lowerText(t, m)
			if !bytes.Equal(tb, tc.want64) {
				t.Errorf("i64: .text bytes = % x, want % x", tb, tc.want64)
			}
			objdumpHas(t, "f", raw, tc.mnem)
		})
	}
}

// Tests truncating stores.
func TestLowerTruncatingStores(t *testing.T) {
	t.Run("store8", func(t *testing.T) {
		tb, raw := lowerSubStore(t, func(b *ir.Block, v ir.I32, p ir.Ptr) { b.I32.Store8(v, p) })
		want := []byte{
			0x40, 0x88, 0x37, // mov [rdi], sil
			0x8b, 0xc6, // mov eax, esi
			0xc3,
		}
		if !bytes.Equal(tb, want) {
			t.Errorf(".text bytes = % x, want % x", tb, want)
		}
		objdumpHas(t, "f", raw, "movb")
	})

	t.Run("store16", func(t *testing.T) {
		tb, raw := lowerSubStore(t, func(b *ir.Block, v ir.I32, p ir.Ptr) { b.I32.Store16(v, p) })
		want := []byte{
			0x66, 0x89, 0x37, // mov [rdi], si
			0x8b, 0xc6, // mov eax, esi
			0xc3,
		}
		if !bytes.Equal(tb, want) {
			t.Errorf(".text bytes = % x, want % x", tb, want)
		}
		objdumpHas(t, "f", raw, "movw")
	})

	t.Run("store32", func(t *testing.T) {
		m := ir.NewModule("t", ir.X86_64Linux)
		fn := m.Func("f").Export()
		p := fn.ParamPtr("p")
		v := fn.ParamI64("v")
		fn.ReturnsI64()

		entry := fn.Entry()
		entry.I64.Store32(v, p)
		entry.Return(v)

		tb, raw := lowerText(t, m)
		want := []byte{
			0x89, 0x37, // mov [rdi], esi
			0x48, 0x8b, 0xc6, // mov rax, rsi
			0xc3,
		}
		if !bytes.Equal(tb, want) {
			t.Errorf(".text bytes = % x, want % x", tb, want)
		}
		objdumpHas(t, "f", raw, "movl")
	})
}

// Helper to lower sub-stores.
func lowerSubStore(t *testing.T, store func(b *ir.Block, v ir.I32, p ir.Ptr)) (text, object []byte) {
	t.Helper()

	m := ir.NewModule("t", ir.X86_64Linux)
	fn := m.Func("f").Export()
	p := fn.ParamPtr("p")
	v := fn.ParamI32("v")
	fn.ReturnsI32()

	entry := fn.Entry()
	store(entry, v, p)
	entry.Return(v)

	return lowerText(t, m)
}
