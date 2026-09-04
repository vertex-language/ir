package amd64_test

// §I's callee half: the save area, the list, and the walk over it.

import (
	"bytes"
	"testing"

	"github.com/vertex-language/ir"
	amd64lower "github.com/vertex-language/ir/lower/amd64"
)

// vaFunc is a variadic function with one named i32 parameter, which is
// the shape every test here starts from. The signature has to be
// finished before the entry block exists, so the caller declares its
// result and then calls vaEntry.
func vaFunc(m *ir.Module, name string) *ir.Func {
	fn := m.Func(name).Export()
	fn.Signature().Variadic()
	fn.ParamI32("n")
	return fn
}

// vaEntry opens fn's entry block with a va_list allocated in its frame.
func vaEntry(fn *ir.Func) (*ir.Block, ir.Ptr) {
	entry := fn.Entry()
	return entry, entry.Ptr.Alloc(24, 8)
}

// All six integer argument registers into the save area, then the eight
// vector ones if AL says the caller used any — which is not the
// optimisation it looks like, a caller that passed none having left
// those registers holding anything at all.
func TestLowerVariadicPrologue(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	fn := vaFunc(m, "sum")
	fn.ReturnsI32()
	entry, ap := vaEntry(fn)
	entry.VaStart(ap)
	entry.Return(entry.I32.VaArg(ap))

	tb, raw := lowerText(t, m)

	// 208 bytes: 176 of save area and 24 for the list, rounded up to
	// the stack alignment. That total is the regression test for the
	// two pieces overlapping — the save area is reserved first and the
	// static allocations continue from where it left the running total,
	// rather than restarting at zero on top of it.
	prologue := []byte{
		0x55,             // push rbp
		0x48, 0x8b, 0xec, // mov rbp, rsp
		0x48, 0x81, 0xec, 0xd0, 0x00, 0x00, 0x00, // sub rsp, 208
		0x48, 0x89, 0xbd, 0x50, 0xff, 0xff, 0xff, // mov [rbp-176], rdi
		0x48, 0x89, 0xb5, 0x58, 0xff, 0xff, 0xff, // mov [rbp-168], rsi
		0x48, 0x89, 0x95, 0x60, 0xff, 0xff, 0xff, // mov [rbp-160], rdx
		0x48, 0x89, 0x8d, 0x68, 0xff, 0xff, 0xff, // mov [rbp-152], rcx
		0x4c, 0x89, 0x85, 0x70, 0xff, 0xff, 0xff, // mov [rbp-144], r8
		0x4c, 0x89, 0x8d, 0x78, 0xff, 0xff, 0xff, // mov [rbp-136], r9
		0x84, 0xc0, // test al, al
		0x0f, 0x84, 0x20, 0x00, 0x00, 0x00, // je +32 (past the vector half)
		0x0f, 0x29, 0x45, 0x80, // movaps [rbp-128], xmm0
		0x0f, 0x29, 0x4d, 0x90, // movaps [rbp-112], xmm1
		0x0f, 0x29, 0x55, 0xa0, // movaps [rbp-96], xmm2
		0x0f, 0x29, 0x5d, 0xb0, // movaps [rbp-80], xmm3
		0x0f, 0x29, 0x65, 0xc0, // movaps [rbp-64], xmm4
		0x0f, 0x29, 0x6d, 0xd0, // movaps [rbp-48], xmm5
		0x0f, 0x29, 0x75, 0xe0, // movaps [rbp-32], xmm6
		0x0f, 0x29, 0x7d, 0xf0, // movaps [rbp-16], xmm7
	}
	if !bytes.HasPrefix(tb, prologue) {
		n := len(prologue)
		if n > len(tb) {
			n = len(tb)
		}
		t.Errorf("prologue does not save the argument registers into the save area\ngot  % x\nwant % x", tb[:n], prologue)
	}
	objdumpHas(t, "sum", raw, "testb", "movaps")
}

// va_start writes the four fields §3.5.7 names, and the two offsets
// start past this function's own declared parameters — those occupied
// argument registers before the tail did.
//
// fp_offset counts from the end of the integer half, which is why it
// starts at 48 rather than at zero.
func TestLowerVaStart(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	fn := m.Func("mixed").Export()
	fn.Signature().Variadic()
	fn.ParamI32("a")
	fn.ParamI32("b")
	fn.ParamF64("x")
	fn.ReturnsI32()
	entry := fn.Entry()
	ap := entry.Ptr.Alloc(24, 8)
	entry.VaStart(ap)
	entry.Return(entry.I32.VaArg(ap))

	tb, raw := lowerText(t, m)

	// Two named integers and one named float: gp_offset starts at 16
	// and fp_offset at 48 + 16.
	for _, want := range [][]byte{
		{0xc7, 0x00, 0x10, 0x00, 0x00, 0x00},       // mov dword [ap], 16
		{0xc7, 0x40, 0x04, 0x40, 0x00, 0x00, 0x00}, // mov dword [ap+4], 64
		{0x48, 0x8d, 0x4d, 0x10},                   // lea rcx, [rbp+16]      (overflow_arg_area)
		{0x48, 0x8d, 0x8d, 0x50, 0xff, 0xff, 0xff}, // lea rcx, [rbp-176]  (reg_save_area)
	} {
		if !bytes.Contains(tb, want) {
			t.Errorf("va_start does not write % x\ngot % x", want, tb)
		}
	}
	objdumpHas(t, "mixed", raw, "retq")
}

// An integer off the list walks gp_offset in steps of eight, and the
// bound is the last offset from which a whole eightbyte still fits in
// the integer half: 48 − 8, so the test is against 41.
func TestLowerVaArgInteger(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	fn := vaFunc(m, "geti")
	fn.ReturnsI64()
	entry, ap := vaEntry(fn)
	entry.VaStart(ap)
	entry.Return(entry.I64.VaArg(ap))

	tb, raw := lowerText(t, m)

	for _, want := range [][]byte{
		{0x81, 0xf9, 0x29, 0x00, 0x00, 0x00}, // cmp ecx, 41
		{0x81, 0x00, 0x08, 0x00, 0x00, 0x00}, // add dword [ap], 8
		{0x48, 0x8d, 0x51, 0x08},             // lea rdx, [rcx+8]   (the overflow arm)
	} {
		if !bytes.Contains(tb, want) {
			t.Errorf("va_arg does not contain % x\ngot % x", want, tb)
		}
	}
	objdumpHas(t, "geti", raw, "jb")
}

// A float walks fp_offset instead, in steps of sixteen: the save area
// holds whole vector registers, so a slot is sixteen bytes wide even for
// an f64. The bound is 176 − 16.
func TestLowerVaArgFloat(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	fn := vaFunc(m, "getf")
	fn.ReturnsF64()
	entry, ap := vaEntry(fn)
	entry.VaStart(ap)
	entry.Return(entry.F64.VaArg(ap))

	tb, raw := lowerText(t, m)

	for _, want := range [][]byte{
		{0x8b, 0x48, 0x04},                         // mov ecx, [ap+4]        (fp_offset, not gp)
		{0x81, 0xf9, 0xa1, 0x00, 0x00, 0x00},       // cmp ecx, 161
		{0x81, 0x40, 0x04, 0x10, 0x00, 0x00, 0x00}, // add dword [ap+4], 16
		{0xf2, 0x0f, 0x10, 0x01},                   // movsd xmm0, [rcx]
	} {
		if !bytes.Contains(tb, want) {
			t.Errorf("va_arg does not contain % x\ngot % x", want, tb)
		}
	}
	objdumpHas(t, "getf", raw, "movsd")
}

// va_copy is twenty-four bytes moved. Three loads and three stores
// rather than a memcpy: the size is a constant this package knows, and
// three eightbytes is below any size at which asking the library would
// be worth the call.
func TestLowerVaCopy(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	fn := m.Func("cp").Export()
	fn.Signature().Variadic()
	fn.ParamI32("n")
	fn.ReturnsI32()
	entry := fn.Entry()
	ap := entry.Ptr.Alloc(24, 8)
	bp := entry.Ptr.Alloc(24, 8)
	entry.VaStart(ap)
	entry.VaCopy(bp, ap)
	entry.Return(entry.I32.VaArg(bp))

	tb, raw := lowerText(t, m)

	for _, want := range [][]byte{
		{0x48, 0x89, 0x51, 0x08}, // mov [rcx+8], rdx
		{0x48, 0x8b, 0x50, 0x10}, // mov rdx, [rax+16]
		{0x48, 0x89, 0x51, 0x10}, // mov [rcx+16], rdx
	} {
		if !bytes.Contains(tb, want) {
			t.Errorf("va_copy does not contain % x\ngot % x", want, tb)
		}
	}
	objdumpHas(t, "cp", raw, "retq")
}

// va_arg_ref on a memory-class aggregate: the overflow pointer is the
// answer, and the list advances past it by whole eightbytes. No branch,
// because there is no register case for a memory-class argument to be
// in.
func TestLowerVaArgRef(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	st := m.Struct("big").
		Field("a", ir.StoreI64.FType()).
		Field("b", ir.StoreI64.FType()).
		Field("c", ir.StoreI64.FType())

	fn := vaFunc(m, "takeref")
	fn.ReturnsI64()
	entry, ap := vaEntry(fn)
	entry.VaStart(ap)
	p := entry.Ptr.VaArgRef(ap, st)
	entry.Return(entry.I64.Load(p))

	tb, raw := lowerText(t, m)

	for _, want := range [][]byte{
		{0x48, 0x8b, 0x48, 0x08}, // mov rcx, [ap+8]     (overflow_arg_area)
		{0x48, 0x8d, 0x51, 0x18}, // lea rdx, [rcx+24]   (past this argument)
		{0x48, 0x89, 0x50, 0x08}, // mov [ap+8], rdx
	} {
		if !bytes.Contains(tb, want) {
			t.Errorf("va_arg_ref does not contain % x\ngot % x", want, tb)
		}
	}
	objdumpHas(t, "takeref", raw, "retq")
}

// An aggregate small enough to have been passed in registers has its
// eightbytes scattered through the save area, so producing one address
// for it means gathering them into a slot first. That is a different
// shape of work from advancing a pointer and is not written yet.
func TestLowerRejectsRegisterClassVaArgRef(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	st := m.Struct("small").
		Field("a", ir.StoreI64.FType()).
		Field("b", ir.StoreI64.FType())

	fn := vaFunc(m, "f")
	fn.ReturnsI64()
	entry, ap := vaEntry(fn)
	entry.VaStart(ap)
	p := entry.Ptr.VaArgRef(ap, st)
	entry.Return(entry.I64.Load(p))

	if _, err := amd64lower.Lower(m, amd64lower.Options{}); err == nil {
		t.Error("Lower should refuse a va_arg_ref of a register-class aggregate")
	}
}

// A function that is not variadic has no tail and no save area, so
// there is nothing for a list to start on.
func TestLowerRejectsVaStartInNonVariadicFunction(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	fn := m.Func("f").Export()
	fn.ParamI32("n")
	fn.ReturnsI32()
	entry := fn.Entry()
	ap := entry.Ptr.Alloc(24, 8)
	entry.VaStart(ap)
	entry.Return(entry.I32.Const(0))

	if _, err := amd64lower.Lower(m, amd64lower.Options{}); err == nil {
		t.Error("Lower should refuse va_start in a function with no variadic tail")
	}
}

// A variadic function that never opens a list gets no save area: 176
// bytes of frame and fourteen stores is a great deal to spend on a list
// nobody opens.
func TestLowerVariadicWithoutVaStartHasNoSaveArea(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	fn := m.Func("quiet").Export()
	fn.Signature().Variadic()
	n := fn.ParamI32("n")
	fn.ReturnsI32()
	entry := fn.Entry()
	entry.Return(n)

	tb, _ := lowerText(t, m)

	want := []byte{
		0x8b, 0xc7, // mov eax, edi
		0xc3, // ret
	}
	if !bytes.Equal(tb, want) {
		t.Errorf(".text bytes = % x, want % x — no frame at all", tb, want)
	}
}
