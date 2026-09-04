package i386_test

// Running the generated code, which on this host takes a whole machine.
//
// There is no way to execute a 32-bit x86 process here: the host is Apple
// Silicon, macOS has no 32-bit runtime, and QEMU's user-mode emulators cannot
// be built on a non-Linux host. So the program under test is not a process —
// it is a multiboot kernel, linked against a freestanding runtime with no
// libc, booted by qemu-system-i386, and printing to the emulated serial port.
//
// That sounds heavier than it is, and it buys the same thing the arm64
// backend's link-and-run check buys: an answer computed by the hardware
// rather than an expectation written by the same author as the code. It
// catches what a byte comparison cannot — a frame that does not balance, a
// callee-saved register that is not given back, a carry that does not cross.
//
// The runtime prints hex rather than decimal on purpose. A decimal conversion
// of a 64-bit value calls __udivdi3, and a freestanding link has no
// compiler-rt to take it from.

import (
	"bytes"
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	i386elf "github.com/vertex-language/i386/obj/elf"

	"github.com/vertex-language/ir"
	i386lower "github.com/vertex-language/ir/lower/i386"
	"github.com/vertex-language/ir/verify"
)

// runtimeC is everything the test's own C does not have to say: the multiboot
// header the loader looks for, a stack, the serial port, and the printers.
const runtimeC = `
__asm__(
".section .multiboot,\"a\"\n"
".align 4\n"
".long 0x1BADB002\n"
".long 0x00000000\n"
".long -(0x1BADB002 + 0x00000000)\n"
".section .text\n"
".globl _start\n"
"_start:\n"
"  movl $stack_top, %esp\n"
// SSE, which the processor comes out of reset refusing to execute.
// OSFXSR is the bit that says the operating system knows how to save
// XMM state, and without it every one of these instructions is a #UD
// — which, with no IDT, is a triple fault and a machine that prints
// nothing at all. MP and the cleared EM are the x87 half of the same
// switch: the float return convention goes through ST(0).
"  movl %cr0, %eax\n"
"  andl $~(1 << 2), %eax\n"
"  orl  $(1 << 1), %eax\n"
"  movl %eax, %cr0\n"
"  movl %cr4, %eax\n"
"  orl  $(3 << 9), %eax\n"
"  movl %eax, %cr4\n"
"  call kmain\n"
"1: hlt\n"
"  jmp 1b\n"
".section .bss\n"
".align 16\n"
"stack_bottom: .skip 16384\n"
"stack_top:\n"
);

static inline void outb(unsigned short p, unsigned char v) {
    __asm__ volatile("outb %0, %1" :: "a"(v), "Nd"(p));
}
static inline void outl(unsigned short p, unsigned int v) {
    __asm__ volatile("outl %0, %1" :: "a"(v), "Nd"(p));
}
static void emit(char c) { outb(0x3F8, (unsigned char)c); }
static void print(const char *s) { while (*s) emit(*s++); }
static void nib(unsigned v) { emit("0123456789abcdef"[v & 15]); }
static void printx32(unsigned v) { for (int i = 28; i >= 0; i -= 4) nib(v >> i); }
static void printx64(unsigned long long v) {
    printx32((unsigned)(v >> 32)); printx32((unsigned)v);
}
/* The four §E verbs lower to calls to these, so a freestanding link has to
   have them. Byte at a time on purpose: what is under test is the call this
   package emits, not how fast the callee is. */
void *memset(void *d, int c, unsigned n) {
    unsigned char *p = d;
    while (n--) *p++ = (unsigned char)c;
    return d;
}
void *memcpy(void *d, const void *s, unsigned n) {
    unsigned char *p = d; const unsigned char *q = s;
    while (n--) *p++ = *q++;
    return d;
}
void *memmove(void *d, const void *s, unsigned n) {
    unsigned char *p = d; const unsigned char *q = s;
    if (p == q || n == 0) return d;
    if (p < q) { while (n--) *p++ = *q++; return d; }
    p += n; q += n;
    while (n--) *--p = *--q;
    return d;
}
int memcmp(const void *a, const void *b, unsigned n) {
    const unsigned char *p = a, *q = b;
    for (; n--; p++, q++) if (*p != *q) return (int)*p - (int)*q;
    return 0;
}
static int failed = 0;
static void chk64(const char *what, unsigned long long got, unsigned long long want) {
    if (got != want) {
        failed = 1;
        print(what); print(": got "); printx64(got);
        print(" want "); printx64(want); print("\n");
    }
}
static void chk32(const char *what, unsigned got, unsigned want) {
    chk64(what, got, want);
}
static void body(void);
void kmain(void) {
    print("\001");
    body();
    print(failed ? "MISMATCH" : "ok");
    print("\001\n");
    outl(0xf4, 0);
}
`

// runKernel lowers m, links it against bodyC, boots the result and returns
// what it printed between the markers.
func runKernel(t *testing.T, m *ir.Module, bodyC string) string {
	t.Helper()

	clang, err := exec.LookPath("clang")
	if err != nil {
		t.Skip("clang not on PATH; skipping the boot-and-run check")
	}
	lld, err := exec.LookPath("ld.lld")
	if err != nil {
		t.Skip("ld.lld not on PATH; skipping the boot-and-run check (brew install lld)")
	}
	qemu, err := exec.LookPath("qemu-system-i386")
	if err != nil {
		t.Skip("qemu-system-i386 not on PATH; skipping the boot-and-run check (brew install qemu)")
	}

	if err := verify.Module(m); err != nil {
		t.Fatalf("verify.Module: %v", err)
	}
	o, err := i386lower.Lower(m, i386lower.Options{})
	if err != nil {
		t.Fatalf("Lower: %v", err)
	}
	var obj bytes.Buffer
	if err := i386elf.Write(&obj, o); err != nil {
		t.Fatalf("elf.Write: %v", err)
	}

	dir := t.TempDir()
	if os.Getenv("I386_KEEP") != "" {
		dir, _ = os.MkdirTemp("", "i386run")
		t.Logf("keeping build in %s", dir)
	}
	write := func(name, content string) string {
		p := filepath.Join(dir, name)
		if err := os.WriteFile(p, []byte(content), 0o644); err != nil {
			t.Fatal(err)
		}
		return p
	}
	objPath := filepath.Join(dir, "lowered.o")
	if err := os.WriteFile(objPath, obj.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}
	rtPath := write("rt.c", runtimeC+bodyC)
	ldPath := write("link.ld", `ENTRY(_start)
SECTIONS {
  . = 1M;
  .text   : { *(.multiboot) *(.text*) }
  .rodata : { *(.rodata*) }
  .data   : { *(.data*) }
  .bss    : { *(.bss*) *(COMMON) }
}
`)

	rtObj := filepath.Join(dir, "rt.o")
	// -march=i386 and no SSE, which is not fussiness: clang will happily
	// copy eight bytes with MOVSD through XMM0 on a target that allows it,
	// and this kernel never enables SSE — so the first 64-bit assignment
	// in the runtime raises #UD, and with no IDT that is a triple fault
	// and a machine that prints nothing at all.
	// -fno-builtin as well as -ffreestanding: without it clang recognizes
	// the loop in memset above as a memset and rewrites it into a call to
	// itself, which is a kernel that hangs rather than one that answers.
	cc := exec.Command(clang, "-target", "i386-unknown-none-elf", "-ffreestanding",
		"-fno-builtin", "-march=i386", "-mno-sse", "-mno-sse2", "-mno-mmx",
		"-fno-pic", "-fno-stack-protector", "-c", "-o", rtObj, rtPath)
	if out, err := cc.CombinedOutput(); err != nil {
		t.Fatalf("compile: %v\n%s", err, out)
	}

	prog := filepath.Join(dir, "prog")
	ln := exec.Command(lld, "-m", "elf_i386", "-T", ldPath, "-o", prog, rtObj, objPath)
	if out, err := ln.CombinedOutput(); err != nil {
		t.Fatalf("link: %v\n%s", err, out)
	}

	// A kernel that faults or loops would otherwise hang the run: qemu has
	// no notion of a failed program, only of one that has not exited.
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	run := exec.CommandContext(ctx, qemu, "-kernel", prog, "-nographic", "-no-reboot",
		"-net", "none", "-device", "isa-debug-exit,iobase=0xf4,iosize=0x04")
	out, _ := run.CombinedOutput()
	if ctx.Err() != nil {
		t.Fatalf("the kernel did not exit within a minute:\n%s", out)
	}

	// Between the markers, which is what separates the program's output
	// from SeaBIOS's.
	s := string(out)
	if os.Getenv("I386_DEBUG") != "" {
		t.Logf("qemu output:\n%s", s)
	}
	i := strings.Index(s, "\x01")
	j := strings.LastIndex(s, "\x01")
	if i < 0 || j <= i {
		t.Fatalf("the kernel printed no marked output; it may not have started:\n%s\nsources kept in %s", s, dir)
	}
	return strings.ReplaceAll(s[i+1:j], "\r", "")
}

// wantOK runs the module and requires the C side to have found no mismatch.
func wantOK(t *testing.T, m *ir.Module, bodyC string) {
	t.Helper()
	if got := runKernel(t, m, bodyC); got != "ok" {
		t.Errorf("kernel reported:\n%s", got)
	}
}

// hex is a 64-bit constant as C source, since the runtime compares in hex.
func hex64(v uint64) string { return fmt.Sprintf("0x%016xULL", v) }

// The additive verbs at both widths, against values chosen so the 64-bit ones
// carry between the halves — which is the one thing a 32-bit machine has to
// get right and cannot get right by accident.
func TestRunArithmetic(t *testing.T) {
	m := ir.NewModule("t", ir.I386Linux)

	for _, tc := range []struct {
		name string
		emit func(b *ir.Block, x, y ir.I64) ir.I64
	}{
		{"vadd", func(b *ir.Block, x, y ir.I64) ir.I64 { return b.I64.Add(x, y) }},
		{"vsub", func(b *ir.Block, x, y ir.I64) ir.I64 { return b.I64.Sub(x, y) }},
		{"vand", func(b *ir.Block, x, y ir.I64) ir.I64 { return b.I64.And(x, y) }},
		{"vor", func(b *ir.Block, x, y ir.I64) ir.I64 { return b.I64.Or(x, y) }},
		{"vxor", func(b *ir.Block, x, y ir.I64) ir.I64 { return b.I64.Xor(x, y) }},
	} {
		fn := m.Func(tc.name).Export()
		x := fn.ParamI64("x")
		y := fn.ParamI64("y")
		fn.ReturnsI64()
		e := fn.Entry()
		e.Return(tc.emit(e, x, y))
	}

	neg := m.Func("vneg").Export()
	negX := neg.ParamI64("x")
	neg.ReturnsI64()
	ne := neg.Entry()
	ne.Return(ne.I64.Neg(negX))

	not := m.Func("vnot").Export()
	notX := not.ParamI64("x")
	not.ReturnsI64()
	nt := not.Entry()
	nt.Return(nt.I64.Not(notX))

	add32 := m.Func("vadd32").Export()
	a32 := add32.ParamI32("a")
	b32 := add32.ParamI32("b")
	add32.ReturnsI32()
	a3 := add32.Entry()
	a3.Return(a3.I32.Mul(a3.I32.Add(a32, b32), a3.I32.Const(3)))

	wantOK(t, m, `
long long vadd(long long, long long), vsub(long long, long long);
long long vand(long long, long long), vor(long long, long long);
long long vxor(long long, long long), vneg(long long), vnot(long long);
int vadd32(int, int);
static void body(void) {
    /* Carries out of the low half in both directions. */
    unsigned long long a = 0x00000000ffffffffULL, b = 1ULL;
    chk64("add carry", vadd(a, b), 0x0000000100000000ULL);
    chk64("sub borrow", vsub(0x0000000100000000ULL, 1ULL), a);
    chk64("add big", vadd(0x123456789abcdef0ULL, 0x1111111111111111ULL),
                     0x23456789abcdf001ULL);
    chk64("sub big", vsub(0x23456789abcdf001ULL, 0x1111111111111111ULL),
                     0x123456789abcdef0ULL);
    chk64("and", vand(0xff00ff00ff00ff00ULL, 0x0f0f0f0f0f0f0f0fULL),
                 0x0f000f000f000f00ULL);
    chk64("or",  vor(0xff00ff00ff00ff00ULL, 0x00ff00ff00ff00ffULL),
                 0xffffffffffffffffULL);
    chk64("xor", vxor(0xffffffffffffffffULL, 0x123456789abcdef0ULL),
                 0xedcba9876543210fULL);
    /* Negation borrows out of the low half exactly when it is non-zero. */
    chk64("neg", vneg(1ULL), 0xffffffffffffffffULL);
    chk64("neg hi only", vneg(0x0000000100000000ULL), 0xffffffff00000000ULL);
    chk64("neg zero", vneg(0ULL), 0ULL);
    chk64("neg min", vneg(0x8000000000000000ULL), 0x8000000000000000ULL);
    chk64("not", vnot(0x123456789abcdef0ULL), 0xedcba9876543210fULL);
    chk32("add32", (unsigned)vadd32(100, 5), 315u);
}
`)
}

// §B at sixty-four bits, where one flag has to come out of two comparisons.
func TestRunCompare64(t *testing.T) {
	m := ir.NewModule("t", ir.I386Linux)

	for _, tc := range []struct {
		name string
		emit func(b *ir.Block, x, y ir.I64) ir.I1
	}{
		{"ceq", func(b *ir.Block, x, y ir.I64) ir.I1 { return b.I64.Eq(x, y) }},
		{"cne", func(b *ir.Block, x, y ir.I64) ir.I1 { return b.I64.Ne(x, y) }},
		{"cslt", func(b *ir.Block, x, y ir.I64) ir.I1 { return b.I64.SLt(x, y) }},
		{"csle", func(b *ir.Block, x, y ir.I64) ir.I1 { return b.I64.SLe(x, y) }},
		{"cult", func(b *ir.Block, x, y ir.I64) ir.I1 { return b.I64.ULt(x, y) }},
		{"cule", func(b *ir.Block, x, y ir.I64) ir.I1 { return b.I64.ULe(x, y) }},
	} {
		fn := m.Func(tc.name).Export()
		x := fn.ParamI64("x")
		y := fn.ParamI64("y")
		fn.ReturnsI32()
		e := fn.Entry()
		e.Return(e.I32.ZExtI1(tc.emit(e, x, y)))
	}

	wantOK(t, m, `
int ceq(long long,long long), cne(long long,long long);
int cslt(long long,long long), csle(long long,long long);
int cult(unsigned long long,unsigned long long), cule(unsigned long long,unsigned long long);
static void body(void) {
    unsigned long long lo = 0x0000000000000001ULL;
    unsigned long long hi = 0x0000000100000000ULL;
    chk32("eq same",   ceq(lo, lo), 1);
    chk32("eq lo diff", ceq(lo, 2), 0);
    /* Differing only in the high half, which a 32-bit compare would miss. */
    chk32("eq hi diff", ceq(lo, hi + 1), 0);
    chk32("ne hi diff", cne(lo, hi + 1), 1);
    chk32("ne same",    cne(hi, hi), 0);

    chk32("slt lo",  cslt(1, 2), 1);
    chk32("slt eq",  cslt(2, 2), 0);
    chk32("slt gt",  cslt(2, 1), 0);
    chk32("slt hi",  cslt((long long)hi, (long long)(hi + 1)), 1);
    /* Signs, which only the high half knows about. */
    chk32("slt neg", cslt(-1LL, 1LL), 1);
    chk32("slt neg2", cslt(-2LL, -1LL), 1);
    chk32("slt pos", cslt(1LL, -1LL), 0);
    chk32("sle eq",  csle(-1LL, -1LL), 1);
    chk32("sle lt",  csle(-2LL, -1LL), 1);
    chk32("sle gt",  csle(-1LL, -2LL), 0);

    /* The same values unsigned, where -1 is the largest there is. */
    chk32("ult",      cult(1ULL, 2ULL), 1);
    chk32("ult big",  cult(0xffffffffffffffffULL, 1ULL), 0);
    chk32("ult big2", cult(1ULL, 0xffffffffffffffffULL), 1);
    chk32("ult hi",   cult(hi, hi + 1), 1);
    chk32("ule eq",   cule(hi, hi), 1);
    chk32("ule gt",   cule(hi + 1, hi), 0);
}
`)
}

// §C's widening and narrowing, which is where the pair is built and taken
// apart.
func TestRunConversions(t *testing.T) {
	m := ir.NewModule("t", ir.I386Linux)

	sx := m.Func("sx").Export()
	sxA := sx.ParamI32("a")
	sx.ReturnsI64()
	e1 := sx.Entry()
	e1.Return(e1.I64.SExtI32(sxA))

	zx := m.Func("zx").Export()
	zxA := zx.ParamI32("a")
	zx.ReturnsI64()
	e2 := zx.Entry()
	e2.Return(e2.I64.ZExtI32(zxA))

	wr := m.Func("wr").Export()
	wrA := wr.ParamI64("a")
	wr.ReturnsI32()
	e3 := wr.Entry()
	e3.Return(e3.I32.WrapI64(wrA))

	wantOK(t, m, `
long long sx(int), zx(unsigned);
int wr(long long);
static void body(void) {
    chk64("sext pos", sx(1), 1ULL);
    chk64("sext neg", sx(-1), 0xffffffffffffffffULL);
    chk64("sext min", sx((int)0x80000000), 0xffffffff80000000ULL);
    chk64("zext pos", zx(1u), 1ULL);
    chk64("zext top", zx(0xffffffffu), 0x00000000ffffffffULL);
    chk32("wrap", (unsigned)wr(0x123456789abcdef0ULL), 0x9abcdef0u);
}
`)
}

// A frame slot, a branch carrying arguments, and a value that has to survive
// in a callee-saved register — which is what proves the prologue gives it back.
func TestRunFrameAndBranch(t *testing.T) {
	m := ir.NewModule("t", ir.I386Linux)
	dbl := m.ImportFunc("dbl", ir.NewSig().Param(ir.TypeI32).Ret(ir.TypeI32))

	fn := m.Func("maxstore").Export()
	a := fn.ParamI32("a")
	b := fn.ParamI32("b")
	fn.ReturnsI32()

	entry := fn.Entry()
	join := fn.Block("join")
	r := join.ParamI32("r")

	slot := entry.Ptr.Alloc(4, 4)
	entry.BrIf(entry.I32.SLt(a, b), join.To(b), join.To(a))
	join.I32.Store(r, slot)
	// A call with a value live across it, then the slot read back.
	d := join.Call(dbl, join.I32.Const(21)).Value(0).(ir.I32)
	join.Return(join.I32.Add(join.I32.Load(slot), d))

	wantOK(t, m, `
int dbl(int x) { return x * 2; }
int maxstore(int, int);
static void body(void) {
    chk32("max a", (unsigned)maxstore(42, 7), 84u);
    chk32("max b", (unsigned)maxstore(7, 42), 84u);
}
`)
}

// A 64-bit value returned in EDX:EAX and one passed as two stack slots, which
// is the whole of the psABI this backend has to agree with the C compiler
// about.
func TestRunAbi64(t *testing.T) {
	m := ir.NewModule("t", ir.I386Linux)

	fn := m.Func("mix").Export()
	a := fn.ParamI64("a")
	n := fn.ParamI32("n")
	b := fn.ParamI64("b")
	fn.ReturnsI64()
	entry := fn.Entry()
	entry.Return(entry.I64.Add(entry.I64.Sub(a, b), entry.I64.SExtI32(n)))

	// And the caller's side of the same convention.
	caller := m.Func("callmix").Export()
	caller.ReturnsI64()
	ce := caller.Entry()
	ce.Return(ce.Call(m.Lookup("mix").(ir.Callee),
		ce.I64.Const(0x0000000200000000),
		ce.I32.Const(-3),
		ce.I64.Const(0x0000000100000001),
	).Value(0).(ir.I64))

	wantOK(t, m, `
long long mix(long long, int, long long);
long long callmix(void);
static void body(void) {
    chk64("mix", mix(0x0000000200000000ULL, -3, 0x0000000100000001ULL),
                 0x00000000fffffffcULL);
    chk64("callmix", callmix(), 0x00000000fffffffcULL);
}
`)
}
