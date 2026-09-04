package arm64_test

// The strongest verification available on this hardware: lower a module, write
// it as a Mach-O object, link it against a C main with the native toolchain,
// and run it.
//
// Disassembly proves the bytes decode. This proves they compute — that the
// ABI is right on both sides of a real call boundary, that the frame is
// balanced, that callee-saved registers really are saved. Nothing else here
// can catch a prologue that leaks a register.
//
// Skipped rather than failed off Apple Silicon, or with no clang: a missing
// native toolchain is an environment fact and not a defect.

import (
	"bytes"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"syscall"
	"testing"

	arm64macho "github.com/vertex-language/arm64/obj/macho"
	machocore "github.com/vertex-language/macho"

	"github.com/vertex-language/ir"
	arm64lower "github.com/vertex-language/ir/lower/arm64"
	"github.com/vertex-language/ir/verify"
)

// runNative lowers m, links it against mainC, runs the result and returns what
// it printed.
func runNative(t *testing.T, m *ir.Module, mainC string) string {
	t.Helper()
	bin := buildNative(t, m, mainC)
	out, err := exec.Command(bin).CombinedOutput()
	if err != nil {
		t.Fatalf("run: %v\n%s", err, out)
	}
	return string(out)
}

// runNativeTrap is runNative for a module that is supposed to trap: it builds
// and runs the same way and requires the process to die of SIGTRAP, which is
// what BRK raises.
//
// Nothing else here can check that a trap traps. A disassembly can show the
// BRK is present and a range check can be tested for the values it lets
// through, but that the instruction actually stops the program is a fact
// about the hardware.
func runNativeTrap(t *testing.T, m *ir.Module, mainC string) {
	t.Helper()
	bin := buildNative(t, m, mainC)
	err := exec.Command(bin).Run()
	if err == nil {
		t.Fatalf("the program exited cleanly; it was supposed to trap")
	}
	var ee *exec.ExitError
	if !errors.As(err, &ee) {
		t.Fatalf("run: %v", err)
	}
	st, ok := ee.Sys().(syscall.WaitStatus)
	if !ok || !st.Signaled() || st.Signal() != syscall.SIGTRAP {
		t.Fatalf("exited with %v; want a SIGTRAP", ee)
	}
}

// buildNative lowers m, writes it as a Mach-O object, links it against mainC
// and returns the path of the executable.
func buildNative(t *testing.T, m *ir.Module, mainC string) string {
	t.Helper()
	if runtime.GOARCH != "arm64" || runtime.GOOS != "darwin" {
		t.Skip("not on Apple Silicon; skipping the link-and-run check")
	}
	clang, err := exec.LookPath("clang")
	if err != nil {
		t.Skip("clang not on PATH; skipping the link-and-run check")
	}

	if err := verify.Module(m); err != nil {
		t.Fatalf("verify.Module: %v", err)
	}
	// The platform this harness actually links against: Mach-O prefixes a
	// C symbol with an underscore, which is why every function here is
	// named with one and why the library calls this package invents have
	// to be told about it — and Apple's variadic variant is the one the C
	// side of every call here was compiled for.
	o, err := arm64lower.Lower(m, arm64lower.Options{
		LibcallPrefix: "_",
		Variadic:      arm64lower.VariadicDarwin,
	})
	if err != nil {
		t.Fatalf("Lower: %v", err)
	}
	var buf bytes.Buffer
	if err := arm64macho.Write(&buf, o, arm64macho.Options{
		Platform: machocore.PlatformMacOS,
		MinOS:    "11.0",
	}); err != nil {
		t.Fatalf("macho.Write: %v", err)
	}

	dir := t.TempDir()
	objPath := filepath.Join(dir, "lowered.o")
	if err := os.WriteFile(objPath, buf.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}
	mainPath := filepath.Join(dir, "main.c")
	if err := os.WriteFile(mainPath, []byte(mainC), 0o644); err != nil {
		t.Fatal(err)
	}

	bin := filepath.Join(dir, "prog")
	if out, err := exec.Command(clang, "-o", bin, mainPath, objPath).CombinedOutput(); err != nil {
		t.Fatalf("link: %v\n%s", err, out)
	}
	return bin
}

// Mach-O prefixes a C symbol with an underscore, so a function this test links
// against C by name is declared with one.
func TestRunAddition(t *testing.T) {
	m := ir.NewModule("t", ir.AArch64Linux)
	fn := m.Func("_sum3").Export()
	a := fn.ParamI64("a")
	b := fn.ParamI64("b")
	c := fn.ParamI64("c")
	fn.ReturnsI64()
	entry := fn.Entry()
	entry.Return(entry.I64.Add(entry.I64.Add(a, b), c))

	got := runNative(t, m, `
#include <stdio.h>
long sum3(long, long, long);
int main(void) { printf("%ld\n", sum3(3, 4, 35)); return 0; }
`)
	if got != "42\n" {
		t.Errorf("printed %q, want %q", got, "42\n")
	}
}

// Nine arguments: eight in registers and one on the stack, which is the case
// that proves the outgoing area and the incoming one agree about where it is.
func TestRunStackArgument(t *testing.T) {
	m := ir.NewModule("t", ir.AArch64Linux)
	fn := m.Func("_ninth").Export()
	var last ir.I64
	for i := 0; i < 9; i++ {
		last = fn.ParamI64("p")
	}
	fn.ReturnsI64()
	entry := fn.Entry()
	entry.Return(last)

	got := runNative(t, m, `
#include <stdio.h>
long ninth(long,long,long,long,long,long,long,long,long);
int main(void) { printf("%ld\n", ninth(1,2,3,4,5,6,7,8,99)); return 0; }
`)
	if got != "99\n" {
		t.Errorf("printed %q, want %q", got, "99\n")
	}
}

// A value live across a call, which is what forces a callee-saved register and
// the prologue that saves it. If the save or the restore were wrong the
// answer would be whatever the callee left behind.
func TestRunValueLiveAcrossCall(t *testing.T) {
	m := ir.NewModule("t", ir.AArch64Linux)
	dbl := m.ImportFunc("_dbl", ir.NewSig().Param(ir.TypeI64).Ret(ir.TypeI64))

	fn := m.Func("_adddbl").Export()
	a := fn.ParamI64("a")
	fn.ReturnsI64()
	entry := fn.Entry()
	r := entry.Call(dbl, a).Value(0).(ir.I64)
	entry.Return(entry.I64.Add(r, a))

	got := runNative(t, m, `
#include <stdio.h>
long dbl(long x) { return x * 2; }
long adddbl(long);
int main(void) { printf("%ld\n", adddbl(14)); return 0; }
`)
	if got != "42\n" {
		t.Errorf("printed %q, want %q", got, "42\n")
	}
}

// A branch with block arguments, and the frame slot behind ptr.alloc.
func TestRunBranchAndFrame(t *testing.T) {
	m := ir.NewModule("t", ir.AArch64Linux)
	fn := m.Func("_maxstore").Export()
	a := fn.ParamI64("a")
	b := fn.ParamI64("b")
	fn.ReturnsI64()

	entry := fn.Entry()
	join := fn.Block("join")
	r := join.ParamI64("r")

	slot := entry.Ptr.Alloc(8, 8)
	entry.BrIf(entry.I64.SLt(a, b), join.To(b), join.To(a))
	join.I64.Store(r, slot)
	join.Return(join.I64.Load(slot))

	got := runNative(t, m, `
#include <stdio.h>
long maxstore(long, long);
int main(void) { printf("%ld %ld\n", maxstore(42, 7), maxstore(7, 42)); return 0; }
`)
	if got != "42 42\n" {
		t.Errorf("printed %q, want %q", got, "42 42\n")
	}
}

// A global, its address, and a load through it.
func TestRunGlobal(t *testing.T) {
	m := ir.NewModule("t", ir.AArch64Linux)
	g := m.Global("_answer", ir.RW, ir.StoreI64.FType()).Init(ir.Lit(ir.Int(42))).Export()

	fn := m.Func("_get").Export()
	fn.ReturnsI64()
	entry := fn.Entry()
	entry.Return(entry.I64.Load(entry.Ptr.GetAddr(g)))

	got := runNative(t, m, `
#include <stdio.h>
long get(void);
int main(void) { printf("%ld\n", get()); return 0; }
`)
	if got != "42\n" {
		t.Errorf("printed %q, want %q", got, "42\n")
	}
}

// Every §A verb this backend lowers, in one function, checked against what C
// computes for the same inputs.
func TestRunArithmetic(t *testing.T) {
	m := ir.NewModule("t", ir.AArch64Linux)
	fn := m.Func("_mix").Export()
	a := fn.ParamI64("a")
	b := fn.ParamI64("b")
	fn.ReturnsI64()
	entry := fn.Entry()

	// (a+b)*(a-b) ^ (a&b) | (a<<1) with a not and a shift right in it.
	sum := entry.I64.Add(a, b)
	dif := entry.I64.Sub(a, b)
	prod := entry.I64.Mul(sum, dif)
	and := entry.I64.And(a, b)
	sh := entry.I64.Shl(a, entry.I64.Const(1))
	x := entry.I64.Xor(prod, and)
	entry.Return(entry.I64.Or(x, entry.I64.UShr(sh, entry.I64.Const(2))))

	got := runNative(t, m, `
#include <stdio.h>
long mix(long, long);
int main(void) {
    long a = 1234, b = 567;
    long want = (((a+b)*(a-b)) ^ (a&b)) | (((unsigned long)(a<<1)) >> 2);
    long got = mix(a, b);
    printf("%s\n", got == want ? "ok" : "MISMATCH");
    return 0;
}
`)
	if got != "ok\n" {
		t.Errorf("printed %q, want %q", got, "ok\n")
	}
}
