package amd64_test

// The link-and-run check, for x86-64.
//
// Every other test in this package proves the bytes decode: it assembles a
// module and reads the disassembly back. That cannot catch a convention that
// is wrong in the same way on both sides of a call, which is exactly what an
// ABI bug is — a private convention is self-consistent, and only another
// compiler's code disagrees with it. So this links what this package lowered
// against a C main clang compiled, and runs it.
//
// x86-64 is not the host here. Apple Silicon runs these under Rosetta, which
// is enough: the instructions and the calling convention are the real ones,
// and the translation is below both. Skipped where that is not available, the
// way the arm64 harness skips off Apple Silicon — a missing toolchain is an
// environment fact and not a defect.

import (
	"bytes"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"sync"
	"testing"

	amd64macho "github.com/vertex-language/amd64/obj/macho"
	machocore "github.com/vertex-language/macho"

	"github.com/vertex-language/ir"
	amd64lower "github.com/vertex-language/ir/lower/amd64"
	"github.com/vertex-language/ir/verify"
)

// canRunX86 reports whether an x86-64 binary built here actually runs, which
// on Apple Silicon means asking Rosetta rather than assuming it. Answered
// once: it costs a compile and a process.
var canRunX86 = sync.OnceValues(func() (bool, error) {
	clang, err := exec.LookPath("clang")
	if err != nil {
		return false, err
	}
	dir, err := os.MkdirTemp("", "x86probe")
	if err != nil {
		return false, err
	}
	defer os.RemoveAll(dir)

	src := filepath.Join(dir, "probe.c")
	if err := os.WriteFile(src, []byte("int main(void){return 0;}\n"), 0o644); err != nil {
		return false, err
	}
	bin := filepath.Join(dir, "probe")
	if out, err := exec.Command(clang, "-arch", "x86_64", "-o", bin, src).CombinedOutput(); err != nil {
		return false, wrap(err, out)
	}
	if out, err := exec.Command(bin).CombinedOutput(); err != nil {
		return false, wrap(err, out)
	}
	return true, nil
})

func wrap(err error, out []byte) error {
	if len(out) == 0 {
		return err
	}
	return &probeError{err: err, out: out}
}

type probeError struct {
	err error
	out []byte
}

func (e *probeError) Error() string { return e.err.Error() + ": " + string(e.out) }

// runNative lowers m, links it against mainC, runs the result and returns what
// it printed.
func runNative(t *testing.T, m *ir.Module, mainC string) string {
	t.Helper()
	if runtime.GOOS != "darwin" {
		t.Skip("not on macOS; skipping the link-and-run check")
	}
	if ok, err := canRunX86(); !ok {
		t.Skipf("cannot build and run x86-64 here (%v); skipping the link-and-run check", err)
	}
	clang, _ := exec.LookPath("clang")

	if err := verify.Module(m); err != nil {
		t.Fatalf("verify.Module: %v", err)
	}
	// Mach-O prefixes a C symbol with an underscore, which is why every
	// function here is named with one and why the library calls this
	// package invents have to be told about it.
	o, err := amd64lower.Lower(m, amd64lower.Options{LibcallPrefix: "_"})
	if err != nil {
		t.Fatalf("Lower: %v", err)
	}
	var buf bytes.Buffer
	if err := amd64macho.Write(&buf, o, amd64macho.Options{
		Platform: machocore.PlatformMacOS,
		MinOS:    "10.13",
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
	if out, err := exec.Command(clang, "-arch", "x86_64", "-w", "-o", bin, mainPath, objPath).CombinedOutput(); err != nil {
		t.Fatalf("link: %v\n%s", err, out)
	}
	out, err := exec.Command(bin).CombinedOutput()
	if err != nil {
		t.Fatalf("run: %v\n%s", err, out)
	}
	return string(out)
}

// The harness proving itself: if this cannot add three numbers across a real
// call, nothing below it means anything.
func TestRunAddition(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64MacOS)
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
int main(void) { printf("%ld\n", sum3(3, 4, 5)); return 0; }
`)
	if want := "12\n"; got != want {
		t.Errorf("printed %q, want %q", got, want)
	}
}
