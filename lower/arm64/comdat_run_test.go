package arm64_test

// §5's comdat attribute: a definition the linker keeps once however many
// objects carry it. A C++ frontend emits every inline function, virtual
// table and template instance this way, so the test is two units that
// both carry the same function and the same global, linked into one
// program. Were either copy an ordinary definition, ld64 would refuse the
// pair as duplicates.

import (
	"bytes"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"testing"

	arm64macho "github.com/vertex-language/arm64/obj/macho"
	machocore "github.com/vertex-language/macho"

	"github.com/vertex-language/ir"
	arm64lower "github.com/vertex-language/ir/lower/arm64"
	"github.com/vertex-language/ir/verify"
)

// comdatUnit is one unit: _inline_f and _limit in comdat groups, and an
// ordinary exported entry that reaches both.
func comdatUnit(t *testing.T, entry string) []byte {
	t.Helper()
	m := ir.NewModule("t", ir.AArch64MacOS)
	g := m.Global("_limit", ir.RO, ir.StoreI32.FType()).Export().Comdat().Init(ir.Lit(ir.Int(35)))

	inl := m.Func("_inline_f").Export().Comdat()
	inl.ReturnsI32()
	ie := inl.Entry()
	ie.Return(ie.I32.Const(7))

	fn := m.Func(entry).Export()
	fn.ReturnsI32()
	e := fn.Entry()
	r := e.Call(inl)
	e.Return(e.I32.Add(r.Value(0).(ir.I32), e.I32.Load(e.Ptr.GetAddr(g))))

	if err := verify.Module(m); err != nil {
		t.Fatalf("verify.Module: %v", err)
	}
	o, err := arm64lower.Lower(m, arm64lower.Options{
		LibcallPrefix: "_",
		Variadic:      arm64lower.VariadicDarwin,
	})
	if err != nil {
		t.Fatalf("Lower: %v", err)
	}
	var buf bytes.Buffer
	if err := arm64macho.Write(&buf, o, arm64macho.Options{
		Platform:    machocore.PlatformMacOS,
		MinOS:       "11.0",
		Subsections: true,
	}); err != nil {
		t.Fatalf("macho.Write: %v", err)
	}
	return buf.Bytes()
}

func TestRunComdat(t *testing.T) {
	if runtime.GOARCH != "arm64" || runtime.GOOS != "darwin" {
		t.Skip("not on Apple Silicon; skipping the link-and-run check")
	}
	clang, err := exec.LookPath("clang")
	if err != nil {
		t.Skip("clang not on PATH; skipping the link-and-run check")
	}

	dir := t.TempDir()
	var args []string
	for _, entry := range []string{"_one", "_two"} {
		path := filepath.Join(dir, entry[1:]+".o")
		if err := os.WriteFile(path, comdatUnit(t, entry), 0o644); err != nil {
			t.Fatal(err)
		}
		args = append(args, path)
	}
	mainPath := filepath.Join(dir, "main.c")
	if err := os.WriteFile(mainPath, []byte(`
#include <stdio.h>
int one(void);
int two(void);
int main(void) { printf("%d %d\n", one(), two()); return 0; }
`), 0o644); err != nil {
		t.Fatal(err)
	}
	bin := filepath.Join(dir, "prog")
	args = append([]string{"-o", bin, mainPath}, args...)
	if out, err := exec.Command(clang, args...).CombinedOutput(); err != nil {
		t.Fatalf("link: %v\n%s", err, out)
	}
	out, err := exec.Command(bin).CombinedOutput()
	if err != nil {
		t.Fatalf("run: %v\n%s", err, out)
	}
	if want := "42 42\n"; string(out) != want {
		t.Errorf("printed %q, want %q", out, want)
	}
}
