package amd64_test

// Tests for the amd64 lowerer, split across focused files.
// This file contains shared helper functions used by the tests.

import (
	"bytes"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	amd64elf "github.com/vertex-language/amd64/obj/elf"
	elfobj "github.com/vertex-language/elf/obj"
	"github.com/vertex-language/ir"
	amd64lower "github.com/vertex-language/ir/lower/amd64"
	"github.com/vertex-language/ir/verify"
)

// lowerText lowers m, writes it as ELF, reads the object straight back, and
// returns @.text's bytes along with the whole object for a disassembly
// cross-check.
func lowerText(t *testing.T, m *ir.Module) (text, object []byte) {
	t.Helper()

	if err := verify.Module(m); err != nil {
		t.Fatalf("verify.Module: %v", err)
	}
	o, err := amd64lower.Lower(m, amd64lower.Options{})
	if err != nil {
		t.Fatalf("Lower: %v", err)
	}

	var buf bytes.Buffer
	if err := amd64elf.Write(&buf, o); err != nil {
		t.Fatalf("elf.Write: %v", err)
	}
	f, err := elfobj.NewFile(bytes.NewReader(buf.Bytes()))
	if err != nil {
		t.Fatalf("NewFile: %v", err)
	}
	sec := f.Section(".text")
	if sec == nil {
		t.Fatal(".text section not found")
	}
	tb, err := sec.Data()
	if err != nil {
		t.Fatalf(".text Data: %v", err)
	}
	return tb, buf.Bytes()
}

// objdumpHas is the same best-effort disassembly cross-check the round-trip
// tests do: skipped, not failed, when objdump is not on PATH, since the
// byte comparison above every call is the authoritative one.
func objdumpHas(t *testing.T, name string, object []byte, mnemonics ...string) {
	t.Helper()

	objdump, err := exec.LookPath("objdump")
	if err != nil {
		t.Skip("objdump not on PATH; skipping the disassembly cross-check")
	}
	path := filepath.Join(t.TempDir(), name+".o")
	if err := os.WriteFile(path, object, 0o644); err != nil {
		t.Fatal(err)
	}
	out, err := exec.Command(objdump, "-d", path).CombinedOutput()
	if err != nil {
		t.Fatalf("objdump -d: %v\n%s", err, out)
	}
	for _, want := range mnemonics {
		if !strings.Contains(string(out), want) {
			t.Errorf("objdump -d output missing %q\n%s", want, out)
		}
	}
}
