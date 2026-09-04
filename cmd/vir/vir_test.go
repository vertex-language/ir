package main

import (
	"bytes"
	"errors"
	"strings"
	"testing"

	"github.com/vertex-language/ir/verify"
)

// Every sample is a module the builder accepted: a sticky error would
// make text.Print refuse it and make every fault verify reported a
// consequence of that one rather than the rule the sample is for.
func TestSamplesBuildClean(t *testing.T) {
	seen := map[string]bool{}
	for _, s := range samples {
		if seen[s.name] {
			t.Errorf("two samples named %q", s.name)
		}
		seen[s.name] = true

		m := s.build()
		if m == nil {
			t.Fatalf("%s: build returned nil", s.name)
		}
		if err := m.Err(); err != nil {
			t.Errorf("%s: sticky builder error: %v", s.name, err)
		}
	}
}

// Every sound sample verifies clean, and the one named unsound does not.
// This is
// the tool's only real claim — that "vir verify" tells you something you
// could not have assumed — so it is worth asserting in both directions.
func TestSamplesVerifyAsAdvertised(t *testing.T) {
	for _, s := range samples {
		err := verify.Module(s.build())
		if s.name == "unsound" {
			if !errors.Is(err, verify.ErrDominance) {
				t.Errorf("unsound: verify.Module = %v, want an ErrDominance fault", err)
			}
			continue
		}
		if err != nil {
			t.Errorf("%s: verify.Module: %v", s.name, err)
		}
	}
}

// cat prints .vir, which every module opens with its own module line.
func TestCat(t *testing.T) {
	var buf bytes.Buffer
	if err := run(&buf, []string{"cat", "add"}); err != nil {
		t.Fatalf("cat add: %v", err)
	}
	out := buf.String()
	if !strings.HasPrefix(out, "module add\n") {
		t.Errorf("cat add starts %q, want a module line", firstLine(out))
	}
	for _, want := range []string{`use "x86_64/linux"`, "export func @add", "i32.add"} {
		if !strings.Contains(out, want) {
			t.Errorf("cat add output missing %q\n%s", want, out)
		}
	}
}

// A clean verify says so on stdout and does not fail: finding nothing is
// a result, not an error.
func TestVerifyClean(t *testing.T) {
	var buf bytes.Buffer
	if err := run(&buf, []string{"verify", "add"}); err != nil {
		t.Fatalf("verify add: %v", err)
	}
	if got := strings.TrimSpace(buf.String()); got != "ok" {
		t.Errorf("verify add = %q, want %q", got, "ok")
	}
}

// And a fault is also a result: printed on stdout, one line per fault,
// with the tool exiting successfully. A verifier that made its own exit
// code the answer would be a linter, and this is a debugging tool — the
// caller decides what a fault means.
func TestVerifyReportsFaults(t *testing.T) {
	var buf bytes.Buffer
	if err := run(&buf, []string{"verify", "unsound"}); err != nil {
		t.Fatalf("verify unsound: %v", err)
	}
	out := strings.TrimSpace(buf.String())
	if !strings.Contains(out, "definition does not dominate use") {
		t.Errorf("verify unsound = %q, want the dominance fault", out)
	}
	// The register keeps its .vir sigil in a fault, the way the block
	// beside it in the same sentence does.
	if !strings.Contains(out, "%") {
		t.Errorf("verify unsound = %q, want the faulting register named", out)
	}
}

func TestList(t *testing.T) {
	var buf bytes.Buffer
	if err := run(&buf, []string{"list"}); err != nil {
		t.Fatalf("list: %v", err)
	}
	for _, s := range samples {
		if !strings.Contains(buf.String(), s.name) {
			t.Errorf("list output missing %q\n%s", s.name, buf.String())
		}
	}
}

func TestRunRejectsBadInput(t *testing.T) {
	for _, args := range [][]string{
		nil,                           // no command
		{"bogus"},                     // no such command
		{"cat"},                       // no sample
		{"cat", "nope"},               // no such sample
		{"verify", "add", "and-more"}, // two samples
		{"list", "add"},               // list takes none
	} {
		var buf bytes.Buffer
		if err := run(&buf, args); err == nil {
			t.Errorf("run(%q) = nil, want an error", args)
		}
	}
}

func firstLine(s string) string {
	if i := strings.IndexByte(s, '\n'); i >= 0 {
		return s[:i]
	}
	return s
}
