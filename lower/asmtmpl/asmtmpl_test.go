package asmtmpl

import "testing"

func refs(t *testing.T, tmpl string, nops int, labels ...string) []Ref {
	t.Helper()
	r, err := Parse(tmpl, nops, labels)
	if err != nil {
		t.Fatalf("Parse(%q, %d): %v", tmpl, nops, err)
	}
	return r
}

func mustFail(t *testing.T, tmpl string, nops int, labels ...string) error {
	t.Helper()
	_, err := Parse(tmpl, nops, labels)
	if err == nil {
		t.Fatalf("Parse(%q, %d): expected an error", tmpl, nops)
	}
	return err
}

// expand is the whole round trip: parse, then substitute a register name per
// operand and a label name per label.
func expand(t *testing.T, tmpl string, nops int, labels ...string) string {
	t.Helper()
	rs := refs(t, tmpl, nops, labels...)
	out, err := Expand(tmpl, rs, func(r Ref) (string, error) {
		if r.IsLabel() {
			return ".L" + r.Label, nil
		}
		if r.Modifier == 'w' {
			return "w" + itoa(r.Operand), nil
		}
		return "x" + itoa(r.Operand), nil
	})
	if err != nil {
		t.Fatalf("Expand(%q): %v", tmpl, err)
	}
	return out
}

func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	var b []byte
	for n > 0 {
		b = append([]byte{byte('0' + n%10)}, b...)
		n /= 10
	}
	return string(b)
}

func TestExpand(t *testing.T) {
	for _, c := range []struct {
		tmpl string
		nops int
		want string
	}{
		{"", 0, ""},
		{"nop", 0, "nop"},
		{"mrs %0, tpidr_el0", 1, "mrs x0, tpidr_el0"},
		{"add %0, %0, #1", 1, "add x0, x0, #1"},
		{"add %0, %1, %2", 3, "add x0, x1, x2"},
		{"add %w0, %w1, %w2", 3, "add w0, w1, w2"},
		{"mov %0, %1", 2, "mov x0, x1"},
		// A literal % is what x86 needs constantly.
		{"movq %%rax, %0", 1, "movq %rax, x0"},
		{"%%%%", 0, "%%"},
		{"a %% b %0 c", 1, "a % b x0 c"},
		// Multi-line, which every non-trivial template is.
		{"1: ldxr %0, [%1]\n   cbnz %0, 1b", 2, "1: ldxr x0, [x1]\n   cbnz x0, 1b"},
		// Ten or more operands: %10 is operand ten, not operand one then a
		// literal zero.
		{"%10", 11, "x10"},
	} {
		if got := expand(t, c.tmpl, c.nops); got != c.want {
			t.Errorf("%q with %d operands\n got %q\nwant %q", c.tmpl, c.nops, got, c.want)
		}
	}
}

func TestLabels(t *testing.T) {
	if got := expand(t, "jnz %l[out]", 0, "out"); got != "jnz .Lout" {
		t.Errorf("got %q", got)
	}
	if got := expand(t, "test %0, %0\n\tjnz %l[taken]", 1, "taken", "other"); got !=
		"test x0, x0\n\tjnz .Ltaken" {
		t.Errorf("got %q", got)
	}
}

// TestArityIsChecked is the point of the package. Every one of these built,
// verified and printed without complaint before it existed.
func TestArityIsChecked(t *testing.T) {
	for _, c := range []struct {
		tmpl string
		nops int
	}{
		{"%0", 0},
		{"%1", 1},
		{"add %0, %1, %2", 2},
		{"ldxr %0, [%1]\nadd x9, %0, %3", 3}, // the case from spec/asm.md
		{"%l[nowhere]", 0},
	} {
		mustFail(t, c.tmpl, c.nops)
	}
	// A label reference in something that is not an asm goto.
	mustFail(t, "b %l[x]", 1)
	// A label this asm goto does not have.
	mustFail(t, "b %l[nope]", 0, "yes")
}

func TestMalformed(t *testing.T) {
	for _, tmpl := range []string{
		"trailing %",
		"%z",
		"% 0",
		"%l[unclosed",
		"%,",
	} {
		mustFail(t, tmpl, 4)
	}
}

func TestErrorsArePositioned(t *testing.T) {
	err := mustFail(t, "nop\nnop\nadd x0, %3, x1", 2)
	e, ok := err.(*Error)
	if !ok {
		t.Fatalf("got %T", err)
	}
	if e.Line != 3 || e.Col != 9 {
		t.Errorf("got line %d col %d, want 3:9 (%v)", e.Line, e.Col, e)
	}
}

// TestMessageNamesTheFix checks that the diagnostic says what is wrong rather
// than only that something is.
func TestMessageNamesTheFix(t *testing.T) {
	err := mustFail(t, "add x0, %3, x1", 3)
	want := "%0 to %2"
	if got := err.Error(); !contains2(got, want) {
		t.Errorf("message %q does not mention %q", got, want)
	}
}

func contains2(s, sub string) bool {
	for i := 0; i+len(sub) <= len(s); i++ {
		if s[i:i+len(sub)] == sub {
			return true
		}
	}
	return false
}
