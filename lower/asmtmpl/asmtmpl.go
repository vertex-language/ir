// Package asmtmpl reads the operand references in an inline assembly
// template.
//
// The template is GCC's, as spec/grammar.md §8b states: %0 and %1 number the
// outputs and then the inputs, %% is a literal %, %l[label] names an asm goto
// label, and a letter before the digits — %w0, %b0 — is a target-specific
// modifier asking for a particular view of the operand. Nothing else is
// special, and in particular a bare % that begins none of those is an error
// rather than a literal, because in the two dialects that matter a stray %
// is either a register sigil someone forgot to double or a typo.
//
// The package does two things and deliberately not a third. It finds the
// references and checks that each one names an operand that exists, which is
// the check nothing in this pipeline was making — a template referring to %3
// when three operands were declared assembled into a wrong object with no
// diagnostic anywhere. And it expands them, once the caller knows what each
// one became. It does not parse the assembly around them: that is the
// architecture's assembler, and it runs on the expanded text.
package asmtmpl

import (
	"fmt"
	"strings"
)

// A Ref is one reference in a template.
type Ref struct {
	// Start and End bound the reference in the template, including the %.
	Start, End int

	// Operand is the index into the combined outputs-then-inputs list, or
	// -1 for a label reference.
	Operand int

	// Label is the block label named by %l[...], empty otherwise.
	Label string

	// Modifier is the letter between the sigil and the digits, or 0. Its
	// meaning is the architecture's: on AArch64 `w` is the 32-bit view of a
	// general register, on x86 `b`, `h`, `w` and `k` are the byte, high
	// byte, word and doubleword views.
	Modifier byte
}

// IsLabel reports whether the reference names a label rather than an operand.
func (r Ref) IsLabel() bool { return r.Operand < 0 }

func (r Ref) String() string {
	if r.IsLabel() {
		return "%l[" + r.Label + "]"
	}
	if r.Modifier != 0 {
		return fmt.Sprintf("%%%c%d", r.Modifier, r.Operand)
	}
	return fmt.Sprintf("%%%d", r.Operand)
}

// An Error is a diagnostic positioned within the template.
//
// Line and Col count from the start of the template rather than from the start
// of a file, because a template has no file: it reached the backend as a Go
// string on an ir.Asm. A frontend that knows where the string came from can
// add that; a backend that invented a filename would be lying.
type Error struct {
	Line, Col int
	Msg       string
}

func (e *Error) Error() string {
	return fmt.Sprintf("template line %d, column %d: %s", e.Line, e.Col, e.Msg)
}

// Parse finds every reference in the template.
//
// nops is the number of operands, outputs and inputs together, and labels are
// an asm goto's block labels. A reference outside either is the error this
// function exists to raise.
func Parse(template string, nops int, labels []string) ([]Ref, error) {
	var refs []Ref
	for i := 0; i < len(template); {
		if template[i] != '%' {
			i++
			continue
		}
		r, next, err := parseRef(template, i, nops, labels)
		if err != nil {
			return nil, err
		}
		if r != nil {
			refs = append(refs, *r)
		}
		i = next
	}
	return refs, nil
}

// parseRef reads the reference starting at the % at i. It returns nil for
// `%%`, which is a literal and not a reference.
func parseRef(t string, i, nops int, labels []string) (*Ref, int, error) {
	fail := func(format string, args ...any) (*Ref, int, error) {
		line, col := position(t, i)
		return nil, 0, &Error{Line: line, Col: col, Msg: fmt.Sprintf(format, args...)}
	}

	j := i + 1
	if j >= len(t) {
		return fail("a template may not end with a bare %%; write %%%% for a literal one")
	}

	if t[j] == '%' {
		return nil, j + 1, nil
	}

	// %l[label]
	if t[j] == 'l' && j+1 < len(t) && t[j+1] == '[' {
		end := strings.IndexByte(t[j+2:], ']')
		if end < 0 {
			return fail("%%l[ with no closing ]")
		}
		name := t[j+2 : j+2+end]
		if !contains(labels, name) {
			if len(labels) == 0 {
				return fail("%%l[%s] names a label, and this is not an asm goto", name)
			}
			return fail("%%l[%s] names no label of this asm goto; it has %s",
				name, quoteList(labels))
		}
		return &Ref{Start: i, End: j + 3 + end, Operand: -1, Label: name}, j + 3 + end, nil
	}

	// An optional modifier letter, then the digits.
	var mod byte
	if isLetter(t[j]) {
		mod = t[j]
		j++
	}
	start := j
	for j < len(t) && t[j] >= '0' && t[j] <= '9' {
		j++
	}
	if j == start {
		if mod != 0 {
			return fail("%%%c is not an operand reference; a modifier is followed by digits, as in %%%c0",
				mod, mod)
		}
		return fail("%%%c is not an operand reference; write %%%% for a literal %%", t[j])
	}

	n := 0
	for _, c := range t[start:j] {
		n = n*10 + int(c-'0')
		if n > 999 {
			return fail("operand number %s is not one this instruction could have", t[start:j])
		}
	}
	if n >= nops {
		switch nops {
		case 0:
			return fail("%%%d names an operand, and this asm declares none", n)
		case 1:
			return fail("%%%d names operand %d, and this asm declares one, numbered %%0", n, n)
		default:
			return fail("%%%d names operand %d, and this asm declares %d, numbered %%0 to %%%d",
				n, n, nops, nops-1)
		}
	}
	return &Ref{Start: i, End: j, Operand: n, Modifier: mod}, j, nil
}

// Expand rewrites the template, replacing each reference with what sub returns
// for it and each %% with a single %.
func Expand(template string, refs []Ref, sub func(Ref) (string, error)) (string, error) {
	var b strings.Builder
	b.Grow(len(template))
	i := 0
	for _, r := range refs {
		b.WriteString(unescape(template[i:r.Start]))
		s, err := sub(r)
		if err != nil {
			line, col := position(template, r.Start)
			return "", &Error{Line: line, Col: col, Msg: fmt.Sprintf("%s: %v", r, err)}
		}
		b.WriteString(s)
		i = r.End
	}
	b.WriteString(unescape(template[i:]))
	return b.String(), nil
}

// unescape collapses %% in a run with no references in it. Parse has already
// established that every % in such a run is one of a pair.
func unescape(s string) string {
	if !strings.Contains(s, "%%") {
		return s
	}
	return strings.ReplaceAll(s, "%%", "%")
}

func position(s string, i int) (line, col int) {
	line, col = 1, 1
	for k := 0; k < i && k < len(s); k++ {
		if s[k] == '\n' {
			line++
			col = 1
			continue
		}
		col++
	}
	return line, col
}

func isLetter(c byte) bool { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') }

func contains(ss []string, s string) bool {
	for _, v := range ss {
		if v == s {
			return true
		}
	}
	return false
}

func quoteList(ss []string) string {
	q := make([]string, len(ss))
	for i, s := range ss {
		q[i] = `"` + s + `"`
	}
	switch len(q) {
	case 1:
		return q[0]
	case 2:
		return q[0] + " and " + q[1]
	}
	return strings.Join(q[:len(q)-1], ", ") + " and " + q[len(q)-1]
}
