package ir

import (
	"errors"
	"fmt"
	"strings"
)

// Builder sentinels. These name faults the builder catches as you emit;
// ir/verify holds the ones only a finished module reveals.
var (
	// ErrLayout is an ext-float namespace absent from the module's layout
	// block, or an operation that needs two such namespaces with one missing.
	ErrLayout = errors.New("namespace not admitted by layout")

	// ErrPoison is a zero Value used as an operand — the residue of an earlier
	// failure, or a value that was never defined.
	ErrPoison = errors.New("operand is a zero Value")

	// ErrTerminated is an instruction emitted into a block that already ends in
	// a terminator.
	ErrTerminated = errors.New("block already terminated")

	// ErrFrozen is a parameter declared after the thing it parameterizes began
	// to be filled: a block parameter after the block's first instruction, or a
	// function parameter after the entry block exists.
	ErrFrozen = errors.New("parameter list already frozen")

	// ErrArity is an argument list whose length does not match the parameter
	// list it feeds.
	ErrArity = errors.New("arity mismatch")

	// ErrType is an operand, argument, or result whose type does not match the
	// signature or block parameter it feeds.
	ErrType = errors.New("type mismatch")

	// ErrPlacement is an instruction in a block that may not hold it, or a
	// modifier on a declaration that may not carry it.
	ErrPlacement = errors.New("invalid placement")

	// ErrAlign is an align attribute that is not a power of two, or exceeds the
	// access width, or appears on an atomic access.
	ErrAlign = errors.New("invalid alignment")

	// ErrOrdering is a memory ordering §19.9 rejects.
	ErrOrdering = errors.New("invalid memory ordering")

	// ErrSRet is an sret parameter that is not the first, or a second one.
	ErrSRet = errors.New("sret must be the first and only such parameter")

	// ErrDuplicate is a name already taken in the namespace it is declared in.
	ErrDuplicate = errors.New("duplicate name")

	// ErrName is a name that is not an identifier, or is empty.
	ErrName = errors.New("invalid name")

	// ErrSignature is a signature or func typedef that cannot be used where it
	// was used.
	ErrSignature = errors.New("invalid signature")
)

// An Error is a builder failure, positioned by the function, block, and
// mnemonic that produced it.
type Error struct {
	Func   string // "" at module scope
	Block  string // "" outside a function body
	Op     Op     // the zero Op outside instruction emission
	Detail string
	Err    error // one of the sentinels above
}

func (e *Error) Error() string {
	var b strings.Builder
	b.WriteString("ir: ")
	if e.Func != "" {
		b.WriteString("@" + e.Func)
		if e.Block != "" {
			b.WriteString(" @" + e.Block)
		}
		b.WriteString(": ")
	}
	if e.Op.Verb != "" {
		b.WriteString(e.Op.String() + ": ")
	}
	b.WriteString(e.Err.Error())
	if e.Detail != "" {
		b.WriteString(": " + e.Detail)
	}
	return b.String()
}

func (e *Error) Unwrap() error { return e.Err }

// fail records err if the module has not already failed. First wins.
func (m *Module) fail(fn, blk string, op Op, err error, format string, args ...any) {
	if m == nil || m.err != nil {
		return
	}
	m.err = &Error{
		Func:   fn,
		Block:  blk,
		Op:     op,
		Detail: fmt.Sprintf(format, args...),
		Err:    err,
	}
}

// failModule records a module-scope failure.
func (m *Module) failModule(err error, format string, args ...any) {
	m.fail("", "", Op{}, err, format, args...)
}

// defer registers a check to run at Err. Branch argument arity and type cannot
// be checked at emission, because a forward branch may name a block whose
// parameters are not declared yet.
func (m *Module) deferCheck(c func() *Error) {
	if m.err == nil {
		m.deferred = append(m.deferred, c)
	}
}

func validIdent(s string) bool {
	if s == "" {
		return false
	}
	for i := 0; i < len(s); i++ {
		c := s[i]
		switch {
		case c == '_':
		case c >= 'a' && c <= 'z':
		case c >= 'A' && c <= 'Z':
		case c >= '0' && c <= '9' && i > 0:
		default:
			return false
		}
	}
	return true
}
