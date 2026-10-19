// Package verify implements §19 IR well-formedness validation for completed modules.
//
// Rules checked here include SSA dominance (§19.1), block terminators and CFG
// reachability (§19.2, §19.17), landing pads and exception unwinding (§19.4–5, §19.16),
// block address usage (§19.6), global initializers (§19.10), and struct field offsets (§19.18).
// Single-instruction and syntactic invariants are checked earlier by the builder.
package verify

import (
	"fmt"

	"github.com/vertex-language/ir"
)

// Options are the knobs a run has. The zero value is what the
// package-level Module and Func use, and reports every fault found.
type Options struct {
	// MaxErrors caps how many faults a run reports; zero is no cap. This
	// is for a driver printing to a terminal, not for correctness: which
	// faults a capped run shows you is declaration order, not severity.
	MaxErrors int
}

// Module checks every function definition in m, after the module's own
// sticky builder error. Function imports have no body and nothing in §19
// to break.
func Module(m *ir.Module) error { return Options{}.Module(m) }

// Func checks one function definition. It is the entry point a pass uses
// on the function it just rewrote, rather than paying for the whole module.
func Func(f *ir.Func) error { return Options{}.Func(f) }

// Module is Module with options.
func (o Options) Module(m *ir.Module) error {
	if m == nil {
		return nil
	}
	if err := m.Err(); err != nil {
		return err
	}

	c := &checker{opts: o}
	c.moduleItems(m)
	for _, f := range m.Funcs() {
		if c.full() {
			break
		}
		c.function(f)
	}
	return c.result()
}

// Func is Func with options.
func (o Options) Func(f *ir.Func) error {
	if f == nil {
		return nil
	}
	// The sticky error belongs to the module, not the function, so a
	// caller with only an *ir.Func in hand still gets it — and gets it
	// first, since a builder failure means every later call was a no-op
	// and the body this package would walk is a partial one.
	if m := f.Module(); m != nil {
		if err := m.Err(); err != nil {
			return err
		}
	}

	c := &checker{opts: o}
	c.function(f)
	return c.result()
}

// A checker accumulates one run's faults. It carries the function being
// walked so the rule checks do not each have to thread a position through.
type checker struct {
	opts Options
	fn   *ir.Func
	errs Errors
}

// function runs every implemented rule over f.
//
// The CFG checks come first and stop the function if they fail, because
// every rule after them reads the graph through the terminators §19.2 is
// about: a block with no terminator has no successors, which would make
// its own successors look unreachable and every value crossing them look
// undominated. One real fault reported as five is worse than one.
func (c *checker) function(f *ir.Func) {
	c.fn = f

	if _, ok := f.AsmBodyText(); ok {
		// A body of assembly has no blocks, no registers and no CFG, so
		// every rule below is about something that is not there. The
		// builder has already refused the one way to break this — blocks
		// and an asm body in one function, ir.ErrPlacement at whichever
		// came second.
		return
	}

	blocks := f.Blocks()
	if f.IsNaked() {
		// Ahead of ErrNoBody, because a naked function with nothing in it
		// is not a definition that should have been an import — it is one
		// whose body was going to be assembly and was never written, and
		// pointing it at ImportFunc would be answering the wrong question.
		had := "blocks"
		if len(blocks) == 0 {
			had = "nothing"
		}
		c.fail(nil, -1, ir.Op{}, ErrNakedBody,
			"naked, with %s for a body; a naked function's body is ir.Func.AsmBody", had)
		return
	}
	if len(blocks) == 0 {
		c.fail(nil, -1, ir.Op{}, ErrNoBody,
			"a definition with no blocks; ir.Module.ImportFunc declares a function this module does not define")
		return
	}

	if !c.terminators(blocks) {
		return
	}
	c.reachability(f, blocks)

	// One dominator tree and one instruction numbering, shared: §19.1 and
	// §19.5 are both dominance questions, and every fault either of them
	// reports has to name a position.
	dt := newDomTree(f)
	pos := instPositions(f)
	c.dominance(f, dt, pos)
	if c.full() {
		return
	}
	c.unwind(f, blocks, dt)
	if c.full() {
		return
	}
	c.blockAddrs(blocks)
	if c.full() {
		return
	}
	c.asm(blocks)
	if c.full() {
		return
	}
	c.asmGotoTargets(blocks)
}

// fail records one fault, positioned at in within blk. Pass a nil blk for
// a fault that is the function's own, and a negative idx for one that is
// the block's own.
func (c *checker) fail(blk *ir.Block, idx int, op ir.Op, err error, format string, args ...any) {
	e := &Error{
		Func:   c.fn.Name(),
		Inst:   idx,
		Op:     op,
		Detail: detail(format, args...),
		Err:    err,
	}
	if blk != nil {
		e.Block = blk.Label()
	}
	c.errs = append(c.errs, e)
}

// failItem records one module-scope fault: a global or a type, which is
// in no function and no block. ir.Error positions these the same way, by
// leaving Func empty and naming the item in the detail.
func (c *checker) failItem(err error, format string, args ...any) {
	c.errs = append(c.errs, &Error{
		Inst:   -1,
		Detail: detail(format, args...),
		Err:    err,
	})
}

// full reports whether the run has hit its cap and should stop looking.
func (c *checker) full() bool {
	return c.opts.MaxErrors > 0 && len(c.errs) >= c.opts.MaxErrors
}

// result is the run's verdict. The nil is explicit: an empty Errors in an
// error interface is not nil, and every caller of this package tests the
// return against nil.
func (c *checker) result() error {
	if len(c.errs) == 0 {
		return nil
	}
	return c.errs
}

// detail formats a fault's detail text. A format with no arguments is
// taken literally, so a rule quoting a mnemonic does not have to escape a
// percent sign that Sprintf would otherwise read as a verb.
func detail(format string, args ...any) string {
	if len(args) == 0 {
		return format
	}
	return fmt.Sprintf(format, args...)
}
