// Package verify is §19: the well-formedness rules the grammar admits and
// a sound module does not break.
//
// A package rather than a method on ir.Module, for three reasons. §19 is
// nineteen rules and wants more than one file. Its failure corpus — one
// case per rule, each expected to fail with a named sentinel at a named
// position — belongs next to the sentinels that name it. And a method
// would put the verifier inside the package it verifies, where every
// unexported field is in reach and no rule has to be sayable through the
// public surface. As a client, the verifier is that surface's most
// demanding reader: a rule it cannot state with exported methods is a
// missing method, not a reason to reach inside.
//
// # What is checked where
//
// Everything the builder can catch while you emit is sticky and
// first-wins on the module itself, and Module returns that before it
// looks at anything else. §19.3 is the clearest case: a branch's argument
// list against its target's parameters is a deferred builder check, since
// a forward branch may name a block whose parameters are not declared
// yet, and ir.Module.Err runs it. That makes soundness one call and not
// two, and it keeps this package to what only a finished module reveals.
//
// # What is checked here
//
// Every §19 rule that survives into a finished module: §19.1 (dominance),
// §19.2 (terminators and reachability), §19.4–5 and §19.16 (the unwind
// rules), §19.6's second clause (a blockaddr some brind targets), §19.10
// (a global initializer against its declared type), §19.17 (the entry
// block is nobody's target), §19.18 (a struct's stated offsets), and
// §19.19 (a naked function's body is assembly).
//
// The rest of §19 is not missing, it is elsewhere, and each file here
// names where: verify/memory.go for §19.6's first clause and §19.7–9,
// verify/module.go for §19.11–15. Every one of those is a fact about a
// single instruction or declaration in isolation, which is what the
// builder catches as you emit and reports through ir.Module.Err — which
// both entry points below return before they look at anything else.
//
// Two clauses are checked by neither, and both are the same kind of gap:
// they need a target ABI, and this repo has none. §19.18's alignment
// clause wants a per-type alignment table, which §3's layout block does
// not carry; §19.10 declines to range-check an initializer whose literal
// may be a symbolic sizeof. Both belong where a Layout becomes a target,
// which is ir/lower.
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
