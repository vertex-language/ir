// Package inline copies a callee's body into its caller.
//
// It exists for the targets with no calling convention yet: a device
// function on AMDGPU is unlowerable until s_swappc_b64 and the scratch
// stack exist, and every __device__ helper would be refused by name.
// Inlined, it is just more of the kernel. PTX has calls and gains only
// speed, so there it is an option.
//
// The transformation is the textbook one. The call's block is split after
// the call; the callee's blocks are cloned into the caller under fresh
// labels, its parameters renamed to the call's arguments; each return
// becomes a branch to the split-off block, whose parameters are the
// call's results. Nothing is simplified afterwards — the branch from the
// call site into the callee's entry, and the entry itself, stay as they
// are, since a backend's block layout folds an empty edge for nothing.
//
// What it refuses, by name: recursion, since a body copied into itself
// never finishes; a callee that returns twice, whose second return has
// no block to come back to. What it leaves alone: a call through a
// pointer, a call to an import, a call to a variadic, a call to a naked
// function or an assembly body, and every invoke — those are calls the
// backend has to lower, or refuse, itself.
package inline

import (
	"fmt"
	"strconv"

	"github.com/vertex-language/ir"
)

// Options say which calls to inline.
type Options struct {
	// Into restricts inlining to callers this returns true for; nil is
	// every function. The AMDGPU backend passes a kernel test, so that a
	// device function called from two kernels is copied into both and
	// itself left for the backend to refuse.
	Into func(*ir.Func) bool

	// MaxDepth bounds how many callees deep a chain of inlining goes
	// through one call site; zero is 64. Recursion is refused before it
	// gets there; the bound is what makes that refusal a guarantee.
	MaxDepth int

	// Should, when set, makes inlining a choice: a call is inlined only
	// where it answers true, and a call this package cannot inline --
	// recursion, a chain past MaxDepth, a callee with a pad block or one
	// that returns twice -- is left as a call rather than an error. A CPU
	// backend, which has calls, sets it to take the small callees only.
	Should func(caller, callee *ir.Func) bool
}

// Module inlines every call it can in every function of m.
func Module(m *ir.Module, opts Options) error {
	for _, f := range m.Funcs() {
		if opts.Into != nil && !opts.Into(f) {
			continue
		}
		if err := Func(f, opts); err != nil {
			return err
		}
	}
	return m.Err()
}

// Func inlines every call it can in f, including the calls that arrive
// with an inlined body, until none is left that this package inlines.
func Func(f *ir.Func, opts Options) error {
	if _, asm := f.AsmBodyText(); asm {
		return nil
	}
	depth := opts.MaxDepth
	if depth <= 0 {
		depth = 64
	}
	choosing := opts.Should != nil
	st := &state{f: f, labels: map[string]bool{}, names: map[string]bool{}}
	for _, b := range f.Blocks() {
		st.labels[b.Label()] = true
	}
	f.WalkDefs(func(d *ir.Def) bool {
		if d.Name() != "" {
			st.names[d.Name()] = true
		}
		return true
	})
	// Each call site carries the chain of callees it came through, which
	// is how recursion is seen: the callee is already on the chain.
	type site struct {
		in    *ir.Inst
		chain []*ir.Func
	}
	var work []site
	for _, b := range f.Blocks() {
		for _, in := range b.Insts() {
			if in.Op().Verb == ir.VCall {
				work = append(work, site{in: in})
			}
		}
	}
	for len(work) > 0 {
		s := work[0]
		work = work[1:]
		callee, ok := inlinable(s.in)
		if !ok {
			continue
		}
		if choosing {
			if len(callee.Blocks()) == 0 || callee == f || onChain(s.chain, callee) || len(s.chain) >= depth ||
				!plain(callee) || !closed(s.in, callee) || !wider(f, callee) || !opts.Should(f, callee) {
				continue
			}
		}
		if !wider(f, callee) {
			return fmt.Errorf("inline: @%s requires (%s), which @%s does not", callee.Name(),
				callee.RequiresAttr()&^f.RequiresAttr(), f.Name())
		}
		if callee == f {
			return fmt.Errorf("inline: @%s calls itself", f.Name())
		}
		for _, c := range s.chain {
			if c == callee {
				return fmt.Errorf("inline: @%s is recursive through @%s", callee.Name(), s.chain[len(s.chain)-1].Name())
			}
		}
		if len(s.chain) >= depth {
			return fmt.Errorf("inline: @%s: a chain of calls %d deep at @%s", f.Name(), len(s.chain), callee.Name())
		}
		calls, err := st.inline(s.in, callee)
		if err != nil {
			return err
		}
		chain := append(append([]*ir.Func(nil), s.chain...), callee)
		for _, c := range calls {
			work = append(work, site{in: c, chain: chain})
		}
	}
	return f.Module().Err()
}

// wider reports whether caller may execute everything callee may: a
// callee that requires a capability (§P) its caller does not is the
// hardware path of a choice the caller makes with i1.cpu_has, and copied
// into the caller its instructions would run before the choice is made.
func wider(caller, callee *ir.Func) bool {
	return callee.RequiresAttr()&^caller.RequiresAttr() == 0
}

// closed reports whether call's callee can be copied whole: every
// parameter has an argument, and every operand in the body is a
// parameter, a block's parameter or a result defined before it in the
// order the copy is made. Anything else -- a value the copy would leave
// pointing into the callee -- makes the call one to keep.
func closed(call *ir.Inst, callee *ir.Func) bool {
	if call.NumArgs() != len(callee.Params()) {
		return false
	}
	known := map[*ir.Def]bool{}
	for _, p := range callee.Params() {
		known[p] = true
	}
	order := callee.RPO()
	for _, b := range order {
		for _, p := range b.Params() {
			known[p] = true
		}
	}
	for _, b := range order {
		insts := append(append([]*ir.Inst(nil), b.Insts()...), b.Term())
		for _, in := range insts {
			for _, a := range in.Args() {
				if a != nil && !known[a] {
					return false
				}
			}
			for _, r := range in.Results() {
				known[r] = true
			}
		}
	}
	return true
}

// onChain reports whether callee is already on a call site's chain: a
// recursion.
func onChain(chain []*ir.Func, callee *ir.Func) bool {
	for _, c := range chain {
		if c == callee {
			return true
		}
	}
	return false
}

// plain reports whether callee has none of what inline refuses: a pad
// block, or a second return.
func plain(callee *ir.Func) bool {
	if callee.IsReturnsTwice() {
		return false
	}
	for _, b := range callee.Blocks() {
		if b.IsPad() {
			return false
		}
	}
	return true
}

// inlinable is the callee, if the call is one this package inlines.
func inlinable(in *ir.Inst) (*ir.Func, bool) {
	callee, ok := in.Callee().(*ir.Func)
	if !ok {
		return nil, false
	}
	if _, asm := callee.AsmBodyText(); asm || callee.IsNaked() || callee.Signature().IsVariadic() {
		return nil, false
	}
	return callee, true
}

type state struct {
	f      *ir.Func
	labels map[string]bool
	names  map[string]bool
	n      int
}

// fresh is a label no block of the caller has.
func (st *state) fresh(base string) string {
	// A callee's name may hold what a label may not -- a lambda's
	// mangled closure carries a '$' -- so the label keeps the letters.
	b := []byte(base)
	for i, c := range b {
		if !(c == '_' || c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z' || c >= '0' && c <= '9' && i > 0) {
			b[i] = '_'
		}
	}
	base = string(b)
	for {
		st.n++
		l := base + "_" + strconv.Itoa(st.n)
		if !st.labels[l] {
			st.labels[l] = true
			return l
		}
	}
}

// inline replaces one call with the callee's body and returns the calls
// the body brought with it.
func (st *state) inline(call *ir.Inst, callee *ir.Func) ([]*ir.Inst, error) {
	f := st.f
	if callee.IsReturnsTwice() {
		return nil, fmt.Errorf("inline: @%s returns twice; a second return has nowhere to come back to", callee.Name())
	}
	for _, b := range callee.Blocks() {
		if b.IsPad() {
			return nil, fmt.Errorf("inline: @%s has a pad block; unwinding through an inlined body is not done yet", callee.Name())
		}
	}
	at := call.Block()

	// The continuation: what followed the call, taking the results.
	cont := at.SplitAfter(call, st.fresh(callee.Name()+"_ret"))
	if cont == nil {
		return nil, f.Module().Err()
	}
	for _, r := range call.Results() {
		p := cont.Param(r.Type(), st.name(r.Name())).Def()
		if p == nil {
			return nil, f.Module().Err()
		}
		f.ReplaceUses(r, p)
	}

	// The renaming: parameters to arguments, blocks to their copies with
	// their parameters, and every result as it is cloned.
	defs := map[*ir.Def]*ir.Def{}
	for i, p := range callee.Params() {
		defs[p] = call.Arg(i)
	}
	blocks := map[*ir.Block]*ir.Block{}
	order := callee.RPO()
	for _, b := range order {
		nb := f.Block(st.fresh(callee.Name() + "_" + b.Label()))
		blocks[b] = nb
		for _, p := range b.Params() {
			np := nb.Param(p.Type(), st.name(p.Name())).Def()
			if np == nil {
				return nil, f.Module().Err()
			}
			defs[p] = np
		}
	}
	rename := func(d *ir.Def) *ir.Def { return defs[d] }
	reblock := func(b *ir.Block) *ir.Block { return blocks[b] }

	var calls []*ir.Inst
	for _, b := range order {
		nb := blocks[b]
		for _, in := range b.Insts() {
			// A frame slot of the callee is a frame slot of the caller:
			// ptr.alloc lives in the entry block, and the copy goes to
			// the caller's.
			into := nb
			if in.Op().Verb == ir.VAlloc {
				into = f.Entry()
			}
			out := into.Clone(in, rename, reblock)
			if out == nil {
				return nil, f.Module().Err()
			}
			for i, r := range in.Results() {
				nr := out.Result(i)
				nr.SetName(st.name(r.Name()))
				defs[r] = nr
			}
			if out.Op().Verb == ir.VCall {
				calls = append(calls, out)
			}
		}
		term := b.Term()
		if term.Op().Verb == ir.VReturn {
			vals := make([]ir.Value, len(term.Args()))
			for i, a := range term.Args() {
				vals[i] = rename(a).Value()
			}
			nb.Br(cont.To(vals...))
			continue
		}
		if nb.Clone(term, rename, reblock) == nil {
			return nil, f.Module().Err()
		}
	}

	// The call itself: gone, its block branching into the copy.
	at.Remove(call)
	at.Br(blocks[callee.Entry()].To())
	return calls, f.Module().Err()
}

// name keeps a callee's register name where the caller has none like it,
// so the copy reads like the original where it can and prints without
// two registers of one name where it cannot.
func (st *state) name(s string) string {
	if s == "" || st.names[s] {
		return ""
	}
	st.names[s] = true
	return s
}
