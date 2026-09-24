package air

import (
	"fmt"

	am "github.com/vertex-language/air"
	"github.com/vertex-language/ir"
)

// Every pointer in AIR is in one address space, and VIR's pointers say
// none. So each pointer value's space is inferred from where it comes from,
// over the kernel after inlining:
//
//   - a kernel's pointer argument is device memory, or constant memory for
//     one passed by value;
//   - the address of a shared global, or a dynamic shared import, is
//     threadgroup memory, and of a read-only global constant memory;
//   - an allocation is thread memory;
//   - a pointer loaded from memory, or made from an integer, is device
//     memory: that is where a pointer stored in a buffer points, as Metal's
//     argument buffers have it;
//   - ptr.add keeps its pointer's space; a select, and a block parameter,
//     take their operands' spaces, which must agree.
//
// Two spaces meeting is an error naming the value. A pointer nothing
// decides -- a block parameter fed only by itself -- is device memory.

// spaceState is the lattice: unknown, one space, or a conflict.
type spaceState struct {
	known    bool
	space    am.Space
	conflict bool
	with     am.Space // the other space, for the message
}

func (s *spaceState) meet(t am.Space) bool {
	switch {
	case s.conflict:
		return false
	case !s.known:
		s.known, s.space = true, t
		return true
	case s.space != t:
		s.conflict, s.with = true, t
		return true
	}
	return false
}

// inferSpaces is every pointer def's space in f.
func (l *lowerer) inferSpaces(f *ir.Func, bind []binding) (map[*ir.Def]am.Space, error) {
	st := map[*ir.Def]*spaceState{}
	get := func(d *ir.Def) *spaceState {
		s, ok := st[d]
		if !ok {
			s = &spaceState{}
			st[d] = s
		}
		return s
	}
	// Sources.
	for i, d := range f.Params() {
		if d.Type() != ir.TypePtr {
			continue
		}
		get(d).meet(bind[i].space)
	}
	var sourceErr error
	f.WalkInsts(func(in *ir.Inst) bool {
		for _, r := range in.Results() {
			if r.Type() != ir.TypePtr {
				continue
			}
			switch in.Op().Verb {
			case ir.VAlloc, ir.VAlloca:
				get(r).meet(am.Thread)
			case ir.VGetAddr:
				sp, err := l.symbolSpace(in.Symbol())
				if err != nil && sourceErr == nil {
					sourceErr = err
				}
				get(r).meet(sp)
			case ir.VLoad, ir.VFromI64, ir.VAtomicLoad, ir.VAtomicRmwXchg, ir.VAtomicCas:
				get(r).meet(am.Device)
			}
		}
		return true
	})
	if sourceErr != nil {
		return nil, sourceErr
	}

	// Propagation, to a fixpoint.
	for changed := true; changed; {
		changed = false
		f.WalkInsts(func(in *ir.Inst) bool {
			switch in.Op().Verb {
			case ir.VAdd, ir.VSelect:
				if in.Op().Type != ir.TypePtr {
					return true
				}
				r := in.Result(0)
				for _, a := range in.Args() {
					if a.Type() == ir.TypePtr {
						if s := get(a); s.known && get(r).meet(s.space) {
							changed = true
						}
					}
				}
			}
			for _, t := range in.Targets() {
				params := t.Block().Params()
				for i, a := range t.Args() {
					if a.Type() != ir.TypePtr {
						continue
					}
					if s := get(a); s.known && get(params[i]).meet(s.space) {
						changed = true
					}
				}
			}
			return true
		})
	}

	out := map[*ir.Def]am.Space{}
	var err error
	f.WalkDefs(func(d *ir.Def) bool {
		if d.Type() != ir.TypePtr {
			return true
		}
		s := get(d)
		switch {
		case s.conflict && err == nil:
			err = fmt.Errorf("%s may point into %v or %v memory; AIR has no generic pointer, so a pointer is one or the other",
				describe(d), s.space, s.with)
		case !s.known:
			out[d] = am.Device
		default:
			out[d] = s.space
		}
		return true
	})
	return out, err
}

// symbolSpace is the space of a symbol's address.
func (l *lowerer) symbolSpace(sym ir.Symbol) (am.Space, error) {
	switch s := sym.(type) {
	case *ir.Global:
		if s.Domain() == ir.Shared {
			return am.Threadgroup, nil
		}
		if s.Domain() == ir.RO {
			return am.Constant, nil
		}
		return am.Device, fmt.Errorf("@%s is writable: Metal has no writable memory at program scope", s.Name())
	case *ir.GlobalImport:
		if s.Domain() == ir.Shared {
			return am.Threadgroup, nil
		}
		return am.Device, fmt.Errorf("@%s is imported: an AIR library has no other module to link against", s.Name())
	}
	return am.Device, fmt.Errorf("the address of %v: only globals have addresses in AIR", sym)
}

// describe is a def as a message names it: %name, or the instruction that
// made it.
func describe(d *ir.Def) string {
	if d.Name() != "" {
		return "%" + d.Name()
	}
	if d.IsParam() {
		if b := d.Block(); b != nil && !b.IsEntry() {
			return fmt.Sprintf("parameter %d of block @%s", d.Index(), b.Label())
		}
		return fmt.Sprintf("argument %d", d.Index())
	}
	return "the result of " + d.Inst().Op().String()
}
