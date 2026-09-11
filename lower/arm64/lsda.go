package arm64

// The language-specific data area: the table a personality routine reads to
// decide whether this frame handles an exception, and where to jump if it
// does.
//
// # The shape
//
// It is the Itanium C++ ABI's, which is what every Unix personality reads
// whatever language threw — __gxx_personality_v0, __objc_personality_v0 and
// Rust's all parse the same four parts:
//
//   - a header, naming how the two tables below are encoded;
//   - the call-site table, mapping a range of the function to a landing pad
//     and an action;
//   - the action table, a chain of type indices per landing pad;
//   - the type table, indexed backwards from its own end, holding the
//     type-info the catch clauses name.
//
// # What this emits, and what it does not
//
// LPStart is omitted, which makes every offset relative to the function's
// first byte. The call-site encoding is uleb128 and the type-table encoding
// is indirect|pcrel|sdata4 — a distance to a GOT slot, because the class a
// @catch names may be in another image and a read-only section cannot hold a
// relocated pointer into one.
//
// The call-site table covers the function end to end. A region with no
// invoke in it gets an entry with no landing pad rather than a gap: an
// address the table does not mention is not "nothing happens here" to every
// personality that reads it, and the cost of saying so is three bytes.
//
// Filter clauses — C++'s exception specifications — are not emitted. They
// need a second table after the type table and a negative index to reach it,
// and no frontend in this tree produces one.
//
// # The selector
//
// §G3 says a pad's second parameter is "the personality's selector value"
// and stops there, which was room to choose. What this backend delivers is
// the 1-based position of the matching clause among the pad's *catch*
// clauses, or zero when the pad was entered to run its cleanup.
//
// The raw value is an index into the function's type table, and a frontend
// that had to predict one would have to know how many clauses every earlier
// pad declared — which is a fact about a table it cannot see. Since a pad's
// clauses are consecutive in that table, the translation is one subtraction,
// and the common case of a single pad needs not even that.

import (
	"fmt"
	"sort"

	arm64asm "github.com/vertex-language/arm64"

	"github.com/vertex-language/ir"
)

// The DWARF exception-header encodings this emits. There are many more and
// none of them is needed: a compiler picks one spelling and every personality
// reads what it was given.
const (
	dwEhPeOmit     = 0xff
	dwEhPeUleb128  = 0x01
	dwEhPeIndirect = 0x80
	dwEhPePcrel    = 0x10
	dwEhPeSdata4   = 0x0b

	// ttypeEncoding is what a type-table entry is: a four-byte signed
	// distance from the entry to a GOT slot holding the type-info.
	ttypeEncoding = dwEhPeIndirect | dwEhPePcrel | dwEhPeSdata4
)

// exceptTabSection is where the tables go.
const exceptTabSection = "__TEXT,__gcc_except_tab"

// ehPad is one pad block's row in the tables.
type ehPad struct {
	blk *ir.Block

	// base is the type-table index just below this pad's own: the pad's
	// first catch clause is base+1. It is what the pad's prologue
	// subtracts to turn the personality's selector into a clause number.
	base int

	// action is the call-site table's action field for an invoke that
	// unwinds here: zero for a pad that only cleans up, and otherwise one
	// more than the byte offset of its first action record.
	action int

	// at is the landing pad's offset from the function's first byte,
	// filled in during emission.
	at    uint32
	found bool
}

// ehSite is one invoke: the range of the function the call occupies and the
// pad it unwinds to.
type ehSite struct {
	pad        int
	begin, end uint32
	closed     bool
}

// ehPlan is one function's exception tables.
//
// Built before selection, because isel has to name a site and a pad by index
// before either has an address, and finished during emission, when they do.
type ehPlan struct {
	pads  []*ehPad
	padOf map[*ir.Block]int

	// types is the type table in index order: types[i-1] is index i, and a
	// nil entry is a catch-all, which is what a clause with no type-info
	// global means.
	types []ir.Symbol

	// actions is the action table, already laid out: the plan knows every
	// clause before it knows a single address.
	actions []byte

	sites []*ehSite

	start, end uint32
}

// planEH builds the tables fn's pads describe, or nil if it has none.
func planEH(fn *ir.Func) (*ehPlan, error) {
	var p *ehPlan
	for _, blk := range fn.Blocks() {
		if !blk.IsPad() {
			continue
		}
		if p == nil {
			p = &ehPlan{padOf: map[*ir.Block]int{}}
		}
		pad := &ehPad{blk: blk, base: len(p.types)}
		if err := p.addClauses(fn, pad); err != nil {
			return nil, err
		}
		p.padOf[blk] = len(p.pads)
		p.pads = append(p.pads, pad)
	}
	if p != nil && fn.PersonalityFn() == nil {
		return nil, fmt.Errorf("%s has a pad block and no personality routine", fn.Name())
	}
	return p, nil
}

// addClauses gives one pad its type-table entries and its action chain.
//
// A pad that only cleans up gets neither: its call sites carry action zero,
// which is what "run the cleanup and keep unwinding" is spelled as, and an
// action record for it would say the same thing in two more bytes.
func (p *ehPlan) addClauses(fn *ir.Func, pad *ehPad) error {
	clauses := pad.blk.Clauses()
	catches := 0
	for _, c := range clauses {
		switch c.Kind() {
		case ir.PadCatch:
			catches++
			p.types = append(p.types, c.TypeInfo())
		case ir.PadCleanup:
		case ir.PadFilter:
			return fmt.Errorf("%s: @%s: a filter clause is not emitted yet",
				fn.Name(), pad.blk.Label())
		default:
			return fmt.Errorf("%s: @%s: unknown pad clause", fn.Name(), pad.blk.Label())
		}
	}
	if catches == 0 {
		return nil
	}

	pad.action = len(p.actions) + 1
	next := pad.base
	for i, c := range clauses {
		idx := 0
		if c.Kind() == ir.PadCatch {
			next++
			idx = next
		}
		p.actions = appendSLEB(p.actions, int64(idx))
		if i == len(clauses)-1 {
			// No further action: the chain ends, and the personality
			// reports no match if it got this far.
			p.actions = append(p.actions, 0)
			break
		}
		// The next record starts one byte past this field, and the field
		// is read relative to its own position.
		p.actions = appendSLEB(p.actions, 1)
	}
	return nil
}

// site registers an invoke and returns its index, which isel puts in the
// pseudo-instructions that bracket the call.
func (p *ehPlan) site(pad *ir.Block) (int, error) {
	i, ok := p.padOf[pad]
	if !ok {
		return 0, fmt.Errorf("unwind edge to @%s, which is not a pad block", pad.Label())
	}
	p.sites = append(p.sites, &ehSite{pad: i})
	return len(p.sites) - 1, nil
}

// callSites renders the call-site table from the offsets emission recorded.
func (p *ehPlan) callSites() ([]byte, error) {
	entry := func(out []byte, begin, length uint32, lp uint32, action int) []byte {
		out = appendULEB(out, uint64(begin))
		out = appendULEB(out, uint64(length))
		out = appendULEB(out, uint64(lp))
		return appendULEB(out, uint64(action))
	}

	// Address order, not the order selection registered them in: blocks are
	// selected in reverse postorder and emitted in the order the function
	// declared them, so the two disagree whenever a frontend appended a
	// block after something already branching to it.
	sites := append([]*ehSite(nil), p.sites...)
	for i, s := range sites {
		if !s.closed {
			return nil, fmt.Errorf("invoke %d was never emitted", i)
		}
	}
	sort.Slice(sites, func(i, j int) bool { return sites[i].begin < sites[j].begin })

	var out []byte
	at := uint32(0)
	for i, s := range sites {
		if s.begin < at {
			return nil, fmt.Errorf("invoke %d begins at 0x%x, inside the site before it", i, s.begin)
		}
		if s.begin > at {
			out = entry(out, at, s.begin-at, 0, 0)
		}
		pad := p.pads[s.pad]
		if !pad.found {
			return nil, fmt.Errorf("@%s is an unwind edge's target and was never emitted", pad.blk.Label())
		}
		out = entry(out, s.begin, s.end-s.begin, pad.at, pad.action)
		at = s.end
	}
	if at < p.end {
		out = entry(out, at, p.end-at, 0, 0)
	}
	return out, nil
}

// emitLSDA writes the tables and returns the symbol naming them.
func emitLSDA(am *arm64asm.Module, fn *ir.Func, p *ehPlan) (string, error) {
	cst, err := p.callSites()
	if err != nil {
		return "", fmt.Errorf("lower: %s: %w", fn.Name(), err)
	}

	// Everything after the type-table base's own field, whose length is the
	// one thing that cannot be known before the rest is laid out -- which
	// is exactly why the field is read relative to the byte after itself.
	var body []byte
	body = append(body, dwEhPeUleb128)
	body = appendULEB(body, uint64(len(cst)))
	body = append(body, cst...)
	body = append(body, p.actions...)

	// No type table means no field to reach one with, and nothing to align.
	head := []byte{dwEhPeOmit, dwEhPeOmit}
	pad := 0
	if len(p.types) > 0 {
		head[1] = ttypeEncoding
		// Sized by fixpoint: the distance is measured from the byte after
		// the field that states it, so the field's own length does not
		// enter the distance -- but it does move everything before the
		// type table, and so decides how much padding there is.
		for n := 1; ; {
			total := len(head) + n + len(body)
			pad = int(alignUp(uint64(total), 4)) - total
			enc := appendULEB(nil, uint64(len(body)+pad+4*len(p.types)))
			if len(enc) == n {
				head = append(head, enc...)
				break
			}
			n = len(enc)
		}
	}

	name := "GCC_except_table." + fn.Name()
	s := am.SectionNamed(exceptTabSection, arm64asm.ROData)
	s.Align(4)
	s.Label(name, arm64asm.Local, arm64asm.ObjectSym)
	s.Data(head)
	s.Data(body)
	s.Zero(pad)

	// Backwards: entry i sits four bytes below entry i-1, so that the base
	// at the end of the table is what an index counts down from.
	for i := len(p.types); i >= 1; i-- {
		ti := p.types[i-1]
		if ti == nil {
			// A catch-all. The personality matches it against anything and
			// there is no type to name.
			s.Long(0)
			continue
		}
		s.Ref(ti.Name(), arm64asm.RefGotPrel32)
	}
	s.EndLabel(name)
	return name, nil
}

// appendULEB appends v as an unsigned LEB128.
func appendULEB(out []byte, v uint64) []byte {
	for {
		b := byte(v & 0x7f)
		v >>= 7
		if v != 0 {
			b |= 0x80
		}
		out = append(out, b)
		if v == 0 {
			return out
		}
	}
}

// appendSLEB appends v as a signed LEB128.
func appendSLEB(out []byte, v int64) []byte {
	for {
		b := byte(v & 0x7f)
		v >>= 7
		done := (v == 0 && b&0x40 == 0) || (v == -1 && b&0x40 != 0)
		if !done {
			b |= 0x80
		}
		out = append(out, b)
		if done {
			return out
		}
	}
}
