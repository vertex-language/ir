package verify

import "github.com/vertex-language/ir"

// §19.6's second clause: a ptr.blockaddr names a block that some brind in
// the same function branches to.
//
// The rest of §19.6–9 is the builder's, and this is where to find out why
// none of it is here:
//
//   - §19.6, first clause — ptr.alloc appears in the entry block only.
//     ir.ErrPlacement, at the call: which block is being emitted into is
//     the builder's own state.
//   - §19.7 — ptr.tlsaddr names a global in domain tls, and tlsmodel
//     appears only on such a global. ir.ErrPlacement, twice: at the
//     instruction and at the declaration, each with the global in hand.
//   - §19.8 — align is a power of two not exceeding the access width.
//     ir.ErrAlign, where the width is the mnemonic's own.
//   - §19.9 — the compare-and-swap and read-modify-write ordering rules.
//     ir.ErrOrdering, where the orderings are literal arguments.
//
// Every one of those is a fact about a single instruction or declaration
// in isolation, which is the whole of what a builder can catch. What is
// left is the one clause of the four that relates two instructions in two
// blocks — and that no call site can see, because the brind making a
// block address a legitimate target may not be written yet.

// blockAddrs is §19.6's second clause. A block address that no brind
// branches to is an address with nowhere to be used: §17 admits
// ptr.blockaddr for brind and for nothing else, so the label would name a
// block that control can only fall into, never jump to.
func (c *checker) blockAddrs(blocks []*ir.Block) {
	targeted := make(map[*ir.Block]bool, len(blocks))
	for _, b := range blocks {
		term := b.Term()
		if term.Op() != (ir.Op{Type: ir.TypeNone, Verb: ir.VBrInd}) {
			continue
		}
		for _, l := range term.Labels() {
			targeted[l] = true
		}
	}

	for _, b := range blocks {
		for i, in := range b.All() {
			if in.Op() != (ir.Op{Type: ir.TypePtr, Verb: ir.VBlockAddr}) {
				continue
			}
			for _, l := range in.Labels() {
				if targeted[l] {
					continue
				}
				c.fail(b, i, in.Op(), ErrBlockAddr,
					"@%s is the target of no brind in this function", l.Label())
				if c.full() {
					return
				}
			}
		}
	}
}
