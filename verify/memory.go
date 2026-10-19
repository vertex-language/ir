package verify

import "github.com/vertex-language/ir"

// blockAddrs checks §19.6's second clause: every ptr.blockaddr must name a block
// targeted by a brind instruction in the same function.
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
