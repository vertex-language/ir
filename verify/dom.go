package verify

import "github.com/vertex-language/ir"

// Dominance lives here, unexported, because §19.1 and §19.5 are its only
// consumers. There is no analysis package: CFG shape is traversal and
// belongs to ir/walk.go, and liveness is a fact about MIR — vregs and
// physregs — so it belongs to whoever allocates registers.

// A domTree is the immediate-dominator map of one function's reachable
// blocks, plus the reverse-postorder numbering the map was built from.
//
// This is Cooper, Harvey, and Kennedy's iterative algorithm: walk the
// blocks in reverse postorder, set each one's immediate dominator to the
// meet of its already-processed predecessors, and repeat until nothing
// moves. It is quadratic in the worst case and linear on real control
// flow, and it needs no data structure beyond the two maps below — which
// is the trade a verifier wants, since it runs once over a function it is
// not going to rewrite.
type domTree struct {
	idom map[*ir.Block]*ir.Block // entry maps to itself
	rpo  map[*ir.Block]int       // reverse-postorder index; absent means unreachable
}

func newDomTree(f *ir.Func) *domTree {
	order := f.RPO()
	d := &domTree{
		idom: make(map[*ir.Block]*ir.Block, len(order)),
		rpo:  make(map[*ir.Block]int, len(order)),
	}
	if len(order) == 0 {
		return d
	}
	for i, b := range order {
		d.rpo[b] = i
	}
	d.idom[order[0]] = order[0]

	preds := f.Preds()
	for changed := true; changed; {
		changed = false
		for _, b := range order[1:] {
			var idom *ir.Block
			for _, p := range preds[b] {
				// An unreachable predecessor says nothing about what
				// dominates b: no path from the entry block runs
				// through it, so no definition of its can be the one
				// a use in b sees.
				if _, ok := d.rpo[p]; !ok {
					continue
				}
				if _, done := d.idom[p]; !done {
					continue
				}
				if idom == nil {
					idom = p
					continue
				}
				idom = d.intersect(idom, p)
			}
			if idom != nil && d.idom[b] != idom {
				d.idom[b] = idom
				changed = true
			}
		}
	}
	return d
}

// intersect is the meet: the nearest block that dominates both a and b,
// found by walking whichever is deeper in reverse postorder up its own
// dominator chain until the two meet.
func (d *domTree) intersect(a, b *ir.Block) *ir.Block {
	for a != b {
		for d.rpo[a] > d.rpo[b] {
			a = d.idom[a]
		}
		for d.rpo[b] > d.rpo[a] {
			b = d.idom[b]
		}
	}
	return a
}

// reachable reports whether any path from the entry block reaches b.
func (d *domTree) reachable(b *ir.Block) bool {
	_, ok := d.rpo[b]
	return ok
}

// dominates reports whether every path from the entry block to b runs
// through a first. A block dominates itself; an unreachable block
// dominates nothing and is dominated by nothing.
func (d *domTree) dominates(a, b *ir.Block) bool {
	if !d.reachable(a) || !d.reachable(b) {
		return false
	}
	for a != b {
		up := d.idom[b]
		if up == nil || up == b {
			return false // walked to the entry block without finding a
		}
		b = up
	}
	return true
}
