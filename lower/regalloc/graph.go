package regalloc

import (
	"sort"

	"github.com/vertex-language/ir/lower/mir"
)

// The interference graph.
//
// A map of maps is the obvious shape for it and the wrong one. The graph is
// built once per spill round, a function whose register pressure is a
// hundred over what the pool has spills about a hundred times, and each
// build inserts an edge for every pair of values live across an
// instruction — so the insertion is the operation that matters, and a map
// pays a hash and a probe for each one.
//
// A vreg is already a dense index into its function, since mir.Func hands
// them out in order, so this keeps one neighbour list per vreg and appends
// to it. Appending admits duplicates, which is why seal exists: the lists
// are sorted and uniqued once, at the end, and the colouring then walks a
// node's real neighbours and nothing else.
//
// The alternative, a bit per pair, answers "do these interfere" in one AND
// but costs n²/8 bytes and makes every insertion a miss in a matrix far
// bigger than the cache. The graph is sparse — a function with twenty
// thousand vregs has perhaps a hundred live at once — so the lists are both
// smaller and quicker, and iterating a node's neighbours costs its degree
// rather than a scan of the whole row.

// graph is an interference graph over the vregs of one mir.Func: an edge
// between two that cannot share a register.
type graph struct {
	n      int
	adj    [][]mir.VReg // neighbours, with duplicates until sealed
	isNode []bool       // whether the function names this vreg at all
	nodes  []mir.VReg
}

func newGraph(n int) *graph {
	return &graph{n: n, adj: make([][]mir.VReg, n), isNode: make([]bool, n)}
}

// addNode records that the function names v, whether or not anything
// interferes with it: a value nothing else is live across still needs a
// register to be written into.
func (g *graph) addNode(v mir.VReg) {
	if g.inRange(v) {
		g.isNode[v] = true
	}
}

// addEdge records that a and b cannot share a register.
//
// A repeat costs an append rather than a lookup; seal is where repeats go.
func (g *graph) addEdge(a, b mir.VReg) {
	if a == b || !g.inRange(a) || !g.inRange(b) {
		return
	}
	g.isNode[a] = true
	g.isNode[b] = true
	g.adj[a] = append(g.adj[a], b)
	g.adj[b] = append(g.adj[b], a)
}

// seal sorts each neighbour list and drops the duplicates, after which the
// graph is read-only and degree and neighbours mean what they say.
func (g *graph) seal() {
	for v, list := range g.adj {
		if len(list) < 2 {
			continue
		}
		sort.Slice(list, func(i, j int) bool { return list[i] < list[j] })
		out := list[:1]
		for _, n := range list[1:] {
			if n != out[len(out)-1] {
				out = append(out, n)
			}
		}
		g.adj[v] = out
	}
}

// degree is how many vregs interfere with v.
func (g *graph) degree(v mir.VReg) int {
	if !g.inRange(v) {
		return 0
	}
	return len(g.adj[v])
}

// neighbours calls fn for each vreg interfering with v, in vreg order.
func (g *graph) neighbours(v mir.VReg, fn func(mir.VReg)) {
	if !g.inRange(v) {
		return
	}
	for _, n := range g.adj[v] {
		fn(n)
	}
}

// Nodes is every vreg the function names, ascending.
//
// Ascending rather than in the order they were added, and the reason is the
// package doc's: the colouring walks this list, so its order decides the
// assignment, and an assignment that depended on when a vreg was first
// mentioned would not be a function of the MIR alone. A vreg is its own
// index, so ascending costs a scan and no sort.
func (g *graph) Nodes() []mir.VReg {
	if g.nodes == nil {
		g.nodes = make([]mir.VReg, 0, g.n)
		for v, ok := range g.isNode {
			if ok {
				g.nodes = append(g.nodes, mir.VReg(v))
			}
		}
	}
	return g.nodes
}

func (g *graph) inRange(v mir.VReg) bool { return v >= 0 && int(v) < g.n }
