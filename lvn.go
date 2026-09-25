package ir

// Two cleanups that find the same value computed twice.
//
// ThreadJoins undoes the join a frontend lowers `a && b` and `a || b`
// through. Each operand's block branches to one shared block with its
// answer as an argument, and that block does nothing but branch on it:
//
//	@b5:    br @j(0)                  @b5:    br @no
//	@c1:    ...; br @j(%21)    ==>    @c1:    ...; brif %21, @yes, @no
//	@j(%p i1): brif %p, @yes, @no
//
// Beyond the branch it saves, the join is a block with two predecessors
// standing between the code that tests `raw[i] == 13` and the code that
// tests `raw[i] == 10`, which hides from NumberValues that the second
// bounds check and load are the first one's again.
//
// NumberValues is local value numbering over extended basic blocks: a
// block with exactly one predecessor carries on with what that
// predecessor knew. What it knows is every pure instruction computed so
// far, keyed by its operation and operands; every load, until something
// that may write memory; and every branch condition's answer on the edge
// taken. A second computation of a known value is replaced by the first,
// and a brif on a condition whose answer is known becomes a br. Applied to
// a bounds-checked byte loop, the count load, the compare, the check, the
// address and the byte load of a repeated `raw[i]` all go.

// ThreadJoins rewrites each branch into a join that only branches on its
// one parameter so that it branches where the join would, and reports how
// many branches it rewrote.
func (f *Func) ThreadJoins() int {
	if f == nil || len(f.blocks) == 0 || f.m == nil || f.m.err != nil {
		return 0
	}
	uses := map[*Def]int{}
	f.WalkUses(func(u Use) bool {
		if d := u.Def(); d != nil {
			uses[d]++
		}
		return true
	})
	n := 0
	for _, j := range f.blocks {
		if j.isEntry || j.isPad || len(j.insts) != 0 || len(j.params) != 1 || j.term == nil {
			continue
		}
		t := j.term
		p := j.params[0]
		if t.op.Verb != VBrIf || t.args[0] != p || uses[p] != 1 || t.im == nil || len(t.im.targets) != 2 {
			continue
		}
		then, els := t.im.targets[0], t.im.targets[1]
		if then.blk == j || els.blk == j {
			continue
		}
		for _, b := range f.blocks {
			pt := b.term
			if b == j || pt == nil || pt.op.Verb != VBr || pt.im == nil || len(pt.im.targets) != 1 {
				continue
			}
			edge := pt.im.targets[0]
			if edge.blk != j || len(edge.args) != 1 {
				continue
			}
			v := edge.args[0]
			if c, ok := constOf(v); ok {
				to := els
				if c != 0 {
					to = then
				}
				b.term = &Inst{op: Op{TypeNone, VBr}, blk: b, im: &imm{targets: []BlockTarget{copyTarget(to)}}}
			} else {
				b.term = &Inst{op: Op{TypeNone, VBrIf}, blk: b, args: []*Def{v},
					im: &imm{targets: []BlockTarget{copyTarget(then), copyTarget(els)}}}
			}
			n++
		}
	}
	if n > 0 {
		f.dropUnreachable()
	}
	return n
}

// dropUnreachable removes the blocks no path from the entry reaches any
// more -- a join every branch now goes around, the arm of a branch whose
// answer became known -- which the verifier (§19.2) does not admit.
func (f *Func) dropUnreachable() {
	live := map[*Block]bool{}
	for _, b := range f.RPO() {
		live[b] = true
	}
	var dead []*Block
	for _, b := range f.blocks {
		if !live[b] {
			dead = append(dead, b)
		}
	}
	for _, b := range dead {
		f.RemoveBlock(b)
	}
}

func copyTarget(t BlockTarget) BlockTarget {
	out := BlockTarget{blk: t.blk, bare: t.bare}
	if len(t.args) > 0 {
		out.args = append([]*Def(nil), t.args...)
	}
	return out
}

// constOf is the integer an i1, i32 or i64 const instruction defines.
func constOf(d *Def) (int64, bool) {
	if d == nil || d.inst == nil || d.inst.op.Verb != VConst || d.inst.im == nil || !d.inst.im.hasLit {
		return 0, false
	}
	if d.inst.im.lit.kind != ConstInt {
		return 0, false
	}
	return d.inst.im.lit.i, true
}

// vnKey is what makes two instructions compute the same value.
type vnKey struct {
	op         Op
	a0, a1, a2 *Def
	nargs      int
	kind       ConstKind
	i          int64
	f          float64
	sym        Symbol
	width      uint64
}

// vnState is what a block knows on entry and adds to as it goes.
type vnState struct {
	pure  map[vnKey]*Def
	loads map[vnKey]*Def
	facts map[*Def]bool
}

func (s *vnState) clone() *vnState {
	c := &vnState{
		pure:  make(map[vnKey]*Def, len(s.pure)),
		loads: make(map[vnKey]*Def, len(s.loads)),
		facts: make(map[*Def]bool, len(s.facts)),
	}
	for k, v := range s.pure {
		c.pure[k] = v
	}
	for k, v := range s.loads {
		c.loads[k] = v
	}
	for k, v := range s.facts {
		c.facts[k] = v
	}
	return c
}

// NumberValues replaces each recomputation of a value its extended basic
// block already has, and each branch whose answer is already known, and
// reports how many instructions it removed.
func (f *Func) NumberValues() int {
	if f == nil || len(f.blocks) == 0 || f.m == nil || f.m.err != nil {
		return 0
	}
	order := f.RPO()
	preds := f.Preds()
	repl := map[*Def]*Def{}
	resolve := func(d *Def) *Def {
		for n := 0; n <= len(repl); n++ {
			r, ok := repl[d]
			if !ok {
				return d
			}
			d = r
		}
		return d
	}
	// Constants stay where they are (see pureKey) but key by their value,
	// so that `raw + 16` in one block and `raw + 16` in the next, each
	// with a const of its own, are known to be one address.
	consts := map[vnKey]*Def{}
	canon := func(d *Def) *Def {
		if d == nil || d.inst == nil || d.inst.op.Verb != VConst || d.inst.im == nil || !d.inst.im.hasLit {
			return d
		}
		lit := d.inst.im.lit
		if lit.kind != ConstInt && lit.kind != ConstFloat {
			return d
		}
		k := vnKey{op: d.inst.op, kind: lit.kind, i: lit.i, f: lit.f}
		if c, ok := consts[k]; ok {
			return c
		}
		consts[k] = d
		return d
	}
	end := map[*Block]*vnState{}
	removed := 0
	folded := false
	for _, b := range order {
		var st *vnState
		if ps := preds[b]; len(ps) == 1 && !b.isPad {
			if pst, ok := end[ps[0]]; ok {
				st = pst.clone()
				// The edge taken decides the predecessor's condition.
				if t := ps[0].term; t != nil && t.op.Verb == VBrIf && t.im != nil && len(t.im.targets) == 2 {
					then, els := t.im.targets[0].blk, t.im.targets[1].blk
					c := resolve(t.args[0])
					if then == b && els != b {
						st.facts[c] = true
					} else if els == b && then != b {
						st.facts[c] = false
					}
				}
			}
		}
		if st == nil {
			st = &vnState{pure: map[vnKey]*Def{}, loads: map[vnKey]*Def{}, facts: map[*Def]bool{}}
		}
		kept := b.insts[:0]
		for _, in := range b.insts {
			for i, a := range in.args {
				in.args[i] = resolve(a)
			}
			if same := identity(in); same != nil {
				// x*1, x+0, x-0, x|0, x^0, x<<0: the operand itself.
				repl[in.results[0]] = same
				removed++
				continue
			}
			if k, ok := pureKey(in, canon); ok {
				if d, seen := st.pure[k]; seen {
					repl[in.results[0]] = d
					removed++
					continue
				}
				st.pure[k] = in.results[0]
			} else if k, ok := loadKey(in); ok {
				if d, seen := st.loads[k]; seen {
					repl[in.results[0]] = d
					removed++
					continue
				}
				st.loads[k] = in.results[0]
			} else if !harmless(in) {
				// Anything else may write memory: what was loaded may
				// no longer be what is there.
				st.loads = map[vnKey]*Def{}
			}
			kept = append(kept, in)
		}
		b.insts = kept
		if t := b.term; t != nil {
			for i, a := range t.args {
				t.args[i] = resolve(a)
			}
			if t.op.Verb == VBrIf && t.im != nil && len(t.im.targets) == 2 {
				c := t.args[0]
				known, ok := st.facts[c]
				if !ok {
					if v, isConst := constOf(c); isConst {
						known, ok = v != 0, true
					}
				}
				if ok {
					to := t.im.targets[1]
					if known {
						to = t.im.targets[0]
					}
					b.term = &Inst{op: Op{TypeNone, VBr}, blk: b, im: &imm{targets: []BlockTarget{to}}}
					folded = true
				}
			}
		}
		end[b] = st
	}
	if folded {
		f.dropUnreachable()
	}
	if len(repl) > 0 {
		f.WalkUses(func(u Use) bool {
			if d := u.Def(); d != nil {
				if _, ok := repl[d]; ok {
					u.Set(resolve(d))
				}
			}
			return true
		})
	}
	return removed + f.dropDead()
}

// dropDead removes pure instructions and plain loads whose results nothing
// reads -- the address of a literal a frontend folded to a constant, the
// operands of an instruction NumberValues replaced -- until none is left.
func (f *Func) dropDead() int {
	removed := 0
	for {
		uses := map[*Def]int{}
		f.WalkUses(func(u Use) bool {
			if d := u.Def(); d != nil {
				uses[d]++
			}
			return true
		})
		n := 0
		for _, b := range f.blocks {
			kept := b.insts[:0]
			for _, in := range b.insts {
				if len(in.results) == 1 && uses[in.results[0]] == 0 {
					if pureVerb(in.op.Verb) && !traps(in.op.Verb) {
						n++
						continue
					}
					if _, isLoad := loadKey(in); isLoad {
						n++
						continue
					}
				}
				kept = append(kept, in)
			}
			b.insts = kept
		}
		if n == 0 {
			return removed
		}
		removed += n
	}
}

// pureKey is the key of an instruction whose one result depends on its
// operands alone -- not on memory, not on the thread -- or false.
func pureKey(in *Inst, canon func(*Def) *Def) (vnKey, bool) {
	// A constant is left where it is: the backend folds one into the
	// instruction that reads it when both are in one block, and a shared
	// one would cost a register across the blocks between.
	if len(in.results) != 1 || len(in.args) > 3 || !pureVerb(in.op.Verb) || in.op.Verb == VConst {
		return vnKey{}, false
	}
	k := vnKey{op: in.op, nargs: len(in.args)}
	if len(in.args) > 0 {
		k.a0 = canon(in.args[0])
	}
	if len(in.args) > 1 {
		k.a1 = canon(in.args[1])
	}
	if len(in.args) > 2 {
		k.a2 = canon(in.args[2])
	}
	if im := in.im; im != nil {
		if len(im.targets) > 0 || len(im.labels) > 0 || im.unwind != nil || im.asm != nil || im.typ != nil {
			return vnKey{}, false
		}
		if im.hasLit {
			if im.lit.kind != ConstInt && im.lit.kind != ConstFloat {
				return vnKey{}, false
			}
			k.kind, k.i, k.f = im.lit.kind, im.lit.i, im.lit.f
		}
		if im.sym != nil {
			k.sym = im.sym
		}
		if im.callee != nil {
			return vnKey{}, false
		}
	}
	return k, true
}

// loadKey is the key of a plain load: its operation and its address.
func loadKey(in *Inst) (vnKey, bool) {
	if len(in.results) != 1 || len(in.args) != 1 {
		return vnKey{}, false
	}
	switch in.op.Verb {
	case VLoad, VSLoad8, VSLoad16, VSLoad32, VULoad8, VULoad16, VULoad32:
	default:
		return vnKey{}, false
	}
	if in.im != nil && in.im.volatile {
		return vnKey{}, false
	}
	k := vnKey{op: in.op, a0: in.args[0], nargs: 1}
	if in.im != nil {
		k.width = in.im.size
	}
	return k, true
}

// harmless reports whether in is known not to write memory, without being
// a value worth keying: nothing here yet beyond the pure verbs and loads
// themselves, so the rest clear what loads are known.
func harmless(in *Inst) bool {
	return pureVerb(in.op.Verb) || in.op.Verb == VAlloc
}

// pureVerb is the set of verbs whose result is a function of their
// operands. A trapping one -- a division, a float-to-int conversion --
// is still that: a second one with the same operands comes after the
// first, which would already have trapped.
func pureVerb(v Verb) bool {
	switch v {
	case VAdd, VSub, VMul, VSMulHi, VUMulHi, VSDiv, VUDiv, VSRem, VURem, VNeg,
		VSAddO, VUAddO, VSSubO, VSMulO, VUMulO,
		VDiv, VFMA, VAbs, VSqrt, VMinimum, VMaximum, VMinNum, VMaxNum, VCopySign,
		VCeil, VFloor, VTrunc, VNearest,
		VNot, VAnd, VOr, VXor, VShl, VSShr, VUShr, VRotL, VRotR,
		VClz, VCtz, VPopcnt, VBswap, VConst,
		VEq, VNe, VSLt, VULt, VSLe, VULe, VLt, VLe, VUno,
		VWrapI64, VSExtI32, VZExtI32, VZExtI1,
		VSCvtI32, VSCvtI64, VUCvtI32, VUCvtI64,
		VSCvtF32, VSCvtF64, VSCvtF80, VSCvtF128, VUCvtF32, VUCvtF64, VUCvtF80, VUCvtF128,
		VSCvtSatF32, VSCvtSatF64, VSCvtSatF80, VSCvtSatF128,
		VUCvtSatF32, VUCvtSatF64, VUCvtSatF80, VUCvtSatF128,
		VFCvtF32, VFCvtF64, VFCvtF80, VFCvtF128,
		VSCvtF16, VSCvtBF16, VUCvtF16, VUCvtBF16,
		VSCvtSatF16, VSCvtSatBF16, VUCvtSatF16, VUCvtSatBF16,
		VFCvtF16, VFCvtBF16,
		VBitcastF32, VBitcastI32, VBitcastF64, VBitcastI64, VBitcastF16, VBitcastBF16,
		VFromI64, VFromPtr, VGetAddr, VSelect:
		return true
	}
	return false
}

// identity is the operand an instruction returns unchanged -- a multiply
// by one, an add, subtract, or, xor or shift of zero -- or nil.
func identity(in *Inst) *Def {
	if len(in.results) != 1 || len(in.args) != 2 {
		return nil
	}
	a, b := in.args[0], in.args[1]
	cb, bConst := constOf(b)
	ca, aConst := constOf(a)
	switch in.op.Verb {
	case VMul:
		if bConst && cb == 1 && in.op.Type != TypeF32 && in.op.Type != TypeF64 {
			return a
		}
		if aConst && ca == 1 && in.op.Type != TypeF32 && in.op.Type != TypeF64 {
			return b
		}
	case VAdd, VOr, VXor:
		if in.op.Type == TypeF32 || in.op.Type == TypeF64 {
			return nil
		}
		if bConst && cb == 0 && (in.op.Type != TypePtr || b.typ != TypePtr) {
			return a
		}
		if aConst && ca == 0 && in.op.Type != TypePtr {
			return b
		}
	case VSub, VShl, VSShr, VUShr:
		if bConst && cb == 0 && in.op.Type != TypeF32 && in.op.Type != TypeF64 {
			return a
		}
	}
	return nil
}

// traps reports whether a pure verb can trap, which the IR defines rather
// than leaving undefined: one whose result nothing reads still has to run.
func traps(v Verb) bool {
	switch v {
	case VSDiv, VUDiv, VSRem, VURem,
		VSCvtF32, VSCvtF64, VSCvtF80, VSCvtF128, VUCvtF32, VUCvtF64, VUCvtF80, VUCvtF128,
		VSCvtF16, VSCvtBF16, VUCvtF16, VUCvtBF16:
		return true
	}
	return false
}
