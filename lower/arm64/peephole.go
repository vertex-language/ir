package arm64

import (
	"github.com/vertex-language/ir"
	"github.com/vertex-language/ir/lower/mir"
	"github.com/vertex-language/ir/lower/regalloc"
)

// The peephole pass: what isel leaves for a second look.
//
// isel selects one instruction at a time and knows nothing of its
// neighbours, which is the right shape for a selector and the wrong one for
// the code: a constant is materialized and then added, a comparison is
// materialized into a register and then compared with zero, an address is
// summed and then loaded through. Every one of these is one instruction on
// this architecture, and the pass here turns the pairs into it.
//
// It runs on MIR, after isel and before register allocation, so that what
// it removes never costs a register, and it works within a block on a value
// with one definition and one reader -- which is nearly every temporary
// isel makes, and the only kind whose producer can be deleted when its
// consumer absorbs it. Nothing here reorders, moves across a block, or
// reasons about flags beyond adjacency: a comparison and the branch that
// reads it are fused only when nothing stands between them.
//
// What it does, in the order it does it:
//
//  1. A constant whose only reader is an add, subtract, compare, shift,
//     multiply or flag-setting add is folded into that instruction's
//     immediate field, where the field can hold it.
//  2. An add whose only reader is a load or store becomes that access's
//     addressing mode: a displacement when the add was by a literal, a
//     register offset when it was of two registers.
//  3. An add followed by the flag-setting add of the same operands -- the
//     sum and its overflow check -- is one ADDS.
//  4. A condition materialized by CSET and at once tested by a branch is
//     the branch on the condition.
//  5. A jump that carries a condition it just decided -- a constant, or a
//     CSET -- into a block that only branches on it is made that branch
//     itself: the shape `&&` and `||` lower to, where each operand's block
//     hands its answer to one shared test.
//
// The emitter finishes the job: a branch to the block emitted next is not
// written.
func peephole(mf *mir.Func) {
	foldConstants(mf)
	foldAddresses(mf)
	mergeOverflowAdds(mf)
	fuseCompareBranches(mf)
	threadJumps(mf)
}

// refCounts is how many instructions define and read each vreg.
type refCounts struct {
	defs, uses []int
}

func countRefs(mf *mir.Func) refCounts {
	n := mf.NumVRegs()
	rc := refCounts{defs: make([]int, n), uses: make([]int, n)}
	for _, b := range mf.Blocks {
		for _, in := range b.Instrs {
			for _, v := range in.Defs {
				rc.defs[v]++
			}
			for _, v := range in.Uses {
				rc.uses[v]++
			}
		}
	}
	return rc
}

// single reports whether v is defined once and read once: the shape a
// producer can be folded into its consumer under.
func (rc refCounts) single(v mir.VReg) bool {
	return rc.defs[v] == 1 && rc.uses[v] == 1
}

// readerOf is the index in instrs, after from, of the one instruction that
// reads v, or -1 when it is not in this block. Only one reader exists when
// this is asked.
func readerOf(instrs []mir.Instr, from int, v mir.VReg) int {
	for j := from + 1; j < len(instrs); j++ {
		for _, u := range instrs[j].Uses {
			if u == v {
				return j
			}
		}
	}
	return -1
}

// compact drops the instructions marked dead.
func compact(instrs []mir.Instr, dead []bool) []mir.Instr {
	out := instrs[:0]
	for i, in := range instrs {
		if !dead[i] {
			out = append(out, in)
		}
	}
	return out
}

// An arithmetic immediate is twelve bits, unsigned; ADD and SUB between
// them cover both signs of it.
const arithImmMax = 4095

func fitsArith(imm int64) bool { return imm >= -arithImmMax && imm <= arithImmMax }

// foldConstants is step 1.
//
// A constant is pure, so it may be folded into some of its readers and
// left standing for the rest; it is deleted only when every reader took
// it. A reader in another block is one that did not.
func foldConstants(mf *mir.Func) {
	rc := countRefs(mf)
	for _, b := range mf.Blocks {
		dead := make([]bool, len(b.Instrs))
		for i := range b.Instrs {
			c, ok := b.Instrs[i].Op.(constOp)
			if !ok || rc.defs[b.Instrs[i].Defs[0]] != 1 {
				continue
			}
			v := b.Instrs[i].Defs[0]
			folded := 0
			for j := i + 1; j < len(b.Instrs); j++ {
				if reads(b.Instrs[j], v) && foldConstantInto(&b.Instrs[j], v, c.imm) {
					folded++
				}
			}
			if folded == rc.uses[v] {
				dead[i] = true
			}
		}
		b.Instrs = compact(b.Instrs, dead)
	}
}

// reads reports whether in has v among its operands.
func reads(in mir.Instr, v mir.VReg) bool {
	for _, u := range in.Uses {
		if u == v {
			return true
		}
	}
	return false
}

// foldConstantInto rewrites in, which reads v, to carry imm as an operand of
// its own, and reports whether it could.
func foldConstantInto(in *mir.Instr, v mir.VReg, imm int64) bool {
	switch op := in.Op.(type) {
	case aluOp:
		a, b := in.Uses[0], in.Uses[1]
		if a == b {
			return false
		}
		switch op.verb {
		case ir.VAdd:
			if !fitsArith(imm) {
				return false
			}
			other := a
			if a == v {
				other = b
			}
			in.Op = addImmOp{imm: imm, w: op.w}
			in.Uses = []mir.VReg{other}
			return true
		case ir.VSub:
			if b != v || a == v || !fitsArith(-imm) {
				return false
			}
			in.Op = addImmOp{imm: -imm, w: op.w}
			in.Uses = []mir.VReg{a}
			return true
		case ir.VMul:
			other := a
			if a == v {
				other = b
			}
			if other == v {
				return false
			}
			if imm == 1 {
				in.Op = movOp{w: op.w}
				in.Uses = []mir.VReg{other}
				in.Copy = true
				return true
			}
			if sh, ok := powerOfTwo(imm, op.w); ok {
				in.Op = shiftImmOp{verb: ir.VShl, amount: sh, w: op.w}
				in.Uses = []mir.VReg{other}
				return true
			}
			return false
		case ir.VShl, ir.VUShr, ir.VSShr:
			bits := int64(64)
			if op.w == w32 {
				bits = 32
			}
			if b != v || a == v || imm < 0 || imm >= bits {
				return false
			}
			in.Op = shiftImmOp{verb: op.verb, amount: uint8(imm), w: op.w}
			in.Uses = []mir.VReg{a}
			return true
		}
		return false

	case flagAluOp:
		a, b := in.Uses[0], in.Uses[1]
		if b != v || a == v {
			return false
		}
		switch op.verb {
		case ir.VAdd:
			if !fitsArith(imm) {
				return false
			}
			in.Op = flagAddImmOp{imm: imm, w: op.w}
		case ir.VSub:
			if !fitsArith(-imm) {
				return false
			}
			in.Op = flagAddImmOp{imm: -imm, w: op.w}
		default:
			return false
		}
		in.Uses = []mir.VReg{a}
		return true

	case cmpOp:
		// Only the right operand: CMP Xn, #imm compares the register
		// with the literal in that order, and a literal on the left
		// would need the condition its readers use turned around.
		a, b := in.Uses[0], in.Uses[1]
		if b != v || a == v || imm < 0 || imm > arithImmMax {
			return false
		}
		in.Op = cmpImmOp{imm: imm, w: op.w}
		in.Uses = []mir.VReg{a}
		return true
	}
	return false
}

// powerOfTwo is the shift a multiply by imm is, where it is one.
func powerOfTwo(imm int64, w width) (uint8, bool) {
	if imm <= 1 || imm&(imm-1) != 0 {
		return 0, false
	}
	sh := uint8(0)
	for imm > 1 {
		imm >>= 1
		sh++
	}
	if w == w32 && sh >= 32 {
		return 0, false
	}
	return sh, true
}

// foldAddresses is step 2.
func foldAddresses(mf *mir.Func) {
	rc := countRefs(mf)
	for _, b := range mf.Blocks {
		dead := make([]bool, len(b.Instrs))
		for i := range b.Instrs {
			in := b.Instrs[i]
			if len(in.Defs) != 1 || !rc.single(in.Defs[0]) {
				continue
			}
			v := in.Defs[0]
			j := readerOf(b.Instrs, i, v)
			if j < 0 {
				continue
			}
			folded := false
			switch op := in.Op.(type) {
			case addImmOp:
				if op.w == w64 {
					folded = foldOffsetInto(&b.Instrs[j], v, in.Uses[0], op.imm)
				}
			case aluOp:
				if op.verb == ir.VAdd && op.w == w64 {
					folded = foldIndexInto(&b.Instrs[j], v, in.Uses[0], in.Uses[1])
				}
			}
			if folded {
				dead[i] = true
			}
		}
		b.Instrs = compact(b.Instrs, dead)
	}
}

// accessBytes is how wide the memory access of a load or store is: the
// scale of its displacement.
func accessBytes(w width) int64 {
	switch w {
	case w32, wf32:
		return 4
	}
	return 8
}

// fitsOffset reports whether off is a displacement the scaled unsigned
// twelve-bit form of a load or store of size bytes can hold.
func fitsOffset(off, size int64) bool {
	return off >= 0 && off%size == 0 && off/size <= arithImmMax
}

// foldOffsetInto makes the access in, whose address is v = base + off, one
// with a displacement, and reports whether it could.
func foldOffsetInto(in *mir.Instr, v, base mir.VReg, off int64) bool {
	switch op := in.Op.(type) {
	case loadOp:
		if in.Uses[0] != v || op.off != 0 || op.indexed || !fitsOffset(off, accessBytes(op.w)) {
			return false
		}
		op.off = off
		in.Op = op
		in.Uses = []mir.VReg{base}
		return true
	case extLoadOp:
		if in.Uses[0] != v || op.off != 0 || op.indexed || !fitsOffset(off, int64(op.from)) {
			return false
		}
		op.off = off
		in.Op = op
		in.Uses = []mir.VReg{base}
		return true
	case storeOp:
		if in.Uses[1] != v || in.Uses[0] == v || op.off != 0 || op.indexed || !fitsOffset(off, accessBytes(op.w)) {
			return false
		}
		op.off = off
		in.Op = op
		in.Uses = []mir.VReg{in.Uses[0], base}
		return true
	case subStoreOp:
		if in.Uses[1] != v || in.Uses[0] == v || op.off != 0 || op.indexed || !fitsOffset(off, int64(op.to)) {
			return false
		}
		op.off = off
		in.Op = op
		in.Uses = []mir.VReg{in.Uses[0], base}
		return true
	}
	return false
}

// foldIndexInto makes the access in, whose address is v = base + index, one
// with a register offset, and reports whether it could. Vector-register
// accesses are left alone: the assembler has no register-offset form for
// them.
func foldIndexInto(in *mir.Instr, v, base, index mir.VReg) bool {
	switch op := in.Op.(type) {
	case loadOp:
		if in.Uses[0] != v || op.off != 0 || op.indexed || op.w.isFloat() {
			return false
		}
		op.indexed = true
		in.Op = op
		in.Uses = []mir.VReg{base, index}
		return true
	case extLoadOp:
		if in.Uses[0] != v || op.off != 0 || op.indexed {
			return false
		}
		op.indexed = true
		in.Op = op
		in.Uses = []mir.VReg{base, index}
		return true
	case storeOp:
		if in.Uses[1] != v || in.Uses[0] == v || op.off != 0 || op.indexed || op.w.isFloat() {
			return false
		}
		op.indexed = true
		in.Op = op
		in.Uses = []mir.VReg{in.Uses[0], base, index}
		return true
	case subStoreOp:
		if in.Uses[1] != v || in.Uses[0] == v || op.off != 0 || op.indexed {
			return false
		}
		op.indexed = true
		in.Op = op
		in.Uses = []mir.VReg{in.Uses[0], base, index}
		return true
	}
	return false
}

// mergeOverflowAdds is step 3: `add d, a, b` at once followed by `adds t,
// a, b` -- the sum, then the same sum for its flags -- with t read by
// nothing, is `adds d, a, b`. The same for a subtraction, and for the
// immediate forms step 1 made.
func mergeOverflowAdds(mf *mir.Func) {
	rc := countRefs(mf)
	for _, b := range mf.Blocks {
		dead := make([]bool, len(b.Instrs))
		for i := 0; i+1 < len(b.Instrs); i++ {
			sum, flags := b.Instrs[i], b.Instrs[i+1]
			if len(flags.Defs) != 1 || rc.uses[flags.Defs[0]] != 0 || len(sum.Defs) != 1 {
				continue
			}
			if !sameUses(sum.Uses, flags.Uses) {
				continue
			}
			switch fop := flags.Op.(type) {
			case flagAluOp:
				sop, ok := sum.Op.(aluOp)
				if !ok || sop.verb != fop.verb || sop.w != fop.w {
					continue
				}
			case flagAddImmOp:
				sop, ok := sum.Op.(addImmOp)
				if !ok || sop.imm != fop.imm || sop.w != fop.w {
					continue
				}
			default:
				continue
			}
			b.Instrs[i+1].Defs = []mir.VReg{sum.Defs[0]}
			dead[i] = true
		}
		b.Instrs = compact(b.Instrs, dead)
	}
}

func sameUses(a, b []mir.VReg) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// fuseCompareBranches is step 4: `cset v, cond; cmp v, #0; b.ne` -- and the
// same through an i1 not -- is `b.cond`, when v is read by nothing else. The
// CSET is at once before the test, so the flags it read are the flags the
// branch will. A test of a value that no CSET in reach made, a block
// parameter say, is CBNZ: the test and the branch in one.
func fuseCompareBranches(mf *mir.Func) {
	rc := countRefs(mf)
	for _, b := range mf.Blocks {
		n := len(b.Instrs)
		if n < 2 {
			continue
		}
		br, ok := b.Instrs[n-1].Op.(bcondOp)
		if !ok || br.cond != condNE {
			continue
		}
		test, ok := b.Instrs[n-2].Op.(cmpImmOp)
		if !ok || test.imm != 0 || test.w != w32 {
			continue
		}
		v := b.Instrs[n-2].Uses[0]
		if !rc.single(v) {
			b.Instrs[n-2] = mir.Instr{Op: cbnzOp{then: br.then, els: br.els}, Uses: []mir.VReg{v}}
			b.Instrs = b.Instrs[:n-1]
			continue
		}
		// The value tested: a CSET, or a CSET through an i1 not, which
		// inverts the condition.
		at := n - 3
		invert := false
		if at >= 0 {
			if _, ok := b.Instrs[at].Op.(i1NotOp); ok && b.Instrs[at].Defs[0] == v && rc.single(b.Instrs[at].Uses[0]) {
				v = b.Instrs[at].Uses[0]
				invert = true
				at--
			}
		}
		var set csetOp
		if at >= 0 {
			set, ok = b.Instrs[at].Op.(csetOp)
		}
		if at < 0 || !ok || b.Instrs[at].Defs[0] != v {
			b.Instrs[n-2] = mir.Instr{Op: cbnzOp{then: br.then, els: br.els}, Uses: []mir.VReg{v}}
			b.Instrs = b.Instrs[:n-1]
			continue
		}
		cond := set.cond
		if invert {
			cond = cond.inverse()
		}
		br.cond = cond
		b.Instrs[at] = mir.Instr{Op: br}
		b.Instrs = b.Instrs[:at+1]
	}
}

// threadJumps is step 5.
//
// A block E that ends `mov p, t; b T`, where T is nothing but `cbnz p, A,
// B` and t was made by a constant or a CSET at once before the copy, knows
// which way T will go, or at least on what flags. E is given T's branch,
// decided where the constant is known, and on the CSET's own condition
// where it is not -- the flags the CSET read are still set, since a copy
// and a jump touch none. p must be read by T alone: a block parameter the
// arms go on to read still has to be written on this edge.
func threadJumps(mf *mir.Func) {
	rc := countRefs(mf)
	for _, e := range mf.Blocks {
		n := len(e.Instrs)
		if n < 3 {
			continue
		}
		jump, ok := e.Instrs[n-1].Op.(bOp)
		if !ok {
			continue
		}
		t := mf.Block(jump.target)
		if t == nil || len(t.Instrs) != 1 {
			continue
		}
		test, ok := t.Instrs[0].Op.(cbnzOp)
		if !ok {
			continue
		}
		p := t.Instrs[0].Uses[0]
		if rc.uses[p] != 1 {
			continue
		}
		copy := e.Instrs[n-2]
		if _, ok := copy.Op.(movOp); !ok || !copy.Copy || copy.Defs[0] != p {
			continue
		}
		src := copy.Uses[0]
		if !rc.single(src) || e.Instrs[n-3].Defs[0] != src {
			continue
		}
		var branch mir.Instr
		switch op := e.Instrs[n-3].Op.(type) {
		case constOp:
			target := test.els
			if op.imm != 0 {
				target = test.then
			}
			branch = mir.Instr{Op: bOp{target: target}}
		case csetOp:
			branch = mir.Instr{Op: bcondOp{cond: op.cond, then: test.then, els: test.els}}
		default:
			continue
		}
		e.Instrs = append(e.Instrs[:n-3], branch)
		e.Succs = replaceSucc(e.Succs, t, mf, branch)
	}
}

// replaceSucc is succs with t taken out and the targets of branch put in.
func replaceSucc(succs []*mir.Block, t *mir.Block, mf *mir.Func, branch mir.Instr) []*mir.Block {
	out := succs[:0]
	for _, s := range succs {
		if s != t {
			out = append(out, s)
		}
	}
	add := func(label string) {
		if b := mf.Block(label); b != nil {
			for _, s := range out {
				if s == b {
					return
				}
			}
			out = append(out, b)
		}
	}
	switch op := branch.Op.(type) {
	case bOp:
		add(op.target)
	case bcondOp:
		add(op.then)
		add(op.els)
	}
	return out
}

// shortcutJumps runs after register allocation, on what the emitter will
// write. It is the part of the job that has to wait for registers: an edge
// block isel made to hold a block parameter's copies is a bare jump once
// the allocator has put both sides of every copy in the same register, and
// only then can a branch into it go straight to where it goes.
//
// A branch to such a block is sent on to the block's own target, and a
// block nothing can reach any more -- the bare jumps, and the tests
// threadJumps took every way in to -- is not emitted. What is reachable is
// worked out from the branches themselves, starting from the entry, every
// block a blockaddr names, and every landing pad: the unwinder enters a pad
// by an address in the LSDA, which no branch here names.
func shortcutJumps(mf *mir.Func, assigned map[mir.VReg]regalloc.PhysReg, roots map[string]bool) {
	if len(mf.Blocks) < 2 {
		return
	}
	// The target of each block that is only a jump, once no-op copies are
	// set aside.
	bare := map[string]string{}
	for i, b := range mf.Blocks {
		if i == 0 || roots[b.Label] {
			continue
		}
		if t, ok := bareJump(b, assigned); ok && t != b.Label {
			bare[b.Label] = t
		}
	}
	resolve := func(l string) string {
		for n := 0; n < len(bare); n++ {
			t, ok := bare[l]
			if !ok {
				break
			}
			l = t
		}
		return l
	}
	if len(bare) > 0 {
		for _, b := range mf.Blocks {
			if len(b.Instrs) == 0 {
				continue
			}
			last := &b.Instrs[len(b.Instrs)-1]
			switch op := last.Op.(type) {
			case bOp:
				op.target = resolve(op.target)
				last.Op = op
			case bcondOp:
				op.then, op.els = resolve(op.then), resolve(op.els)
				if op.then == op.els {
					last.Op = bOp{target: op.then}
				} else {
					last.Op = op
				}
			case cbnzOp:
				op.then, op.els = resolve(op.then), resolve(op.els)
				if op.then == op.els {
					*last = mir.Instr{Op: bOp{target: op.then}}
				} else {
					last.Op = op
				}
			case brTableOp:
				targets := make([]string, len(op.targets))
				for i, t := range op.targets {
					targets[i] = resolve(t)
				}
				op.targets, op.dflt = targets, resolve(op.dflt)
				last.Op = op
			}
		}
	}

	index := make(map[string]int, len(mf.Blocks))
	for i, b := range mf.Blocks {
		index[b.Label] = i
	}
	live := make([]bool, len(mf.Blocks))
	var work []int
	mark := func(i int) {
		if i >= 0 && i < len(live) && !live[i] {
			live[i] = true
			work = append(work, i)
		}
	}
	markLabel := func(l string) {
		if i, ok := index[l]; ok {
			mark(i)
		}
	}
	mark(0)
	for i, b := range mf.Blocks {
		if roots[b.Label] {
			mark(i)
		}
	}
	for len(work) > 0 {
		i := work[len(work)-1]
		work = work[:len(work)-1]
		b := mf.Blocks[i]
		ends := false
		for k, in := range b.Instrs {
			last := k == len(b.Instrs)-1
			switch op := in.Op.(type) {
			case bOp:
				markLabel(op.target)
				ends = last
			case bcondOp:
				markLabel(op.then)
				markLabel(op.els)
				ends = last
			case cbnzOp:
				markLabel(op.then)
				markLabel(op.els)
				ends = last
			case brTableOp:
				for _, t := range op.targets {
					markLabel(t)
				}
				markLabel(op.dflt)
				ends = last
			case blockAddrOp:
				markLabel(op.label)
			case retOp, trapOp, tailOp, tailIndOp, brIndOp:
				ends = last
			}
		}
		// A block that does not end in a jump of its own runs on into
		// the next.
		if !ends {
			mark(i + 1)
		}
	}
	out := mf.Blocks[:0]
	for i, b := range mf.Blocks {
		if live[i] {
			out = append(out, b)
		}
	}
	mf.Blocks = out
}

// bareJump reports the target of b when b is a jump and nothing else the
// emitter would write: every instruction before it a copy the allocator
// made a no-op.
func bareJump(b *mir.Block, assigned map[mir.VReg]regalloc.PhysReg) (string, bool) {
	if len(b.Instrs) == 0 {
		return "", false
	}
	j, ok := b.Instrs[len(b.Instrs)-1].Op.(bOp)
	if !ok {
		return "", false
	}
	for _, in := range b.Instrs[:len(b.Instrs)-1] {
		if _, ok := in.Op.(movOp); !ok || assigned[in.Defs[0]] != assigned[in.Uses[0]] {
			return "", false
		}
	}
	return j.target, true
}

// jumpRoots are the blocks entered other than by a branch this package
// emits: those a blockaddr or an asm goto names, and the landing pads.
func jumpRoots(mf *mir.Func) map[string]bool {
	roots := labeledBlocks(mf)
	for _, b := range mf.Blocks {
		for _, in := range b.Instrs {
			if _, ok := in.Op.(padEntryOp); ok {
				roots[b.Label] = true
			}
		}
	}
	return roots
}

// layoutBlocks orders the blocks for the emitter, which leaves out a branch
// to the block it writes next. isel emits blocks in the order the IR
// declared them, which is source order, and in a loop that puts every arm
// of an `if` a taken branch away from the code that runs after it. Here
// each block is followed, where it can be, by the successor it most likely
// runs next: the target of its jump, or the arm of a conditional branch
// that is not a trap. Blocks that end in a trap -- the calls to a fatal
// error that every bounds and overflow check branches to -- are cold and go
// last, out of the way of the code around them.
//
// The entry stays first. A function with a block whose end this pass does
// not recognize as a jump, a return or a trap is left in the order it has,
// since such a block may run on into the one after it.
func layoutBlocks(mf *mir.Func) {
	n := len(mf.Blocks)
	if n < 3 {
		return
	}
	index := make(map[string]int, n)
	cold := make([]bool, n)
	for i, b := range mf.Blocks {
		index[b.Label] = i
		if len(b.Instrs) == 0 {
			return
		}
		switch b.Instrs[len(b.Instrs)-1].Op.(type) {
		case trapOp:
			cold[i] = true
		case bOp, bcondOp, cbnzOp, brTableOp, retOp, tailOp, tailIndOp, brIndOp:
		default:
			return
		}
	}
	at := func(l string) int {
		if i, ok := index[l]; ok {
			return i
		}
		return -1
	}
	// likely is the successor worth falling into, or -1.
	likely := func(i int) int {
		pick := func(first, second string) int {
			if f := at(first); f >= 0 && !cold[f] {
				return f
			}
			return at(second)
		}
		switch op := mf.Blocks[i].Instrs[len(mf.Blocks[i].Instrs)-1].Op.(type) {
		case bOp:
			return at(op.target)
		case bcondOp:
			return pick(op.els, op.then)
		case cbnzOp:
			return pick(op.els, op.then)
		}
		return -1
	}
	placed := make([]bool, n)
	order := make([]*mir.Block, 0, n)
	chain := func(i int) {
		for i >= 0 && !placed[i] {
			placed[i] = true
			order = append(order, mf.Blocks[i])
			next := likely(i)
			if next < 0 || next == 0 || (cold[next] && !cold[i]) {
				return
			}
			i = next
		}
	}
	for _, wantCold := range []bool{false, true} {
		for i := range mf.Blocks {
			if cold[i] == wantCold {
				chain(i)
			}
		}
	}
	mf.Blocks = order
}
