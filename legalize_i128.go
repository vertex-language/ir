package ir

// Legalizing i128 into pairs of i64.
//
// No machine this targets has a 128-bit general register, so no backend is
// asked to find one. This pass runs before instruction selection and takes
// the width out of the module: every i128 value becomes two i64 halves, and
// every i128 instruction becomes the i64 instructions that do the same work
// over them. What reaches a backend afterwards is a module in the reg-types
// it already had.
//
// It is the same arrangement i386 makes for i64 in its own isel, hoisted to
// where one implementation serves every target. The halves are in the
// module's byte order: little-endian keeps the low half first, which is what
// decides the two offsets a load becomes.
//
// The four divisions are the exception, as §E says they are for every width
// a target is short of: a pair cannot divide in line, and each becomes a
// call to the helper a C compiler would call. Their signatures are the ABI's
// own -- an i128 argument is a pair of registers and an i128 result is a
// pair, which is exactly a call of four i64 arguments returning two.

// I128Options settles what the legalizer cannot read off the module.
type I128Options struct {
	// SymbolPrefix is prepended to the division helpers' names.
	//
	// Stated rather than derived, for the reason a backend's libcall
	// prefix is: nothing else in this package mangles a symbol, because
	// the names a module writes are the names it gets. These four names
	// have no author but this pass, and the platforms disagree -- Mach-O
	// prefixes a C symbol with an underscore and ELF does not -- so the
	// caller, which knows the platform, says which.
	SymbolPrefix string
}

// LegalizeI128 rewrites every i128 value in the module into pairs of i64,
// and reports whether it changed anything. A module with no i128 is walked
// and left alone.
//
// Run it before lowering. It is safe to run on a module that has already
// been legalized, and on one that never had an i128 at all.
func (m *Module) LegalizeI128() bool { return m.LegalizeI128Opts(I128Options{}) }

// LegalizeI128Opts is LegalizeI128 with the options stated.
func (m *Module) LegalizeI128Opts(o I128Options) bool {
	if m == nil || m.err != nil {
		return false
	}
	changed := false
	for _, f := range m.Funcs() {
		if f.legalizeI128(o) {
			changed = true
		}
	}
	return changed
}

// LegalizeI128 rewrites one function.
func (f *Func) LegalizeI128() bool { return f.legalizeI128(I128Options{}) }

func (f *Func) legalizeI128(o I128Options) bool {
	if f == nil || f.m == nil || f.m.err != nil || len(f.blocks) == 0 {
		return false
	}
	if !f.usesI128() {
		return false
	}
	p := &i128Pass{f: f, opt: o, half: map[*Def]halves{}, wrap: map[*Def]*Def{}}
	p.run()
	return true
}

// usesI128 reports whether any definition in f is an i128, which is what
// makes the walk worth doing.
func (f *Func) usesI128() bool {
	for _, d := range f.params {
		if d.typ == TypeI128 {
			return true
		}
	}
	for _, b := range f.blocks {
		for _, d := range b.params {
			if d.typ == TypeI128 {
				return true
			}
		}
		for _, in := range b.insts {
			if in.op.Type == TypeI128 {
				return true
			}
			for _, r := range in.results {
				if r.typ == TypeI128 {
					return true
				}
			}
			for _, a := range in.args {
				if a != nil && a.typ == TypeI128 {
					return true
				}
			}
		}
	}
	return false
}

// halves is where an i128 value lives once it is two registers.
type halves struct{ lo, hi *Def }

type i128Pass struct {
	f    *Func
	opt  I128Options
	half map[*Def]halves

	// wrap maps a result this pass replaced outright -- i64.wrap_i128 is
	// its operand's low half and no instruction at all -- to what its uses
	// should read instead.
	wrap map[*Def]*Def

	blk *Block
	out []*Inst
}

func (p *i128Pass) run() {
	// The parameters first: an instruction that reads one needs its halves
	// already recorded.
	p.splitSignature()
	for _, b := range p.f.blocks {
		p.blk = b
		p.out = make([]*Inst, 0, len(b.insts)+8)
		for _, in := range b.insts {
			if !p.expand(in) {
				p.out = append(p.out, in)
			}
		}
		b.insts = p.out
	}
	// Uses of a value this pass replaced outright now read the
	// replacement.
	if len(p.wrap) > 0 {
		p.f.WalkUses(func(u Use) bool {
			if d := u.Def(); d != nil {
				if r := p.cur(d); r != d {
					u.Set(r)
				}
			}
			return true
		})
	}

	// The terminators last: a return or a branch argument is an i128
	// whose halves the block's own instructions defined.
	for _, b := range p.f.blocks {
		p.blk = b
		p.splitEdges(b)
	}
}

// --- emission ---------------------------------------------------------

// emit appends one instruction to the block being rewritten and returns its
// single result.
func (p *i128Pass) emit(t RegType, v Verb, args ...*Def) *Def {
	return p.emitImm(t, v, nil, args...)
}

func (p *i128Pass) emitImm(t RegType, v Verb, im *imm, args ...*Def) *Def {
	in := &Inst{op: Op{t, v}, blk: p.blk, args: args, im: im}
	d := p.f.newDef(t, "", in, 0)
	in.results = []*Def{d}
	p.out = append(p.out, in)
	return d
}

// emitVoid appends an instruction with no result.
func (p *i128Pass) emitVoid(op Op, im *imm, args ...*Def) {
	p.out = append(p.out, &Inst{op: op, blk: p.blk, args: args, im: im})
}

// konst is an i64 literal.
func (p *i128Pass) konst(v int64) *Def {
	return p.emitImm(TypeI64, VConst, &imm{lit: Int(v), hasLit: true})
}

// i64 shorthands.
func (p *i128Pass) add(a, b *Def) *Def  { return p.emit(TypeI64, VAdd, a, b) }
func (p *i128Pass) sub(a, b *Def) *Def  { return p.emit(TypeI64, VSub, a, b) }
func (p *i128Pass) mul(a, b *Def) *Def  { return p.emit(TypeI64, VMul, a, b) }
func (p *i128Pass) or(a, b *Def) *Def   { return p.emit(TypeI64, VOr, a, b) }
func (p *i128Pass) and(a, b *Def) *Def  { return p.emit(TypeI64, VAnd, a, b) }
func (p *i128Pass) shl(a, b *Def) *Def  { return p.emit(TypeI64, VShl, a, b) }
func (p *i128Pass) ushr(a, b *Def) *Def { return p.emit(TypeI64, VUShr, a, b) }
func (p *i128Pass) sshr(a, b *Def) *Def { return p.emit(TypeI64, VSShr, a, b) }

// sel chooses between two i64s on an i1.
func (p *i128Pass) sel(c, a, b *Def) *Def { return p.emit(TypeI64, VSelect, c, a, b) }

// zext1 widens an i1 to an i64 0 or 1, which is how a carry is added.
func (p *i128Pass) zext1(c *Def) *Def { return p.emit(TypeI64, VZExtI1, c) }

// hv is the pair a value lives in.
//
// Every i128 the pass meets should have been split by the time something
// reads it: the parameters before the walk, each instruction as it is
// reached. One that was not is a verb this pass has no expansion for, and
// saying so here is the whole diagnosis -- carrying on would hand the
// backend a 128-bit register it has no way to name.
func (p *i128Pass) hv(d *Def) halves {
	d = p.cur(d)
	if h, ok := p.half[d]; ok {
		return h
	}
	verb := "a parameter"
	if in := d.inst; in != nil {
		verb = in.op.String()
	}
	p.f.m.fail(p.f.name, p.blk.Label(), Op{TypeI128, VConst}, ErrType,
		"legalizing i128: no expansion for %s", verb)
	return halves{lo: p.konst(0), hi: p.konst(0)}
}

// cur is what a value reads as now.
//
// A value this pass replaced outright -- i64.wrap_i128 is its operand's low
// half and no instruction at all -- stands for its replacement, and that
// replacement may itself have been replaced. Reading an operand without
// following the chain leaves a definition behind that nothing defines any
// more, which a backend meets as a register it cannot name.
func (p *i128Pass) cur(d *Def) *Def {
	for n := 0; d != nil && n <= len(p.wrap); n++ {
		r, ok := p.wrap[d]
		if !ok {
			return d
		}
		d = r
	}
	return d
}

// define records the halves of an instruction's i128 result.
func (p *i128Pass) define(in *Inst, lo, hi *Def) bool {
	if len(in.results) == 1 {
		p.half[in.results[0]] = halves{lo: lo, hi: hi}
	}
	return true
}

// --- expansion --------------------------------------------------------

// expand rewrites one instruction, and reports whether it did. An
// instruction it leaves alone is kept as it stands.
func (p *i128Pass) expand(in *Inst) bool {
	// Every operand reads as whatever it stands for now, before anything
	// below looks at one. An instruction this pass keeps is updated in
	// place; one it expands is read through the same resolution.
	for i, a := range in.args {
		if a != nil {
			in.args[i] = p.cur(a)
		}
	}

	// Ops in other namespaces that read an i128.
	switch in.op.Verb {
	case VWrapI128:
		a := p.hv(in.args[0])
		switch in.op.Type {
		case TypeI64:
			// The low half itself; no instruction at all.
			p.wrap[in.results[0]] = a.lo
			return true
		case TypeI32:
			p.wrap[in.results[0]] = p.emit(TypeI32, VWrapI64, a.lo)
			return true
		}
	}
	switch in.op.Verb {
	case VCall, VCallInd, VInvoke, VInvokeInd:
		// A call carries i128 across a boundary, where the ABI has
		// always made it a pair.
		p.splitCall(in)
		return false
	}
	if in.op.Type != TypeI128 {
		return false
	}

	switch in.op.Verb {
	case VConst:
		lo, hi := p.konst(0), p.konst(0)
		if in.im != nil && in.im.hasLit && in.im.lit.kind == ConstInt {
			v := in.im.lit.i
			lo, hi = p.konst(v), p.konst(v>>63)
		}
		return p.define(in, lo, hi)

	case VAdd:
		a, b := p.hv(in.args[0]), p.hv(in.args[1])
		lo := p.add(a.lo, b.lo)
		// The carry out of the low half is the wrap: a sum that came out
		// below one of its addends is one that wrapped.
		carry := p.zext1(p.emit(TypeI1, VULt, lo, a.lo))
		hi := p.add(p.add(a.hi, b.hi), carry)
		return p.define(in, lo, hi)

	case VSub:
		a, b := p.hv(in.args[0]), p.hv(in.args[1])
		lo := p.sub(a.lo, b.lo)
		borrow := p.zext1(p.emit(TypeI1, VULt, a.lo, b.lo))
		hi := p.sub(p.sub(a.hi, b.hi), borrow)
		return p.define(in, lo, hi)

	case VNeg:
		a := p.hv(in.args[0])
		zero := p.konst(0)
		lo := p.sub(zero, a.lo)
		borrow := p.zext1(p.emit(TypeI1, VULt, zero, a.lo))
		hi := p.sub(p.sub(zero, a.hi), borrow)
		return p.define(in, lo, hi)

	case VMul:
		// The low half is the low product; the high half is that
		// product's own high half plus the two cross terms. The fourth
		// term, hi*hi, is entirely above the width.
		a, b := p.hv(in.args[0]), p.hv(in.args[1])
		lo := p.mul(a.lo, b.lo)
		hi := p.add(p.emit(TypeI64, VUMulHi, a.lo, b.lo),
			p.add(p.mul(a.lo, b.hi), p.mul(a.hi, b.lo)))
		return p.define(in, lo, hi)

	case VAnd, VOr, VXor:
		a, b := p.hv(in.args[0]), p.hv(in.args[1])
		return p.define(in,
			p.emit(TypeI64, in.op.Verb, a.lo, b.lo),
			p.emit(TypeI64, in.op.Verb, a.hi, b.hi))

	case VNot:
		a := p.hv(in.args[0])
		return p.define(in, p.emit(TypeI64, VNot, a.lo), p.emit(TypeI64, VNot, a.hi))

	case VShl, VUShr, VSShr:
		lo, hi := p.shift(in.op.Verb, p.hv(in.args[0]), p.hv(in.args[1]))
		return p.define(in, lo, hi)

	case VSelect:
		c := in.args[0]
		a, b := p.hv(in.args[1]), p.hv(in.args[2])
		return p.define(in, p.sel(c, a.lo, b.lo), p.sel(c, a.hi, b.hi))

	case VEq, VNe, VSLt, VULt, VSLe, VULe:
		p.wrap[in.results[0]] = p.compare(in.op.Verb, p.hv(in.args[0]), p.hv(in.args[1]))
		return true

	case VSExtI64:
		a := in.args[0]
		return p.define(in, a, p.sshr(a, p.konst(63)))
	case VZExtI64:
		return p.define(in, in.args[0], p.konst(0))
	case VSExtI32:
		lo := p.emit(TypeI64, VSExtI32, in.args[0])
		return p.define(in, lo, p.sshr(lo, p.konst(63)))
	case VZExtI32:
		return p.define(in, p.emit(TypeI64, VZExtI32, in.args[0]), p.konst(0))
	case VZExtI1:
		return p.define(in, p.emit(TypeI64, VZExtI1, in.args[0]), p.konst(0))

	case VLoad:
		lo, hi := p.loadHalves(in)
		return p.define(in, lo, hi)

	case VStore:
		p.storeHalves(in)
		return true

	case VSDiv, VUDiv, VSRem, VURem:
		lo, hi := p.divide(in)
		return p.define(in, lo, hi)
	}

	// A verb with no expansion is left as it stands; isel will refuse it
	// by name, which says more than anything this pass could.
	return false
}

// shift is shl, ushr or sshr over a pair.
//
// The amount is taken modulo 128, which splits the work in two: an amount
// below 64 moves bits between the halves, and one at or above 64 moves a
// whole half across and fills behind it. Both are computed and the right one
// chosen, rather than branched on, so the expansion stays one basic block.
//
// The bits that cross are the awkward part. Shifting a half by 64-n is a
// shift by 64 when n is zero, which §A5 takes modulo 64 and so leaves the
// half untouched instead of clearing it. Shifting by 63-n and then by one
// more is the same count for every n that matters and never reaches 64.
func (p *i128Pass) shift(v Verb, a, amt halves) (lo, hi *Def) {
	n := p.and(amt.lo, p.konst(127))
	small := p.emit(TypeI1, VULt, n, p.konst(64))
	far := p.sub(n, p.konst(64))  // the amount once a whole half has moved
	back := p.sub(p.konst(63), n) // 63-n, for the bits that cross
	zero := p.konst(0)

	switch v {
	case VShl:
		crossed := p.shl(p.ushr(p.ushr(a.lo, back), p.konst(1)), p.konst(0))
		smallLo := p.shl(a.lo, n)
		smallHi := p.or(p.shl(a.hi, n), crossed)
		return p.sel(small, smallLo, zero), p.sel(small, smallHi, p.shl(a.lo, far))

	case VUShr:
		crossed := p.shl(p.shl(a.hi, back), p.konst(1))
		smallLo := p.or(p.ushr(a.lo, n), crossed)
		smallHi := p.ushr(a.hi, n)
		return p.sel(small, smallLo, p.ushr(a.hi, far)), p.sel(small, smallHi, zero)

	default: // VSShr
		crossed := p.shl(p.shl(a.hi, back), p.konst(1))
		smallLo := p.or(p.ushr(a.lo, n), crossed)
		smallHi := p.sshr(a.hi, n)
		// Past 64 the whole value is the high half shifted, and what
		// fills behind it is the sign, which is that half shifted by 63.
		sign := p.sshr(a.hi, p.konst(63))
		return p.sel(small, smallLo, p.sshr(a.hi, far)), p.sel(small, smallHi, sign)
	}
}

// compare is one of §B's six predicates over a pair.
//
// The high halves decide unless they are equal, and only then does the low
// pair matter -- and it matters unsigned whatever the predicate is, because
// the sign lives in the high half alone.
func (p *i128Pass) compare(v Verb, a, b halves) *Def {
	hiEq := p.emit(TypeI1, VEq, a.hi, b.hi)
	switch v {
	case VEq:
		return p.emit(TypeI1, VAnd, hiEq, p.emit(TypeI1, VEq, a.lo, b.lo))
	case VNe:
		return p.emit(TypeI1, VNot,
			p.emit(TypeI1, VAnd, hiEq, p.emit(TypeI1, VEq, a.lo, b.lo)))
	}
	hiLt, loCmp := VULt, VULt
	switch v {
	case VSLt:
		hiLt = VSLt
	case VSLe:
		hiLt, loCmp = VSLt, VULe
	case VULe:
		loCmp = VULe
	}
	return p.emit(TypeI1, VOr,
		p.emit(TypeI1, hiLt, a.hi, b.hi),
		p.emit(TypeI1, VAnd, hiEq, p.emit(TypeI1, loCmp, a.lo, b.lo)))
}

// halfPtr is the address of one half: the base for the first, eight bytes on
// for the second, in the module's byte order.
func (p *i128Pass) halfPtr(base *Def, high bool) *Def {
	first := !high
	if p.f.m.layout.Endian == BigEndian {
		first = high
	}
	if first {
		return base
	}
	return p.emit(TypePtr, VAdd, base, p.konst(8))
}

// halfAttrs is a half's memory attributes: the whole access's, with the
// alignment only the first half can claim.
func (p *i128Pass) halfAttrs(src *imm, high bool) *imm {
	im := &imm{}
	if src != nil {
		*im = *src
		im.targets, im.labels = nil, nil
	}
	first := !high
	if p.f.m.layout.Endian == BigEndian {
		first = high
	}
	if !first {
		// Eight bytes into a 16-aligned object is 8-aligned, and no more.
		if im.hasAlign && im.align > 8 {
			im.align = 8
		}
	}
	return im
}

// loadHalves reads a pair from memory as two i64 loads.
func (p *i128Pass) loadHalves(in *Inst) (lo, hi *Def) {
	base := in.args[0]
	lo = p.emitImm(TypeI64, VLoad, p.halfAttrs(in.im, false), p.halfPtr(base, false))
	hi = p.emitImm(TypeI64, VLoad, p.halfAttrs(in.im, true), p.halfPtr(base, true))
	return lo, hi
}

// storeHalves writes a pair to memory as two i64 stores.
func (p *i128Pass) storeHalves(in *Inst) {
	v := p.hv(in.args[0])
	base := in.args[1]
	p.emitVoid(Op{TypeI64, VStore}, p.halfAttrs(in.im, false), v.lo, p.halfPtr(base, false))
	p.emitVoid(Op{TypeI64, VStore}, p.halfAttrs(in.im, true), v.hi, p.halfPtr(base, true))
}

// --- division ---------------------------------------------------------

// divHelper is the compiler-rt name each division verb becomes.
var divHelper = map[Verb]string{
	VSDiv: "__divti3",
	VUDiv: "__udivti3",
	VSRem: "__modti3",
	VURem: "__umodti3",
}

// divide calls the helper for one of §A's four divisions.
//
// The helper's signature is the ABI's own reading of `__int128 f(__int128,
// __int128)`: each argument is a pair of registers and the result is a pair,
// which is a call of four i64 arguments returning two. The order within a
// pair is the register order, which follows the module's endianness the same
// way the halves in memory do.
func (p *i128Pass) divide(in *Inst) (lo, hi *Def) {
	a, b := p.hv(in.args[0]), p.hv(in.args[1])
	fn := p.helper(divHelper[in.op.Verb])
	if fn == nil {
		return a.lo, a.hi
	}
	args := []*Def{a.lo, a.hi, b.lo, b.hi}
	if p.f.m.layout.Endian == BigEndian {
		args = []*Def{a.hi, a.lo, b.hi, b.lo}
	}
	call := &Inst{op: Op{TypeNone, VCall}, blk: p.blk, args: args, im: &imm{callee: fn, sym: fn}}
	r0 := p.f.newDef(TypeI64, "", call, 0)
	r1 := p.f.newDef(TypeI64, "", call, 1)
	call.results = []*Def{r0, r1}
	p.out = append(p.out, call)
	if p.f.m.layout.Endian == BigEndian {
		return r1, r0
	}
	return r0, r1
}

// helper imports a runtime division helper, once per module.
func (p *i128Pass) helper(name string) Callee {
	if name == "" {
		return nil
	}
	m := p.f.m
	name = p.opt.SymbolPrefix + name
	if s := m.Lookup(name); s != nil {
		if c, ok := s.(Callee); ok {
			return c
		}
		return nil
	}
	sig := NewSig().
		Param(TypeI64).Param(TypeI64).Param(TypeI64).Param(TypeI64).
		Ret(TypeI64).Ret(TypeI64)
	return m.ImportFunc(name, sig)
}

// --- parameters, results and edges ------------------------------------

// splitSignature rewrites every i128 parameter and result into the pair of
// i64s it occupies.
//
// This is the ABI's own reading of the type and not an invention: AAPCS64
// and the System V psABI both carry a 128-bit integer in two registers, so a
// parameter that was one i128 is two i64s in the same place, and a result
// that was one is two. A call's arguments are split the same way, and its
// two results reassembled into the pair its uses read.
func (p *i128Pass) splitSignature() {
	f := p.f
	if n := p.splitDefs(&f.params, func(name string, i int) *Def {
		d := f.newDef(TypeI64, name, nil, i)
		d.isParam = true
		return d
	}); n {
		f.sig.params = splitParams(f.sig.params)
		for i, d := range f.params {
			d.idx = i
		}
	}
	f.sig.rets = splitRets(f.sig.rets)

	for _, b := range f.blocks {
		if b.isEntry || b.isPad {
			continue
		}
		if p.splitDefs(&b.params, func(name string, i int) *Def {
			d := f.newDef(TypeI64, name, nil, i)
			d.isParam = true
			d.blk = b
			return d
		}) {
			for i, d := range b.params {
				d.idx = i
			}
		}
	}
}

// splitDefs replaces each i128 in a parameter list with two i64s, recording
// the halves, and reports whether it changed the list.
func (p *i128Pass) splitDefs(list *[]*Def, mk func(string, int) *Def) bool {
	has := false
	for _, d := range *list {
		if d != nil && d.typ == TypeI128 {
			has = true
			break
		}
	}
	if !has {
		return false
	}
	out := make([]*Def, 0, len(*list)+1)
	for _, d := range *list {
		if d == nil || d.typ != TypeI128 {
			out = append(out, d)
			continue
		}
		lo := mk(halfName(d.name, "lo"), len(out))
		hi := mk(halfName(d.name, "hi"), len(out)+1)
		p.half[d] = halves{lo: lo, hi: hi}
		out = append(out, lo, hi)
	}
	*list = out
	return true
}

func halfName(base, suffix string) string {
	if base == "" {
		return ""
	}
	return base + "." + suffix
}

func splitParams(in []Param) []Param {
	out := make([]Param, 0, len(in))
	for _, p := range in {
		if p.Type != TypeI128 {
			out = append(out, p)
			continue
		}
		out = append(out,
			Param{Name: halfName(p.Name, "lo"), Type: TypeI64, Attrs: p.Attrs},
			Param{Name: halfName(p.Name, "hi"), Type: TypeI64})
	}
	return out
}

func splitRets(in []RetItem) []RetItem {
	out := make([]RetItem, 0, len(in))
	for _, r := range in {
		if r.Type != TypeI128 {
			out = append(out, r)
			continue
		}
		out = append(out, RetItem{Type: TypeI64, Attrs: r.Attrs}, RetItem{Type: TypeI64})
	}
	return out
}

// widen is the pair an i128 operand travels as, in register order.
func (p *i128Pass) widen(d *Def) []*Def {
	h := p.hv(d)
	if p.f.m.layout.Endian == BigEndian {
		return []*Def{h.hi, h.lo}
	}
	return []*Def{h.lo, h.hi}
}

// splitArgs expands every i128 in an argument list into its pair.
func (p *i128Pass) splitArgs(args []*Def) []*Def {
	need := false
	for _, a := range args {
		if a != nil && a.typ == TypeI128 {
			need = true
			break
		}
	}
	if !need {
		return args
	}
	out := make([]*Def, 0, len(args)+1)
	for _, a := range args {
		if a == nil || a.typ != TypeI128 {
			out = append(out, a)
			continue
		}
		out = append(out, p.widen(a)...)
	}
	return out
}

// splitCall rewrites a call's i128 arguments and results.
func (p *i128Pass) splitCall(in *Inst) bool {
	touched := false
	if args := p.splitArgs(in.args); len(args) != len(in.args) {
		in.args = args
		touched = true
	}
	for _, r := range in.results {
		if r.typ == TypeI128 {
			touched = true
			break
		}
	}
	if !touched {
		return false
	}
	// Results: an i128 becomes two i64s in its place, and the pair is what
	// its uses read.
	out := make([]*Def, 0, len(in.results)+1)
	for _, r := range in.results {
		if r.typ != TypeI128 {
			out = append(out, r)
			continue
		}
		lo := p.f.newDef(TypeI64, halfName(r.name, "lo"), in, len(out))
		hi := p.f.newDef(TypeI64, halfName(r.name, "hi"), in, len(out)+1)
		if p.f.m.layout.Endian == BigEndian {
			lo, hi = hi, lo
		}
		p.half[r] = halves{lo: lo, hi: hi}
		out = append(out, in.results[0:0]...)
		if p.f.m.layout.Endian == BigEndian {
			out = append(out, hi, lo)
		} else {
			out = append(out, lo, hi)
		}
	}
	in.results = out
	for i, d := range in.results {
		d.idx = i
	}
	return true
}

// splitEdges expands the i128 arguments a terminator carries: a return's
// operands, and the argument list on every block target.
func (p *i128Pass) splitEdges(b *Block) {
	t := b.term
	if t == nil {
		return
	}
	t.args = p.splitArgs(t.args)
	if t.im == nil {
		return
	}
	for i := range t.im.targets {
		tg := &t.im.targets[i]
		if len(tg.args) == 0 {
			continue
		}
		if args := p.splitArgs(tg.args); len(args) != len(tg.args) {
			tg.args = args
			tg.bare = len(args) == 0
		}
	}
}
