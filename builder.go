package ir

// A Builder emits instructions into one block. Block embeds one, so the
// namespaces and the bare verbs are reached through the block itself:
//
//	x := blk.I32.Add(a, b)
//	blk.Br(next.To(x))
//
// Type.Verb is namespace value, method name. Bare mnemonics are builder
// methods, matching §17's bare set exactly.
type Builder struct {
	blk *Block

	I1  I1NS
	I32 I32NS
	I64 I64NS
	F32 F32NS
	F64 F64NS
	Ptr PtrNS
}

func (b *Builder) init(blk *Block) {
	b.blk = blk
	b.I1 = I1NS{b}
	b.I32 = I32NS{b}
	b.I64 = I64NS{b}
	b.F32 = F32NS{b}
	b.F64 = F64NS{b}
	b.Ptr = PtrNS{b}
}

// F80 returns the f80 namespace. Availability is a run-time property of the
// layout block — the one thing Go's type system cannot carry — so a module
// whose layout omits f80 records ErrLayout here rather than emulating it.
func (b *Builder) F80() F80NS {
	b.requireExtFloat(TypeF80)
	return F80NS{b}
}

// V128 returns the v128 namespace, under the same rule as F80: the layout
// block says whether the target has a vector register file at all.
func (b *Builder) V128() V128NS {
	if m := b.mod(); m != nil && !m.layout.Vector {
		b.fail(Op{Type: TypeV128}, ErrLayout, "the layout block does not list a vector register file")
	}
	return V128NS{b}
}

// F128 returns the f128 namespace, under the same rule as F80.
func (b *Builder) F128() F128NS {
	b.requireExtFloat(TypeF128)
	return F128NS{b}
}

func (b *Builder) requireExtFloat(t RegType) bool {
	m := b.mod()
	if m == nil {
		return false
	}
	if !m.layout.HasExtFloat(t) {
		b.fail(Op{Type: t}, ErrLayout, "the layout block does not list %s", t)
		return false
	}
	return true
}

// Block returns the block being filled.
func (b *Builder) Block() *Block { return b.blk }

// Func returns the function being filled.
func (b *Builder) Func() *Func {
	if b.blk == nil {
		return nil
	}
	return b.blk.fn
}

// Module returns the module being built.
func (b *Builder) Module() *Module { return b.mod() }

func (b *Builder) mod() *Module {
	if b.blk == nil || b.blk.fn == nil {
		return nil
	}
	return b.blk.fn.m
}

// Name names a register after the fact.
//
// A name is made unique within the function by appending a counter where it
// is already taken. Two registers with one name print as two definitions of
// the same register, which is not a module the text format can read back —
// and a frontend naming a temporary after the construct that produced it has
// no way to know whether an enclosing scope already used the name.
func (b *Builder) Name(v Value, name string) {
	d := defOf(v)
	if d == nil {
		return
	}
	if b.blk == nil || b.blk.fn == nil || name == "" {
		d.name = name
		return
	}
	f := b.blk.fn
	if f.names == nil {
		f.names = map[string]int{}
	}
	if n, taken := f.names[name]; taken {
		for {
			n++
			cand := name + "_" + itoa(n)
			if _, clash := f.names[cand]; !clash {
				f.names[name] = n
				f.names[cand] = 0
				d.name = cand
				return
			}
		}
	}
	f.names[name] = 0
	d.name = name
}

// Last returns the most recently emitted instruction in the block — the
// terminator if there is one — for attaching metadata.
func (b *Builder) Last() *Inst {
	if b.blk == nil {
		return nil
	}
	if b.blk.term != nil {
		return b.blk.term
	}
	if n := len(b.blk.insts); n > 0 {
		return b.blk.insts[n-1]
	}
	return nil
}

func (b *Builder) fail(op Op, kind error, format string, args ...any) {
	m := b.mod()
	if m == nil {
		return
	}
	m.fail(b.blk.fn.name, b.blk.label, op, kind, format, args...)
}

// emit is the one path every instruction takes. It enforces the two faults the
// builder is in a position to catch immediately — a terminated block and a
// poison operand — panics on a value from another function, which is a Go bug
// rather than IR data, and freezes the block's parameter list.
func (b *Builder) emit(op Op, res []RegType, args []*Def, im *imm) *Inst {
	blk := b.blk
	if blk == nil || blk.fn == nil {
		return nil
	}
	m := blk.fn.m
	if m.err != nil {
		return nil
	}
	if blk.term != nil {
		m.fail(blk.fn.name, blk.label, op, ErrTerminated, "block already ends in %s", blk.term.op)
		return nil
	}
	for _, a := range args {
		if a == nil {
			m.fail(blk.fn.name, blk.label, op, ErrPoison, "operand is a zero Value")
			return nil
		}
		if a.fn != blk.fn {
			panic("ir: value %" + a.String() + " from @" + a.fn.name +
				" used in @" + blk.fn.name)
		}
	}
	blk.frozen = true

	in := &Inst{op: op, blk: blk, args: args, im: im}
	if len(res) > 0 {
		in.results = make([]*Def, len(res))
		for i, t := range res {
			in.results[i] = blk.fn.newDef(t, "", in, i)
		}
	}
	if op.IsTerminator() {
		blk.term = in
	} else {
		blk.insts = append(blk.insts, in)
	}
	return in
}

// def1 emits an instruction with exactly one result.
func (b *Builder) def1(op Op, t RegType, args ...*Def) *Def {
	in := b.emit(op, []RegType{t}, args, nil)
	if in == nil {
		return nil
	}
	return in.results[0]
}

// def1i is def1 with non-register operands.
func (b *Builder) def1i(op Op, t RegType, args []*Def, im *imm) *Def {
	in := b.emit(op, []RegType{t}, args, im)
	if in == nil {
		return nil
	}
	return in.results[0]
}

// void emits an instruction with no result.
func (b *Builder) void(op Op, args ...*Def) *Inst {
	return b.emit(op, nil, args, nil)
}

// voidi is void with non-register operands.
func (b *Builder) voidi(op Op, args []*Def, im *imm) *Inst {
	return b.emit(op, nil, args, im)
}
