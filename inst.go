package ir

// A Def is a single SSA definition: one virtual register, assigned exactly
// once. It is either a parameter — of a signature, of a block, or of a pad —
// or one result of one instruction.
type Def struct {
	id      int32
	typ     RegType
	name    string
	fn      *Func
	blk     *Block
	inst    *Inst
	idx     int
	isParam bool
}

func (f *Func) newDef(t RegType, name string, in *Inst, idx int) *Def {
	d := &Def{id: f.nextID, typ: t, name: name, fn: f, inst: in, idx: idx}
	f.nextID++
	if in != nil {
		d.blk = in.blk
	}
	return d
}

// ID is the definition's dense index within its function. It is the printer's
// fallback name, not a stable identity across rewrites.
func (d *Def) ID() int32 { return d.id }

func (d *Def) Type() RegType { return d.typ }

// Name is the name the builder was given, or "" for a temporary. Names you
// supply survive; temporaries get %0…
func (d *Def) Name() string { return d.name }

// SetName names a temporary after the fact.
func (d *Def) SetName(s string) {
	if d != nil {
		d.name = s
	}
}

// Inst returns the defining instruction, or nil for a parameter.
func (d *Def) Inst() *Inst { return d.inst }

// Block returns the defining block: the block holding the instruction, or the
// block the parameter belongs to. It is nil for a signature parameter, whose
// home is the function.
func (d *Def) Block() *Block { return d.blk }

func (d *Def) Func() *Func   { return d.fn }
func (d *Def) IsParam() bool { return d.isParam }

// Index is the result index within the defining instruction, or the position
// within the parameter list.
func (d *Def) Index() int { return d.idx }

func (d *Def) String() string {
	if d == nil {
		return "<poison>"
	}
	if d.name != "" {
		return d.name
	}
	return itoa(int(d.id))
}

// Value wraps the definition in the Go type of its reg-type.
func (d *Def) Value() Value { return Wrap(d) }

// A Value is an SSA value of one of the seven-or-nine reg-types. The zero value
// of each is poison: using one records ErrPoison, since it is either the
// residue of an earlier failure or a register that was never defined.
type Value interface {
	// Def returns the underlying definition, or nil for a zero Value.
	Def() *Def
	// RegType returns the value's static type.
	RegType() RegType
}

// The one Go type per reg-type. There is no I8 or I16, because §2 makes those
// storage-only widths.
type (
	I1   struct{ d *Def }
	I32  struct{ d *Def }
	I64  struct{ d *Def }
	F32  struct{ d *Def }
	F64  struct{ d *Def }
	F80  struct{ d *Def }
	F128 struct{ d *Def }
	V128 struct{ d *Def }
	Ptr  struct{ d *Def }
)

func (v I1) Def() *Def   { return v.d }
func (v I32) Def() *Def  { return v.d }
func (v I64) Def() *Def  { return v.d }
func (v F32) Def() *Def  { return v.d }
func (v F64) Def() *Def  { return v.d }
func (v F80) Def() *Def  { return v.d }
func (v F128) Def() *Def { return v.d }
func (v V128) Def() *Def { return v.d }
func (v Ptr) Def() *Def  { return v.d }

func (v I1) RegType() RegType   { return TypeI1 }
func (v I32) RegType() RegType  { return TypeI32 }
func (v I64) RegType() RegType  { return TypeI64 }
func (v F32) RegType() RegType  { return TypeF32 }
func (v F64) RegType() RegType  { return TypeF64 }
func (v F80) RegType() RegType  { return TypeF80 }
func (v F128) RegType() RegType { return TypeF128 }
func (v V128) RegType() RegType { return TypeV128 }
func (v Ptr) RegType() RegType  { return TypePtr }

func (v I1) IsZero() bool   { return v.d == nil }
func (v I32) IsZero() bool  { return v.d == nil }
func (v I64) IsZero() bool  { return v.d == nil }
func (v F32) IsZero() bool  { return v.d == nil }
func (v F64) IsZero() bool  { return v.d == nil }
func (v F80) IsZero() bool  { return v.d == nil }
func (v F128) IsZero() bool { return v.d == nil }
func (v V128) IsZero() bool { return v.d == nil }
func (v Ptr) IsZero() bool  { return v.d == nil }

// Named names the register. Go cannot see the name of the variable a value is
// assigned to, so a name worth keeping in the text form is stated here.
func (v I1) Named(s string) I1     { v.d.SetName(s); return v }
func (v I32) Named(s string) I32   { v.d.SetName(s); return v }
func (v I64) Named(s string) I64   { v.d.SetName(s); return v }
func (v F32) Named(s string) F32   { v.d.SetName(s); return v }
func (v F64) Named(s string) F64   { v.d.SetName(s); return v }
func (v F80) Named(s string) F80   { v.d.SetName(s); return v }
func (v F128) Named(s string) F128 { v.d.SetName(s); return v }
func (v V128) Named(s string) V128 { v.d.SetName(s); return v }
func (v Ptr) Named(s string) Ptr   { v.d.SetName(s); return v }

// Wrap returns d in the Go type of its reg-type, or nil if d is nil.
func Wrap(d *Def) Value {
	if d == nil {
		return nil
	}
	switch d.typ {
	case TypeI1:
		return I1{d}
	case TypeI32:
		return I32{d}
	case TypeI64:
		return I64{d}
	case TypeF32:
		return F32{d}
	case TypeF64:
		return F64{d}
	case TypeF80:
		return F80{d}
	case TypeF128:
		return F128{d}
	case TypeV128:
		return V128{d}
	case TypePtr:
		return Ptr{d}
	}
	return nil
}

func defOf(v Value) *Def {
	if v == nil {
		return nil
	}
	return v.Def()
}

func defsOf(vs []Value) []*Def {
	if len(vs) == 0 {
		return nil
	}
	out := make([]*Def, len(vs))
	for i, v := range vs {
		out[i] = defOf(v)
	}
	return out
}

// imm holds an instruction's non-register operands. It is nil for the many
// instructions that have none.
type imm struct {
	lit    Const
	hasLit bool

	sym    Symbol // getaddr, tlsaddr, call, invoke
	callee Callee
	typ    *Type // callind's func typedef, alloc's @Type, va_arg_ref's @Type

	targets []BlockTarget // br, brif, br_table, invoke normal edge, asm goto
	labels  []*Block      // brind, asm goto
	unwind  *Block        // invoke, invokeind

	size     uint64
	align    uint64
	hasAlign bool
	zeroed   bool
	volatile bool

	ord    [2]Ordering
	nord   int
	single bool

	asm *Asm
}

// An Inst is one instruction, terminators included.
type Inst struct {
	op      Op
	blk     *Block
	args    []*Def
	results []*Def
	im      *imm
	meta    []Attach
}

func (in *Inst) Op() Op             { return in.op }
func (in *Inst) Block() *Block      { return in.blk }
func (in *Inst) Args() []*Def       { return in.args }
func (in *Inst) NumArgs() int       { return len(in.args) }
func (in *Inst) Results() []*Def    { return in.results }
func (in *Inst) Attached() []Attach { return in.meta }

// Func returns the function the instruction belongs to.
func (in *Inst) Func() *Func {
	if in.blk == nil {
		return nil
	}
	return in.blk.fn
}

// Arg returns the i'th register operand, or nil.
func (in *Inst) Arg(i int) *Def {
	if i < 0 || i >= len(in.args) {
		return nil
	}
	return in.args[i]
}

// Result returns the i'th result, or nil.
func (in *Inst) Result(i int) *Def {
	if i < 0 || i >= len(in.results) {
		return nil
	}
	return in.results[i]
}

// Meta attaches metadata to the instruction.
func (in *Inst) Meta(a ...Attach) *Inst {
	if in != nil {
		in.meta = append(in.meta, a...)
	}
	return in
}

// Lit returns the instruction's literal operand.
func (in *Inst) Lit() (Const, bool) {
	if in.im == nil {
		return Const{}, false
	}
	return in.im.lit, in.im.hasLit
}

// Symbol returns the module-scope symbol the instruction names.
func (in *Inst) Symbol() Symbol {
	if in.im == nil {
		return nil
	}
	if in.im.callee != nil {
		return in.im.callee
	}
	return in.im.sym
}

// Callee returns the function a call or invoke names, or nil for the indirect
// forms.
func (in *Inst) Callee() Callee {
	if in.im == nil {
		return nil
	}
	return in.im.callee
}

// NamedType returns the TypeName operand: callind's func typedef, ptr.alloc's
// @Type, ptr.va_arg_ref's @Type.
func (in *Inst) NamedType() *Type {
	if in.im == nil {
		return nil
	}
	return in.im.typ
}

// Targets returns the branch targets, in mnemonic order. For br_table the
// default edge is last; for invoke the normal edge is the only one.
func (in *Inst) Targets() []BlockTarget {
	if in.im == nil {
		return nil
	}
	return in.im.targets
}

// Labels returns the parameterless label list of brind or asm goto.
func (in *Inst) Labels() []*Block {
	if in.im == nil {
		return nil
	}
	return in.im.labels
}

// Unwind returns the pad block an invoke names.
func (in *Inst) Unwind() *Block {
	if in.im == nil {
		return nil
	}
	return in.im.unwind
}

// Size returns ptr.alloc's stated size.
func (in *Inst) Size() uint64 {
	if in.im == nil {
		return 0
	}
	return in.im.size
}

// Align returns the align attribute and whether one was stated. Absence asserts
// natural alignment.
func (in *Inst) Align() (uint64, bool) {
	if in.im == nil {
		return 0, false
	}
	return in.im.align, in.im.hasAlign
}

// Zeroed reports whether an allocation guarantees all-zero bytes on entry to
// its live range.
func (in *Inst) Zeroed() bool { return in.im != nil && in.im.zeroed }

// Volatile reports whether the access is observable.
func (in *Inst) Volatile() bool { return in.im != nil && in.im.volatile }

// Orderings returns an atomic's orderings: one for load, store and rmw, two for
// compare-and-swap, success first.
func (in *Inst) Orderings() []Ordering {
	if in.im == nil || in.im.nord == 0 {
		return nil
	}
	return in.im.ord[:in.im.nord]
}

// SingleThread reports whether a fence is a compiler barrier — C11's
// atomic_signal_fence, which emits no machine barrier.
func (in *Inst) SingleThread() bool { return in.im != nil && in.im.single }

// Asm returns the inline-assembly payload.
func (in *Inst) Asm() *Asm {
	if in.im == nil {
		return nil
	}
	return in.im.asm
}

// Results is a call's result list. Its typed accessors check the callee's
// signature, since a call's result types are not knowable to Go's type system.
type Results struct {
	m    *Module
	fn   string
	blk  string
	op   Op
	defs []*Def
}

func (r Results) Len() int     { return len(r.defs) }
func (r Results) Defs() []*Def { return r.defs }

// Value returns the i'th result in the Go type of its reg-type.
func (r Results) Value(i int) Value {
	if i < 0 || i >= len(r.defs) {
		return nil
	}
	return Wrap(r.defs[i])
}

func (r Results) at(i int, t RegType) *Def {
	if r.m == nil || r.m.err != nil {
		return nil
	}
	if i < 0 || i >= len(r.defs) {
		r.m.fail(r.fn, r.blk, r.op, ErrArity, "result %d of %d", i, len(r.defs))
		return nil
	}
	if r.defs[i].typ != t {
		r.m.fail(r.fn, r.blk, r.op, ErrType, "result %d is %s, read as %s", i, r.defs[i].typ, t)
		return nil
	}
	return r.defs[i]
}

func (r Results) I1(i int) I1     { return I1{r.at(i, TypeI1)} }
func (r Results) I32(i int) I32   { return I32{r.at(i, TypeI32)} }
func (r Results) I64(i int) I64   { return I64{r.at(i, TypeI64)} }
func (r Results) F32(i int) F32   { return F32{r.at(i, TypeF32)} }
func (r Results) F64(i int) F64   { return F64{r.at(i, TypeF64)} }
func (r Results) F80(i int) F80   { return F80{r.at(i, TypeF80)} }
func (r Results) F128(i int) F128 { return F128{r.at(i, TypeF128)} }
func (r Results) Ptr(i int) Ptr   { return Ptr{r.at(i, TypePtr)} }
