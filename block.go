// block.go
package ir

// A PadKind distinguishes §G3's three pad clauses.
type PadKind uint8

const (
	PadCleanup PadKind = iota + 1
	PadCatch
	PadFilter
)

// A PadClause is a pad block's handling rule. cleanup is sufficient for all of
// C; catch and filter cost C nothing and leave C++ reachable.
type PadClause struct {
	kind PadKind
	ti   Symbol
	set  []Symbol
}

// Cleanup runs destructors and re-raises.
var Cleanup = PadClause{kind: PadCleanup}

// Catch handles the type described by a type-info global.
func Catch(ti Symbol) PadClause { return PadClause{kind: PadCatch, ti: ti} }

// Filter handles anything not in the listed set.
func Filter(tis ...Symbol) PadClause { return PadClause{kind: PadFilter, set: tis} }

func (p PadClause) Kind() PadKind    { return p.kind }
func (p PadClause) TypeInfo() Symbol { return p.ti }
func (p PadClause) Set() []Symbol    { return p.set }

// A Block is a basic block. Blocks are declared, parameterized, then filled: a
// block freezes its parameter list at its first instruction. There are no phi
// nodes to construct and no predecessor-indexed operand lists to keep in sync,
// which is also why predecessors are computed by walk.go rather than stored.
type Block struct {
	Builder

	fn      *Func
	label   string
	index   int
	isEntry bool

	params []*Def
	insts  []*Inst
	term   *Inst
	frozen bool

	isPad   bool
	clauses []PadClause

	meta []Attach
}

func (b *Block) Label() string        { return b.label }
func (b *Block) Func() *Func          { return b.fn }
func (b *Block) Index() int           { return b.index }
func (b *Block) IsEntry() bool        { return b.isEntry }
func (b *Block) IsPad() bool          { return b.isPad }
func (b *Block) Clauses() []PadClause { return b.clauses }
func (b *Block) Params() []*Def       { return b.params }
func (b *Block) Attached() []Attach   { return b.meta }

// Insts returns the block's non-terminator instructions.
func (b *Block) Insts() []*Inst { return b.insts }

// Term returns the block's terminator, or nil if it has none yet. A block with
// no terminator is a §19.2 failure, which is the verifier's to report.
func (b *Block) Term() *Inst { return b.term }

// Meta attaches metadata to the block header.
func (b *Block) Meta(a ...Attach) *Block { b.meta = append(b.meta, a...); return b }

// Entry returns the function's entry block, creating it on first call and
// freezing the signature. The entry block takes no parameters: its inputs are
// the signature's parameter registers.
func (f *Func) Entry() *Block {
	if f.hasAsmBody {
		f.m.fail(f.name, "", Op{}, ErrPlacement, "a block in a function whose body is assembly")
		return nil
	}
	if f.entry == nil {
		f.frozen = true
		f.entry = f.newBlock("entry")
		if f.entry != nil {
			f.entry.isEntry = true
			f.entry.frozen = true
		}
	}
	return f.entry
}

// Block declares an ordinary block. The entry block is created first if it does
// not exist yet, so that it is always blocks[0].
func (f *Func) Block(label string) *Block {
	f.Entry()
	return f.newBlock(label)
}

// Pad declares a pad block. Its two parameters are supplied by the personality
// routine — the exception object pointer and the personality's selector — not
// by any branch, which is why an unwind edge carries no argument list.
func (f *Func) Pad(label string, clauses ...PadClause) *Block {
	f.Entry()
	b := f.newBlock(label)
	if b == nil {
		return nil
	}
	if len(clauses) == 0 {
		f.m.fail(f.name, label, Op{}, ErrPlacement, "pad block with no clause")
		return b
	}
	b.isPad = true
	b.clauses = clauses
	b.params = []*Def{
		f.newDef(TypePtr, "exn", nil, 0),
		f.newDef(TypeI32, "sel", nil, 1),
	}
	b.params[0].isParam = true
	b.params[0].blk = b
	b.params[1].isParam = true
	b.params[1].blk = b
	b.frozen = true
	return b
}

// Exn returns a pad block's exception object parameter.
func (b *Block) Exn() Ptr {
	if b == nil || !b.isPad {
		return Ptr{}
	}
	return Ptr{b.params[0]}
}

// Sel returns a pad block's personality selector parameter.
func (b *Block) Sel() I32 {
	if b == nil || !b.isPad {
		return I32{}
	}
	return I32{b.params[1]}
}

// newBlock allocates and registers a block, or — once the module has already
// failed — hands back an inert stand-in so a frontend already mid-walk has
// something to keep calling into.
//
// Even the stand-in gets Builder.init: every builder method funnels through
// Builder.mod, which dereferences blk.fn.m before anything else runs, and a
// zero-value Builder's namespaces (I32NS{}, PtrNS{}, ...) hold a nil *Builder.
// A method call on that nil receiver panics before it ever reaches the
// m.err != nil checks that are supposed to make post-failure calls no-ops.
// Initializing the stand-in costs nothing — it just makes it point back at
// the same already-failed module — and is what actually makes "every call
// after a failure is a no-op" true instead of aspirational.
func (f *Func) newBlock(label string) *Block {
	if f.m.err != nil && f.entry != nil {
		b := &Block{fn: f, label: label}
		b.Builder.init(b)
		return b
	}
	if !validIdent(label) {
		f.m.fail(f.name, "", Op{}, ErrName, "block label %q", label)
	} else if _, dup := f.labels[label]; dup {
		f.m.fail(f.name, "", Op{}, ErrDuplicate, "block @%s", label)
	}
	b := &Block{fn: f, label: label, index: len(f.blocks)}
	b.Builder.init(b)
	f.labels[label] = b
	f.blocks = append(f.blocks, b)
	return b
}

// blockParam appends a parameter to the block.
func (b *Block) blockParam(t RegType, name string) *Def {
	f := b.fn
	if f.m.err != nil {
		return nil
	}
	if b.isEntry {
		f.m.fail(f.name, b.label, Op{}, ErrPlacement, "the entry block takes no parameters")
		return nil
	}
	if b.isPad {
		f.m.fail(f.name, b.label, Op{}, ErrPlacement, "a pad block's parameters are the personality's")
		return nil
	}
	if b.frozen {
		f.m.fail(f.name, b.label, Op{}, ErrFrozen, "parameter %%%s after the block's first instruction", name)
		return nil
	}
	if !f.m.layout.Admits(t) {
		f.m.fail(f.name, b.label, Op{}, ErrLayout, "block parameter of type %s", t)
		return nil
	}
	d := f.newDef(t, name, nil, len(b.params))
	d.isParam = true
	d.blk = b
	b.params = append(b.params, d)
	return d
}

// Param declares a block parameter of a type decided at run time, in the Go
// type of that reg-type.
//
// The typed spellings below are what a hand-written builder wants; this is
// what a frontend wants, which knows a parameter's type from the source it is
// translating and not from the line it is on. An asm goto's outputs are the
// case that needs it: §14 makes them this block's trailing parameters, and
// their types come from the C declarations the operands named.
func (b *Block) Param(t RegType, name string) Value { return Wrap(b.blockParam(t, name)) }

func (b *Block) ParamI1(name string) I1   { return I1{b.blockParam(TypeI1, name)} }
func (b *Block) ParamI32(name string) I32 { return I32{b.blockParam(TypeI32, name)} }
func (b *Block) ParamI64(name string) I64 { return I64{b.blockParam(TypeI64, name)} }
func (b *Block) ParamF32(name string) F32 { return F32{b.blockParam(TypeF32, name)} }
func (b *Block) ParamF64(name string) F64 { return F64{b.blockParam(TypeF64, name)} }
func (b *Block) ParamF80(name string) F80 { return F80{b.blockParam(TypeF80, name)} }
func (b *Block) ParamV128(name string) V128 {
	return V128{b.blockParam(TypeV128, name)}
}

func (b *Block) ParamF128(name string) F128 {
	return F128{b.blockParam(TypeF128, name)}
}
func (b *Block) ParamPtr(name string) Ptr { return Ptr{b.blockParam(TypePtr, name)} }

// A BlockTarget is §7's target: a label and the argument list a branch supplies
// to its parameters. To with no arguments emits the bare @label form.
type BlockTarget struct {
	blk  *Block
	args []*Def
	bare bool
}

// To names this block as a branch target, with the arguments its parameters
// take.
func (b *Block) To(args ...Value) BlockTarget {
	t := BlockTarget{blk: b, bare: len(args) == 0}
	if len(args) > 0 {
		t.args = make([]*Def, len(args))
		for i, a := range args {
			t.args[i] = defOf(a)
		}
	}
	return t
}

func (t BlockTarget) Block() *Block { return t.blk }
func (t BlockTarget) Args() []*Def  { return t.args }

// Bare reports whether the target was written without an argument list.
func (t BlockTarget) Bare() bool { return t.bare }

// check validates a branch edge. Arity and type against the target's parameter
// list are deferred to Module.Err, because a forward branch may name a block
// whose parameters are not declared yet.
func (b *Builder) checkTarget(op Op, t BlockTarget) bool {
	return b.checkTargetPlus(op, t, 0)
}

// checkTargetPlus is checkTarget for an edge whose target takes trailing
// parameters from somewhere other than the branch. extra is how many.
//
// An asm goto's outputs are the one case: they arrive on the fallthrough
// edge, after the arguments the edge itself carries, because the terminator
// defines no register of its own. Invoke's results are the same shape and
// reach it a different way, by not coming through here at all.
func (b *Builder) checkTargetPlus(op Op, t BlockTarget, extra int) bool {
	blk := b.blk
	if t.blk == nil {
		blk.fn.m.fail(blk.fn.name, blk.label, op, ErrPoison, "branch to the zero BlockTarget")
		return false
	}
	if t.blk.fn != blk.fn {
		panic("ir: branch to a block from another function: @" + t.blk.fn.name +
			" @" + t.blk.label + " used in @" + blk.fn.name)
	}
	for _, a := range t.args {
		if a == nil {
			blk.fn.m.fail(blk.fn.name, blk.label, op, ErrPoison, "branch argument is a zero Value")
			return false
		}
		if a.fn != blk.fn {
			panic("ir: value %" + a.String() + " from another function")
		}
	}
	m := blk.fn.m
	fname, lname, tname := blk.fn.name, blk.label, t.blk.label
	m.deferCheck(func() *Error {
		if len(t.args)+extra != len(t.blk.params) {
			d := "@" + tname + " takes " + itoa(len(t.blk.params)) +
				" parameters, branch supplies " + itoa(len(t.args))
			if extra > 0 {
				d += " and the instruction " + itoa(extra) + " more"
			}
			return &Error{Func: fname, Block: lname, Op: op, Err: ErrArity, Detail: d}
		}
		for i, a := range t.args {
			if a.typ != t.blk.params[i].typ {
				return &Error{Func: fname, Block: lname, Op: op, Err: ErrType,
					Detail: "@" + tname + " parameter " + itoa(i) + " is " +
						t.blk.params[i].typ.String() + ", branch supplies " + a.typ.String()}
			}
		}
		return nil
	})
	return true
}

func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	neg := n < 0
	if neg {
		n = -n
	}
	var buf [20]byte
	i := len(buf)
	for n > 0 {
		i--
		buf[i] = byte('0' + n%10)
		n /= 10
	}
	if neg {
		i--
		buf[i] = '-'
	}
	return string(buf[i:])
}
