package ir

// A MetaArgKind distinguishes §16's meta-arg forms.
type MetaArgKind uint8

const (
	MetaInt MetaArgKind = iota + 1
	MetaUint
	MetaFloat
	MetaString
	MetaSymbol
	MetaIdent
	MetaRefKind
	MetaNodeKind
)

// A MetaArg is one argument of a metadata node.
type MetaArg struct {
	kind MetaArgKind
	i    int64
	u    uint64
	f    float64
	s    string
	sym  Symbol
	ref  *MetaDecl
	node []MetaArg
}

func MInt(v int64) MetaArg     { return MetaArg{kind: MetaInt, i: v} }
func MUint(v uint64) MetaArg   { return MetaArg{kind: MetaUint, u: v} }
func MFloat(v float64) MetaArg { return MetaArg{kind: MetaFloat, f: v} }
func MStr(s string) MetaArg    { return MetaArg{kind: MetaString, s: s} }
func MSym(s Symbol) MetaArg    { return MetaArg{kind: MetaSymbol, sym: s} }

// MIdent is a bare identifier argument.
func MIdent(s string) MetaArg { return MetaArg{kind: MetaIdent, s: s} }

// MRef references another metadata declaration. Nodes referencing one another is
// what makes a debug-information graph — scope trees, type DAGs, location chains
// — expressible without a further grammar change.
func MRef(d *MetaDecl) MetaArg { return MetaArg{kind: MetaRefKind, ref: d} }

// MNode is an inline node.
func MNode(args ...MetaArg) MetaArg { return MetaArg{kind: MetaNodeKind, node: args} }

func (a MetaArg) Kind() MetaArgKind { return a.kind }
func (a MetaArg) Int() int64        { return a.i }
func (a MetaArg) Uint() uint64      { return a.u }
func (a MetaArg) Float() float64    { return a.f }
func (a MetaArg) String_() string   { return a.s }
func (a MetaArg) Symbol() Symbol    { return a.sym }
func (a MetaArg) Ref() *MetaDecl    { return a.ref }
func (a MetaArg) Node() []MetaArg   { return a.node }

// A MetaDecl is a module-scope metadata node: !name = { ... }.
type MetaDecl struct {
	m    *Module
	name string
	args []MetaArg
}

func (d *MetaDecl) ItemKind() ItemKind { return ItemMeta }
func (d *MetaDecl) Name() string       { return d.name }
func (d *MetaDecl) Args() []MetaArg    { return d.args }

// MetaDecl declares a metadata node.
func (m *Module) MetaDecl(name string, args ...MetaArg) *MetaDecl {
	d := &MetaDecl{m: m, name: name, args: args}
	if !validIdent(name) {
		m.failModule(ErrName, "metadata name %q", name)
		return d
	}
	if _, dup := m.metas[name]; dup {
		m.failModule(ErrDuplicate, "metadata !%s", name)
		return d
	}
	m.metas[name] = d
	m.items = append(m.items, d)
	return d
}

// LookupMeta resolves a metadata name.
func (m *Module) LookupMeta(name string) *MetaDecl { return m.metas[name] }

// An Attach is metadata attached to an instruction, terminator, block header,
// type declaration, global declaration, import, or function definition.
type Attach struct {
	Name string
	Args []MetaArg
}

// Attached builds an attachment: !name followed by its arguments.
func Attached(name string, args ...MetaArg) Attach {
	return Attach{Name: name, Args: args}
}

// AttachRef is the common case: !name !node.
func AttachRef(name string, d *MetaDecl) Attach {
	return Attach{Name: name, Args: []MetaArg{MRef(d)}}
}
