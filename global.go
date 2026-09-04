// global.go
package ir

// A Domain is a global's storage domain (§5).
type Domain uint8

const (
	RO Domain = iota + 1
	RW
	TLS
)

func (d Domain) String() string {
	switch d {
	case RO:
		return "ro"
	case RW:
		return "rw"
	case TLS:
		return "tls"
	}
	return "<invalid domain>"
}

// A Binding is §3's binding modifier.
type Binding uint8

const (
	NoBinding Binding = iota
	Weak
	Common
)

func (b Binding) String() string {
	switch b {
	case Weak:
		return "weak"
	case Common:
		return "common"
	}
	return ""
}

// A TLSModel is §5's tls-model.
type TLSModel uint8

const (
	NoTLSModel TLSModel = iota
	GlobalDynamic
	LocalDynamic
	InitialExec
	LocalExec
)

func (t TLSModel) String() string {
	switch t {
	case GlobalDynamic:
		return "global-dynamic"
	case LocalDynamic:
		return "local-dynamic"
	case InitialExec:
		return "initial-exec"
	case LocalExec:
		return "local-exec"
	}
	return ""
}

// An InitKind distinguishes the initializer forms of §5.
type InitKind uint8

const (
	InitLiteral InitKind = iota + 1
	InitString
	InitZeroed
	InitRelocKind
	InitList
	InitFields
)

// A Reloc is a relocation, not an expression: a symbol, optionally minus a
// second symbol, plus one assemble-time-known displacement. Admitting one
// operator would mean owning a constant evaluator.
type Reloc struct {
	Sym       Symbol
	Minus     Symbol
	Addend    Const
	HasAddend bool
}

// A FieldVal is one named element of a struct initializer.
type FieldVal struct {
	Name string
	Init Init
}

// An Init is a global initializer. Its structure must match the declared type
// exactly; §19.10 checks that, which is why §5's ftype is required.
type Init struct {
	kind   InitKind
	c      Const
	s      string
	elems  []Init
	fields []FieldVal
	reloc  Reloc
}

// ZeroInit is the zeroed initializer. It takes its width from the declared
// type, which is why an initializer that states none cannot be inferred from.
var ZeroInit = Init{kind: InitZeroed}

// Lit is a literal initializer, including the symbolic constants.
func Lit(c Const) Init { return Init{kind: InitLiteral, c: c} }

// Str is a string initializer.
func Str(s string) Init { return Init{kind: InitString, s: s} }

// List is a positional aggregate initializer.
func List(items ...Init) Init { return Init{kind: InitList, elems: items} }

// Fields is a named-field aggregate initializer.
func Fields(f ...FieldVal) Init { return Init{kind: InitFields, fields: f} }

// Val pairs a field name with its initializer.
func Val(name string, i Init) FieldVal { return FieldVal{Name: name, Init: i} }

// RelocInit is the address of a symbol.
func RelocInit(s Symbol) Init { return Init{kind: InitRelocKind, reloc: Reloc{Sym: s}} }

// Minus subtracts a second symbol from a reloc initializer.
func (i Init) Minus(s Symbol) Init {
	if i.kind == InitRelocKind {
		i.reloc.Minus = s
	}
	return i
}

// Plus adds a displacement to a reloc initializer. &arr[3] is
// RelocInit(arr).Plus(OffsetOf(arrTy, IndexPath(3))); no multiplication is
// required or provided.
func (i Init) Plus(c Const) Init {
	if i.kind == InitRelocKind {
		i.reloc.Addend = c
		i.reloc.HasAddend = true
	}
	return i
}

func (i Init) Kind() InitKind        { return i.kind }
func (i Init) Const() Const          { return i.c }
func (i Init) String_() string       { return i.s }
func (i Init) Elems() []Init         { return i.elems }
func (i Init) FieldVals() []FieldVal { return i.fields }
func (i Init) Reloc() Reloc          { return i.reloc }
func (i Init) IsZero() bool          { return i.kind == 0 }

// A Global is a module-scope data definition.
type Global struct {
	m       *Module
	name    string
	domain  Domain
	typ     FType
	linkage Linkage
	vis     Visibility
	binding Binding

	section   string
	comdat    string
	hasComdat bool
	align     uint64
	tlsModel  TLSModel

	init Init
	meta []Attach
}

func (g *Global) ItemKind() ItemKind     { return ItemGlobal }
func (g *Global) Name() string           { return g.name }
func (g *Global) SymbolKind() SymbolKind { return SymGlobal }
func (g *Global) Domain() Domain         { return g.domain }
func (g *Global) Type() FType            { return g.typ }
func (g *Global) Linkage() Linkage       { return g.linkage }
func (g *Global) Visibility() Visibility { return g.vis }
func (g *Global) Binding() Binding       { return g.binding }

// SectionAttr is the section this global was placed in, or "" if none was
// stated. Named -Attr, not Section, because Section(s) is the builder method
// that sets it.
func (g *Global) SectionAttr() string { return g.section }

// ComdatAttr reports the comdat key and whether one was stated. See
// SectionAttr for why this isn't named Comdat.
func (g *Global) ComdatAttr() (string, bool) { return g.comdat, g.hasComdat }

func (g *Global) AlignAttr() uint64      { return g.align }
func (g *Global) TLSModelAttr() TLSModel { return g.tlsModel }

// Initializer is the global's initializer. Named -Initializer, not Init,
// because Init(i) is the builder method that sets it.
func (g *Global) Initializer() Init { return g.init }

func (g *Global) Attached() []Attach { return g.meta }

// Global declares a module-scope datum. The ftype is required: §19.10 checks an
// initializer's structure against the declared type, which is unenforceable
// without one, and zeroed has no width to take from an initializer that states
// none.
func (m *Module) Global(name string, d Domain, t FType) *Global {
	g := &Global{m: m, name: name, domain: d, typ: t, init: ZeroInit}
	if t.IsZero() {
		m.failModule(ErrType, "global @%s declares no type", name)
		return g
	}
	if d == 0 {
		m.failModule(ErrPlacement, "global @%s declares no domain", name)
		return g
	}
	if m.declare(name, g) {
		m.items = append(m.items, g)
	}
	return g
}

func (g *Global) Export() *Global    { g.linkage = Export; return g }
func (g *Global) Internal() *Global  { g.linkage = Internal; return g }
func (g *Global) Hidden() *Global    { g.vis = Hidden; return g }
func (g *Global) Protected() *Global { g.vis = Protected; return g }

// DLLExport marks the definition exported from a DLL. dllimport is a property
// of a reference, so it lives on imports only (§19.11).
func (g *Global) DLLExport() *Global { g.vis = DLLExport; return g }

func (g *Global) Weak() *Global { g.binding = Weak; return g }

// Common marks a tentative allocation. §19.11 admits it on domain rw only.
func (g *Global) Common() *Global {
	if g.domain != RW {
		g.m.failModule(ErrPlacement, "common on @%s in domain %s", g.name, g.domain)
		return g
	}
	g.binding = Common
	return g
}

// Section sets the section this global is placed in.
func (g *Global) Section(s string) *Global { g.section = s; return g }

// Comdat sets the comdat key. With no key it defaults to the symbol's own name.
func (g *Global) Comdat(key ...string) *Global {
	g.hasComdat = true
	if len(key) > 0 {
		g.comdat = key[0]
	}
	return g
}

func (g *Global) Align(n uint64) *Global {
	if !isPow2(n) {
		g.m.failModule(ErrAlign, "global @%s align %d", g.name, n)
		return g
	}
	g.align = n
	return g
}

// TLSModel states the model. §19.7 admits it on domain tls globals only.
func (g *Global) TLSModel(t TLSModel) *Global {
	if g.domain != TLS {
		g.m.failModule(ErrPlacement, "tlsmodel on @%s in domain %s", g.name, g.domain)
		return g
	}
	g.tlsModel = t
	return g
}

// Init sets the initializer.
func (g *Global) Init(i Init) *Global {
	if i.IsZero() {
		g.m.failModule(ErrPoison, "global @%s given the zero Init", g.name)
		return g
	}
	g.init = i
	return g
}

func (g *Global) Meta(a ...Attach) *Global { g.meta = append(g.meta, a...); return g }

// A GlobalImport is a reference to a datum another module defines. It carries
// visibility and weak because both are facts about the reference; linkage and
// common are absent because an import defines nothing.
type GlobalImport struct {
	m         *Module
	name      string
	typ       FType
	vis       Visibility
	weak      bool
	section   string
	comdat    string
	hasComdat bool
	align     uint64
	tlsModel  TLSModel
	meta      []Attach
}

func (g *GlobalImport) ItemKind() ItemKind     { return ItemGlobalImport }
func (g *GlobalImport) Name() string           { return g.name }
func (g *GlobalImport) SymbolKind() SymbolKind { return SymGlobal }
func (g *GlobalImport) Type() FType            { return g.typ }
func (g *GlobalImport) Visibility() Visibility { return g.vis }
func (g *GlobalImport) IsWeak() bool           { return g.weak }

// SectionAttr is the section this import is bound to, or "" if none was
// stated. Named -Attr for the same reason as Global.SectionAttr.
func (g *GlobalImport) SectionAttr() string { return g.section }

// ComdatAttr reports the comdat key and whether one was stated.
func (g *GlobalImport) ComdatAttr() (string, bool) { return g.comdat, g.hasComdat }

func (g *GlobalImport) AlignAttr() uint64      { return g.align }
func (g *GlobalImport) TLSModelAttr() TLSModel { return g.tlsModel }
func (g *GlobalImport) Attached() []Attach     { return g.meta }

// ImportGlobal declares a reference to a datum another module defines.
func (m *Module) ImportGlobal(name string, t FType) *GlobalImport {
	g := &GlobalImport{m: m, name: name, typ: t}
	if t.IsZero() {
		m.failModule(ErrType, "import global @%s declares no type", name)
		return g
	}
	if m.declare(name, g) {
		m.items = append(m.items, g)
	}
	return g
}

func (g *GlobalImport) Hidden() *GlobalImport    { g.vis = Hidden; return g }
func (g *GlobalImport) Protected() *GlobalImport { g.vis = Protected; return g }

// DLLImport describes how this module reaches a symbol another module defines.
func (g *GlobalImport) DLLImport() *GlobalImport { g.vis = DLLImport; return g }

// Weak makes the reference one that may go unresolved — &g != NULL in C.
func (g *GlobalImport) Weak() *GlobalImport { g.weak = true; return g }

func (g *GlobalImport) Section(s string) *GlobalImport { g.section = s; return g }

func (g *GlobalImport) Comdat(key ...string) *GlobalImport {
	g.hasComdat = true
	if len(key) > 0 {
		g.comdat = key[0]
	}
	return g
}

func (g *GlobalImport) Align(n uint64) *GlobalImport {
	if !isPow2(n) {
		g.m.failModule(ErrAlign, "import global @%s align %d", g.name, n)
		return g
	}
	g.align = n
	return g
}

func (g *GlobalImport) TLSModel(t TLSModel) *GlobalImport { g.tlsModel = t; return g }

func (g *GlobalImport) Meta(a ...Attach) *GlobalImport {
	g.meta = append(g.meta, a...)
	return g
}
