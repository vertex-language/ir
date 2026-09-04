package ir

// An Endian is a byte order.
type Endian uint8

const (
	LittleEndian Endian = iota
	BigEndian
)

func (e Endian) String() string {
	if e == BigEndian {
		return "big"
	}
	return "little"
}

// A Layout is §3's layout block: the five attributes that make sizeof, alignof
// and offsetof determinate and that admit or reject an ext-float namespace.
type Layout struct {
	ABI        string // the module's default calling convention; ccc means this
	Endian     Endian
	PtrBits    uint
	StackAlign uint
	ExtFloat   []RegType // any of TypeF80, TypeF128; may be empty
	Vector     bool      // whether the target has a 128-bit vector register file
}

// HasExtFloat reports whether the layout admits an ext-float namespace. A
// namespace it admits is usable whether or not the target implements it in
// silicon; where it does not, lowering supplies a runtime call.
func (l Layout) HasExtFloat(t RegType) bool {
	for _, e := range l.ExtFloat {
		if e == t {
			return true
		}
	}
	return false
}

// Admits reports whether the layout provides the namespace t belongs to. Only
// the gated ones can answer no; the core reg-types are on every target.
func (l Layout) Admits(t RegType) bool {
	switch {
	case t.IsExtFloat():
		return l.HasExtFloat(t)
	case t.IsVector():
		return l.Vector
	}
	return true
}

func (l Layout) clone() Layout {
	if l.ExtFloat != nil {
		e := make([]RegType, len(l.ExtFloat))
		copy(e, l.ExtFloat)
		l.ExtFloat = e
	}
	return l
}

// A Target pairs §3's use string with the layout that usually accompanies it.
// It is a convenience over NewModuleLayout and carries no other authority.
type Target struct {
	use    string
	layout Layout
}

// NewTarget pairs a use path with a layout.
func NewTarget(use string, l Layout) Target { return Target{use: use, layout: l.clone()} }

func (t Target) Use() string    { return t.use }
func (t Target) Layout() Layout { return t.layout.clone() }
func (t Target) String() string { return t.use }

// Stock targets. long double and __float128 decide the ext-float list; the
// vector flag is whether the architecture's *baseline* has a 128-bit register
// file, which x86-64 and AArch64 both do — SSE2 is part of the former's
// definition and Advanced SIMD of the latter's. i386 is the one that does not:
// SSE2 exists on the chips anyone still runs but is not in the architecture a
// plain i386 target names, and a compiler that assumed it would produce
// binaries that fault on the machines the target is for.
var (
	X86_64Linux = NewTarget("x86_64/linux", Layout{
		ABI: "sysv", Endian: LittleEndian, PtrBits: 64, StackAlign: 16,
		ExtFloat: []RegType{TypeF80, TypeF128}, Vector: true,
	})
	I386Linux = NewTarget("i386/linux", Layout{
		ABI: "sysv", Endian: LittleEndian, PtrBits: 32, StackAlign: 16,
		ExtFloat: []RegType{TypeF80},
	})
	AArch64Linux = NewTarget("aarch64/linux", Layout{
		ABI: "aapcs", Endian: LittleEndian, PtrBits: 64, StackAlign: 16,
		ExtFloat: []RegType{TypeF128}, Vector: true,
	})
	X86_64MacOS = NewTarget("x86_64/macos", Layout{
		ABI: "sysv", Endian: LittleEndian, PtrBits: 64, StackAlign: 16,
		ExtFloat: []RegType{TypeF80}, Vector: true,
	})
	AArch64MacOS = NewTarget("aarch64/macos", Layout{
		ABI: "aapcs", Endian: LittleEndian, PtrBits: 64, StackAlign: 16,
		Vector: true,
	})
	X86_64Windows = NewTarget("x86_64/windows", Layout{
		ABI: "ms", Endian: LittleEndian, PtrBits: 64, StackAlign: 16,
		Vector: true,
	})
)

// An ItemKind distinguishes the module-item forms of §3.
type ItemKind uint8

const (
	ItemType ItemKind = iota + 1
	ItemGlobal
	ItemFuncImport
	ItemGlobalImport
	ItemFunc
	ItemAlias
	ItemMeta
	ItemAsm
)

// An Item is a module-scope declaration, in declaration order.
type Item interface {
	ItemKind() ItemKind
}

// A Linkage is §3's linkage modifier. The zero value is the file default.
type Linkage uint8

const (
	NoLinkage Linkage = iota
	Export
	Internal
)

func (l Linkage) String() string {
	switch l {
	case Export:
		return "export"
	case Internal:
		return "internal"
	}
	return ""
}

// A Visibility is §3's visibility modifier.
type Visibility uint8

const (
	NoVisibility Visibility = iota
	Hidden
	Protected
	DLLImport
	DLLExport
)

func (v Visibility) String() string {
	switch v {
	case Hidden:
		return "hidden"
	case Protected:
		return "protected"
	case DLLImport:
		return "dllimport"
	case DLLExport:
		return "dllexport"
	}
	return ""
}

// A SymbolKind says which half of the module-scope value namespace a symbol
// occupies. Globals and functions share one namespace, since they share one at
// link time.
type SymbolKind uint8

const (
	SymGlobal SymbolKind = iota + 1
	SymFunc
)

// A Symbol is anything nameable in the module-scope value namespace: a global,
// a function definition, either import form, or an alias.
type Symbol interface {
	Name() string
	SymbolKind() SymbolKind
}

// A Module is everything a frontend decided and nothing a backend will decide.
// It has no Finalize, because an IR exists to be rewritten.
type Module struct {
	name   string
	use    string
	layout Layout

	items  []Item
	types  map[string]*Type
	values map[string]Symbol
	metas  map[string]*MetaDecl

	err      error
	deferred []func() *Error
}

// NewModule creates a module for a stock target. module, use and layout appear
// exactly once each, in order, ahead of every module item, so they are
// constructor arguments and not builder methods — which is what stops §19.15
// from being a check a verifier can fail.
func NewModule(name string, t Target) *Module {
	return NewModuleLayout(name, t.use, t.layout)
}

// NewModuleLayout creates a module for a use path and an explicit layout.
func NewModuleLayout(name, use string, l Layout) *Module {
	m := &Module{
		name:   name,
		use:    use,
		layout: l.clone(),
		types:  make(map[string]*Type),
		values: make(map[string]Symbol),
		metas:  make(map[string]*MetaDecl),
	}
	if !validIdent(name) {
		m.failModule(ErrName, "module name %q is not an identifier", name)
	}
	if use == "" {
		m.failModule(ErrName, "empty use path")
	}
	return m
}

func (m *Module) Name() string   { return m.name }
func (m *Module) Use() string    { return m.use }
func (m *Module) Layout() Layout { return m.layout.clone() }

// Items returns the module's declarations in declaration order.
func (m *Module) Items() []Item { return m.items }

// Funcs returns the function definitions, in declaration order.
func (m *Module) Funcs() []*Func {
	var out []*Func
	for _, it := range m.items {
		if f, ok := it.(*Func); ok {
			out = append(out, f)
		}
	}
	return out
}

// Globals returns the global definitions, in declaration order.
func (m *Module) Globals() []*Global {
	var out []*Global
	for _, it := range m.items {
		if g, ok := it.(*Global); ok {
			out = append(out, g)
		}
	}
	return out
}

// FuncImports returns the function imports, in declaration order.
func (m *Module) FuncImports() []*FuncImport {
	var out []*FuncImport
	for _, it := range m.items {
		if f, ok := it.(*FuncImport); ok {
			out = append(out, f)
		}
	}
	return out
}

// GlobalImports returns the global imports, in declaration order.
func (m *Module) GlobalImports() []*GlobalImport {
	var out []*GlobalImport
	for _, it := range m.items {
		if g, ok := it.(*GlobalImport); ok {
			out = append(out, g)
		}
	}
	return out
}

// Types returns the named type declarations, in declaration order.
func (m *Module) Types() []*Type {
	var out []*Type
	for _, it := range m.items {
		if t, ok := it.(*Type); ok {
			out = append(out, t)
		}
	}
	return out
}

// Lookup resolves a name in the module-scope value namespace.
func (m *Module) Lookup(name string) Symbol { return m.values[name] }

// LookupType resolves a name in the type namespace, which is disjoint from the
// value namespace.
func (m *Module) LookupType(name string) *Type { return m.types[name] }

// Err returns the first builder failure, or nil. It runs any deferred checks —
// branch argument arity and type against block parameters, which a forward
// branch cannot be checked against at emission — before it answers, and is
// idempotent afterwards. verify.Module calls this first, so soundness is one
// call and not two.
func (m *Module) Err() error {
	if m.err != nil {
		return m.err
	}
	deferred := m.deferred
	m.deferred = nil
	for _, c := range deferred {
		if e := c(); e != nil {
			m.err = e
			return m.err
		}
	}
	return nil
}

// declare records a symbol in the module-scope value namespace.
func (m *Module) declare(name string, s Symbol) bool {
	if !validSymbol(name) {
		m.failModule(ErrName, "symbol name %q", name)
		return false
	}
	if _, dup := m.values[name]; dup {
		m.failModule(ErrDuplicate, "symbol @%s", name)
		return false
	}
	m.values[name] = s
	return true
}

// An Alias binds a second name to an existing definition and initializes
// nothing of its own (§5b).
type Alias struct {
	m       *Module
	name    string
	target  Symbol
	kind    SymbolKind
	linkage Linkage
	vis     Visibility
	weak    bool
}

func (a *Alias) ItemKind() ItemKind     { return ItemAlias }
func (a *Alias) Name() string           { return a.name }
func (a *Alias) SymbolKind() SymbolKind { return a.kind }
func (a *Alias) Target() Symbol         { return a.target }
func (a *Alias) Linkage() Linkage       { return a.linkage }
func (a *Alias) Visibility() Visibility { return a.vis }
func (a *Alias) IsWeak() bool           { return a.weak }

// AliasFunc binds name to an existing function.
func (m *Module) AliasFunc(name string, to Callee) *Alias {
	return m.newAlias(name, to, SymFunc)
}

// AliasGlobal binds name to an existing global.
func (m *Module) AliasGlobal(name string, to Symbol) *Alias {
	if to != nil && to.SymbolKind() != SymGlobal {
		m.failModule(ErrType, "alias global @%s names a function", name)
	}
	return m.newAlias(name, to, SymGlobal)
}

func (m *Module) newAlias(name string, to Symbol, k SymbolKind) *Alias {
	a := &Alias{m: m, name: name, target: to, kind: k}
	if to == nil {
		m.failModule(ErrPoison, "alias @%s has no target", name)
		return a
	}
	if m.declare(name, a) {
		m.items = append(m.items, a)
	}
	return a
}

func (a *Alias) Export() *Alias    { a.linkage = Export; return a }
func (a *Alias) Internal() *Alias  { a.linkage = Internal; return a }
func (a *Alias) Hidden() *Alias    { a.vis = Hidden; return a }
func (a *Alias) Protected() *Alias { a.vis = Protected; return a }
func (a *Alias) Weak() *Alias      { a.weak = true; return a }
