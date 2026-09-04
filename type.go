package ir

import "strconv"

// A RegType is a type a value can live in a register as (§2). i8 and i16 are
// absent: they are storage-only widths, reached through the sub-width load and
// store verbs.
type RegType uint8

const (
	TypeNone RegType = iota // a bare mnemonic's namespace
	TypeI1
	TypeI32
	TypeI64
	TypeF32
	TypeF64
	TypeF80
	TypeF128

	// TypeV128 is a 128-bit vector register. It is one type and not six
	// because the lane shape is not a property of the register: the same
	// sixteen bytes are eight words to one instruction and four
	// doublewords to the next, and C's own __m128i is written that way —
	// one type used at every lane width, with the width in the intrinsic's
	// name. So the shape rides in the verb, which is where §1 already puts
	// signedness for the same reason: the hardware has one register file
	// and several ways of reading it.
	TypeV128

	TypePtr
)

var regTypeText = [...]string{
	TypeNone: "<none>", TypeI1: "i1", TypeI32: "i32", TypeI64: "i64",
	TypeF32: "f32", TypeF64: "f64", TypeF80: "f80", TypeF128: "f128",
	TypeV128: "v128",
	TypePtr:  "ptr",
}

func (t RegType) String() string {
	if int(t) < len(regTypeText) {
		return regTypeText[t]
	}
	return "RegType(" + strconv.Itoa(int(t)) + ")"
}

// IsInt reports whether t is one of i1, i32, i64.
func (t RegType) IsInt() bool { return t == TypeI1 || t == TypeI32 || t == TypeI64 }

// IsFloat reports whether t is any float namespace, extended ones included.
func (t RegType) IsFloat() bool {
	return t == TypeF32 || t == TypeF64 || t == TypeF80 || t == TypeF128
}

// IsExtFloat reports whether t is an ext-float namespace, whose availability
// the layout block decides.
func (t RegType) IsExtFloat() bool { return t == TypeF80 || t == TypeF128 }

// IsVector reports whether t is a vector namespace, whose availability the
// layout block decides the same way.
func (t RegType) IsVector() bool { return t == TypeV128 }

// A StoreType is a scalar type a value can live in memory as (§2).
type StoreType uint8

const (
	StoreNone StoreType = iota
	StoreI8
	StoreI16
	StoreI32
	StoreI64
	StoreF32
	StoreF64
	StoreF80
	StoreF128
	StoreV128
	StorePtr
)

var storeTypeText = [...]string{
	StoreNone: "<none>", StoreI8: "i8", StoreI16: "i16", StoreI32: "i32",
	StoreI64: "i64", StoreF32: "f32", StoreF64: "f64", StoreF80: "f80",
	StoreF128: "f128", StoreV128: "v128", StorePtr: "ptr",
}

func (s StoreType) String() string {
	if int(s) < len(storeTypeText) {
		return storeTypeText[s]
	}
	return "StoreType(" + strconv.Itoa(int(s)) + ")"
}

// RegType returns the register type of the same width, or TypeNone for i8 and
// i16, which have none.
func (s StoreType) RegType() RegType {
	switch s {
	case StoreI32:
		return TypeI32
	case StoreI64:
		return TypeI64
	case StoreF32:
		return TypeF32
	case StoreF64:
		return TypeF64
	case StoreF80:
		return TypeF80
	case StoreF128:
		return TypeF128
	case StoreV128:
		return TypeV128
	case StorePtr:
		return TypePtr
	}
	return TypeNone
}

// FType wraps s as an ftype.
func (s StoreType) FType() FType { return FType{kind: FTypeScalar, scalar: s} }

// An FTypeKind distinguishes the three shapes an ftype can take.
type FTypeKind uint8

const (
	FTypeScalar FTypeKind = iota + 1
	FTypeNamed
	FTypeArray
)

// An FType is a storage-shaped type: a store-type, a named type, or a fixed
// array of one (§2's ftype).
type FType struct {
	kind   FTypeKind
	scalar StoreType
	named  *Type
	elem   *FType
	n      uint64
}

// Array returns the ftype for a fixed array of n elements.
func Array(n uint64, e FType) FType {
	c := e
	return FType{kind: FTypeArray, elem: &c, n: n}
}

func (f FType) Kind() FTypeKind   { return f.kind }
func (f FType) Scalar() StoreType { return f.scalar }
func (f FType) Named() *Type      { return f.named }
func (f FType) IsZero() bool      { return f.kind == 0 }
func (f FType) Len() uint64       { return f.n }

// Elem returns the element type of an array ftype.
func (f FType) Elem() FType {
	if f.elem == nil {
		return FType{}
	}
	return *f.elem
}

func (f FType) String() string {
	switch f.kind {
	case FTypeScalar:
		return f.scalar.String()
	case FTypeNamed:
		if f.named == nil {
			return "@<nil>"
		}
		return "@" + f.named.name
	case FTypeArray:
		return "[" + strconv.FormatUint(f.n, 10) + "]" + f.Elem().String()
	}
	return "<invalid ftype>"
}

// A TypeKind distinguishes the four typedef forms of §4.
type TypeKind uint8

const (
	KindStruct TypeKind = iota + 1
	KindUnion
	KindAlias // a typedef of a plain ftype
	KindFunc  // a func typedef, which callind names
)

// A Field is one member of a struct or union. Offset is meaningful only where
// HasOffset is set, which §19.18 makes an all-or-none property of the struct.
type Field struct {
	Name      string
	Type      FType
	Offset    uint64
	HasOffset bool
}

// A Type is a named type declaration: a layout description the assembler
// consults, not a value the instruction stream manipulates.
type Type struct {
	m       *Module
	name    string
	linkage Linkage
	kind    TypeKind
	packed  bool
	align   uint64
	fields  []Field
	alias   FType
	sig     *Sig
	meta    []Attach
}

func (t *Type) ItemKind() ItemKind { return ItemType }

func (t *Type) Name() string       { return t.name }
func (t *Type) Kind() TypeKind     { return t.kind }
func (t *Type) Linkage() Linkage   { return t.linkage }
func (t *Type) IsPacked() bool     { return t.packed }
func (t *Type) Fields() []Field    { return t.fields }
func (t *Type) Aliased() FType     { return t.alias }
func (t *Type) Sig() *Sig          { return t.sig }
func (t *Type) Attached() []Attach { return t.meta }

// AlignAttr reports the struct-layout align, or zero if none was stated.
func (t *Type) AlignAttr() uint64 { return t.align }

// FType wraps t for use where an ftype is expected.
func (t *Type) FType() FType { return FType{kind: FTypeNamed, named: t} }

// Struct declares a struct type.
func (m *Module) Struct(name string) *Type { return m.newType(name, KindStruct) }

// Union declares a union type. A union's fields all begin at zero.
func (m *Module) Union(name string) *Type { return m.newType(name, KindUnion) }

// TypeOf declares a typedef of a plain ftype.
func (m *Module) TypeOf(name string, f FType) *Type {
	t := m.newType(name, KindAlias)
	t.alias = f
	return t
}

// FuncType declares a func typedef. callind names one of these, which is what
// keeps an indirect call well-typed and carries its calling convention.
func (m *Module) FuncType(name string, sig *Sig) *Type {
	t := m.newType(name, KindFunc)
	if sig == nil {
		sig = NewSig()
	}
	t.sig = sig
	return t
}

func (m *Module) newType(name string, k TypeKind) *Type {
	t := &Type{m: m, name: name, kind: k}
	if !validIdent(name) {
		m.failModule(ErrName, "type name %q", name)
		return t
	}
	if _, dup := m.types[name]; dup {
		m.failModule(ErrDuplicate, "type @%s", name)
		return t
	}
	m.types[name] = t
	m.items = append(m.items, t)
	return t
}

// Field appends a field with a computed offset.
func (t *Type) Field(name string, f FType) *Type {
	t.fields = append(t.fields, Field{Name: name, Type: f})
	return t
}

// FieldAt appends a field with a stated byte offset. §19.18 admits at on all
// of a struct's fields or none of them, and never on a union's.
func (t *Type) FieldAt(name string, f FType, off uint64) *Type {
	if t.kind == KindUnion {
		t.m.failModule(ErrPlacement, "at on a union field @%s.%s", t.name, name)
		return t
	}
	t.fields = append(t.fields, Field{Name: name, Type: f, Offset: off, HasOffset: true})
	return t
}

// Pack marks the aggregate packed.
func (t *Type) Pack() *Type { t.packed = true; return t }

// Align states the aggregate's alignment.
func (t *Type) Align(n uint64) *Type {
	if !isPow2(n) {
		t.m.failModule(ErrAlign, "type @%s align %d", t.name, n)
		return t
	}
	t.align = n
	return t
}

func (t *Type) Export() *Type   { t.linkage = Export; return t }
func (t *Type) Internal() *Type { t.linkage = Internal; return t }

// Meta attaches metadata to the declaration.
func (t *Type) Meta(a ...Attach) *Type { t.meta = append(t.meta, a...); return t }

// A ConstKind distinguishes the literal forms of §8.
type ConstKind uint8

const (
	ConstInt ConstKind = iota + 1
	ConstFloat
	ConstSizeOf
	ConstAlignOf
	ConstOffsetOf

	// ConstBytes is a literal run of bytes in memory order. It exists for
	// v128.const, whose value has no scalar spelling: sixteen bytes are
	// eight words to one instruction and four doublewords to the next, and
	// writing them as a number would mean choosing one of those readings
	// and an endianness to go with it.
	ConstBytes
)

// A PathElem is one step of an offsetof path: a field name or an array index.
type PathElem struct {
	name    string
	index   uint64
	isIndex bool
}

func (p PathElem) IsIndex() bool { return p.isIndex }
func (p PathElem) Name() string  { return p.name }
func (p PathElem) Index() uint64 { return p.index }

// FieldPath names a field in an offsetof path.
func FieldPath(name string) PathElem { return PathElem{name: name} }

// IndexPath indexes an array in an offsetof path.
func IndexPath(i uint64) PathElem { return PathElem{index: i, isIndex: true} }

// A Const is a literal: an integer, a float, or one of the three symbolic
// constants that make a frontend's flattening into byte offsets checkable
// rather than hand-computed and silently target-dependent.
type Const struct {
	kind  ConstKind
	i     int64
	f     float64
	typ   *Type
	sym   Symbol
	path  []PathElem
	bytes []byte
}

func Int(v int64) Const     { return Const{kind: ConstInt, i: v} }
func Uint(v uint64) Const   { return Const{kind: ConstInt, i: int64(v)} }
func Float(v float64) Const { return Const{kind: ConstFloat, f: v} }

// Bytes is a literal run of bytes in memory order.
func Bytes(b []byte) Const {
	c := make([]byte, len(b))
	copy(c, b)
	return Const{kind: ConstBytes, bytes: c}
}

// SizeOf is sizeof @T.
func SizeOf(t *Type) Const { return Const{kind: ConstSizeOf, typ: t} }

// SizeOfSym is sizeof @g, for a global.
func SizeOfSym(s Symbol) Const { return Const{kind: ConstSizeOf, sym: s} }

// AlignOf is alignof @T.
func AlignOf(t *Type) Const { return Const{kind: ConstAlignOf, typ: t} }

// AlignOfSym is alignof @g, for a global.
func AlignOfSym(s Symbol) Const { return Const{kind: ConstAlignOf, sym: s} }

// OffsetOf is offsetof @T followed by a path.
func OffsetOf(t *Type, path ...PathElem) Const {
	return Const{kind: ConstOffsetOf, typ: t, path: path}
}

func (c Const) Kind() ConstKind  { return c.kind }
func (c Const) Int() int64       { return c.i }
func (c Const) Float() float64   { return c.f }
func (c Const) Type() *Type      { return c.typ }
func (c Const) Symbol() Symbol   { return c.sym }
func (c Const) Path() []PathElem { return c.path }

// Bytes returns a ConstBytes literal's bytes. The slice is the caller's to
// read and not to write.
func (c Const) Bytes() []byte { return c.bytes }
func (c Const) IsZero() bool  { return c.kind == 0 }

func isPow2(n uint64) bool { return n != 0 && n&(n-1) == 0 }
