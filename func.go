// func.go
package ir

import "strconv"

// A CallConv is §6's callconv. The zero value is CCC, which is the abi named in
// the module's layout block — a module has no convention-free default.
type CallConv uint8

const (
	CCC CallConv = iota
	FastCC
	PreserveMost
	PreserveAll
	StdCall
	FastCall
	ThisCall
	VectorCall
	MSABI
	SysVABI

	// Kernel is a GPU entry point: the function a host launches over a
	// grid of work-items. It is a convention and not a placement because
	// it is one — the parameters arrive in the kernel-argument buffer
	// rather than in registers, nothing on the device can call it, and it
	// returns nothing. §19.20 states those three as rules.
	Kernel
)

var callConvText = [...]string{
	CCC: "ccc", FastCC: "fastcc", PreserveMost: "preserve_most",
	PreserveAll: "preserve_all", StdCall: "stdcall", FastCall: "fastcall",
	ThisCall: "thiscall", VectorCall: "vectorcall", MSABI: "ms_abi",
	SysVABI: "sysv_abi", Kernel: "kernel",
}

func (c CallConv) String() string {
	if int(c) < len(callConvText) {
		return callConvText[c]
	}
	return "ccc"
}

type paKind uint8

const (
	paByVal paKind = iota + 1
	paSRet
	paSRetMem
	paZExt
	paSExt
	paNoAlias
	paSwiftSelf
	paSwiftError
	paSwiftOut
	paSwiftAsync
	paNarrow
)

// A ParamAttr is a param-attr or a ret-attr (§6). ZExt and SExt are the two a
// ret-item admits.
type ParamAttr struct {
	kind paKind
	typ  *Type

	// For Narrow: the source type's size in bytes, and its signedness.
	bytes  uint8
	signed bool
}

var (
	ZExt    = ParamAttr{kind: paZExt}
	SExt    = ParamAttr{kind: paSExt}
	NoAlias = ParamAttr{kind: paNoAlias}

	// SwiftSelf marks the parameter that travels in the self register
	// rather than in the argument sequence.
	//
	// It is a convention rather than a hint: a caller that puts the value
	// in an argument register instead passes it where the callee does not
	// look, and the callee reads whatever was in the self register. That
	// compiles and links and answers, which is why this exists — the two
	// sides have to agree, and the only way to agree is to say so in the
	// signature. AAPCS64 has no such register; this is Swift's, and on
	// AArch64 it is X20.
	SwiftSelf = ParamAttr{kind: paSwiftSelf}

	// SwiftError marks the result that travels in the error register
	// rather than in the return sequence.
	//
	// It is how Swift says a call failed: the caller clears the
	// register before the call, the callee writes to it on the path
	// that fails, and the caller reads it afterwards -- swiftc's own
	// code does exactly that, `mov x21, #0` before the branch and
	// `cbnz x21` after it. A function that returns normally leaves it
	// as it found it.
	//
	// On a result rather than a parameter, because that is what it
	// is: the value travels out. The register is cleared for the call
	// whether or not anything reads what comes back, since the callee
	// only writes it when it fails and the caller has no other way to
	// tell the two apart.
	SwiftError = ParamAttr{kind: paSwiftError}

	// SwiftIndirectResult marks the parameter carrying the address of
	// the storage a caller set aside for a result that is returned
	// indirectly by convention rather than by size.
	//
	// It is SRet's answer without SRet's question. AAPCS64 decides
	// indirection from the aggregate: a two-eightbyte result comes
	// back in X0 and X1, so a caller supplies no address at all, and
	// SRet says so by naming the type and letting §6.9 re-derive it.
	// Swift decides it from the declaration instead. A generic
	// function returns `@out T` through the indirect result register
	// whatever T is, because the caller is the only one who knows how
	// big T became -- Array's subscript getter hands an Int32 back
	// through a four-byte slot in X8, which no size rule would ask
	// for.
	//
	// So this is a statement and not a hint, the way SwiftSelf is: the
	// register is X8 because the declaration says so.
	SwiftIndirectResult = ParamAttr{kind: paSwiftOut}

	// SwiftAsync marks the parameter carrying an async function's
	// context: the frame holding everything that has to survive a
	// suspension, and the chain back to the caller's.
	//
	// Swift's async functions do not keep their state on the stack,
	// because a suspended one has no stack -- it has given its thread
	// back. What it has instead is this pointer, and a frame reached
	// through it that the task's allocator owns. So the register is
	// the one thing every piece of a split function is handed, and it
	// is the first thing each of them reads.
	//
	// The same statement SwiftSelf is, for the same reason: AAPCS64 has
	// no such register, the two sides have to agree, and the only way
	// to agree is to say so in the signature. On AArch64 it is X22,
	// which is what swiftc emits.
	SwiftAsync = ParamAttr{kind: paSwiftAsync}
)

// Narrow says an i32 parameter is a source-language integer of one or two
// bytes -- a C char, short or bool -- which is a fact the register type does
// not carry and a calling convention may need. Apple's arm64 packs a stack
// argument at its own size and alignment: a bool takes one byte and the int
// after it the next four-byte boundary, where the base AAPCS64 gives each an
// eightbyte. The caller stores exactly that many bytes and the callee loads
// them extended as signed says. A convention with no such packing ignores it;
// in a register the value is extended to 32 bits either way.
func Narrow(bytes int, signed bool) ParamAttr {
	return ParamAttr{kind: paNarrow, bytes: uint8(bytes), signed: signed}
}

// NarrowWidth reports a Narrow attribute's size and signedness.
func (a ParamAttr) NarrowWidth() (bytes int, signed, ok bool) {
	if a.kind != paNarrow {
		return 0, false, false
	}
	return int(a.bytes), a.signed, true
}

// ByVal passes the aggregate the pointer names by value.
func ByVal(t *Type) ParamAttr { return ParamAttr{kind: paByVal, typ: t} }

// SRet names the aggregate the callee writes its result into. §19.13 admits it
// on at most one parameter, which is the first.
//
// Whether the result comes back in registers instead is the target's to
// decide from the aggregate's shape. A front end that knows the language
// forbids it -- C++ returns a class with a non-trivial copy constructor or
// destructor in memory however small it is -- says so with SRetMemory.
func SRet(t *Type) ParamAttr { return ParamAttr{kind: paSRet, typ: t} }

// SRetMemory is SRet for a result the language requires in memory. The
// target places the pointer where its convention puts an indirect result
// -- x8 on AArch64, the first integer register on x86-64 -- and never
// brings the aggregate back in registers.
func SRetMemory(t *Type) ParamAttr { return ParamAttr{kind: paSRetMem, typ: t} }

func (a ParamAttr) IsByVal() bool   { return a.kind == paByVal }
func (a ParamAttr) IsSRet() bool    { return a.kind == paSRet || a.kind == paSRetMem }

// IsSRetMemory reports whether the result must come back in memory rather
// than in registers, whatever its shape would allow.
func (a ParamAttr) IsSRetMemory() bool { return a.kind == paSRetMem }
func (a ParamAttr) IsZExt() bool    { return a.kind == paZExt }
func (a ParamAttr) IsSExt() bool    { return a.kind == paSExt }
func (a ParamAttr) IsNoAlias() bool { return a.kind == paNoAlias }

// IsSwiftSelf reports whether this parameter travels in the self register.
func (a ParamAttr) IsSwiftSelf() bool { return a.kind == paSwiftSelf }

// IsSwiftAsync reports whether this parameter carries an async
// function's context, and so travels in the async context register.
func (a ParamAttr) IsSwiftAsync() bool { return a.kind == paSwiftAsync }

// IsSwiftError reports whether this result travels in the error
// register.
func (a ParamAttr) IsSwiftError() bool { return a.kind == paSwiftError }

// IsSwiftIndirectResult reports whether this parameter carries the
// address of storage a result is written into, by convention rather
// than by size.
func (a ParamAttr) IsSwiftIndirectResult() bool { return a.kind == paSwiftOut }
func (a ParamAttr) Type() *Type                 { return a.typ }

func (a ParamAttr) String() string {
	switch a.kind {
	case paByVal:
		return "byval"
	case paSRet:
		return "sret"
	case paSRetMem:
		return "sret_memory"
	case paZExt:
		return "zext"
	case paSExt:
		return "sext"
	case paNoAlias:
		return "noalias"
	case paSwiftSelf:
		return "swiftself"
	case paSwiftError:
		return "swifterror"
	case paSwiftOut:
		return "swiftindirect"
	case paSwiftAsync:
		return "swiftasync"
	case paNarrow:
		s := "narrow" + strconv.Itoa(8*int(a.bytes))
		if a.signed {
			return s + "s"
		}
		return s + "u"
	}
	return ""
}

// A Param is one entry of a signature's parameter list. Name is empty for an
// import or a func typedef, neither of which has a body to reference it.
type Param struct {
	Name  string
	Type  RegType
	Attrs []ParamAttr
}

// A RetItem is one entry of a signature's result list. A signature may return
// several registers as an ABI-level multi-value result; that is a call-boundary
// shape, not an aggregate register type.
type RetItem struct {
	Type  RegType
	Attrs []ParamAttr
}

// A Sig is an abs-signature: convention, parameters, variadic tail, results.
type Sig struct {
	conv     CallConv
	params   []Param
	variadic bool
	rets     []RetItem
}

// NewSig returns an empty signature, for a func typedef or an import.
func NewSig() *Sig { return &Sig{} }

// Conv sets the calling convention. Absent one, a signature's convention is
// ccc, which is the abi named in the module's layout block.
func (s *Sig) Conv(c CallConv) *Sig { s.conv = c; return s }

// Param appends an unnamed parameter.
func (s *Sig) Param(t RegType, attrs ...ParamAttr) *Sig {
	s.params = append(s.params, Param{Type: t, Attrs: attrs})
	return s
}

// Variadic marks the var-tail.
func (s *Sig) Variadic() *Sig { s.variadic = true; return s }

// Ret appends a ret-item. Calling it twice gives a multi-value result.
func (s *Sig) Ret(t RegType, attrs ...ParamAttr) *Sig {
	s.rets = append(s.rets, RetItem{Type: t, Attrs: attrs})
	return s
}

func (s *Sig) CallConv() CallConv { return s.conv }
func (s *Sig) Params() []Param    { return s.params }
func (s *Sig) IsVariadic() bool   { return s.variadic }
func (s *Sig) Rets() []RetItem    { return s.rets }

func (s *Sig) retTypes() []RegType {
	if len(s.rets) == 0 {
		return nil
	}
	out := make([]RegType, len(s.rets))
	for i, r := range s.rets {
		out[i] = r.Type
	}
	return out
}

// A Callee is anything call may name: a definition or a function import.
type Callee interface {
	Symbol
	Signature() *Sig
}

// A Func is a function definition.
type Func struct {
	m    *Module
	name string
	sig  *Sig

	linkage Linkage
	vis     Visibility
	weak    bool

	section      string
	comdat       string
	hasComdat    bool
	nounwind     bool
	personality  Callee
	returnsTwice bool
	naked        bool
	noreturn     bool
	align        uint64

	asmBody    string
	hasAsmBody bool

	params []*Def
	blocks []*Block

	// names are the register names taken in this function, so Name can make
	// a repeat unique instead of printing two definitions of one register.
	names  map[string]int
	entry  *Block
	labels map[string]*Block

	frozen bool
	nextID int32
	meta   []Attach
}

func (f *Func) ItemKind() ItemKind     { return ItemFunc }
func (f *Func) Name() string           { return f.name }
func (f *Func) SymbolKind() SymbolKind { return SymFunc }
func (f *Func) Signature() *Sig        { return f.sig }
func (f *Func) Module() *Module        { return f.m }
func (f *Func) Linkage() Linkage       { return f.linkage }
func (f *Func) Visibility() Visibility { return f.vis }
func (f *Func) IsWeak() bool           { return f.weak }

// SectionAttr is the section this function was placed in, or "" if none was
// stated. Named -Attr, not Section, because Section(s) is the builder method
// that sets it — Go allows one name per receiver, not one per direction.
func (f *Func) SectionAttr() string { return f.section }

// ComdatAttr reports the comdat key and whether one was stated. See
// SectionAttr for why this isn't named Comdat.
func (f *Func) ComdatAttr() (string, bool) { return f.comdat, f.hasComdat }

func (f *Func) IsNoUnwind() bool      { return f.nounwind }
func (f *Func) PersonalityFn() Callee { return f.personality }
func (f *Func) IsReturnsTwice() bool  { return f.returnsTwice }
func (f *Func) IsNaked() bool         { return f.naked }
func (f *Func) IsNoReturn() bool      { return f.noreturn }
func (f *Func) AlignAttr() uint64     { return f.align }
func (f *Func) Attached() []Attach    { return f.meta }

// Params returns the signature's parameter registers, which are the entry
// block's inputs and which no branch can supply.
func (f *Func) Params() []*Def { return f.params }

// Blocks returns the function's blocks in declaration order, entry first.
func (f *Func) Blocks() []*Block { return f.blocks }

// Func declares a function definition.
func (m *Module) Func(name string) *Func {
	f := &Func{m: m, name: name, sig: NewSig(), labels: make(map[string]*Block)}
	if m.declare(name, f) {
		m.items = append(m.items, f)
	}
	return f
}

func (f *Func) Export() *Func    { f.linkage = Export; return f }
func (f *Func) Internal() *Func  { f.linkage = Internal; return f }
func (f *Func) Hidden() *Func    { f.vis = Hidden; return f }
func (f *Func) Protected() *Func { f.vis = Protected; return f }
func (f *Func) DLLExport() *Func { f.vis = DLLExport; return f }
func (f *Func) Weak() *Func      { f.weak = true; return f }

// Section sets the section this function is placed in.
func (f *Func) Section(s string) *Func { f.section = s; return f }

// Comdat sets the comdat key. With no key it defaults to the symbol's own name.
func (f *Func) Comdat(key ...string) *Func {
	f.hasComdat = true
	if len(key) > 0 {
		f.comdat = key[0]
	}
	return f
}

func (f *Func) NoUnwind() *Func     { f.nounwind = true; return f }
func (f *Func) ReturnsTwice() *Func { f.returnsTwice = true; return f }
func (f *Func) Naked() *Func        { f.naked = true; return f }
func (f *Func) NoReturn() *Func     { f.noreturn = true; return f }

// Personality names the personality routine. A function containing invoke,
// invokeind, or a pad block declares one (§19.4).
func (f *Func) Personality(p Callee) *Func { f.personality = p; return f }

func (f *Func) Align(n uint64) *Func {
	if !isPow2(n) {
		f.m.failModule(ErrAlign, "@%s align %d", f.name, n)
		return f
	}
	f.align = n
	return f
}

// CallConv sets the convention, overriding the layout block's abi for this
// signature and for nothing else.
func (f *Func) CallConv(c CallConv) *Func { f.sig.conv = c; return f }

// Variadic marks the var-tail.
func (f *Func) Variadic() *Func { f.sig.variadic = true; return f }

func (f *Func) Meta(a ...Attach) *Func { f.meta = append(f.meta, a...); return f }

// param appends a named parameter and its register.
func (f *Func) param(t RegType, name string, attrs []ParamAttr) *Def {
	if f.m.err != nil {
		return nil
	}
	if f.frozen {
		after := "the entry block"
		if f.hasAsmBody {
			after = "the asm body"
		}
		f.m.fail(f.name, "", Op{}, ErrFrozen, "parameter %%%s after %s", name, after)
		return nil
	}
	for _, a := range attrs {
		if a.IsSRet() && len(f.sig.params) != 0 {
			f.m.fail(f.name, "", Op{}, ErrSRet, "sret on parameter %d", len(f.sig.params))
			return nil
		}
		if (a.kind == paByVal || a.IsSRet()) && t != TypePtr {
			f.m.fail(f.name, "", Op{}, ErrType, "%s on a %s parameter", a, t)
			return nil
		}
	}
	if !f.m.layout.Admits(t) {
		f.m.fail(f.name, "", Op{}, ErrLayout, "parameter of type %s", t)
		return nil
	}
	f.sig.params = append(f.sig.params, Param{Name: name, Type: t, Attrs: attrs})
	d := f.newDef(t, name, nil, len(f.params))
	d.isParam = true
	f.params = append(f.params, d)
	return d
}

func (f *Func) ParamI1(name string, a ...ParamAttr) I1   { return I1{f.param(TypeI1, name, a)} }
func (f *Func) ParamI32(name string, a ...ParamAttr) I32 { return I32{f.param(TypeI32, name, a)} }
func (f *Func) ParamI64(name string, a ...ParamAttr) I64 { return I64{f.param(TypeI64, name, a)} }
func (f *Func) ParamI128(name string, a ...ParamAttr) I128 {
	return I128{f.param(TypeI128, name, a)}
}
func (f *Func) ParamF32(name string, a ...ParamAttr) F32 { return F32{f.param(TypeF32, name, a)} }
func (f *Func) ParamF64(name string, a ...ParamAttr) F64 { return F64{f.param(TypeF64, name, a)} }
func (f *Func) ParamF80(name string, a ...ParamAttr) F80 { return F80{f.param(TypeF80, name, a)} }
func (f *Func) ParamF128(name string, a ...ParamAttr) F128 {
	return F128{f.param(TypeF128, name, a)}
}
func (f *Func) ParamV128(name string, a ...ParamAttr) V128 {
	return V128{f.param(TypeV128, name, a)}
}
func (f *Func) ParamPtr(name string, a ...ParamAttr) Ptr { return Ptr{f.param(TypePtr, name, a)} }

// ParamOf declares a parameter whose register type is worked out rather
// than written down, which is what a pass copying one signature into
// another has. The typed ParamI64 and its siblings are the ordinary way
// to declare one.
func (f *Func) ParamOf(t RegType, name string, a ...ParamAttr) Value {
	return Wrap(f.param(t, name, a))
}

// ret appends a ret-item. Called twice, it declares a multi-value result.
func (f *Func) ret(t RegType, attrs []ParamAttr) *Func {
	if f.m.err != nil {
		return f
	}
	if f.frozen {
		f.m.fail(f.name, "", Op{}, ErrFrozen, "result after the entry block")
		return f
	}
	for _, a := range attrs {
		// swifterror names the result that leaves in the error
		// register, as Sig.Ret says for an import.
		if a.kind == paSwiftError && t == TypePtr {
			continue
		}
		if a.kind != paZExt && a.kind != paSExt {
			f.m.fail(f.name, "", Op{}, ErrPlacement, "%s on a result", a)
			return f
		}
	}
	if !f.m.layout.Admits(t) {
		f.m.fail(f.name, "", Op{}, ErrLayout, "result of type %s", t)
		return f
	}
	f.sig.rets = append(f.sig.rets, RetItem{Type: t, Attrs: attrs})
	return f
}

func (f *Func) ReturnsI1(a ...ParamAttr) *Func   { return f.ret(TypeI1, a) }
func (f *Func) ReturnsI32(a ...ParamAttr) *Func  { return f.ret(TypeI32, a) }
func (f *Func) ReturnsI64(a ...ParamAttr) *Func  { return f.ret(TypeI64, a) }
func (f *Func) ReturnsI128(a ...ParamAttr) *Func { return f.ret(TypeI128, a) }
func (f *Func) ReturnsF32(a ...ParamAttr) *Func  { return f.ret(TypeF32, a) }
func (f *Func) ReturnsF64(a ...ParamAttr) *Func  { return f.ret(TypeF64, a) }
func (f *Func) ReturnsF80(a ...ParamAttr) *Func  { return f.ret(TypeF80, a) }
func (f *Func) ReturnsF128(a ...ParamAttr) *Func { return f.ret(TypeF128, a) }
func (f *Func) ReturnsV128(a ...ParamAttr) *Func { return f.ret(TypeV128, a) }
func (f *Func) ReturnsPtr(a ...ParamAttr) *Func  { return f.ret(TypePtr, a) }

// A FuncImport is a reference to a function another module defines.
type FuncImport struct {
	m            *Module
	name         string
	sig          *Sig
	vis          Visibility
	weak         bool
	nounwind     bool
	returnsTwice bool
	noreturn     bool
	meta         []Attach
}

func (f *FuncImport) ItemKind() ItemKind     { return ItemFuncImport }
func (f *FuncImport) Name() string           { return f.name }
func (f *FuncImport) SymbolKind() SymbolKind { return SymFunc }
func (f *FuncImport) Signature() *Sig        { return f.sig }
func (f *FuncImport) Visibility() Visibility { return f.vis }
func (f *FuncImport) IsWeak() bool           { return f.weak }
func (f *FuncImport) IsNoUnwind() bool       { return f.nounwind }
func (f *FuncImport) IsReturnsTwice() bool   { return f.returnsTwice }
func (f *FuncImport) IsNoReturn() bool       { return f.noreturn }
func (f *FuncImport) Attached() []Attach     { return f.meta }

// ImportFunc declares a reference to a function another module defines. An
// import need not name its parameter registers, having no body to reference
// them.
func (m *Module) ImportFunc(name string, sig *Sig) *FuncImport {
	if sig == nil {
		sig = NewSig()
	}
	f := &FuncImport{m: m, name: name, sig: sig}
	if m.declare(name, f) {
		m.items = append(m.items, f)
	}
	return f
}

func (f *FuncImport) Hidden() *FuncImport    { f.vis = Hidden; return f }
func (f *FuncImport) Protected() *FuncImport { f.vis = Protected; return f }
func (f *FuncImport) DLLImport() *FuncImport { f.vis = DLLImport; return f }

// Weak makes the reference one that may go unresolved.
func (f *FuncImport) Weak() *FuncImport { f.weak = true; return f }

func (f *FuncImport) NoUnwind() *FuncImport     { f.nounwind = true; return f }
func (f *FuncImport) ReturnsTwice() *FuncImport { f.returnsTwice = true; return f }
func (f *FuncImport) NoReturn() *FuncImport     { f.noreturn = true; return f }

func (f *FuncImport) Meta(a ...Attach) *FuncImport { f.meta = append(f.meta, a...); return f }
