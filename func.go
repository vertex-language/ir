// func.go
package ir

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
)

var callConvText = [...]string{
	CCC: "ccc", FastCC: "fastcc", PreserveMost: "preserve_most",
	PreserveAll: "preserve_all", StdCall: "stdcall", FastCall: "fastcall",
	ThisCall: "thiscall", VectorCall: "vectorcall", MSABI: "ms_abi",
	SysVABI: "sysv_abi",
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
	paZExt
	paSExt
	paNoAlias
)

// A ParamAttr is a param-attr or a ret-attr (§6). ZExt and SExt are the two a
// ret-item admits.
type ParamAttr struct {
	kind paKind
	typ  *Type
}

var (
	ZExt    = ParamAttr{kind: paZExt}
	SExt    = ParamAttr{kind: paSExt}
	NoAlias = ParamAttr{kind: paNoAlias}
)

// ByVal passes the aggregate the pointer names by value.
func ByVal(t *Type) ParamAttr { return ParamAttr{kind: paByVal, typ: t} }

// SRet names the aggregate the callee writes its result into. §19.13 admits it
// on at most one parameter, which is the first.
func SRet(t *Type) ParamAttr { return ParamAttr{kind: paSRet, typ: t} }

func (a ParamAttr) IsByVal() bool   { return a.kind == paByVal }
func (a ParamAttr) IsSRet() bool    { return a.kind == paSRet }
func (a ParamAttr) IsZExt() bool    { return a.kind == paZExt }
func (a ParamAttr) IsSExt() bool    { return a.kind == paSExt }
func (a ParamAttr) IsNoAlias() bool { return a.kind == paNoAlias }
func (a ParamAttr) Type() *Type     { return a.typ }

func (a ParamAttr) String() string {
	switch a.kind {
	case paByVal:
		return "byval"
	case paSRet:
		return "sret"
	case paZExt:
		return "zext"
	case paSExt:
		return "sext"
	case paNoAlias:
		return "noalias"
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
		if a.kind == paSRet && len(f.sig.params) != 0 {
			f.m.fail(f.name, "", Op{}, ErrSRet, "sret on parameter %d", len(f.sig.params))
			return nil
		}
		if (a.kind == paByVal || a.kind == paSRet) && t != TypePtr {
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
