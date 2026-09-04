package text

import (
	"bytes"
	"fmt"
	"io"
	"math"
	"sort"
	"strconv"
	"strings"

	"github.com/vertex-language/ir"
)

// A Printer writes a module as .vir text.
//
// The zero Printer is not the form Format produces: Indent falls back to two
// spaces, but Metadata must be set for !nodes to appear. NewPrinter returns the
// round-trip form, which is what Print and Format use.
type Printer struct {
	Indent     string
	NameTemps  bool // %0.. vs. keeping builder-supplied names
	Metadata   bool // omit !nodes for a readable diff
	SortModule bool // canonical item order, for golden files
}

// NewPrinter returns the printer Print and Format use: two-space indent,
// builder-supplied names kept, metadata printed, declaration order preserved.
func NewPrinter() *Printer { return &Printer{Indent: "  ", Metadata: true} }

// Print writes m to w.
func Print(w io.Writer, m *ir.Module) error { return NewPrinter().Print(w, m) }

// Format returns m as .vir text.
func Format(m *ir.Module) ([]byte, error) { return NewPrinter().Format(m) }

// Print writes m to w under p's options.
func (p *Printer) Print(w io.Writer, m *ir.Module) error {
	b, err := p.Format(m)
	if err != nil {
		return err
	}
	_, err = w.Write(b)
	return err
}

// Format returns m as .vir text under p's options. A module carrying a builder
// error is not printed: there is nothing faithful to print.
func (p *Printer) Format(m *ir.Module) ([]byte, error) {
	if err := m.Err(); err != nil {
		return nil, err
	}
	pr := &printer{opt: p, indent: p.Indent}
	if pr.indent == "" {
		pr.indent = "  "
	}
	pr.module(m)
	return pr.buf.Bytes(), nil
}

type printer struct {
	opt    *Printer
	indent string
	buf    bytes.Buffer
	names  map[*ir.Def]string
}

func (pr *printer) s(x string)                { pr.buf.WriteString(x) }
func (pr *printer) f(format string, a ...any) { fmt.Fprintf(&pr.buf, format, a...) }
func (pr *printer) nl()                       { pr.buf.WriteByte('\n') }

// —— module ——

var layoutKeys = []string{"abi", "endian", "ptrbits", "stackalign", "extfloat"}

func (pr *printer) module(m *ir.Module) {
	pr.f("module %s\n\nuse %s\n\nlayout {\n", m.Name(), strconv.Quote(m.Use()))
	l := m.Layout()
	vals := []string{
		l.ABI,
		l.Endian.String(),
		strconv.FormatUint(uint64(l.PtrBits), 10),
		strconv.FormatUint(uint64(l.StackAlign), 10),
		extFloatList(l),
	}
	for i, k := range layoutKeys {
		pr.f("%s%-10s %s,\n", pr.indent, k, vals[i])
	}
	pr.s("}\n")

	for _, it := range pr.items(m) {
		pr.nl()
		pr.item(it)
	}
}

func extFloatList(l ir.Layout) string {
	if len(l.ExtFloat) == 0 {
		return "none"
	}
	parts := make([]string, len(l.ExtFloat))
	for i, t := range l.ExtFloat {
		parts[i] = t.String()
	}
	return strings.Join(parts, ", ")
}

func (pr *printer) items(m *ir.Module) []ir.Item {
	items := m.Items()
	if !pr.opt.SortModule {
		return items
	}
	out := make([]ir.Item, len(items))
	copy(out, items)
	sort.SliceStable(out, func(i, j int) bool {
		ki, kj := out[i].ItemKind(), out[j].ItemKind()
		if ki != kj {
			return ki < kj
		}
		return itemName(out[i]) < itemName(out[j])
	})
	return out
}

func itemName(it ir.Item) string {
	switch x := it.(type) {
	case *ir.Type:
		return x.Name()
	case *ir.Global:
		return x.Name()
	case *ir.GlobalImport:
		return x.Name()
	case *ir.Func:
		return x.Name()
	case *ir.FuncImport:
		return x.Name()
	case *ir.Alias:
		return x.Name()
	case *ir.MetaDecl:
		return x.Name()
	}
	return ""
}

func (pr *printer) item(it ir.Item) {
	switch x := it.(type) {
	case *ir.Type:
		pr.typeDecl(x)
	case *ir.Global:
		pr.global(x)
	case *ir.GlobalImport:
		pr.globalImport(x)
	case *ir.FuncImport:
		pr.funcImport(x)
	case *ir.Func:
		pr.fn(x)
	case *ir.Alias:
		pr.alias(x)
	case *ir.MetaDecl:
		pr.metaDecl(x)
	case *ir.ModuleAsm:
		pr.moduleAsm(x)
	}
}

// moduleAsm prints a module-scope assembly block. It has no name, so under
// SortModule every one of them compares equal and the stable sort leaves them
// in declaration order — which is the order they have to keep, since it is
// the order their bytes land in.
func (pr *printer) moduleAsm(a *ir.ModuleAsm) {
	pr.f("asm %s\n", strconv.Quote(a.Text()))
}

// —— declarations ——

func (pr *printer) typeDecl(t *ir.Type) {
	pr.mod(t.Linkage().String())
	pr.f("type @%s ", t.Name())
	switch t.Kind() {
	case ir.KindStruct, ir.KindUnion:
		if t.IsPacked() {
			pr.s("packed ")
		} else if n := t.AlignAttr(); n != 0 {
			pr.f("align %d ", n)
		}
		if t.Kind() == ir.KindStruct {
			pr.s("struct {")
		} else {
			pr.s("union {")
		}
		fs := t.Fields()
		if len(fs) == 0 {
			pr.s("}")
			break
		}
		pr.nl()
		for i, f := range fs {
			pr.f("%s%s %s", pr.indent, f.Name, f.Type.String())
			if f.HasOffset {
				pr.f(" at %d", f.Offset)
			}
			if i < len(fs)-1 {
				pr.s(",")
			}
			pr.nl()
		}
		pr.s("}")
	case ir.KindFunc:
		pr.s("func ")
		pr.absSig(t.Sig())
	default:
		pr.s(t.Aliased().String())
	}
	pr.attached(t.Attached())
	pr.nl()
}

func (pr *printer) global(g *ir.Global) {
	pr.mod(g.Linkage().String(), g.Visibility().String(), g.Binding().String())
	pr.f("global %s @%s %s", g.Domain(), g.Name(), g.Type().String())
	pr.globalPlacements(g.SectionAttr(), g.ComdatAttr, g.AlignAttr(), g.TLSModelAttr())
	pr.s(" = ")
	pr.init(g.Initializer())
	pr.attached(g.Attached())
	pr.nl()
}

func (pr *printer) globalImport(g *ir.GlobalImport) {
	pr.mod(g.Visibility().String(), weakMod(g.IsWeak()))
	pr.f("import global @%s %s", g.Name(), g.Type().String())
	pr.globalPlacements(g.SectionAttr(), g.ComdatAttr, g.AlignAttr(), g.TLSModelAttr())
	pr.attached(g.Attached())
	pr.nl()
}

func (pr *printer) globalPlacements(section string, comdat func() (string, bool), align uint64, tls ir.TLSModel) {
	if section != "" {
		pr.f(" section %s", strconv.Quote(section))
	}
	if key, ok := comdat(); ok {
		pr.s(" comdat")
		if key != "" {
			pr.f(" %s", strconv.Quote(key))
		}
	}
	if align != 0 {
		pr.f(" align %d", align)
	}
	if tls != ir.NoTLSModel {
		pr.f(" tlsmodel %s", tls)
	}
}

func (pr *printer) funcImport(f *ir.FuncImport) {
	pr.mod(f.Visibility().String(), weakMod(f.IsWeak()))
	pr.f("import func @%s", f.Name())
	pr.absSig(f.Signature())
	if f.IsNoUnwind() {
		pr.s(" nounwind")
	}
	if f.IsReturnsTwice() {
		pr.s(" returns_twice")
	}
	if f.IsNoReturn() {
		pr.s(" noreturn")
	}
	pr.attached(f.Attached())
	pr.nl()
}

func (pr *printer) alias(a *ir.Alias) {
	pr.mod(a.Linkage().String(), a.Visibility().String(), weakMod(a.IsWeak()))
	kind := "global"
	if a.SymbolKind() == ir.SymFunc {
		kind = "func"
	}
	pr.f("alias %s @%s @%s\n", kind, a.Name(), a.Target().Name())
}

func (pr *printer) metaDecl(d *ir.MetaDecl) {
	if !pr.opt.Metadata {
		return
	}
	pr.f("!%s = {", d.Name())
	for i, a := range d.Args() {
		if i > 0 {
			pr.s(",")
		}
		pr.s(" ")
		pr.metaArg(a)
	}
	if len(d.Args()) > 0 {
		pr.s(" ")
	}
	pr.s("}\n")
}

func (pr *printer) mod(words ...string) {
	for _, w := range words {
		if w != "" {
			pr.s(w + " ")
		}
	}
}

func weakMod(w bool) string {
	if w {
		return "weak"
	}
	return ""
}

// —— signatures ——

func (pr *printer) absSig(sig *ir.Sig) {
	pr.callconv(sig)
	pr.s("(")
	for i, p := range sig.Params() {
		if i > 0 {
			pr.s(", ")
		}
		pr.s(p.Type.String())
		pr.paramAttrs(p.Attrs)
	}
	pr.varTail(sig)
	pr.s(")")
	pr.ret(sig)
}

func (pr *printer) defSig(f *ir.Func) {
	sig := f.Signature()
	pr.callconv(sig)
	pr.s("(")
	for i, d := range f.Params() {
		if i > 0 {
			pr.s(", ")
		}
		pr.f("%%%s %s", pr.reg(d), d.Type())
		pr.paramAttrs(sig.Params()[i].Attrs)
	}
	pr.varTail(sig)
	pr.s(")")
	pr.ret(sig)
}

func (pr *printer) callconv(sig *ir.Sig) {
	if c := sig.CallConv(); c != ir.CCC {
		pr.s(c.String())
	}
}

func (pr *printer) varTail(sig *ir.Sig) {
	if !sig.IsVariadic() {
		return
	}
	if len(sig.Params()) > 0 {
		pr.s(", ")
	}
	pr.s("...")
}

func (pr *printer) ret(sig *ir.Sig) {
	rs := sig.Rets()
	switch len(rs) {
	case 0:
	case 1:
		pr.s(" " + rs[0].Type.String())
		pr.paramAttrs(rs[0].Attrs)
	default:
		pr.s(" (")
		for i, r := range rs {
			if i > 0 {
				pr.s(", ")
			}
			pr.s(r.Type.String())
			pr.paramAttrs(r.Attrs)
		}
		pr.s(")")
	}
}

func (pr *printer) paramAttrs(as []ir.ParamAttr) {
	for _, a := range as {
		pr.s(" " + a.String())
		if t := a.Type(); t != nil {
			pr.f(" @%s", t.Name())
		}
	}
}

// —— functions ——

func (pr *printer) fn(f *ir.Func) {
	pr.assignNames(f)
	pr.mod(f.Linkage().String(), f.Visibility().String(), weakMod(f.IsWeak()))
	pr.f("func @%s", f.Name())
	pr.defSig(f)

	if s := f.SectionAttr(); s != "" {
		pr.f(" section %s", strconv.Quote(s))
	}
	if key, ok := f.ComdatAttr(); ok {
		pr.s(" comdat")
		if key != "" {
			pr.f(" %s", strconv.Quote(key))
		}
	}
	if n := f.AlignAttr(); n != 0 {
		pr.f(" align %d", n)
	}
	if f.IsNoUnwind() {
		pr.s(" nounwind")
	}
	if p := f.PersonalityFn(); p != nil {
		pr.f(" personality @%s", p.Name())
	}
	if f.IsReturnsTwice() {
		pr.s(" returns_twice")
	}
	if f.IsNaked() {
		pr.s(" naked")
	}
	if f.IsNoReturn() {
		pr.s(" noreturn")
	}
	pr.attached(f.Attached())
	pr.s(" {\n")

	if body, ok := f.AsmBodyText(); ok {
		pr.f("%sasm %s\n}\n", pr.indent, strconv.Quote(body))
		return
	}

	for i, b := range f.Blocks() {
		if i > 0 {
			pr.nl()
		}
		pr.block(b)
	}
	pr.s("}\n")
}

// assignNames fixes the printed name of every register in the function.
// Supplied names survive; temporaries get %0.. in emission order, with the
// counter advancing only over the unnamed — unless NameTemps numbers all of
// them, which is what a golden file wants.
func (pr *printer) assignNames(f *ir.Func) {
	pr.names = make(map[*ir.Def]string)
	n := 0
	f.WalkDefs(func(d *ir.Def) bool {
		if !pr.opt.NameTemps && d.Name() != "" {
			pr.names[d] = d.Name()
			return true
		}
		pr.names[d] = strconv.Itoa(n)
		n++
		return true
	})
}

func (pr *printer) reg(d *ir.Def) string {
	if d == nil {
		return "<poison>"
	}
	if s, ok := pr.names[d]; ok {
		return s
	}
	return d.String()
}

func (pr *printer) block(b *ir.Block) {
	pr.f("@%s", b.Label())
	switch {
	case b.IsPad():
		ps := b.Params()
		pr.f(" pad (%%%s ptr, %%%s i32)", pr.reg(ps[0]), pr.reg(ps[1]))
		for _, c := range b.Clauses() {
			pr.padClause(c)
		}
	case len(b.Params()) > 0:
		pr.s("(")
		for i, d := range b.Params() {
			if i > 0 {
				pr.s(", ")
			}
			pr.f("%%%s %s", pr.reg(d), d.Type())
		}
		pr.s(")")
	}
	pr.s(":")
	pr.attached(b.Attached())
	pr.nl()

	for _, in := range b.Insts() {
		pr.s(pr.indent)
		pr.inst(in)
		pr.nl()
	}
	if t := b.Term(); t != nil {
		pr.s(pr.indent)
		pr.inst(t)
		pr.nl()
	}
}

func (pr *printer) padClause(c ir.PadClause) {
	switch c.Kind() {
	case ir.PadCleanup:
		pr.s(" cleanup")
	case ir.PadCatch:
		pr.f(" catch @%s", c.TypeInfo().Name())
	case ir.PadFilter:
		pr.s(" filter [")
		for i, ti := range c.Set() {
			if i > 0 {
				pr.s(", ")
			}
			pr.f("@%s", ti.Name())
		}
		pr.s("]")
	}
}

// —— instructions ——

func (pr *printer) inst(in *ir.Inst) {
	op := in.Op()
	if op.IsBare() {
		switch op.Verb {
		case ir.VCall, ir.VCallInd:
			pr.call(in)
			pr.attached(in.Attached())
			return
		case ir.VAsm, ir.VAsmGoto:
			pr.asm(in)
			pr.attached(in.Attached())
			return
		case ir.VBr, ir.VBrIf, ir.VBrTable, ir.VBrInd, ir.VReturn, ir.VTrap,
			ir.VInvoke, ir.VInvokeInd, ir.VResume:
			pr.term(in)
			pr.attached(in.Attached())
			return
		}
	}

	pr.results(in)
	pr.s(op.String())
	pr.args(in.Args())

	// The immediate suffixes, in the one order the grammar admits them.
	if c, ok := in.Lit(); ok {
		pr.s(" ")
		pr.constant(c)
	}
	switch op.Verb {
	case ir.VGetAddr, ir.VTLSAddr:
		pr.f(" @%s", in.Symbol().Name())
	case ir.VBlockAddr:
		pr.f(" @%s", in.Labels()[0].Label())
	case ir.VAlloc:
		if t := in.NamedType(); t != nil {
			pr.f(" @%s", t.Name())
		} else {
			pr.f(" %d", in.Size())
		}
	case ir.VVaArgRef:
		pr.f(", @%s", in.NamedType().Name())
	}
	if n, ok := in.Align(); ok {
		pr.f(" align %d", n)
	}
	if in.Zeroed() {
		pr.s(" zeroed")
	}
	for _, o := range in.Orderings() {
		pr.f(" %s", o)
	}
	if in.SingleThread() {
		pr.s(" singlethread")
	}
	if in.Volatile() {
		pr.s(" volatile")
	}
	pr.attached(in.Attached())
}

func (pr *printer) results(in *ir.Inst) {
	rs := in.Results()
	if len(rs) == 0 {
		return
	}
	for i, d := range rs {
		if i > 0 {
			pr.s(", ")
		}
		pr.f("%%%s", pr.reg(d))
	}
	pr.s(" = ")
}

func (pr *printer) args(args []*ir.Def) {
	for i, d := range args {
		if i == 0 {
			pr.s(" ")
		} else {
			pr.s(", ")
		}
		pr.f("%%%s", pr.reg(d))
	}
}

func (pr *printer) argList(args []*ir.Def) {
	pr.s("(")
	for i, d := range args {
		if i > 0 {
			pr.s(", ")
		}
		pr.f("%%%s", pr.reg(d))
	}
	pr.s(")")
}

func (pr *printer) call(in *ir.Inst) {
	pr.results(in)
	if in.Op().Verb == ir.VCall {
		pr.f("call @%s", in.Callee().Name())
		pr.argList(in.Args())
		return
	}
	args := in.Args()
	pr.f("callind %%%s : @%s", pr.reg(args[0]), in.NamedType().Name())
	pr.argList(args[1:])
}

func (pr *printer) term(in *ir.Inst) {
	op := in.Op()
	args := in.Args()
	ts := in.Targets()

	switch op.Verb {
	case ir.VBr:
		pr.s("br ")
		pr.target(ts[0])
	case ir.VBrIf:
		pr.f("brif %%%s, ", pr.reg(args[0]))
		pr.target(ts[0])
		pr.s(", ")
		pr.target(ts[1])
	case ir.VBrTable:
		pr.f("br_table %%%s, [", pr.reg(args[0]))
		for i, t := range ts[:len(ts)-1] {
			if i > 0 {
				pr.s(", ")
			}
			pr.target(t)
		}
		pr.s("], ")
		pr.target(ts[len(ts)-1])
	case ir.VBrInd:
		pr.f("brind %%%s, [", pr.reg(args[0]))
		for i, l := range in.Labels() {
			if i > 0 {
				pr.s(", ")
			}
			pr.f("@%s", l.Label())
		}
		pr.s("]")
	case ir.VReturn:
		pr.s("return")
		pr.args(args)
	case ir.VTrap:
		pr.s("trap")
	case ir.VResume:
		pr.f("resume %%%s", pr.reg(args[0]))
	case ir.VInvoke:
		pr.f("invoke @%s", in.Callee().Name())
		pr.argList(args)
		pr.s(" to ")
		pr.target(ts[0])
		pr.f(" unwind @%s", in.Unwind().Label())
	case ir.VInvokeInd:
		pr.f("invokeind %%%s : @%s", pr.reg(args[0]), in.NamedType().Name())
		pr.argList(args[1:])
		pr.s(" to ")
		pr.target(ts[0])
		pr.f(" unwind @%s", in.Unwind().Label())
	}
}

func (pr *printer) target(t ir.BlockTarget) {
	pr.f("@%s", t.Block().Label())
	if t.Bare() {
		return
	}
	pr.s("(")
	for i, d := range t.Args() {
		if i > 0 {
			pr.s(", ")
		}
		pr.f("%%%s", pr.reg(d))
	}
	pr.s(")")
}

func (pr *printer) asm(in *ir.Inst) {
	a := in.Asm()
	if in.Op().Verb == ir.VAsmGoto {
		// The outputs print where §8b's do, and for the reader's sake: the
		// template numbers them first, so seeing them first is what makes
		// %0 findable. They carry no register name because they define
		// none — the values are the fallthrough target's trailing
		// parameters, which is where the reader looks for them.
		if outs := a.Outs; len(outs) > 0 {
			pr.s("(")
			for i, o := range outs {
				if i > 0 {
					pr.s(", ")
				}
				pr.f("%s %s", o.Type, constraint(o.Constraint))
			}
			pr.s(") = ")
		}
		pr.f("asm goto %s", strconv.Quote(a.Template))
		pr.asmArgs(a)
		pr.s(" to ")
		pr.target(in.Targets()[0])
		pr.s(", [")
		for i, l := range in.Labels() {
			if i > 0 {
				pr.s(", ")
			}
			pr.f("@%s", l.Label())
		}
		pr.s("]")
		return
	}

	if outs := a.Outs; len(outs) > 0 {
		pr.s("(")
		for i, o := range outs {
			if i > 0 {
				pr.s(", ")
			}
			pr.f("%%%s %s %s", pr.reg(in.Result(i)), o.Type, constraint(o.Constraint))
		}
		pr.s(") = ")
	}
	pr.s("asm ")
	if a.Volatile {
		pr.s("volatile ")
	}
	pr.s(strconv.Quote(a.Template))
	pr.asmArgs(a)
}

func (pr *printer) asmArgs(a *ir.Asm) {
	pr.s(" (")
	for i, x := range a.Args {
		if i > 0 {
			pr.s(", ")
		}
		pr.f("%%%s %s", pr.reg(x.Def), constraint(x.Constraint))
	}
	pr.s(")")
	if len(a.Clobbers) > 0 {
		pr.s(" clobber ")
		for i, c := range a.Clobbers {
			if i > 0 {
				pr.s(", ")
			}
			pr.s(strconv.Quote(c))
		}
	}
}

func constraint(c ir.Constraint) string {
	if c.IsKeyword() {
		return c.String()
	}
	return strconv.Quote(c.String())
}

// —— literals, initializers, metadata ——

func (pr *printer) constant(c ir.Const) {
	switch c.Kind() {
	case ir.ConstInt:
		pr.s(strconv.FormatInt(c.Int(), 10))
	case ir.ConstFloat:
		pr.s(floatLit(c.Float()))
	case ir.ConstSizeOf:
		pr.f("sizeof %s", symconstTarget(c))
	case ir.ConstAlignOf:
		pr.f("alignof %s", symconstTarget(c))
	case ir.ConstBytes:
		pr.s("0x")
		for _, b := range c.Bytes() {
			pr.f("%02x", b)
		}
	case ir.ConstOffsetOf:
		pr.f("offsetof @%s", c.Type().Name())
		for _, e := range c.Path() {
			if e.IsIndex() {
				pr.f("[%d]", e.Index())
			} else {
				pr.f(".%s", e.Name())
			}
		}
	}
}

func symconstTarget(c ir.Const) string {
	if t := c.Type(); t != nil {
		return "@" + t.Name()
	}
	return "@" + c.Symbol().Name()
}

func floatLit(v float64) string {
	switch {
	case math.IsInf(v, 1):
		return "inf"
	case math.IsInf(v, -1):
		return "-inf"
	case math.IsNaN(v):
		return "nan"
	}
	return strconv.FormatFloat(v, 'g', -1, 64)
}

func (pr *printer) init(i ir.Init) {
	switch i.Kind() {
	case ir.InitLiteral:
		pr.constant(i.Const())
	case ir.InitString:
		pr.s(strconv.Quote(i.String_()))
	case ir.InitZeroed:
		pr.s("zeroed")
	case ir.InitRelocKind:
		r := i.Reloc()
		pr.f("@%s", r.Sym.Name())
		if r.Minus != nil {
			pr.f(" - @%s", r.Minus.Name())
		}
		if r.HasAddend {
			pr.s(" + ")
			pr.constant(r.Addend)
		}
	case ir.InitList:
		pr.s("{ ")
		for n, e := range i.Elems() {
			if n > 0 {
				pr.s(", ")
			}
			pr.init(e)
		}
		pr.s(" }")
	case ir.InitFields:
		pr.s("{ ")
		for n, fv := range i.FieldVals() {
			if n > 0 {
				pr.s(", ")
			}
			pr.f("%s = ", fv.Name)
			pr.init(fv.Init)
		}
		pr.s(" }")
	}
}

func (pr *printer) attached(as []ir.Attach) {
	if !pr.opt.Metadata {
		return
	}
	for _, a := range as {
		pr.f(" !%s", a.Name)
		for _, x := range a.Args {
			pr.s(" ")
			pr.metaArg(x)
		}
	}
}

func (pr *printer) metaArg(a ir.MetaArg) {
	switch a.Kind() {
	case ir.MetaInt:
		pr.s(strconv.FormatInt(a.Int(), 10))
	case ir.MetaUint:
		pr.s(strconv.FormatUint(a.Uint(), 10))
	case ir.MetaFloat:
		pr.s(floatLit(a.Float()))
	case ir.MetaString:
		pr.s(strconv.Quote(a.String_()))
	case ir.MetaSymbol:
		pr.f("@%s", a.Symbol().Name())
	case ir.MetaIdent:
		pr.s(a.String_())
	case ir.MetaRefKind:
		pr.f("!%s", a.Ref().Name())
	case ir.MetaNodeKind:
		pr.s("{")
		for i, x := range a.Node() {
			if i > 0 {
				pr.s(",")
			}
			pr.s(" ")
			pr.metaArg(x)
		}
		if len(a.Node()) > 0 {
			pr.s(" ")
		}
		pr.s("}")
	}
}
