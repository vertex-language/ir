package ptx

import (
	"fmt"

	"github.com/vertex-language/ir"
	"github.com/vertex-language/ir/lower/globals"
	"github.com/vertex-language/ptx"
)

// linkage is a function's or global's PTX linkage from its VIR linkage
// and binding. A definition that states none is visible, on the terms
// globals.FuncBinding gives: a function is a module's interface unless
// ir.Internal says otherwise.
func funcLinkage(f *ir.Func) ptx.Linkage {
	switch globals.FuncBinding(f) {
	case globals.Weak:
		return ptx.Weak
	case globals.Local:
		return ptx.Default
	}
	return ptx.Visible
}

// declareImport declares a function another module defines: .extern .func
// with no body. A kernel import is §19.20's to refuse and has been.
func (l *lowerer) declareImport(f *ir.FuncImport) error {
	pf, err := l.newFunc(f.Name(), f.Signature())
	if err != nil {
		return fmt.Errorf("lower: import @%s: %w", f.Name(), err)
	}
	pf.Linkage = ptx.Extern
	pf.Body = nil
	pf.NoReturn = f.IsNoReturn()
	l.funcs[f] = pf
	l.pm.Add(pf)
	return nil
}

// declareProto declares a function's prototype, for a body that names it
// before its definition.
func (l *lowerer) declareProto(f *ir.Func) error {
	pf, err := l.newFunc(f.Name(), f.Signature())
	if err != nil {
		return fmt.Errorf("lower: @%s: %w", f.Name(), err)
	}
	pf.Linkage = funcLinkage(f)
	pf.Body = nil
	pf.NoReturn = f.IsNoReturn()
	l.pm.Add(pf)
	return nil
}

// declareFunc declares a definition ahead of its body, so that bodies may
// name one another in any order.
func (l *lowerer) declareFunc(f *ir.Func) error {
	if _, ok := f.AsmBodyText(); ok {
		return fmt.Errorf("lower: @%s: a function whose body is assembly is not emitted yet", f.Name())
	}
	if f.Signature().CallConv() == ir.Kernel {
		k := ptx.NewKernel(symName(f.Name()))
		k.Linkage = funcLinkage(f)
		for i, p := range f.Signature().Params() {
			k.Param(paramName(p, i), paramType(p.Type))
		}
		l.kernels[f] = k
		l.pm.Add(k)
		return nil
	}
	pf, err := l.newFunc(f.Name(), f.Signature())
	if err != nil {
		return fmt.Errorf("lower: @%s: %w", f.Name(), err)
	}
	pf.Linkage = funcLinkage(f)
	pf.NoReturn = f.IsNoReturn()
	l.funcs[f] = pf
	l.pm.Add(pf)
	return nil
}

// retShape is how a device function's results travel: one .param of
// the result's own type, or — for more than one, which ptxas refuses as
// separate .param results and accepts as .reg only outside the ABI that
// indirect calls need — one .param byte array holding every result at its
// natural offset, which is also how nvcc returns a struct.
type retShape struct {
	single bool
	typ    ptx.Type // single
	size   int      // array
	align  int
	offs   []int64
}

func retShapeOf(sig *ir.Sig) retShape {
	rets := sig.Rets()
	if len(rets) == 1 {
		return retShape{single: true, typ: paramType(rets[0].Type)}
	}
	var sh retShape
	sh.align = 1
	var off int64
	for _, r := range rets {
		w := int64(paramType(r.Type).Bits() / 8)
		off = int64(alignUp(uint64(off), uint64(w)))
		sh.offs = append(sh.offs, off)
		off += w
		if int(w) > sh.align {
			sh.align = int(w)
		}
	}
	sh.size = int(alignUp(uint64(off), uint64(sh.align)))
	return sh
}

// newFunc is a .func with the signature's parameters and results as .param
// declarations: the ABI nvcc uses and ptxas requires for an indirect call.
// The convention is ccc, which is the device convention; a signature
// naming a CPU convention is a module built for a CPU.
func (l *lowerer) newFunc(name string, sig *ir.Sig) (*ptx.Func, error) {
	switch sig.CallConv() {
	case ir.CCC:
	case ir.Kernel:
		return nil, fmt.Errorf("a kernel where a function was expected")
	default:
		return nil, fmt.Errorf("callconv %s is a CPU convention", sig.CallConv())
	}
	if sig.IsVariadic() {
		return nil, fmt.Errorf("variadic; PTX has no va_list")
	}
	for i, p := range sig.Params() {
		for _, a := range p.Attrs {
			if a.IsByVal() || a.IsSRet() {
				return nil, fmt.Errorf("parameter %d carries %s, which is not lowered yet", i, a)
			}
		}
	}
	pf := ptx.NewFunc(symName(name))
	if sh := retShapeOf(sig); sh.single {
		pf.Return("_r", sh.typ)
	} else if len(sig.Rets()) > 1 {
		pf.Ret = append(pf.Ret, &ptx.Param{Name: "_r", Type: ptx.B8, Align: sh.align, Len: sh.size})
	}
	for i, p := range sig.Params() {
		pf.Param(paramName(p, i), paramType(p.Type))
	}
	return pf, nil
}

func paramName(p ir.Param, i int) string {
	if p.Name != "" {
		return "_p_" + symName(p.Name)
	}
	return "_p" + itoa(i)
}

// proto is the .callprototype for a func typedef, declared in the body
// ahead of the first callind that names it — a prototype is a labelled
// directive and PTX reads top to bottom.
func (x *fn) proto(t *ir.Type) (*ptx.Proto, error) {
	if t == nil {
		return nil, fmt.Errorf("callind names no type")
	}
	if p, ok := x.protos[t]; ok {
		return p, nil
	}
	sig := t.Sig()
	if sig == nil {
		return nil, fmt.Errorf("@%s is not a func typedef", t.Name())
	}
	if sig.CallConv() != ir.CCC {
		return nil, fmt.Errorf("@%s names callconv %s, a CPU convention", t.Name(), sig.CallConv())
	}
	if sig.IsVariadic() {
		return nil, fmt.Errorf("@%s is variadic; PTX has no va_list", t.Name())
	}
	var rets, params []ptx.Type
	for _, r := range sig.Rets() {
		rets = append(rets, paramType(r.Type))
	}
	for _, p := range sig.Params() {
		params = append(params, paramType(p.Type))
	}
	if len(rets) > 1 {
		sh := retShapeOf(sig)
		rets = []ptx.Type{ptx.B8}
		p := x.b.CallProto(rets, params)
		p.RetArray = [2]int{sh.align, sh.size}
		x.protos[t] = p
		return p, nil
	}
	p := x.b.CallProto(rets, params)
	x.protos[t] = p
	return p, nil
}

// fn is one function's lowering state.
type fn struct {
	l *lowerer
	f *ir.Func

	kernel *ptx.Kernel
	pf     *ptx.Func
	body   *ptx.Body // the function's body; b is where emission currently goes
	b      *ptx.Body

	regs   map[*ir.Def]ptx.Reg
	labels map[*ir.Block]*ptx.Label
	protos map[*ir.Type]*ptx.Proto

	// The local depot: one .local array per function holding every
	// ptr.alloc, addressed as a generic pointer plus a constant offset.
	depot   ptx.Reg
	allocAt map[*ir.Inst]int64
}

func (l *lowerer) lowerFunc(f *ir.Func) error {
	x := &fn{
		l:       l,
		f:       f,
		regs:    map[*ir.Def]ptx.Reg{},
		labels:  map[*ir.Block]*ptx.Label{},
		protos:  map[*ir.Type]*ptx.Proto{},
		allocAt: map[*ir.Inst]int64{},
	}
	if k, ok := l.kernels[f]; ok {
		x.kernel = k
		x.body = k.Body
	} else {
		x.pf = l.funcs[f]
		x.body = x.pf.Body
	}
	x.b = x.body
	if err := x.lower(); err != nil {
		return fmt.Errorf("lower: @%s: %w", f.Name(), err)
	}
	return nil
}

func (x *fn) lower() error {
	for _, blk := range x.f.Blocks() {
		if blk.IsPad() {
			return fmt.Errorf("@%s is a pad block; there is no unwinding on the device", blk.Label())
		}
		x.labels[blk] = x.body.Label("$L_" + symName(blk.Label()))
	}
	if err := x.prologue(); err != nil {
		return err
	}
	for _, blk := range x.f.Blocks() {
		if err := x.block(blk); err != nil {
			return err
		}
	}
	return nil
}

// prologue loads the parameters out of .param space and lays out the
// depot, at the top of the entry block — which is blocks[0] and the target
// of no edge, so this is emitted exactly once. An i1 arrives as a .b32
// and becomes a predicate here.
func (x *fn) prologue() error {
	var params []*ptx.Param
	if x.kernel != nil {
		params = x.kernel.Params
	} else {
		params = x.pf.Params
	}
	for i, d := range x.f.Params() {
		p := params[i]
		r := x.v(d)
		if d.Type() == ir.TypeI1 {
			t := x.body.Regs.New(ptx.B32)
			x.body.Ld(ptx.B32, t, ptx.At(p), ptx.ParamSpace)
			x.body.Setp(ptx.B32, ptx.Ne, r, t, ptx.Imm(0))
			continue
		}
		x.body.Ld(memT(d.Type()), r, ptx.At(p), ptx.ParamSpace)
	}
	return x.layoutDepot()
}

// layoutDepot sizes the local depot from the entry block's allocations,
// which §19.6 confines there, and gives each its offset.
func (x *fn) layoutDepot() error {
	var size, align uint64 = 0, 1
	entry := x.f.Blocks()[0]
	for _, in := range entry.Insts() {
		if in.Op().Verb != ir.VAlloc {
			continue
		}
		s, a, err := allocShape(in)
		if err != nil {
			return err
		}
		size = alignUp(size, a)
		x.allocAt[in] = int64(size)
		size += s
		if a > align {
			align = a
		}
	}
	if size == 0 {
		return nil
	}
	if align < 16 {
		align = 16
	}
	v := x.body.Local(ptx.Var{
		Space: ptx.Local, Align: int(align), Type: ptx.B8,
		Name: "__local_depot", Len: int(alignUp(size, align)),
	})
	raw := x.body.Regs.New(ptx.B64)
	x.depot = x.body.Regs.New(ptx.B64)
	x.body.Mov(ptx.U64, raw, v)
	x.body.Cvta(ptx.U64, x.depot, raw, ptx.Local)
	return nil
}

// allocShape is a ptr.alloc's size and alignment, from the literal form or
// from the named type.
func allocShape(in *ir.Inst) (size, align uint64, err error) {
	if t := in.NamedType(); t != nil {
		size, align, err = sizeAlign(t.FType())
		if err != nil {
			return 0, 0, fmt.Errorf("%s: %w", in.Op(), err)
		}
	} else {
		size = in.Size()
		align, _ = in.Align()
		if align == 0 {
			align = 1
		}
	}
	if size == 0 {
		size = 1
	}
	return size, align, nil
}

// v is the register a value lives in, allocated at first mention.
func (x *fn) v(d *ir.Def) ptx.Reg {
	if r, ok := x.regs[d]; ok {
		return r
	}
	r := x.body.Regs.New(regType(d.Type()))
	x.regs[d] = r
	return r
}

// temp is a fresh register of a PTX type.
func (x *fn) temp(t ptx.Type) ptx.Reg { return x.body.Regs.New(t) }

// arg is the i'th register operand of an instruction.
func (x *fn) arg(in *ir.Inst, i int) ptx.Reg { return x.v(in.Arg(i)) }

// res is an instruction's single result register.
func (x *fn) res(in *ir.Inst) ptx.Reg { return x.v(in.Result(0)) }

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
