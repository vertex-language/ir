package ir

// The bare mnemonics that are not memory or assembly: terminators, calls,
// unwinding, and the variadic verbs. §17's bare set, minus §E's bulk memory
// (mem.go), fence (atomic.go), and asm (asm.go).

// —— §G. Calls ——

// Call names a function definition or import. The calling convention comes from
// the declaration.
func (b *Builder) Call(fn Callee, args ...Value) Results {
	op := Op{TypeNone, VCall}
	if fn == nil {
		b.fail(op, ErrPoison, "no callee named")
		return Results{}
	}
	sig := fn.Signature()
	if !b.checkArgs(op, sig, args) {
		return Results{}
	}
	in := b.emit(op, sig.retTypes(), defsOf(args), &imm{callee: fn, sym: fn})
	return b.results(op, in)
}

// CallInd calls through a pointer. Its named func type carries the convention,
// which is what keeps an indirect call well-typed. A named type that disagrees
// with the callee's actual convention is a program error the IR cannot detect,
// exactly as in C.
func (b *Builder) CallInd(p Ptr, t *Type, args ...Value) Results {
	op := Op{TypeNone, VCallInd}
	sig := b.funcTypeSig(op, t)
	if sig == nil || !b.checkArgs(op, sig, args) {
		return Results{}
	}
	in := b.emit(op, sig.retTypes(), append([]*Def{p.d}, defsOf(args)...), &imm{typ: t})
	return b.results(op, in)
}

func (b *Builder) funcTypeSig(op Op, t *Type) *Sig {
	if t == nil {
		b.fail(op, ErrPoison, "no func type named")
		return nil
	}
	if t.kind != KindFunc {
		b.fail(op, ErrSignature, "@%s is not a func typedef", t.name)
		return nil
	}
	return t.sig
}

func (b *Builder) checkArgs(op Op, sig *Sig, args []Value) bool {
	if sig == nil {
		return false
	}
	np := len(sig.params)
	if len(args) < np || (len(args) > np && !sig.variadic) {
		b.fail(op, ErrArity, "signature takes %d arguments, %d supplied", np, len(args))
		return false
	}
	for i, p := range sig.params {
		d := defOf(args[i])
		if d == nil {
			b.fail(op, ErrPoison, "argument %d is a zero Value", i)
			return false
		}
		if d.typ != p.Type {
			b.fail(op, ErrType, "argument %d is %s, signature says %s", i, d.typ, p.Type)
			return false
		}
	}
	return true
}

func (b *Builder) results(op Op, in *Inst) Results {
	if in == nil {
		return Results{}
	}
	return Results{m: b.mod(), fn: b.blk.fn.name, blk: b.blk.label, op: op, defs: in.results}
}

// —— §G2. Terminators ——

// Br branches unconditionally.
func (b *Builder) Br(t BlockTarget) {
	op := Op{TypeNone, VBr}
	if !b.checkTarget(op, t) {
		return
	}
	b.emit(op, nil, nil, &imm{targets: []BlockTarget{t}})
}

// BrIf branches on an i1. There is no other condition type: a non-i1 condition
// is a Go type error.
func (b *Builder) BrIf(c I1, then, els BlockTarget) {
	op := Op{TypeNone, VBrIf}
	if !b.checkTarget(op, then) || !b.checkTarget(op, els) {
		return
	}
	b.emit(op, nil, []*Def{c.d}, &imm{targets: []BlockTarget{then, els}})
}

// BrTable indexes the table from zero with an i32 selector; an out-of-range
// selector takes the default edge. A switch on a wider or offset type is a
// frontend-emitted subtract and range check — work the frontend already does to
// find the default edge.
func (b *Builder) BrTable(sel I32, cases []BlockTarget, dflt BlockTarget) {
	op := Op{TypeNone, VBrTable}
	for _, t := range cases {
		if !b.checkTarget(op, t) {
			return
		}
	}
	if !b.checkTarget(op, dflt) {
		return
	}
	all := make([]BlockTarget, 0, len(cases)+1)
	all = append(all, cases...)
	all = append(all, dflt)
	b.emit(op, nil, []*Def{sel.d}, &imm{targets: all})
}

// BrInd is computed goto over block addresses. Its targets take no parameters.
func (b *Builder) BrInd(p Ptr, labels ...*Block) {
	op := Op{TypeNone, VBrInd}
	for _, l := range labels {
		if l == nil {
			b.fail(op, ErrPoison, "nil label")
			return
		}
		if l.fn != b.blk.fn {
			panic("ir: brind label @" + l.label + " from another function")
		}
	}
	b.emit(op, nil, []*Def{p.d}, &imm{labels: labels})
}

// Return returns one register per ret-item.
func (b *Builder) Return(vals ...Value) {
	op := Op{TypeNone, VReturn}
	f := b.blk.fn
	if f == nil {
		return
	}
	if len(vals) != len(f.sig.rets) {
		b.fail(op, ErrArity, "signature returns %d values, %d supplied", len(f.sig.rets), len(vals))
		return
	}
	for i, r := range f.sig.rets {
		d := defOf(vals[i])
		if d == nil {
			b.fail(op, ErrPoison, "result %d is a zero Value", i)
			return
		}
		if d.typ != r.Type {
			b.fail(op, ErrType, "result %d is %s, signature says %s", i, d.typ, r.Type)
			return
		}
	}
	b.emit(op, nil, defsOf(vals), nil)
}

// Trap terminates abnormally. A path a frontend believes cannot be taken ends
// here: there is no unreachable, which is undefined behaviour under a friendlier
// name, and this IR has none.
func (b *Builder) Trap() { b.emit(Op{TypeNone, VTrap}, nil, nil, nil) }

// —— §G3. Unwinding ——

// Invoke calls with an unwind edge. It has no result list of its own, and must
// not have one: a register it defined would have to dominate both edges, and on
// the unwind edge the call did not complete. Its results are the trailing
// parameters of the normal target, which is where they are live. §19.16 states
// the arity rule, which the verifier checks.
func (b *Builder) Invoke(fn Callee, args []Value, normal BlockTarget, pad *Block) {
	op := Op{TypeNone, VInvoke}
	if fn == nil {
		b.fail(op, ErrPoison, "no callee named")
		return
	}
	if !b.checkArgs(op, fn.Signature(), args) || !b.checkPad(op, pad) {
		return
	}
	if normal.blk == nil || normal.blk.fn != b.blk.fn {
		b.fail(op, ErrPoison, "invalid normal edge")
		return
	}
	b.emit(op, nil, defsOf(args), &imm{
		callee: fn, sym: fn, targets: []BlockTarget{normal}, unwind: pad,
	})
}

// InvokeInd is Invoke through a pointer and a func typedef.
func (b *Builder) InvokeInd(p Ptr, t *Type, args []Value, normal BlockTarget, pad *Block) {
	op := Op{TypeNone, VInvokeInd}
	sig := b.funcTypeSig(op, t)
	if sig == nil || !b.checkArgs(op, sig, args) || !b.checkPad(op, pad) {
		return
	}
	if normal.blk == nil || normal.blk.fn != b.blk.fn {
		b.fail(op, ErrPoison, "invalid normal edge")
		return
	}
	b.emit(op, nil, append([]*Def{p.d}, defsOf(args)...), &imm{
		typ: t, targets: []BlockTarget{normal}, unwind: pad,
	})
}

func (b *Builder) checkPad(op Op, pad *Block) bool {
	if pad == nil {
		b.fail(op, ErrPoison, "no unwind edge")
		return false
	}
	if pad.fn != b.blk.fn {
		panic("ir: unwind edge to @" + pad.label + " in another function")
	}
	if !pad.isPad {
		b.fail(op, ErrPlacement, "unwind edge names @%s, which is not a pad block", pad.label)
		return false
	}
	return true
}

// Resume takes the exception object parameter of a dominating pad and returns
// control to the unwinder. §19.5 checks the dominance.
func (b *Builder) Resume(exn Ptr) {
	b.emit(Op{TypeNone, VResume}, nil, []*Def{exn.d}, nil)
}

// —— §I. Variadics ——

func (b *Builder) VaStart(ap Ptr) { b.void(Op{TypeNone, VVaStart}, ap.d) }
func (b *Builder) VaEnd(ap Ptr)   { b.void(Op{TypeNone, VVaEnd}, ap.d) }
func (b *Builder) VaCopy(dst, src Ptr) {
	b.void(Op{TypeNone, VVaCopy}, dst.d, src.d)
}

// There is no f32.va_arg and no i8 or i16 form: the default argument promotions
// mean no such argument can be present.
func (n I32NS) VaArg(ap Ptr) I32 { return I32{n.b.def1(Op{TypeI32, VVaArg}, TypeI32, ap.d)} }
func (n I64NS) VaArg(ap Ptr) I64 { return I64{n.b.def1(Op{TypeI64, VVaArg}, TypeI64, ap.d)} }
func (n F64NS) VaArg(ap Ptr) F64 { return F64{n.b.def1(Op{TypeF64, VVaArg}, TypeF64, ap.d)} }
func (n F80NS) VaArg(ap Ptr) F80 { return F80{n.b.def1(Op{TypeF80, VVaArg}, TypeF80, ap.d)} }
func (n F128NS) VaArg(ap Ptr) F128 {
	return F128{n.b.def1(Op{TypeF128, VVaArg}, TypeF128, ap.d)}
}
func (n PtrNS) VaArg(ap Ptr) Ptr { return Ptr{n.b.def1(Op{TypePtr, VVaArg}, TypePtr, ap.d)} }

// VaArgRef advances the list past one argument of the named type and yields its
// address, which is how va_arg(ap, struct S) is expressed. The ABI knowledge it
// requires is the same knowledge byval already demands.
func (n PtrNS) VaArgRef(ap Ptr, t *Type) Ptr {
	op := Op{TypePtr, VVaArgRef}
	if t == nil {
		n.b.fail(op, ErrPoison, "no type named")
		return Ptr{}
	}
	return Ptr{n.b.def1i(op, TypePtr, []*Def{ap.d}, &imm{typ: t})}
}
