package ir

// Legalizing §V5's and §A8's verbs where the processor may not have them.
//
// §0 defines every one of these on every target with v128, and leaves to
// lowering what it is on a processor without the instruction: a call that
// answers the same, in time that does not depend on the operands. This
// pass makes the calls, before instruction selection, as LegalizeHalf
// makes the half floats into i32: a backend runs it with the capabilities
// its target's baseline has, and what reaches its selector afterwards is
// a module in which every verb of a capability is one the function may
// execute -- the baseline's, or one the function Requires (§P).
//
// A v128 operand travels to the routine in memory, through a scratch slot
// of the function's frame: the routine takes a pointer for each v128
// operand and one for its result, then its i32 and i64 operands as they
// are, then the verb's literal, if it has one, as an i32. That is a C
// signature on every target, which a vector argument is not, and the
// routine is the slow path in any case. The routine is vertex_<verb>
// (vertex_v128_aes_enc, vertex_crc32c_u8); the runtime defines them.
//
// i1.cpu_has of a capability the baseline has is 1. Otherwise it is a call
// of vertex_cpu_has(c), which asks the processor the first time and
// answers from what it said after: a program's main is C's own, so there
// is no moment before it the runtime could be sure to run in, and the
// question is asked once per operation rather than once per block.

// CPUHasSymbol is the routine i1.cpu_has calls where the baseline cannot
// answer: i32 (c) -> i32, nonzero where the processor has capability c.
const CPUHasSymbol = "vertex_cpu_has"

// LegalizeCrypto rewrites every §V5 and §A8 instruction a function may not
// execute -- its capability is neither in baseline nor in the function's
// requires -- into a call of the runtime's routine for it, and every
// i1.cpu_has into its answer. It reports whether it changed anything, and
// is safe to run on a module with none of these verbs.
func (m *Module) LegalizeCrypto(baseline Capabilities) bool {
	return m.legalizeCrypto(baseline, allCapabilities)
}

// LegalizeCryptoOn is LegalizeCrypto for a backend that selects
// instructions for the capabilities in lowers and no others: a function's
// requires outside lowers are calls, as if it did not require them, and
// i1.cpu_has of one is 0. A program asks cpu_has to choose a path built on
// the instructions, and on this backend that path would run each of them
// as a call -- slower than the path it guards. x86-64 is such a target: it
// has no CRC-32, SHA-512 or SM3 instruction the IR's verbs are.
func (m *Module) LegalizeCryptoOn(baseline, lowers Capabilities) bool {
	return m.legalizeCrypto(baseline&lowers, lowers)
}

// LegalizeCryptoAll is LegalizeCryptoOn for a backend that lowers none of
// these instructions yet: every one is a call, a function's requires
// notwithstanding, and every i1.cpu_has is 0.
func (m *Module) LegalizeCryptoAll() bool { return m.legalizeCrypto(0, 0) }

// allCapabilities is every capability §P names.
var allCapabilities = func() Capabilities {
	var s Capabilities
	for c := CapAES; c < capCount; c++ {
		s = s.With(c)
	}
	return s
}()

func (m *Module) legalizeCrypto(baseline, lowers Capabilities) bool {
	if m == nil || m.err != nil {
		return false
	}
	changed := false
	for _, it := range m.items {
		if f, ok := it.(*Func); ok && f.legalizeCrypto(baseline, lowers) {
			changed = true
		}
	}
	return changed
}

func (f *Func) legalizeCrypto(baseline, lowers Capabilities) bool {
	if len(f.blocks) == 0 {
		return false
	}
	may := baseline | f.requires&lowers
	p := &cryptoPass{f: f, baseline: baseline, lowers: lowers, repl: map[*Def]*Def{}}
	for _, b := range f.blocks {
		out := make([]*Inst, 0, len(b.insts))
		for _, in := range b.insts {
			for i, a := range in.args {
				if r, ok := p.repl[a]; ok {
					in.args[i] = r
				}
			}
			v := in.op.Verb
			c, gated := v.Capability()
			switch {
			case v == VCpuHas:
			case gated && !may.Has(c):
			default:
				out = append(out, in)
				continue
			}
			// The block's own builder emits the replacement, so that it
			// is checked as anything built is; the block is opened for
			// it and closed again after.
			term := b.term
			b.insts, b.term = out, nil
			d := p.replace(b, in)
			out, b.term = b.insts, term
			if d == nil {
				return p.changed
			}
			p.repl[in.results[0]] = d
			p.changed = true
		}
		b.insts = out
	}
	if len(p.repl) > 0 {
		f.WalkUses(func(u Use) bool {
			if d := u.Def(); d != nil {
				if r, ok := p.repl[d]; ok {
					u.Set(r)
				}
			}
			return true
		})
	}
	return p.changed
}

type cryptoPass struct {
	f        *Func
	baseline Capabilities
	lowers   Capabilities // what the backend selects; cpu_has of the rest is 0
	repl     map[*Def]*Def
	changed  bool
	scratch  *Def // the frame slot v128 operands travel in, once made
}

// slot is the scratch slot: room for a result and three operands,
// allocated once at the head of the entry block.
func (p *cryptoPass) slot() Ptr {
	if p.scratch != nil {
		return Ptr{p.scratch}
	}
	e := p.f.blocks[0]
	insts, term := e.insts, e.term
	e.insts, e.term = nil, nil
	s := e.Ptr.Alloc(64, 16)
	made := e.insts
	e.insts, e.term = append(made, insts...), term
	// The block being rewritten may be the entry block, whose instruction
	// list the caller holds; it sees the slot through b.insts below.
	if p.f.m.err != nil {
		return Ptr{}
	}
	p.scratch = s.d
	return s
}

// routine is the runtime's routine for v, declared once per module.
func (p *cryptoPass) routine(name string, sig *Sig) *FuncImport {
	if fi, ok := p.f.m.Lookup(name).(*FuncImport); ok {
		return fi
	}
	return p.f.m.ImportFunc(name, sig).NoUnwind()
}

func (p *cryptoPass) replace(b *Block, in *Inst) *Def {
	v := in.op.Verb
	if v == VCpuHas {
		c, _ := in.Capability()
		if !p.lowers.Has(c) {
			return b.I1.Const(false).d
		}
		if p.baseline.Has(c) {
			return b.I1.Const(true).d
		}
		sig := NewSig().Param(TypeI32).Ret(TypeI32)
		got := b.Call(p.routine(CPUHasSymbol, sig), b.I32.Const(int64(c))).I32(0)
		return b.I32.Ne(got, b.I32.Const(0)).d
	}

	name := "vertex_" + string(v)
	if in.op.Type == TypeI32 {
		// §A8: two scalars in, one out, which every C ABI passes as is.
		sig := NewSig().Param(TypeI32).Param(in.args[1].typ).Ret(TypeI32)
		args := []Value{I32{in.args[0]}, valueOf(in.args[1])}
		return b.Call(p.routine(name, sig), args...).Value(0).Def()
	}

	// §V5: v128 operands and the result through the scratch slot.
	base := p.slot()
	if base.d == nil {
		return nil
	}
	at := func(i int) Ptr {
		if i == 0 {
			return base
		}
		return b.Ptr.Add(base, b.I64.Const(int64(16*i)))
	}
	sig := NewSig().Param(TypePtr)
	args := []Value{at(0)}
	vecs := 0
	for _, a := range in.args {
		if a.typ == TypeV128 {
			vecs++
			slot := at(vecs)
			b.V128().Store(V128{a}, slot)
			sig.Param(TypePtr)
			args = append(args, slot)
			continue
		}
		sig.Param(a.typ)
		args = append(args, valueOf(a))
	}
	if k, ok := in.Lit(); ok && k.Kind() == ConstInt {
		sig.Param(TypeI32)
		args = append(args, b.I32.Const(k.Int()))
	}
	b.Call(p.routine(name, sig), args...)
	return b.V128().Load(base).d
}

// valueOf is a definition as the Value of its type.
func valueOf(d *Def) Value {
	switch d.typ {
	case TypeI32:
		return I32{d}
	case TypeI64:
		return I64{d}
	case TypeV128:
		return V128{d}
	case TypePtr:
		return Ptr{d}
	case TypeI1:
		return I1{d}
	}
	return nil
}

// ExpandVectorVerbs rewrites the §V verbs a target has no instruction for
// into the ones it has, and reports whether it changed anything: xor3 and
// xor_andnot into two of §V1's, a lane rotate into two shifts and an or,
// the byte swaps into shifts and word shuffles, and i8x16_swizzle -- which
// no shift reaches -- into a call of vertex_i8x16_swizzle, as LegalizeCrypto
// makes its calls. expand says which verbs to rewrite; x86-64 without
// SSSE3 asks for the byte swaps and the swizzle.
func (m *Module) ExpandVectorVerbs(expand func(Verb) bool) bool {
	if m == nil || m.err != nil {
		return false
	}
	changed := false
	for _, it := range m.items {
		if f, ok := it.(*Func); ok && f.ExpandVectorVerbs(expand) {
			changed = true
		}
	}
	return changed
}

// ExpandVectorVerbs is the module's pass on one function, for a target
// whose answer differs by function: one that requires aes on x86-64 runs
// on a processor with SSSE3, whatever the module's baseline.
func (f *Func) ExpandVectorVerbs(expand func(Verb) bool) bool {
	if f == nil || f.m == nil || f.m.err != nil || len(f.blocks) == 0 {
		return false
	}
	changed := false
	p := &cryptoPass{f: f, repl: map[*Def]*Def{}}
	for _, b := range f.blocks {
		out := make([]*Inst, 0, len(b.insts))
		for _, in := range b.insts {
			for i, a := range in.args {
				if r, ok := p.repl[a]; ok {
					in.args[i] = r
				}
			}
			if in.op.Type != TypeV128 || !expandable[in.op.Verb] || !expand(in.op.Verb) {
				out = append(out, in)
				continue
			}
			term := b.term
			b.insts, b.term = out, nil
			d := p.expandVector(b, in)
			out, b.term = b.insts, term
			if d == nil {
				return changed
			}
			p.repl[in.results[0]] = d
			changed = true
		}
		b.insts = out
	}
	if len(p.repl) > 0 {
		f.WalkUses(func(u Use) bool {
			if d := u.Def(); d != nil {
				if r, ok := p.repl[d]; ok {
					u.Set(r)
				}
			}
			return true
		})
	}
	return changed
}

var expandable = map[Verb]bool{
	VVecXor3: true, VVecXorAndNot: true, VI32x4Rotl: true, VI64x2Rotl: true,
	VI16x8Bswap: true, VI32x4Bswap: true, VI64x2Bswap: true, VI8x16Swizzle: true,
}

func (p *cryptoPass) expandVector(b *Block, in *Inst) *Def {
	v := b.V128()
	arg := func(i int) V128 { return V128{in.args[i]} }
	k := int64(0)
	if lit, ok := in.Lit(); ok && lit.Kind() == ConstInt {
		k = lit.Int()
	}
	bswap16 := func(x V128) V128 { return v.Or(v.I16x8Shl(x, b.I32.Const(8)), v.I16x8ShrU(x, b.I32.Const(8))) }
	switch in.op.Verb {
	case VVecXor3:
		return v.Xor(v.Xor(arg(0), arg(1)), arg(2)).d
	case VVecXorAndNot:
		return v.Xor(arg(0), v.AndNot(arg(1), arg(2))).d
	case VI32x4Rotl:
		if k%32 == 0 {
			return v.Or(arg(0), arg(0)).d
		}
		return v.Or(v.I32x4Shl(arg(0), b.I32.Const(k%32)), v.I32x4ShrU(arg(0), b.I32.Const(32-k%32))).d
	case VI64x2Rotl:
		if k%64 == 0 {
			return v.Or(arg(0), arg(0)).d
		}
		return v.Or(v.I64x2Shl(arg(0), b.I32.Const(k%64)), v.I64x2ShrU(arg(0), b.I32.Const(64-k%64))).d
	case VI16x8Bswap:
		return bswap16(arg(0)).d
	case VI32x4Bswap:
		// The bytes of each halfword swapped, then the halfwords of each word.
		return v.I16x8ShuffleHigh(v.I16x8ShuffleLow(bswap16(arg(0)), 0xb1), 0xb1).d
	case VI64x2Bswap:
		return v.I16x8ShuffleHigh(v.I16x8ShuffleLow(bswap16(arg(0)), 0x1b), 0x1b).d
	case VI8x16Swizzle:
		base := p.slot()
		if base.d == nil {
			return nil
		}
		at := func(i int64) Ptr { return b.Ptr.Add(base, b.I64.Const(16*i)) }
		v.Store(arg(0), at(1))
		v.Store(arg(1), at(2))
		sig := NewSig().Param(TypePtr).Param(TypePtr).Param(TypePtr)
		b.Call(p.routine("vertex_"+string(VI8x16Swizzle), sig), base, at(1), at(2))
		return v.Load(base).d
	}
	return nil
}
