package ir

// A MemAttr is §D4's access attribute.
type MemAttr struct {
	align    uint64
	volatile bool
}

// Align overrides natural alignment downward. N is a power of two no greater
// than the access width; absence asserts natural alignment, and a misaligned
// access traps.
func Align(n uint64) MemAttr { return MemAttr{align: n} }

// Volatile marks the access observable: not elidable, duplicable, reorderable
// across other volatiles, or widenable.
var Volatile = MemAttr{volatile: true}

// An AllocOpt is an option on ptr.alloc and ptr.alloca.
type AllocOpt uint8

// Zeroed guarantees the storage reads as all-zero bytes on entry to the
// allocation's live range. Absent it the contents are unspecified but defined —
// arbitrary bytes, never a trap and never UB.
const Zeroed AllocOpt = 1

// accessWidth is the natural width of an access in that namespace, in bytes.
// The align attribute may not exceed it.
func (b *Builder) accessWidth(t RegType) uint64 {
	switch t {
	case TypeI32, TypeF32:
		return 4
	case TypeI64, TypeF64:
		return 8
	case TypeF80, TypeF128, TypeV128:
		return 16
	case TypePtr:
		if m := b.mod(); m != nil && m.layout.PtrBits > 0 {
			return uint64(m.layout.PtrBits) / 8
		}
		return 8
	}
	return 0
}

// memAttrs folds the attribute list into an imm, checking §19.8.
func (b *Builder) memAttrs(op Op, width uint64, attrs []MemAttr) *imm {
	if len(attrs) == 0 {
		return nil
	}
	im := &imm{}
	for _, a := range attrs {
		if a.volatile {
			im.volatile = true
			continue
		}
		if a.align == 0 {
			continue
		}
		if !isPow2(a.align) {
			b.fail(op, ErrAlign, "align %d is not a power of two", a.align)
			return nil
		}
		if width != 0 && a.align > width {
			b.fail(op, ErrAlign, "align %d exceeds the %d-byte access width", a.align, width)
			return nil
		}
		im.align = a.align
		im.hasAlign = true
	}
	return im
}

// —— §D. Full-width load and store ——

func (b *Builder) load(t RegType, p Ptr, attrs []MemAttr) *Def {
	op := Op{t, VLoad}
	im := b.memAttrs(op, b.accessWidth(t), attrs)
	if b.mod() != nil && b.mod().err != nil {
		return nil
	}
	return b.def1i(op, t, []*Def{p.d}, im)
}

func (b *Builder) store(t RegType, val, addr *Def, attrs []MemAttr) {
	op := Op{t, VStore}
	im := b.memAttrs(op, b.accessWidth(t), attrs)
	if b.mod() != nil && b.mod().err != nil {
		return
	}
	b.voidi(op, []*Def{val, addr}, im)
}

func (n I32NS) Load(p Ptr, a ...MemAttr) I32   { return I32{n.b.load(TypeI32, p, a)} }
func (n I64NS) Load(p Ptr, a ...MemAttr) I64   { return I64{n.b.load(TypeI64, p, a)} }
func (n F32NS) Load(p Ptr, a ...MemAttr) F32   { return F32{n.b.load(TypeF32, p, a)} }
func (n F64NS) Load(p Ptr, a ...MemAttr) F64   { return F64{n.b.load(TypeF64, p, a)} }
func (n F80NS) Load(p Ptr, a ...MemAttr) F80   { return F80{n.b.load(TypeF80, p, a)} }
func (n F128NS) Load(p Ptr, a ...MemAttr) F128 { return F128{n.b.load(TypeF128, p, a)} }
func (n V128NS) Load(p Ptr, a ...MemAttr) V128 { return V128{n.b.load(TypeV128, p, a)} }
func (n PtrNS) Load(p Ptr, a ...MemAttr) Ptr   { return Ptr{n.b.load(TypePtr, p, a)} }

// Store is value-first, address-last. Getting the order wrong is a Go type
// error, not a runtime refusal.
func (n I32NS) Store(v I32, dst Ptr, a ...MemAttr)   { n.b.store(TypeI32, v.d, dst.d, a) }
func (n I64NS) Store(v I64, dst Ptr, a ...MemAttr)   { n.b.store(TypeI64, v.d, dst.d, a) }
func (n F32NS) Store(v F32, dst Ptr, a ...MemAttr)   { n.b.store(TypeF32, v.d, dst.d, a) }
func (n F64NS) Store(v F64, dst Ptr, a ...MemAttr)   { n.b.store(TypeF64, v.d, dst.d, a) }
func (n F80NS) Store(v F80, dst Ptr, a ...MemAttr)   { n.b.store(TypeF80, v.d, dst.d, a) }
func (n F128NS) Store(v F128, dst Ptr, a ...MemAttr) { n.b.store(TypeF128, v.d, dst.d, a) }
func (n V128NS) Store(v V128, dst Ptr, a ...MemAttr) { n.b.store(TypeV128, v.d, dst.d, a) }
func (n PtrNS) Store(v Ptr, dst Ptr, a ...MemAttr)   { n.b.store(TypePtr, v.d, dst.d, a) }

// —— §D2. Sub-width load and store ——
//
// These verbs are the only way i8 and i16 are reachable, since §2 makes them
// storage-only widths with no register type.

func (b *Builder) subload(t RegType, v Verb, width uint64, p Ptr, attrs []MemAttr) *Def {
	op := Op{t, v}
	im := b.memAttrs(op, width, attrs)
	if b.mod() != nil && b.mod().err != nil {
		return nil
	}
	return b.def1i(op, t, []*Def{p.d}, im)
}

func (b *Builder) substore(t RegType, v Verb, width uint64, val, addr *Def, attrs []MemAttr) {
	op := Op{t, v}
	im := b.memAttrs(op, width, attrs)
	if b.mod() != nil && b.mod().err != nil {
		return
	}
	b.voidi(op, []*Def{val, addr}, im)
}

func (n I32NS) SLoad8(p Ptr, a ...MemAttr) I32 {
	return I32{n.b.subload(TypeI32, VSLoad8, 1, p, a)}
}
func (n I32NS) SLoad16(p Ptr, a ...MemAttr) I32 {
	return I32{n.b.subload(TypeI32, VSLoad16, 2, p, a)}
}
func (n I32NS) ULoad8(p Ptr, a ...MemAttr) I32 {
	return I32{n.b.subload(TypeI32, VULoad8, 1, p, a)}
}
func (n I32NS) ULoad16(p Ptr, a ...MemAttr) I32 {
	return I32{n.b.subload(TypeI32, VULoad16, 2, p, a)}
}
func (n I64NS) SLoad8(p Ptr, a ...MemAttr) I64 {
	return I64{n.b.subload(TypeI64, VSLoad8, 1, p, a)}
}
func (n I64NS) SLoad16(p Ptr, a ...MemAttr) I64 {
	return I64{n.b.subload(TypeI64, VSLoad16, 2, p, a)}
}
func (n I64NS) SLoad32(p Ptr, a ...MemAttr) I64 {
	return I64{n.b.subload(TypeI64, VSLoad32, 4, p, a)}
}
func (n I64NS) ULoad8(p Ptr, a ...MemAttr) I64 {
	return I64{n.b.subload(TypeI64, VULoad8, 1, p, a)}
}
func (n I64NS) ULoad16(p Ptr, a ...MemAttr) I64 {
	return I64{n.b.subload(TypeI64, VULoad16, 2, p, a)}
}
func (n I64NS) ULoad32(p Ptr, a ...MemAttr) I64 {
	return I64{n.b.subload(TypeI64, VULoad32, 4, p, a)}
}

func (n I32NS) Store8(v I32, dst Ptr, a ...MemAttr) {
	n.b.substore(TypeI32, VStore8, 1, v.d, dst.d, a)
}
func (n I32NS) Store16(v I32, dst Ptr, a ...MemAttr) {
	n.b.substore(TypeI32, VStore16, 2, v.d, dst.d, a)
}
func (n I64NS) Store8(v I64, dst Ptr, a ...MemAttr) {
	n.b.substore(TypeI64, VStore8, 1, v.d, dst.d, a)
}
func (n I64NS) Store16(v I64, dst Ptr, a ...MemAttr) {
	n.b.substore(TypeI64, VStore16, 2, v.d, dst.d, a)
}
func (n I64NS) Store32(v I64, dst Ptr, a ...MemAttr) {
	n.b.substore(TypeI64, VStore32, 4, v.d, dst.d, a)
}

// —— §D3. Pointer ops ——

// Alloc reserves frame storage. §19.6 admits it in the entry block only, which
// the builder catches here rather than deferring.
func (n PtrNS) Alloc(size, align uint64, opts ...AllocOpt) Ptr {
	op := Op{TypePtr, VAlloc}
	if !n.b.entryOnly(op) {
		return Ptr{}
	}
	if !isPow2(align) {
		n.b.fail(op, ErrAlign, "align %d is not a power of two", align)
		return Ptr{}
	}
	return Ptr{n.b.def1i(op, TypePtr, nil, &imm{
		size: size, align: align, hasAlign: true, zeroed: zeroed(opts),
	})}
}

// AllocType reserves frame storage sized and aligned by a named type.
func (n PtrNS) AllocType(t *Type, opts ...AllocOpt) Ptr {
	op := Op{TypePtr, VAlloc}
	if !n.b.entryOnly(op) {
		return Ptr{}
	}
	if t == nil {
		n.b.fail(op, ErrPoison, "no type named")
		return Ptr{}
	}
	return Ptr{n.b.def1i(op, TypePtr, nil, &imm{typ: t, zeroed: zeroed(opts)})}
}

// Alloca reserves a dynamically sized frame region. Unlike Alloc it may appear
// in any block.
func (n PtrNS) Alloca(size I64, align uint64, opts ...AllocOpt) Ptr {
	op := Op{TypePtr, VAlloca}
	if !isPow2(align) {
		n.b.fail(op, ErrAlign, "align %d is not a power of two", align)
		return Ptr{}
	}
	return Ptr{n.b.def1i(op, TypePtr, []*Def{size.d}, &imm{
		align: align, hasAlign: true, zeroed: zeroed(opts),
	})}
}

func zeroed(opts []AllocOpt) bool {
	for _, o := range opts {
		if o == Zeroed {
			return true
		}
	}
	return false
}

func (b *Builder) entryOnly(op Op) bool {
	if b.blk == nil {
		return false
	}
	if !b.blk.isEntry {
		b.fail(op, ErrPlacement, "outside the entry block")
		return false
	}
	return true
}

// StackSave yields an opaque token.
func (n PtrNS) StackSave() Ptr {
	return Ptr{n.b.def1(Op{TypePtr, VStackSave}, TypePtr)}
}

// StackRestore consumes a token from StackSave.
func (n PtrNS) StackRestore(tok Ptr) { n.b.void(Op{TypePtr, VStackRestore}, tok.d) }

// GetAddr is the address of a global or a function.
func (n PtrNS) GetAddr(s Symbol) Ptr {
	op := Op{TypePtr, VGetAddr}
	if s == nil {
		n.b.fail(op, ErrPoison, "no symbol named")
		return Ptr{}
	}
	return Ptr{n.b.def1i(op, TypePtr, nil, &imm{sym: s})}
}

// TLSAddr is the address in the calling thread of a thread-local.
//
// The operand is a global in domain tls, or an import of one. A thread-local
// defined in another unit is reached exactly the same way — the sequence
// finds the calling thread's block and adds an offset the linker fills, and
// neither half cares which object file the definition was in. An import
// carries no domain, having no storage here to place; what marks it is the
// model attribute, which is why the two are asked different questions.
func (n PtrNS) TLSAddr(s Symbol) Ptr {
	op := Op{TypePtr, VTLSAddr}
	switch g := s.(type) {
	case nil:
		n.b.fail(op, ErrPoison, "no global named")
		return Ptr{}
	case *Global:
		if g.domain != TLS {
			n.b.fail(op, ErrPlacement, "@%s is in domain %s", g.name, g.domain)
			return Ptr{}
		}
	case *GlobalImport:
		if g.tlsModel == NoTLSModel {
			n.b.fail(op, ErrPlacement, "@%s is imported without a tlsmodel", g.name)
			return Ptr{}
		}
	default:
		n.b.fail(op, ErrPlacement, "%s is not a global", s.Name())
		return Ptr{}
	}
	return Ptr{n.b.def1i(op, TypePtr, nil, &imm{sym: s})}
}

// BlockAddr is a block's address, for brind only. §19.6 requires the label to
// be a target of some brind in the same function.
func (n PtrNS) BlockAddr(blk *Block) Ptr {
	op := Op{TypePtr, VBlockAddr}
	if blk == nil {
		n.b.fail(op, ErrPoison, "no block named")
		return Ptr{}
	}
	if blk.fn != n.b.blk.fn {
		panic("ir: blockaddr of @" + blk.label + " from another function")
	}
	return Ptr{n.b.def1i(op, TypePtr, nil, &imm{labels: []*Block{blk}})}
}

// FrameAddr describes the current frame only. Walking outward is a runtime's
// job: a frame-pointer-omitting target has no reliable answer for level > 0.
func (n PtrNS) FrameAddr() Ptr {
	return Ptr{n.b.def1(Op{TypePtr, VFrameAddr}, TypePtr)}
}

// ReturnAddr is the current frame's return address, level 0 only.
func (n PtrNS) ReturnAddr() Ptr {
	return Ptr{n.b.def1(Op{TypePtr, VReturnAddr}, TypePtr)}
}

// —— §E. Bulk memory ——
//
// A zero len is well-defined and touches nothing, including when a pointer is
// null.

// MemCpy copies non-overlapping ranges.
func (b *Builder) MemCpy(dst, src Ptr, n I64, attrs ...MemAttr) {
	op := Op{TypeNone, VMemCpy}
	b.voidi(op, []*Def{dst.d, src.d, n.d}, b.memAttrs(op, 0, attrs))
}

// MemMove copies overlap-safely.
func (b *Builder) MemMove(dst, src Ptr, n I64, attrs ...MemAttr) {
	op := Op{TypeNone, VMemMove}
	b.voidi(op, []*Def{dst.d, src.d, n.d}, b.memAttrs(op, 0, attrs))
}

// MemSet writes the low byte of val.
func (b *Builder) MemSet(dst Ptr, val I32, n I64, attrs ...MemAttr) {
	op := Op{TypeNone, VMemSet}
	b.voidi(op, []*Def{dst.d, val.d, n.d}, b.memAttrs(op, 0, attrs))
}

// MemCmp is 0 if equal, and otherwise ordered by the first differing byte,
// compared unsigned.
func (b *Builder) MemCmp(x, y Ptr, n I64, attrs ...MemAttr) I32 {
	op := Op{TypeNone, VMemCmp}
	return I32{b.def1i(op, TypeI32, []*Def{x.d, y.d, n.d}, b.memAttrs(op, 0, attrs))}
}
