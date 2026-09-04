package verify

import "github.com/vertex-language/ir"

// §19.10 (a global initializer's structure against its declared type) and
// §19.18 (a struct's stated field offsets).
//
// §19.11 to §19.15 are not here, and this is where to find out why:
//
//   - §19.11 — dllimport on imports and dllexport on definitions is Go's
//     type system, not a check: DLLImport is a method on ir.GlobalImport
//     and ir.FuncImport, DLLExport is one on ir.Global and ir.Func, and
//     neither type has the other's. `common` on domain rw only is
//     ir.ErrPlacement at the declaration.
//   - §19.12 — an ext-float namespace absent from the layout block is
//     ir.ErrLayout, the moment b.F80() is called. Availability is the one
//     thing Go's type system cannot carry, so it is the one that is
//     sticky rather than a compile error.
//   - §19.13 — sret on at most one parameter, which is the first:
//     ir.ErrSRet, at the parameter.
//   - §19.14 — a callind's TypeName resolves to a func typedef and its
//     arguments match that signature: ir.ErrSignature and
//     ir.ErrArity/ir.ErrType at the call, where the typedef is in hand.
//   - §19.15 — module, use, and layout appear exactly once and precede
//     every item. They are NewModule's parameters, so this stopped being
//     a rule anything could break the day it became a constructor.

// moduleItems runs the module-scope rules. Their faults carry no function
// name: a global or a type is not in a function body, and Error prints
// them at module scope the way ir.Error does.
func (c *checker) moduleItems(m *ir.Module) {
	for _, g := range m.Globals() {
		c.initializer(g)
		if c.full() {
			return
		}
	}
	layout := m.Layout()
	for _, t := range m.Types() {
		c.structOffsets(t, layout)
		if c.full() {
			return
		}
	}
}

// initializer is §19.10: a global's initializer has the declared type's
// structure — arity, nesting, and element widths.
//
// What it does not check is a value's range against its width. An
// initializer's literal may be a symbolic constant — sizeof, alignof,
// offsetof — whose value is the target's to compute, so "does 4096 fit in
// an i8" is a question this repo cannot answer for the general case and
// declines to answer for the easy one. Structure is checkable everywhere,
// which is what §19.10 asks for.
func (c *checker) initializer(g *ir.Global) {
	c.initMatches(g.Type(), g.Initializer(), "@"+g.Name())
}

func (c *checker) initMatches(t ir.FType, i ir.Init, path string) {
	if c.full() {
		return
	}

	switch i.Kind() {
	case 0:
		// No initializer at all. ir.NewGlobal writes ZeroInit rather than
		// leaving this, so it means a zero ir.Init someone built by hand.
		return

	case ir.InitZeroed:
		// zeroed takes its width from the declared type and so matches
		// every one of them. This is why §5 requires the ftype: there is
		// nothing here to infer a width from.
		return

	case ir.InitLiteral, ir.InitRelocKind:
		if r := resolveFType(t); r.Kind() != ir.FTypeScalar {
			c.failItem(ErrInit, "%s is %s; a %s initializer fills one scalar", path, t, initKindName(i.Kind()))
		}

	case ir.InitString:
		r := resolveFType(t)
		if r.Kind() != ir.FTypeArray || resolveFType(r.Elem()).Scalar() != ir.StoreI8 {
			c.failItem(ErrInit, "%s is %s; a string initializer fills an array of i8", path, t)
			return
		}
		// The declared length is the storage; a shorter string leaves the
		// rest zero, which is how a NUL gets there. A longer one has
		// nowhere to put its tail.
		if n := uint64(len(i.String_())); r.Len() < n {
			c.failItem(ErrInit, "%s is %s; the string is %d bytes", path, t, n)
		}

	case ir.InitList:
		c.listMatches(t, i, path)

	case ir.InitFields:
		c.fieldsMatch(t, i, path)
	}
}

// listMatches checks a positional aggregate initializer. Arity is the
// whole of it for an array and a struct; a union holds one member, so a
// positional list names its first.
func (c *checker) listMatches(t ir.FType, i ir.Init, path string) {
	r := resolveFType(t)
	elems := i.Elems()

	if r.Kind() == ir.FTypeArray {
		if uint64(len(elems)) != r.Len() {
			c.failItem(ErrInit, "%s is %s; the initializer has %d elements", path, t, len(elems))
			return
		}
		for n, e := range elems {
			c.initMatches(r.Elem(), e, itemPath(path, "[", n, "]"))
		}
		return
	}

	named := r.Named()
	if r.Kind() != ir.FTypeNamed || named == nil {
		c.failItem(ErrInit, "%s is %s; a braced initializer fills an array, a struct, or a union", path, t)
		return
	}

	switch named.Kind() {
	case ir.KindStruct:
		fields := named.Fields()
		if len(elems) != len(fields) {
			c.failItem(ErrInit, "%s is @%s, which has %d fields; the initializer has %d elements",
				path, named.Name(), len(fields), len(elems))
			return
		}
		for n, e := range elems {
			c.initMatches(fields[n].Type, e, path+"."+fields[n].Name)
		}

	case ir.KindUnion:
		fields := named.Fields()
		if len(elems) != 1 || len(fields) == 0 {
			c.failItem(ErrInit, "%s is the union @%s; a union initializer names one member, and this one has %d",
				path, named.Name(), len(elems))
			return
		}
		c.initMatches(fields[0].Type, elems[0], path+"."+fields[0].Name)

	default:
		c.failItem(ErrInit, "%s is @%s; a braced initializer fills an array, a struct, or a union",
			path, named.Name())
	}
}

// fieldsMatch checks a named-field aggregate initializer.
//
// A struct's list may be partial: naming fields is what this form is for,
// and the fields it does not name are zero. What it may not do is name a
// field the type does not have, or name one twice — the second would make
// the initializer's own order the tie-breaker for a value, which is not a
// thing §5 gives a meaning to.
func (c *checker) fieldsMatch(t ir.FType, i ir.Init, path string) {
	r := resolveFType(t)
	named := r.Named()
	if r.Kind() != ir.FTypeNamed || named == nil ||
		(named.Kind() != ir.KindStruct && named.Kind() != ir.KindUnion) {
		c.failItem(ErrInit, "%s is %s; a field initializer fills a struct or a union", path, t)
		return
	}

	vals := i.FieldVals()
	if named.Kind() == ir.KindUnion && len(vals) != 1 {
		c.failItem(ErrInit, "%s is the union @%s; a union initializer names one member, and this one names %d",
			path, named.Name(), len(vals))
		return
	}

	seen := make(map[string]bool, len(vals))
	for _, v := range vals {
		f, ok := field(named, v.Name)
		if !ok {
			c.failItem(ErrInit, "@%s has no field %q, which %s names", named.Name(), v.Name, path)
			continue
		}
		if seen[v.Name] {
			c.failItem(ErrInit, "%s names field %q twice", path, v.Name)
			continue
		}
		seen[v.Name] = true
		c.initMatches(f.Type, v.Init, path+"."+v.Name)
		if c.full() {
			return
		}
	}
}

// structOffsets is §19.18: within one struct, either every field carries
// at or none does, and where they do the offsets strictly increase and no
// field runs into its successor.
//
// The rule's last clause — that each offset satisfies its field type's
// alignment unless the struct is packed — is not checked, and cannot be
// from here. A type's alignment is the target ABI's fact, not the
// module's: i64 aligns to 8 under SysV AMD64 and to 4 under SysV i386,
// and §3's layout block states the pointer width, the endianness, the
// stack alignment, and the ext-float namespaces — not a per-type
// alignment table. The check belongs where a Layout becomes a target,
// which is ir/lower, not here.
func (c *checker) structOffsets(t *ir.Type, layout ir.Layout) {
	if t.Kind() != ir.KindStruct {
		// A union's fields all begin at zero, and ir.Type.FieldAt refuses
		// at on one with ir.ErrPlacement.
		return
	}

	fields := t.Fields()
	stated := 0
	for _, f := range fields {
		if f.HasOffset {
			stated++
		}
	}
	if stated == 0 {
		return
	}
	if stated != len(fields) {
		c.failItem(ErrStructOffset,
			"@%s states an offset for %d of its %d fields; a struct with some computed offsets has no determinate offsetof",
			t.Name(), stated, len(fields))
		return
	}

	for n := 1; n < len(fields); n++ {
		prev, f := fields[n-1], fields[n]
		if f.Offset <= prev.Offset {
			c.failItem(ErrStructOffset, "@%s field %q is at %d, after %q at %d",
				t.Name(), f.Name, f.Offset, prev.Name, prev.Offset)
			if c.full() {
				return
			}
			continue
		}
		// Overlap is only checkable where the earlier field's width is:
		// an f80's in-memory padding and an aggregate's own layout are
		// the target's to decide, and sizeOf says so by declining.
		size, ok := sizeOf(prev.Type, layout)
		if !ok || prev.Offset+size <= f.Offset {
			continue
		}
		c.failItem(ErrStructOffset, "@%s field %q is %d bytes at %d, running into %q at %d",
			t.Name(), prev.Name, size, prev.Offset, f.Name, f.Offset)
		if c.full() {
			return
		}
	}
}

// resolveFType follows alias typedefs to the shape underneath. A struct,
// a union, a func typedef, an array, and a scalar are all shapes in their
// own right; only KindAlias is a name for another one.
func resolveFType(t ir.FType) ir.FType {
	for i := 0; i < 64; i++ { // a cycle through a typedef is not this rule's fault to report
		if t.Kind() != ir.FTypeNamed || t.Named() == nil || t.Named().Kind() != ir.KindAlias {
			return t
		}
		t = t.Named().Aliased()
	}
	return t
}

// sizeOf is an ftype's width in bytes, and whether that width is knowable
// from this module alone.
//
// It is not: an f80 occupies ten bytes of value and however much padding
// the target adds, a union's size depends on the alignment of its widest
// member, and a struct whose offsets are computed is laid out by the ABI.
// Each of those returns false rather than a guess. A struct that states
// every offset is the case this rule needs, and that one is determinate.
func sizeOf(t ir.FType, l ir.Layout) (uint64, bool) {
	t = resolveFType(t)
	switch t.Kind() {
	case ir.FTypeScalar:
		switch t.Scalar() {
		case ir.StoreI8:
			return 1, true
		case ir.StoreI16:
			return 2, true
		case ir.StoreI32, ir.StoreF32:
			return 4, true
		case ir.StoreI64, ir.StoreF64:
			return 8, true
		case ir.StoreF128:
			return 16, true
		case ir.StorePtr:
			if l.PtrBits == 0 {
				return 0, false
			}
			return uint64(l.PtrBits) / 8, true
		}
		return 0, false // f80, and the zero StoreType

	case ir.FTypeArray:
		e, ok := sizeOf(t.Elem(), l)
		if !ok {
			return 0, false
		}
		return t.Len() * e, true

	case ir.FTypeNamed:
		named := t.Named()
		if named == nil || named.Kind() != ir.KindStruct {
			return 0, false
		}
		var end uint64
		for _, f := range named.Fields() {
			if !f.HasOffset {
				return 0, false
			}
			s, ok := sizeOf(f.Type, l)
			if !ok {
				return 0, false
			}
			if f.Offset+s > end {
				end = f.Offset + s
			}
		}
		return end, true
	}
	return 0, false
}

// field looks one up by name.
func field(t *ir.Type, name string) (ir.Field, bool) {
	for _, f := range t.Fields() {
		if f.Name == name {
			return f, true
		}
	}
	return ir.Field{}, false
}

// initKindName is what to call an initializer form in a fault.
func initKindName(k ir.InitKind) string {
	if k == ir.InitRelocKind {
		return "reloc"
	}
	return "literal"
}

// itemPath is path with an index appended, without pulling in a formatter
// for three pieces.
func itemPath(path, open string, n int, close string) string {
	return path + open + itoa(n) + close
}

func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	var buf [20]byte
	i := len(buf)
	for n > 0 {
		i--
		buf[i] = byte('0' + n%10)
		n /= 10
	}
	return string(buf[i:])
}
