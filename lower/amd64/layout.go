package amd64

// The type layout SysV AMD64 gives §5's types: how many bytes each
// occupies and what it has to be aligned to.
//
// This is the table §19.18's last clause is waiting on and the one
// ir/verify says it cannot have. A type's alignment is the target ABI's
// fact, not the module's: i64 aligns to 8 here and to 4 under SysV
// i386, and §3's layout block states the pointer width, the endianness
// and the stack alignment — not a per-type alignment table. So it lives
// where a Layout becomes a target, which is here.
//
// It is the psABI's table and not a reasonable-looking one. Two entries
// are worth naming because they are the ones a reader expects to be
// wrong: long double occupies sixteen bytes of which ten are the value,
// and __int128 and __float128 align to sixteen, which is what makes a
// struct containing one align to sixteen too.

import (
	"fmt"

	"github.com/vertex-language/ir"
)

// maxNesting bounds how far resolveAlias and the layout walk will follow
// a type into itself. A typedef cycle is a module the builder should not
// have produced; refusing to loop on one is not the same as checking for
// it, and this package is not where that check belongs.
const maxNesting = 64

// sizeAlign is a type's size in bytes and the alignment it requires.
//
// Both, together, because they are computed together: a struct's size
// depends on its own alignment, since it is padded to a multiple of it
// so that an array of the struct keeps every element aligned.
func sizeAlign(t ir.FType) (size, align uint64, err error) {
	return sizeAlignAt(t, 0)
}

func sizeAlignAt(t ir.FType, depth int) (uint64, uint64, error) {
	if depth > maxNesting {
		return 0, 0, fmt.Errorf("type nests more than %d deep", maxNesting)
	}
	t = resolveAlias(t)

	switch t.Kind() {
	case ir.FTypeScalar:
		return scalarSizeAlign(t.Scalar())

	case ir.FTypeArray:
		// The element's size already includes whatever padding keeps the
		// next element aligned — that is what padding a struct to its
		// own alignment is for — so an array is a multiplication and
		// nothing more.
		size, align, err := sizeAlignAt(t.Elem(), depth+1)
		if err != nil {
			return 0, 0, err
		}
		return size * t.Len(), align, nil

	case ir.FTypeNamed:
		return namedSizeAlign(t.Named(), depth)
	}
	return 0, 0, fmt.Errorf("%s has no layout", t)
}

// scalarSizeAlign is the psABI's table for the widths §2 admits.
func scalarSizeAlign(s ir.StoreType) (uint64, uint64, error) {
	switch s {
	case ir.StoreI8:
		return 1, 1, nil
	case ir.StoreI16:
		return 2, 2, nil
	case ir.StoreI32, ir.StoreF32:
		return 4, 4, nil
	case ir.StoreI64, ir.StoreF64, ir.StorePtr:
		return 8, 8, nil
	case ir.StoreF80:
		// Ten bytes of value in sixteen bytes of storage, aligned to
		// sixteen. The six bytes are padding the ABI requires and not
		// slack a packing pass may take back.
		return 16, 16, nil
	case ir.StoreF128:
		return 16, 16, nil
	}
	return 0, 0, fmt.Errorf("no layout for storage type %v", s)
}

// namedSizeAlign lays out a struct or a union.
func namedSizeAlign(t *ir.Type, depth int) (uint64, uint64, error) {
	if t == nil {
		return 0, 0, fmt.Errorf("a named type with no definition")
	}
	switch t.Kind() {
	case ir.KindStruct:
		return structSizeAlign(t, depth)
	case ir.KindUnion:
		return unionSizeAlign(t, depth)
	}
	return 0, 0, fmt.Errorf("@%s is a %v, which has no storage layout", t.Name(), t.Kind())
}

// structSizeAlign lays fields out in declaration order, each at the next
// offset satisfying its own alignment, and rounds the total up to the
// struct's alignment — which is the strictest of its fields'.
//
// The rounding is not decoration. An array of the struct puts the next
// element immediately after this one, and only a size that is a multiple
// of the alignment keeps that element aligned too.
func structSizeAlign(t *ir.Type, depth int) (uint64, uint64, error) {
	_, size, align, err := structLayout(t, depth)
	return size, align, err
}

// structLayout is where each field begins, how many bytes the struct
// occupies, and what it aligns to — one function, because the three are
// one computation: where a field lands depends on where the last one
// ended, the alignment on the fields, and the size on the alignment.
//
// A field that states its own offset is placed there instead of being
// computed. §19.18 requires that a struct state all of them or none, so
// a module the verifier accepted is one where offsets are either
// entirely the author's or entirely this function's.
//
// §4's struct-layout is the author overriding this table, and it admits
// one clause or the other. packed drops every field to alignment one,
// which takes out the padding between them and leaves the struct itself
// aligned to one. align states the struct's alignment outright, downward
// included: it is a statement about the aggregate rather than a floor
// under what its fields want, which is what packed is too.
func structLayout(t *ir.Type, depth int) (offsets []uint64, size, align uint64, err error) {
	packed := t.IsPacked()
	fields := t.Fields()
	offsets = make([]uint64, len(fields))

	var end uint64
	align = 1
	for i, f := range fields {
		fsize, a, err := sizeAlignAt(f.Type, depth+1)
		if err != nil {
			return nil, 0, 0, fmt.Errorf("@%s field %q: %w", t.Name(), f.Name, err)
		}
		if packed {
			a = 1
		}
		if a > align {
			align = a
		}
		at := alignUp(end, a)
		if f.HasOffset {
			at = f.Offset
		}
		offsets[i] = at
		if at+fsize > end {
			end = at + fsize
		}
	}
	if stated := t.AlignAttr(); stated > 0 {
		align = stated
	}
	return offsets, alignUp(end, align), align, nil
}

// unionSizeAlign is the widest member, rounded to the strictest
// alignment. Every member begins at offset zero, which is what a union
// is — so packed changes no offset here and says only that the union
// itself aligns to one.
func unionSizeAlign(t *ir.Type, depth int) (uint64, uint64, error) {
	var size, align uint64 = 0, 1
	for _, f := range t.Fields() {
		s, a, err := sizeAlignAt(f.Type, depth+1)
		if err != nil {
			return 0, 0, fmt.Errorf("@%s member %q: %w", t.Name(), f.Name, err)
		}
		if s > size {
			size = s
		}
		if a > align && !t.IsPacked() {
			align = a
		}
	}
	if stated := t.AlignAttr(); stated > 0 {
		align = stated
	}
	return alignUp(size, align), align, nil
}

// fieldOffsets is where each of a struct's fields begins, by the rules
// structLayout counts with.
func fieldOffsets(t *ir.Type) ([]uint64, error) {
	offsets, _, _, err := structLayout(t, 0)
	return offsets, err
}

// resolveAlias follows a typedef to the type it names.
//
// Here rather than in global.go, which is now an adapter around the shared
// walk: the layout table is what an alias has to be resolved for.
func resolveAlias(t ir.FType) ir.FType {
	for i := 0; i < maxNesting; i++ {
		if t.Kind() != ir.FTypeNamed || t.Named() == nil || t.Named().Kind() != ir.KindAlias {
			return t
		}
		t = t.Named().Aliased()
	}
	return t
}
