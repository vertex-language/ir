package i386

import (
	"fmt"

	"github.com/vertex-language/ir"
)

// The type layout the Intel386 psABI gives §5's types.
//
// The difference from the two 64-bit backends is one number and it is easy to
// get wrong: an eight-byte scalar aligns to *four* here, not eight. long long
// and double are 4-aligned in the i386 psABI, which is why a struct of an i32
// and an i64 is twelve bytes on this target and sixteen on the others.
const maxNesting = 64

func sizeAlign(t ir.FType) (size, align uint64, err error) { return sizeAlignAt(t, 0) }

func sizeAlignAt(t ir.FType, depth int) (uint64, uint64, error) {
	if depth > maxNesting {
		return 0, 0, fmt.Errorf("type nests more than %d deep", maxNesting)
	}
	t = resolveAlias(t)

	switch t.Kind() {
	case ir.FTypeScalar:
		return scalarSizeAlign(t.Scalar())
	case ir.FTypeArray:
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

func scalarSizeAlign(s ir.StoreType) (uint64, uint64, error) {
	switch s {
	case ir.StoreI8:
		return 1, 1, nil
	case ir.StoreI16:
		return 2, 2, nil
	case ir.StoreI32, ir.StoreF32, ir.StorePtr:
		return 4, 4, nil
	case ir.StoreI64, ir.StoreF64:
		// Four, not eight. See the note at the top of the file.
		return 8, 4, nil
	case ir.StoreF80:
		// Twelve bytes in the i386 psABI, four-aligned: ten bytes of
		// x87 register and two of padding.
		return 12, 4, nil
	case ir.StoreF128:
		return 0, 0, fmt.Errorf("f128 is not a type this target has; Intel386's extended float is f80")
	}
	return 0, 0, fmt.Errorf("no layout for storage type %v", s)
}

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

func structSizeAlign(t *ir.Type, depth int) (uint64, uint64, error) {
	_, size, align, err := structLayout(t, depth)
	return size, align, err
}

// structLayout is where each field begins, how many bytes the struct
// occupies, and what it aligns to — one function, because the three are one
// computation.
func structLayout(t *ir.Type, depth int) ([]uint64, uint64, uint64, error) {
	fields := t.Fields()
	offsets := make([]uint64, len(fields))
	var end, align uint64 = 0, 1

	for i, f := range fields {
		size, a, err := sizeAlignAt(f.Type, depth+1)
		if err != nil {
			return nil, 0, 0, err
		}
		if a > align {
			align = a
		}
		at := alignUp(end, a)
		if f.HasOffset {
			at = f.Offset
		}
		offsets[i] = at
		if at+size > end {
			end = at + size
		}
	}
	return offsets, alignUp(end, align), align, nil
}

// fieldOffsets is where each of a struct's fields begins.
func fieldOffsets(t *ir.Type) ([]uint64, error) {
	offsets, _, _, err := structLayout(t, 0)
	return offsets, err
}

func unionSizeAlign(t *ir.Type, depth int) (uint64, uint64, error) {
	var size, align uint64 = 0, 1
	for _, f := range t.Fields() {
		s, a, err := sizeAlignAt(f.Type, depth+1)
		if err != nil {
			return 0, 0, err
		}
		if s > size {
			size = s
		}
		if a > align {
			align = a
		}
	}
	return alignUp(size, align), align, nil
}

// resolveAlias follows a typedef to the type it names.
//
// Aliased, not FType: a named type's FType is the wrapper around itself, so
// following that resolves nothing and spins until the bound.
func resolveAlias(t ir.FType) ir.FType {
	for i := 0; i < maxNesting; i++ {
		if t.Kind() != ir.FTypeNamed {
			return t
		}
		n := t.Named()
		if n == nil || n.Kind() != ir.KindAlias {
			return t
		}
		t = n.Aliased()
	}
	return t
}

func alignUp(n, a uint64) uint64 {
	if a == 0 {
		return n
	}
	return (n + a - 1) &^ (a - 1)
}
