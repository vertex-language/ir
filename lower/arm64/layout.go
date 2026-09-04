package arm64

import (
	"fmt"

	"github.com/vertex-language/ir"
)

// The type layout AAPCS64 gives §5's types.
//
// It agrees with SysV AMD64 on everything this package lays out, which is not
// a coincidence and not something to rely on: both are LP64 with natural
// alignment. Where they part is long double — ten bytes in sixteen on x86-64,
// and a true binary128 here — which is why f80 is not an ext-float this target
// admits at all and f128 is.
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
	case ir.StoreI32, ir.StoreF32:
		return 4, 4, nil
	case ir.StoreI64, ir.StoreF64, ir.StorePtr:
		return 8, 8, nil
	case ir.StoreF128:
		return 16, 16, nil
	case ir.StoreF80:
		// There is no x87 here and no ten-byte float. AArch64's long double
		// is binary128, which is StoreF128; a module naming f80 against this
		// target has named a type the target does not have.
		return 0, 0, fmt.Errorf("f80 is not a type this target has; AArch64's extended float is f128")
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

// structLayout is where each field begins, how many bytes the struct occupies,
// and what it aligns to — one function, because the three are one computation.
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
// following that resolves nothing and spins until the bound. This read
// n.FType() until it was noticed that no global of a typedef'd type could be
// laid out at all.
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
