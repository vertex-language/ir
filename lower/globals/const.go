package globals

// §2's three symbolic constants, resolved.
//
// sizeof, alignof and offsetof exist so that a frontend's flattening into
// byte offsets is checkable rather than hand-computed and silently
// target-dependent. That only works if something resolves them, and until
// this file nothing did: they were constructible, they printed, and every
// backend's isel and this package's initializer walk alike refused them.
//
// Resolving one is a question about the target and about nothing else, which
// is why it lives beside the walk that already asks the target the same
// question for the same reason.

import (
	"fmt"

	"github.com/vertex-language/ir"
)

// ConstInt is c as the integer it names.
//
// A float literal is not one: it is a value rather than a count, and a
// caller that wants one asks c for it directly.
func ConstInt(l Layout, c ir.Const) (int64, error) {
	switch c.Kind() {
	case ir.ConstInt:
		return c.Int(), nil

	case ir.ConstSizeOf, ir.ConstAlignOf:
		t, err := measured(c)
		if err != nil {
			return 0, err
		}
		size, align, err := l.SizeAlign(t)
		if err != nil {
			return 0, err
		}
		if c.Kind() == ir.ConstAlignOf {
			return int64(align), nil
		}
		return int64(size), nil

	case ir.ConstOffsetOf:
		if c.Type() == nil {
			return 0, fmt.Errorf("offsetof names no type")
		}
		return offsetOf(l, c.Type().FType(), c.Path())

	case ir.ConstFloat:
		return 0, fmt.Errorf("a float literal is not an integer constant")
	}
	return 0, fmt.Errorf("constant kind %v is not resolved", c.Kind())
}

// measured is the ftype a sizeof or an alignof asks about: a named type
// stated outright, or the declared type of a global it names.
//
// A function symbol has neither a size nor an alignment §5 knows — what
// sizeof @f would mean is the target's business and not the IR's — so it is
// refused rather than answered with the pointer's width.
func measured(c ir.Const) (ir.FType, error) {
	if t := c.Type(); t != nil {
		return t.FType(), nil
	}
	s := c.Symbol()
	if s == nil {
		return ir.FType{}, fmt.Errorf("sizeof names neither a type nor a symbol")
	}
	g, ok := s.(*ir.Global)
	if !ok {
		return ir.FType{}, fmt.Errorf("@%s is not a global; only a global declares storage to measure", s.Name())
	}
	return g.Type(), nil
}

// offsetOf walks a path from a type to the byte offset it names.
//
// A union contributes nothing: §5 begins all of its members at zero, so
// naming one is a change of type and not a change of offset. An array index
// contributes the element's size, and the index may be the array's length —
// one past the end is the address &arr[n] names and is what Init.Plus was
// written for.
func offsetOf(l Layout, t ir.FType, path []ir.PathElem) (int64, error) {
	var off uint64
	for _, e := range path {
		r := resolveAlias(t)

		if e.IsIndex() {
			if r.Kind() != ir.FTypeArray {
				return 0, fmt.Errorf("offsetof indexes %s, which is not an array", t)
			}
			if e.Index() > r.Len() {
				return 0, fmt.Errorf("offsetof [%d] of %s, which has %d elements", e.Index(), t, r.Len())
			}
			elem := r.Elem()
			size, _, err := l.SizeAlign(elem)
			if err != nil {
				return 0, err
			}
			off += e.Index() * size
			t = elem
			continue
		}

		named := r.Named()
		if r.Kind() != ir.FTypeNamed || named == nil ||
			(named.Kind() != ir.KindStruct && named.Kind() != ir.KindUnion) {
			return 0, fmt.Errorf("offsetof names field %q of %s, which is not a struct or a union", e.Name(), t)
		}
		fields := named.Fields()
		i := fieldIndex(fields, e.Name())
		if i < 0 {
			return 0, fmt.Errorf("@%s has no field %q", named.Name(), e.Name())
		}
		if named.Kind() == ir.KindStruct {
			offs, err := l.FieldOffsets(named)
			if err != nil {
				return 0, err
			}
			off += offs[i]
		}
		t = fields[i].Type
	}
	return int64(off), nil
}
