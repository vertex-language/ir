package globals

import (
	"fmt"
	"strings"
	"testing"

	"github.com/vertex-language/ir"
)

// A layout that answers by rule rather than by any psABI: every scalar is its
// own width, every struct packs its fields end to end, and alignment is one.
// What it is for is the walk over a path — which field, which element, and
// what a union does — none of which is a question about a real target.
type flatLayout struct{}

func (flatLayout) SizeAlign(t ir.FType) (uint64, uint64, error) {
	switch t.Kind() {
	case ir.FTypeScalar:
		switch t.Scalar() {
		case ir.StoreI8:
			return 1, 1, nil
		case ir.StoreI32:
			return 4, 1, nil
		case ir.StoreI64:
			return 8, 1, nil
		}
	case ir.FTypeArray:
		n, _, err := flatLayout{}.SizeAlign(t.Elem())
		return n * t.Len(), 1, err
	case ir.FTypeNamed:
		named := t.Named()
		if named == nil {
			return 0, 0, fmt.Errorf("a named type with no definition")
		}
		if named.Kind() == ir.KindAlias {
			return flatLayout{}.SizeAlign(named.Aliased())
		}
		var total, widest uint64
		for _, f := range named.Fields() {
			n, _, err := flatLayout{}.SizeAlign(f.Type)
			if err != nil {
				return 0, 0, err
			}
			total += n
			if n > widest {
				widest = n
			}
		}
		if named.Kind() == ir.KindUnion {
			return widest, 1, nil
		}
		return total, 1, nil
	}
	return 0, 0, fmt.Errorf("no layout for %s", t)
}

func (flatLayout) FieldOffsets(t *ir.Type) ([]uint64, error) {
	offs := make([]uint64, len(t.Fields()))
	var at uint64
	for i, f := range t.Fields() {
		offs[i] = at
		n, _, err := flatLayout{}.SizeAlign(f.Type)
		if err != nil {
			return nil, err
		}
		at += n
	}
	return offs, nil
}

func TestConstIntPaths(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	i8, i32, i64 := ir.StoreI8.FType(), ir.StoreI32.FType(), ir.StoreI64.FType()

	rec := m.Struct("rec")
	rec.Field("a", i8)
	rec.Field("b", i64)
	rec.Field("c", i32)

	un := m.Union("un")
	un.Field("small", i32)
	un.Field("big", i64)

	arr := m.TypeOf("arr", ir.Array(3, rec.FType()))
	nest := m.Struct("nest")
	nest.Field("head", i32)
	nest.Field("tail", arr.FType())

	g := m.Global("g", ir.RW, i64).Export().Init(ir.Lit(ir.Int(0)))

	for _, tc := range []struct {
		name string
		c    ir.Const
		want int64
	}{
		{"plain", ir.Int(-7), -7},
		{"sizeof struct", ir.SizeOf(rec), 13},
		{"sizeof array", ir.SizeOf(arr), 39},
		{"sizeof global", ir.SizeOfSym(g), 8},
		{"alignof", ir.AlignOf(rec), 1},
		{"offsetof first", ir.OffsetOf(rec, ir.FieldPath("a")), 0},
		{"offsetof middle", ir.OffsetOf(rec, ir.FieldPath("b")), 1},
		{"offsetof last", ir.OffsetOf(rec, ir.FieldPath("c")), 9},
		{"offsetof indexed", ir.OffsetOf(arr, ir.IndexPath(2)), 26},
		{"offsetof through", ir.OffsetOf(arr, ir.IndexPath(2), ir.FieldPath("c")), 35},
		// One past the end, which is the address &arr[n] names.
		{"offsetof end", ir.OffsetOf(arr, ir.IndexPath(3)), 39},
		// A union's members all begin at zero, so naming one changes the
		// type and not the offset.
		{"offsetof union", ir.OffsetOf(un, ir.FieldPath("big")), 0},
		{"offsetof nested", ir.OffsetOf(nest, ir.FieldPath("tail"), ir.IndexPath(1), ir.FieldPath("b")), 18},
	} {
		got, err := ConstInt(flatLayout{}, tc.c)
		if err != nil {
			t.Errorf("%s: %v", tc.name, err)
			continue
		}
		if got != tc.want {
			t.Errorf("%s = %d, want %d", tc.name, got, tc.want)
		}
	}

	// And what it refuses, each by name.
	fn := m.Func("f")
	for _, tc := range []struct {
		name string
		c    ir.Const
		want string
	}{
		{"float", ir.Float(1.5), "not an integer constant"},
		{"no such field", ir.OffsetOf(rec, ir.FieldPath("nope")), `has no field "nope"`},
		{"index a struct", ir.OffsetOf(rec, ir.IndexPath(0)), "not an array"},
		{"field of an array", ir.OffsetOf(arr, ir.FieldPath("a")), "not a struct or a union"},
		{"past the end", ir.OffsetOf(arr, ir.IndexPath(4)), "which has 3 elements"},
		{"sizeof a function", ir.SizeOfSym(fn), "not a global"},
	} {
		_, err := ConstInt(flatLayout{}, tc.c)
		if err == nil {
			t.Errorf("%s: no error", tc.name)
			continue
		}
		if !strings.Contains(err.Error(), tc.want) {
			t.Errorf("%s: %v, want it to mention %q", tc.name, err, tc.want)
		}
	}
}
