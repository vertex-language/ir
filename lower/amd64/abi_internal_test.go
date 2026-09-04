package amd64

import (
	"testing"

	"github.com/vertex-language/ir"
)

func i8() ir.FType  { return ir.StoreI8.FType() }
func i32() ir.FType { return ir.StoreI32.FType() }
func i64() ir.FType { return ir.StoreI64.FType() }
func f32() ir.FType { return ir.StoreF32.FType() }
func f64() ir.FType { return ir.StoreF64.FType() }

func TestClassifyAggregate(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)

	for _, tc := range []struct {
		name    string
		typ     ir.FType
		size    uint64
		classes []abiClass // nil means MEMORY
	}{
		{"two i32 in one eightbyte", m.Struct("a").Field("x", i32()).Field("y", i32()).FType(),
			8, []abiClass{classInteger}},
		{"two f32 in one eightbyte", m.Struct("b").Field("x", f32()).Field("y", f32()).FType(),
			8, []abiClass{classSSE}},
		{"i32 and f32 share an eightbyte, integer wins", m.Struct("c").Field("x", i32()).Field("y", f32()).FType(),
			8, []abiClass{classInteger}},
		{"two f64 are two SSE eightbytes", m.Struct("d").Field("x", f64()).Field("y", f64()).FType(),
			16, []abiClass{classSSE, classSSE}},
		{"i64 then f64", m.Struct("e").Field("x", i64()).Field("y", f64()).FType(),
			16, []abiClass{classInteger, classSSE}},
		{"three eightbytes is memory", m.Struct("f").Field("x", i64()).Field("y", i64()).Field("z", i64()).FType(),
			24, nil},
		{"a byte is one integer eightbyte", m.Struct("g").Field("x", i8()).FType(),
			1, []abiClass{classInteger}},
		{"an array of two f64", ir.Array(2, f64()),
			16, []abiClass{classSSE, classSSE}},
		{"an array of four f32 is two SSE eightbytes", ir.Array(4, f32()),
			16, []abiClass{classSSE, classSSE}},
		{"a nested struct is flattened to absolute offsets",
			m.Struct("outer").Field("a", i64()).Field("b", m.Struct("inner").Field("p", f32()).Field("q", f32()).FType()).FType(),
			16, []abiClass{classInteger, classSSE}},
		{"a union merges its members", m.Union("u").Field("n", i32()).Field("x", f32()).FType(),
			4, []abiClass{classInteger}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got, err := classifyAggregate(tc.typ)
			if err != nil {
				t.Fatalf("classifyAggregate: %v", err)
			}
			if got.size != tc.size {
				t.Errorf("size = %d, want %d", got.size, tc.size)
			}
			if tc.classes == nil {
				if !got.inMemory() {
					t.Errorf("classes = %v, want MEMORY", got.classes)
				}
				return
			}
			if got.inMemory() {
				t.Fatalf("classified as MEMORY, want %v", tc.classes)
			}
			if len(got.classes) != len(tc.classes) {
				t.Fatalf("classes = %v, want %v", got.classes, tc.classes)
			}
			for i := range tc.classes {
				if got.classes[i] != tc.classes[i] {
					t.Errorf("eightbyte %d = %v, want %v", i, got.classes[i], tc.classes[i])
				}
			}
		})
	}
}

// A field placed by hand so that it straddles an eightbyte boundary.
// There is no register pair that can hold half a value in each.
func TestClassifyStraddlingFieldIsMemory(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	st := m.Struct("straddle").FieldAt("x", i64(), 4).FType()

	got, err := classifyAggregate(st)
	if err != nil {
		t.Fatalf("classifyAggregate: %v", err)
	}
	if !got.inMemory() {
		t.Errorf("classes = %v, want MEMORY: the i64 at offset 4 spans both eightbytes", got.classes)
	}
}
