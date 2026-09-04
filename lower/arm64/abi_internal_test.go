package arm64

// §5.4's classification, against the answers clang gives for the same shapes.
// Every expectation here was read out of clang's own codegen for a callee
// taking the struct and one trailing argument: which register the struct
// arrived in, and which one the trailing argument did.

import (
	"testing"

	"github.com/vertex-language/ir"
)

func f32() ir.FType { return ir.StoreF32.FType() }
func f64() ir.FType { return ir.StoreF64.FType() }
func i64() ir.FType { return ir.StoreI64.FType() }
func i8() ir.FType  { return ir.StoreI8.FType() }
func i32() ir.FType { return ir.StoreI32.FType() }

func TestClassifyAggregate(t *testing.T) {
	m := ir.NewModule("t", ir.AArch64Linux)

	st := func(name string, fs ...ir.FType) ir.FType {
		s := m.Struct(name)
		for i, f := range fs {
			s.Field(string(rune('a'+i)), f)
		}
		return s.FType()
	}
	un := func(name string, fs ...ir.FType) ir.FType {
		u := m.Union(name)
		for i, f := range fs {
			u.Field(string(rune('a'+i)), f)
		}
		return u.FType()
	}

	for _, tc := range []struct {
		name string
		typ  ir.FType
		kind aggKind
		n    int
		w    width
		step uint64
	}{
		// One general-purpose register: eight bytes or fewer, not homogeneous.
		{"S1", st("S1", i8()), aggGPR, 1, w64, 8},
		{"S8", st("S8", i64()), aggGPR, 1, w64, 8},
		{"MIX", st("MIX", f32(), i32()), aggGPR, 1, w64, 8},

		// Two, up to and including sixteen bytes.
		{"S16", st("S16", i64(), i64()), aggGPR, 2, w64, 8},
		{"S9", st("S9", i64(), i8()), aggGPR, 2, w64, 8},

		// Past sixteen and not homogeneous: the caller's copy, by reference.
		{"S17", st("S17", i64(), i64(), i8()), aggIndirect, 1, w64, 8},

		// Homogeneous, one register per member, at the member's width.
		{"F2", st("F2", f32(), f32()), aggHFA, 2, wf32, 4},
		{"F4", st("F4", f32(), f32(), f32(), f32()), aggHFA, 4, wf32, 4},
		{"D2", st("D2", f64(), f64()), aggHFA, 2, wf64, 8},

		// Twenty-four bytes and still homogeneous: §5.4's size rule does not
		// reach an HFA, which is the case a size-first classifier gets wrong.
		{"D3", st("D3", f64(), f64(), f64()), aggHFA, 3, wf64, 8},

		// A fifth member is one too many, and twenty bytes decides it.
		{"F5", st("F5", f32(), f32(), f32(), f32(), f32()), aggIndirect, 1, w64, 8},

		// Nesting and arrays are flattened before counting.
		{"NEST", st("NEST", st("F2n", f32(), f32())), aggHFA, 2, wf32, 4},
		{"ARR", st("ARR", ir.Array(3, f32())), aggHFA, 3, wf32, 4},

		// A union counts as its widest member, not the sum of them.
		{"UF", un("UF", f32(), f32()), aggHFA, 1, wf32, 4},
		{"UM", un("UM", f32(), i32()), aggGPR, 1, w64, 8},
		{"WRAP", st("WRAP", un("UF2", f32(), f32())), aggHFA, 1, wf32, 4},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got, err := classifyAggregate(tc.typ)
			if err != nil {
				t.Fatalf("classifyAggregate: %v", err)
			}
			if got.kind != tc.kind || got.n != tc.n || got.w != tc.w || got.step != tc.step {
				t.Errorf("kind %v n %d w %v step %d; want kind %v n %d w %v step %d",
					got.kind, got.n, got.w, got.step, tc.kind, tc.n, tc.w, tc.step)
			}
		})
	}
}

// An aggregate with a quadword float in it keeps the by-reference passing it
// has always had here, rather than being classified into S and D registers
// that cannot hold one.
func TestClassifyF128StaysIndirect(t *testing.T) {
	m := ir.NewModule("t", ir.AArch64Linux)
	q := m.Struct("Q").Field("a", ir.StoreF128.FType()).FType()
	got, err := classifyAggregate(q)
	if err != nil {
		t.Fatalf("classifyAggregate: %v", err)
	}
	if got.kind != aggIndirect {
		t.Errorf("kind %v, want %v", got.kind, aggIndirect)
	}
}
