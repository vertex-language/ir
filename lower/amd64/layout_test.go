package amd64

// Milestone 17: the ABI layout table, tested against the psABI's own
// numbers rather than through a lowered object.
//
// These are white-box because the table is: sizeAlign answers a question
// no exported entry point asks on its own, and every caller of it —
// ptr.alloc's frame slot, a global's alignment, a struct initializer's
// padding — reaches it through a lowering that would obscure which of
// the two numbers was wrong.

import (
	"strings"
	"testing"

	"github.com/vertex-language/ir"
)

func TestScalarLayout(t *testing.T) {
	cases := []struct {
		s           ir.StoreType
		size, align uint64
	}{
		{ir.StoreI8, 1, 1},
		{ir.StoreI16, 2, 2},
		{ir.StoreI32, 4, 4},
		{ir.StoreI64, 8, 8},
		{ir.StorePtr, 8, 8},
		{ir.StoreF32, 4, 4},
		{ir.StoreF64, 8, 8},
		// Ten bytes of value in sixteen bytes of storage: the six are
		// padding SysV requires, not slack.
		{ir.StoreF80, 16, 16},
		{ir.StoreF128, 16, 16},
	}
	for _, tc := range cases {
		t.Run(tc.s.String(), func(t *testing.T) {
			size, align, err := sizeAlign(tc.s.FType())
			if err != nil {
				t.Fatalf("sizeAlign: %v", err)
			}
			if size != tc.size || align != tc.align {
				t.Errorf("sizeAlign = %d/%d, want %d/%d", size, align, tc.size, tc.align)
			}
		})
	}
}

// An array is its element size times its length, and it aligns the way
// one element does — no more, because an element's own size already
// includes whatever padding keeps the next one aligned.
func TestArrayLayout(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	rec := m.Struct("rec").
		Field("tag", ir.StoreI8.FType()).
		Field("big", ir.StoreI64.FType())

	cases := []struct {
		name        string
		t           ir.FType
		size, align uint64
	}{
		{"i8", ir.Array(3, ir.StoreI8.FType()), 3, 1},
		{"i32", ir.Array(4, ir.StoreI32.FType()), 16, 4},
		// The struct is 9 bytes of field padded to 16, so the array is
		// 32 and not 18: the padding is inside each element.
		{"struct", ir.Array(2, rec.FType()), 32, 8},
		{"nested", ir.Array(2, ir.Array(3, ir.StoreI16.FType())), 12, 2},
		{"empty", ir.Array(0, ir.StoreI64.FType()), 0, 8},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			size, align, err := sizeAlign(tc.t)
			if err != nil {
				t.Fatalf("sizeAlign: %v", err)
			}
			if size != tc.size || align != tc.align {
				t.Errorf("sizeAlign = %d/%d, want %d/%d", size, align, tc.size, tc.align)
			}
		})
	}
}

// { i8, i64, i32 } is the shape that shows all three of the struct rules
// at once: a field at the next offset its own type admits, an alignment
// taken from the strictest field, and a size rounded up to that so an
// array of the struct keeps every element aligned.
func TestStructLayout(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	rec := m.Struct("rec").
		Field("tag", ir.StoreI8.FType()).
		Field("big", ir.StoreI64.FType()).
		Field("n", ir.StoreI32.FType())

	size, align, err := sizeAlign(rec.FType())
	if err != nil {
		t.Fatalf("sizeAlign: %v", err)
	}
	if size != 24 || align != 8 {
		t.Errorf("sizeAlign = %d/%d, want 24/8", size, align)
	}

	offsets, err := fieldOffsets(rec)
	if err != nil {
		t.Fatalf("fieldOffsets: %v", err)
	}
	want := []uint64{0, 8, 16}
	if !equalOffsets(offsets, want) {
		t.Errorf("fieldOffsets = %v, want %v", offsets, want)
	}
}

// A struct with no fields is zero bytes aligned to one, which is what
// makes an array of it zero bytes rather than an error.
func TestEmptyStructLayout(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	e := m.Struct("empty")

	size, align, err := sizeAlign(e.FType())
	if err != nil {
		t.Fatalf("sizeAlign: %v", err)
	}
	if size != 0 || align != 1 {
		t.Errorf("sizeAlign = %d/%d, want 0/1", size, align)
	}
}

// A nested struct contributes its own alignment to the outer one, which
// is how a struct four levels down decides where an outer field lands.
func TestNestedStructLayout(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	inner := m.Struct("inner").
		Field("a", ir.StoreI32.FType()).
		Field("b", ir.StoreI64.FType()) // 16 bytes, aligned 8
	outer := m.Struct("outer").
		Field("tag", ir.StoreI8.FType()).
		Field("in", inner.FType())

	size, align, err := sizeAlign(outer.FType())
	if err != nil {
		t.Fatalf("sizeAlign: %v", err)
	}
	if size != 24 || align != 8 {
		t.Errorf("sizeAlign = %d/%d, want 24/8", size, align)
	}
	offsets, err := fieldOffsets(outer)
	if err != nil {
		t.Fatalf("fieldOffsets: %v", err)
	}
	if !equalOffsets(offsets, []uint64{0, 8}) {
		t.Errorf("fieldOffsets = %v, want [0 8]", offsets)
	}
}

// Fields that state their own offsets are placed where they say, and the
// size still rounds to the alignment the field types imply. §19.18 makes
// this all-or-none, so a module reaching here states every one.
func TestStatedOffsets(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	rec := m.Struct("rec").
		FieldAt("a", ir.StoreI8.FType(), 0).
		FieldAt("b", ir.StoreI32.FType(), 16)

	size, align, err := sizeAlign(rec.FType())
	if err != nil {
		t.Fatalf("sizeAlign: %v", err)
	}
	if size != 20 || align != 4 {
		t.Errorf("sizeAlign = %d/%d, want 20/4", size, align)
	}
	offsets, err := fieldOffsets(rec)
	if err != nil {
		t.Fatalf("fieldOffsets: %v", err)
	}
	if !equalOffsets(offsets, []uint64{0, 16}) {
		t.Errorf("fieldOffsets = %v, want [0 16]", offsets)
	}
}

// packed takes every field down to alignment one: no padding between
// them, and the struct itself aligned to one.
func TestPackedStruct(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	rec := m.Struct("rec").
		Field("tag", ir.StoreI8.FType()).
		Field("big", ir.StoreI64.FType()).
		Field("n", ir.StoreI32.FType()).
		Pack()

	size, align, err := sizeAlign(rec.FType())
	if err != nil {
		t.Fatalf("sizeAlign: %v", err)
	}
	if size != 13 || align != 1 {
		t.Errorf("sizeAlign = %d/%d, want 13/1", size, align)
	}
	offsets, err := fieldOffsets(rec)
	if err != nil {
		t.Fatalf("fieldOffsets: %v", err)
	}
	if !equalOffsets(offsets, []uint64{0, 1, 9}) {
		t.Errorf("fieldOffsets = %v, want [0 1 9]", offsets)
	}
}

// align states the aggregate's alignment, and the size rounds up to it
// so that an array of the struct still lands every element on it.
func TestAlignAttrRaisesStruct(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	rec := m.Struct("rec").
		Field("a", ir.StoreI32.FType()).
		Align(16)

	size, align, err := sizeAlign(rec.FType())
	if err != nil {
		t.Fatalf("sizeAlign: %v", err)
	}
	if size != 16 || align != 16 {
		t.Errorf("sizeAlign = %d/%d, want 16/16", size, align)
	}
}

// A union is its widest member rounded to its strictest alignment, every
// member beginning at zero.
func TestUnionLayout(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	u := m.Union("u").
		Field("b", ir.StoreI8.FType()).
		Field("q", ir.StoreI64.FType()).
		Field("arr", ir.Array(9, ir.StoreI8.FType()))

	size, align, err := sizeAlign(u.FType())
	if err != nil {
		t.Fatalf("sizeAlign: %v", err)
	}
	// The nine-byte array is the widest member and eight is the
	// strictest alignment, so the size rounds nine up to sixteen.
	if size != 16 || align != 8 {
		t.Errorf("sizeAlign = %d/%d, want 16/8", size, align)
	}
}

// An alias typedef is transparent to the table: it lays out as whatever
// it names, however many typedefs deep that is.
func TestAliasLayout(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	a := m.TypeOf("word", ir.StoreI64.FType())
	b := m.TypeOf("alias_of_word", a.FType())

	size, align, err := sizeAlign(b.FType())
	if err != nil {
		t.Fatalf("sizeAlign: %v", err)
	}
	if size != 8 || align != 8 {
		t.Errorf("sizeAlign = %d/%d, want 8/8", size, align)
	}
}

// The forms with no storage layout are refused rather than guessed at: a
// func typedef is a signature, and a struct nested inside itself is a
// module the builder should not have produced.
func TestLayoutRefusals(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)

	ft := m.FuncType("fn", ir.NewSig())
	if _, _, err := sizeAlign(ft.FType()); err == nil {
		t.Error("a func typedef has no storage layout")
	}

	if _, _, err := sizeAlign(ir.FType{}); err == nil {
		t.Error("the zero ftype has no layout")
	}

	self := m.Struct("self")
	self.Field("me", self.FType())
	_, _, err := sizeAlign(self.FType())
	if err == nil {
		t.Fatal("a struct containing itself should not lay out")
	}
	if !strings.Contains(err.Error(), "nests") {
		t.Errorf("err = %v, want the nesting bound", err)
	}
}

func equalOffsets(got, want []uint64) bool {
	if len(got) != len(want) {
		return false
	}
	for i := range got {
		if got[i] != want[i] {
			return false
		}
	}
	return true
}
