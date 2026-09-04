package verify_test

// §19.10 and §19.18: the two module-scope rules that survive into a
// finished module. The others are named in verify/module.go with where
// they are caught instead.

import (
	"errors"
	"testing"

	"github.com/vertex-language/ir"
	"github.com/vertex-language/ir/verify"
)

// wantItemFault checks that err is one module-scope Error — no function,
// no block — naming sentinel.
func wantItemFault(t *testing.T, err error, sentinel error) *verify.Error {
	t.Helper()

	if err == nil {
		t.Fatalf("verify: no error, want %v", sentinel)
	}
	if !errors.Is(err, sentinel) {
		t.Fatalf("verify: %v, want %v", err, sentinel)
	}
	var e *verify.Error
	if !errors.As(err, &e) {
		t.Fatalf("verify: %v is not a *verify.Error", err)
	}
	if e.Func != "" {
		t.Errorf("Func = %q, want \"\": a global is in no function", e.Func)
	}
	return e
}

func i32() ir.FType { return ir.StoreI32.FType() }
func i8() ir.FType  { return ir.StoreI8.FType() }

// §19.10, the shapes that match.
func TestInitializerAccepts(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)

	pair := m.Struct("pair").Field("a", i32()).Field("b", i8())
	choice := m.Union("choice").Field("n", i32()).Field("c", i8())

	m.Global("scalar", ir.RO, i32()).Init(ir.Lit(ir.Int(42)))
	m.Global("zeroed", ir.RW, ir.Array(4, i32())).Init(ir.ZeroInit)
	m.Global("array", ir.RO, ir.Array(3, i32())).
		Init(ir.List(ir.Lit(ir.Int(1)), ir.Lit(ir.Int(2)), ir.Lit(ir.Int(3))))
	m.Global("text", ir.RO, ir.Array(6, i8())).Init(ir.Str("hello"))
	m.Global("positional", ir.RO, pair.FType()).
		Init(ir.List(ir.Lit(ir.Int(1)), ir.Lit(ir.Int(2))))
	m.Global("named", ir.RO, pair.FType()).
		Init(ir.Fields(ir.Val("b", ir.Lit(ir.Int(2))))) // partial: @a is zero
	m.Global("member", ir.RO, choice.FType()).
		Init(ir.Fields(ir.Val("c", ir.Lit(ir.Int(1)))))
	m.Global("nested", ir.RO, ir.Array(2, pair.FType())).
		Init(ir.List(
			ir.List(ir.Lit(ir.Int(1)), ir.Lit(ir.Int(2))),
			ir.ZeroInit,
		))
	m.Global("address", ir.RW, ir.StorePtr.FType()).
		Init(ir.RelocInit(m.Lookup("scalar")))

	if err := verify.Module(m); err != nil {
		t.Fatalf("verify.Module: %v", err)
	}
}

// §19.10: arity, in the form the rule is most often broken in.
func TestInitializerArrayArity(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	m.Global("g", ir.RO, ir.Array(3, i32())).
		Init(ir.List(ir.Lit(ir.Int(1)), ir.Lit(ir.Int(2))))

	wantItemFault(t, verify.Module(m), verify.ErrInit)
}

// §19.10: nesting. A scalar initializer does not fill an aggregate.
func TestInitializerScalarForAggregate(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	m.Global("g", ir.RO, ir.Array(3, i32())).Init(ir.Lit(ir.Int(0)))

	wantItemFault(t, verify.Module(m), verify.ErrInit)
}

// §19.10: element widths. A string fills an array of i8 and nothing else.
func TestInitializerStringType(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	m.Global("wide", ir.RO, ir.Array(6, i32())).Init(ir.Str("hello"))
	m.Global("short", ir.RO, ir.Array(2, i8())).Init(ir.Str("hello"))

	err := verify.Module(m)
	var es verify.Errors
	if !errors.As(err, &es) {
		t.Fatalf("verify.Module = %v (%T), want verify.Errors", err, err)
	}
	if len(es) != 2 {
		t.Fatalf("verify.Module found %d faults, want 2: %v", len(es), es)
	}
	for _, e := range es {
		if !errors.Is(e, verify.ErrInit) {
			t.Errorf("%v, want ErrInit", e)
		}
	}
}

// §19.10: a field initializer names fields the type has, once each.
func TestInitializerUnknownField(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	pair := m.Struct("pair").Field("a", i32()).Field("b", i32())
	m.Global("g", ir.RO, pair.FType()).
		Init(ir.Fields(ir.Val("c", ir.Lit(ir.Int(1)))))

	e := wantItemFault(t, verify.Module(m), verify.ErrInit)
	if e.Detail == "" {
		t.Error("Detail is empty; the fault has to name the field")
	}
}

// §19.18: at on all of a struct's fields or none of them.
func TestStructOffsetsPartial(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	m.Struct("half").FieldAt("a", i32(), 0).Field("b", i32())

	wantItemFault(t, verify.Module(m), verify.ErrStructOffset)
}

// §19.18: offsets strictly increase.
func TestStructOffsetsNotIncreasing(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	m.Struct("backwards").FieldAt("a", i32(), 8).FieldAt("b", i32(), 4)

	wantItemFault(t, verify.Module(m), verify.ErrStructOffset)
}

// §19.18: no field runs into its successor.
func TestStructOffsetsOverlap(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	m.Struct("overlap").
		FieldAt("wide", ir.StoreI64.FType(), 0).
		FieldAt("narrow", i32(), 4)

	e := wantItemFault(t, verify.Module(m), verify.ErrStructOffset)
	if e.Detail == "" {
		t.Error("Detail is empty; the fault has to say which field runs into which")
	}
}

// §19.18 accepts: every offset stated, none overlapping — and a struct
// with no offsets at all, which is the ordinary case the rule leaves to
// the target's own layout.
func TestStructOffsetsAccepts(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	m.Struct("stated").
		FieldAt("a", i32(), 0).
		FieldAt("b", ir.StoreI64.FType(), 8).
		FieldAt("tail", ir.Array(4, i8()), 16)
	m.Struct("computed").Field("a", i32()).Field("b", ir.StoreI64.FType())
	// An f80's padding is the target's, so nothing here can say whether
	// the field after one overlaps it. The rule declines rather than
	// guesses.
	m.Struct("extended").
		FieldAt("x", ir.StoreF80.FType(), 0).
		FieldAt("y", i32(), 12)

	if err := verify.Module(m); err != nil {
		t.Fatalf("verify.Module: %v", err)
	}
}
