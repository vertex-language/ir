package amd64

import (
	"testing"

	"github.com/vertex-language/amd64/reg"

	"github.com/vertex-language/ir"
)

// TestSwiftRegistersBesideTheSequence: a parameter Swift gives a register of
// its own takes that register and no position, so the argument after it is
// placed as though it were not there -- under both conventions. The
// registers are the ones swiftc's code uses: self R13, async context R14,
// and an indirect result in RAX under SysV but in the first position under
// the Microsoft convention, where LLVM gives it no register of its own.
func TestSwiftRegistersBesideTheSequence(t *testing.T) {
	ptr := func(role swiftRole) abiArg { return abiArg{t: ir.TypePtr, role: role} }
	i64 := abiArg{t: ir.TypeI64}

	for _, tc := range []struct {
		name string
		abi  string
		args []abiArg
		// want is each argument's register: a Swift one by declaration, or
		// the convention's register at the position it was given.
		want []reg.R64
	}{
		{"ms: self takes no position", abiMS,
			[]abiArg{ptr(roleSelf), i64}, []reg.R64{reg.R13Q, reg.RCX}},
		{"ms: async context takes no position", abiMS,
			[]abiArg{ptr(roleAsync), i64, i64}, []reg.R64{reg.R14Q, reg.RCX, reg.RDX}},
		{"ms: an indirect result is the first argument", abiMS,
			[]abiArg{ptr(roleOut), i64, ptr(roleSelf)}, []reg.R64{reg.RCX, reg.RDX, reg.R13Q}},
		{"ms: all three", abiMS,
			[]abiArg{ptr(roleOut), ptr(roleAsync), i64, ptr(roleSelf), i64},
			[]reg.R64{reg.RCX, reg.R14Q, reg.RDX, reg.R13Q, reg.R8Q}},
		{"sysv: self takes no position", abiSysV,
			[]abiArg{ptr(roleSelf), i64}, []reg.R64{reg.R13Q, reg.RDI}},
		{"sysv: an indirect result is RAX", abiSysV,
			[]abiArg{ptr(roleOut), i64, ptr(roleAsync)}, []reg.R64{reg.RAX, reg.RDI, reg.R14Q}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			places, err := classify(tc.abi, tc.args)
			if err != nil {
				t.Fatalf("classify: %v", err)
			}
			for i, pl := range places {
				if len(pl.regs) != 1 {
					t.Fatalf("argument %d: %d registers, want 1", i, len(pl.regs))
				}
				if got := argIntReg(tc.abi, pl.regs[0]); got != tc.want[i] {
					t.Errorf("argument %d: register %d, want %d", i, got, tc.want[i])
				}
			}
		})
	}
}

// TestSwiftRegisterRefusesAFloat: a Swift register is a general-purpose one,
// and a float that claimed it would have nowhere to go.
func TestSwiftRegisterRefusesAFloat(t *testing.T) {
	for _, abi := range []string{abiMS, abiSysV} {
		if _, err := classify(abi, []abiArg{{t: ir.TypeF64, role: roleSelf}}); err == nil {
			t.Errorf("%s: a swiftself double was placed", abi)
		}
	}
}

// TestSwiftErrorIsBesideTheReturn: the swifterror result is found wherever
// it is in the list, and the results the return sequence places are the
// others, in order.
func TestSwiftErrorIsBesideTheReturn(t *testing.T) {
	sig := ir.NewSig().Ret(ir.TypeI64).Ret(ir.TypePtr, ir.SwiftError)
	i := errorResult(sig)
	if i != 1 {
		t.Fatalf("errorResult = %d, want 1", i)
	}
	rest := withoutIndex([]ir.RegType{ir.TypeI64, ir.TypePtr}, i)
	if len(rest) != 1 || rest[0] != ir.TypeI64 {
		t.Fatalf("the placed results are %v, want [i64]", rest)
	}
	// Under the Microsoft convention one result is all there is room for,
	// so a throwing function's two would be refused if the error were not
	// set aside first.
	if _, err := classifyRet(abiMS, rest); err != nil {
		t.Fatalf("classifyRet: %v", err)
	}
	if errorResult(ir.NewSig().Ret(ir.TypeI64)) != -1 {
		t.Error("a signature with no swifterror result has one")
	}
}

// TestSysVReturnSlotsCarryTheTail: an aggregate that comes back in registers
// says how much of its last eightbyte is real, so the caller stores that much
// and no more -- a three-byte struct is three bytes of RAX, and a store of all
// eight would write past the storage set aside for it.
func TestSysVReturnSlotsCarryTheTail(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64Linux)
	for _, tc := range []struct {
		name  string
		typ   ir.FType
		bytes []uint64 // each slot's bytes, zero for a whole eightbyte
	}{
		{"three bytes", m.Struct("a").Field("x", i8()).Field("y", i8()).Field("z", i8()).FType(), []uint64{3}},
		{"eight bytes", m.Struct("b").Field("x", i64()).FType(), []uint64{0}},
		{"twelve bytes", m.Struct("c").Field("x", i32()).Field("y", i32()).Field("z", i32()).FType(), []uint64{0, 4}},
		{"three floats", m.Struct("d").Field("x", f32()).Field("y", f32()).Field("z", f32()).FType(), []uint64{0, 4}},
		// Padded to its alignment, so its second eightbyte is whole.
		{"a float after a double", m.Struct("e").Field("x", f64()).Field("y", f32()).FType(), []uint64{0, 0}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			agg, inRegs, err := sretInRegs(tc.typ, false)
			if err != nil || !inRegs {
				t.Fatalf("sretInRegs = %v, %v", inRegs, err)
			}
			slots := sretRetSlots(agg)
			if len(slots) != len(tc.bytes) {
				t.Fatalf("%d slots, want %d", len(slots), len(tc.bytes))
			}
			for i, s := range slots {
				if s.bytes != tc.bytes[i] {
					t.Errorf("slot %d carries %d bytes, want %d", i, s.bytes, tc.bytes[i])
				}
			}
		})
	}
}
