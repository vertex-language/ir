package ptx

// The type table: what a VIR register type is in PTX.
//
// A register is declared by size and lives in a class by prefix — %r for
// 32 bits, %rd for 64, %f and %fd for the floats, %p for predicates. An
// instruction names its own type, and PTX's rule is that a bit-size
// register is compatible with any instruction type of the same size, so an
// i32 in a .b32 register can be added as .s32, shifted as .b32 and compared
// as .u32 without a move. Signedness rides in the verb here exactly as it
// does in VIR.

import (
	"fmt"

	"github.com/vertex-language/ir"
	"github.com/vertex-language/ptx"
)

// regType is the declared type of the register a VIR value lives in.
func regType(t ir.RegType) ptx.Type {
	switch t {
	case ir.TypeI1:
		return ptx.Pred
	case ir.TypeI32:
		return ptx.B32
	case ir.TypeI64, ir.TypePtr:
		return ptx.B64
	case ir.TypeF32:
		return ptx.F32
	case ir.TypeF64:
		return ptx.F64
	}
	panic(fmt.Sprintf("ptx: no register type for %s", t))
}

// paramType is the .param type a VIR value crosses a call boundary as. A
// predicate cannot be a parameter, so an i1 travels as a .b32 holding 0 or
// 1 and is converted on either side.
func paramType(t ir.RegType) ptx.Type {
	if t == ir.TypeI1 {
		return ptx.B32
	}
	return regType(t)
}

// The instruction types by namespace and interpretation.
func bitsT(t ir.RegType) ptx.Type {
	switch t {
	case ir.TypeI32, ir.TypeF32:
		return ptx.B32
	case ir.TypeI64, ir.TypePtr, ir.TypeF64:
		return ptx.B64
	case ir.TypeI1:
		return ptx.Pred
	}
	panic(fmt.Sprintf("ptx: no bit type for %s", t))
}

func signedT(t ir.RegType) ptx.Type {
	if t == ir.TypeI64 || t == ir.TypePtr {
		return ptx.S64
	}
	return ptx.S32
}

func unsignedT(t ir.RegType) ptx.Type {
	if t == ir.TypeI64 || t == ir.TypePtr {
		return ptx.U64
	}
	return ptx.U32
}

func floatT(t ir.RegType) ptx.Type {
	if t == ir.TypeF64 {
		return ptx.F64
	}
	return ptx.F32
}

// memT is the type a §D full-width access names: the float types for
// floats, the unsigned widths for everything else.
func memT(t ir.RegType) ptx.Type {
	switch t {
	case ir.TypeF32:
		return ptx.F32
	case ir.TypeF64:
		return ptx.F64
	}
	return unsignedT(t)
}

// width is the register width in bits.
func width(t ir.RegType) int {
	switch t {
	case ir.TypeI32, ir.TypeF32:
		return 32
	case ir.TypeI64, ir.TypePtr, ir.TypeF64:
		return 64
	}
	return 1
}
