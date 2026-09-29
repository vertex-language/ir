package amd64

// Swift's registers: the ones a value travels in by declaration rather than
// by its position in the argument sequence. See ir.SwiftSelf, ir.SwiftAsync,
// ir.SwiftIndirectResult and ir.SwiftError.
//
// Neither SysV nor the Microsoft convention has any of them; they are
// Swift's, laid over both:
//
//	self             R13
//	async context    R14
//	error            R12
//	indirect result  RAX under SysV; the first argument under Microsoft
//
// which is what LLVM's X86 calling conventions assign for swiftself,
// swiftasync, swifterror and a Swift sret, and so what swiftc's own code
// expects on either host. The first three are callee-saved under both
// conventions, which is why Swift chose them: a value passed in one
// survives calls to ordinary functions for free. The indirect result is
// the one that differs. SysV gives it RAX, which is not an argument
// register there; LLVM's Win64 convention has no such rule, and swiftc's
// code for Windows takes it in RCX like any first argument -- a generic
// `func g<T>(_ x: T) -> T` reads its metadata from R8, the third.
//
// None of them takes a place in the argument sequence. An argument after
// one is placed as though it were not there -- under the Microsoft
// convention that means it does not spend one of the four positions
// either, and on the stack it does not own an eightbyte of home space.

import (
	"fmt"

	"github.com/vertex-language/amd64/reg"

	"github.com/vertex-language/ir"
)

// A swiftRole is which of Swift's registers a parameter travels in, if any.
type swiftRole uint8

const (
	roleNone swiftRole = iota
	roleSelf
	roleAsync
	roleOut
)

// The registers themselves, spelled once.
const (
	swiftSelfReg  = reg.R13Q
	swiftAsyncReg = reg.R14Q
	swiftOutReg   = reg.RAX
	swiftErrorReg = reg.R12Q
)

// register is the register a role names under the named convention, or
// false when the role is an ordinary argument there.
func (r swiftRole) register(abi string) (reg.R64, bool) {
	switch r {
	case roleSelf:
		return swiftSelfReg, true
	case roleAsync:
		return swiftAsyncReg, true
	case roleOut:
		if abi == abiMS {
			return 0, false
		}
		return swiftOutReg, true
	}
	return 0, false
}

func (r swiftRole) String() string {
	switch r {
	case roleSelf:
		return "self"
	case roleAsync:
		return "async context"
	case roleOut:
		return "indirect result"
	}
	return "none"
}

// roleOf reads a parameter's Swift attribute.
func roleOf(attrs []ir.ParamAttr) swiftRole {
	for _, a := range attrs {
		switch {
		case a.IsSwiftSelf():
			return roleSelf
		case a.IsSwiftAsync():
			return roleAsync
		case a.IsSwiftIndirectResult():
			return roleOut
		}
	}
	return roleNone
}

// swiftPlace is where a parameter with a Swift role travels under the
// named convention: its register, beside the argument sequence rather than
// in it. Only an integer or a pointer fits in one. It reports false for a
// parameter that is placed as an ordinary argument, which is one with no
// role, or a role the convention gives no register of its own.
func swiftPlace(abi string, a abiArg) (place, bool, error) {
	r, ok := a.role.register(abi)
	if !ok {
		return place{}, false, nil
	}
	w, ok := widthOf(a.t)
	if !ok || w.isFloat() || w == wv128 {
		return place{}, false, fmt.Errorf("%s is not a value the Swift %s register carries", a.t, a.role)
	}
	return place{kind: placeFixed, regs: []regSlot{{kind: placeFixed, w: w, fixed: r}}, scalarW: w}, true, nil
}

// argIntReg is the integer register an argument slot names: a Swift
// register by declaration, or the convention's register at the slot's
// position.
func argIntReg(abi string, slot regSlot) reg.R64 {
	if slot.kind == placeFixed {
		return slot.fixed
	}
	return intArgReg(abi, slot.i)
}

// errorResult is the index of a signature's swifterror result, or -1.
//
// It travels in R12 rather than in the return sequence, so it is left out
// of the ordinary placement at both ends: the results before and after it
// are placed as though it were not there.
func errorResult(sig *ir.Sig) int {
	if sig == nil {
		return -1
	}
	for i, r := range sig.Rets() {
		for _, a := range r.Attrs {
			if a.IsSwiftError() {
				return i
			}
		}
	}
	return -1
}

// funcErrorResult is errorResult for a function's own signature.
func funcErrorResult(fn *ir.Func) int {
	if fn == nil {
		return -1
	}
	return errorResult(fn.Signature())
}

// withoutIndex is types with the one at i left out, or types itself when
// i is negative: the results the return sequence places, once the one in
// the error register is set aside.
func withoutIndex[T any](types []T, i int) []T {
	if i < 0 || i >= len(types) {
		return types
	}
	out := make([]T, 0, len(types)-1)
	out = append(out, types[:i]...)
	return append(out, types[i+1:]...)
}
