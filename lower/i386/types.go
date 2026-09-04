package i386

import (
	"github.com/vertex-language/ir"
	"github.com/vertex-language/ir/lower/asmtmpl"
	"github.com/vertex-language/ir/lower/mir"
	"github.com/vertex-language/ir/lower/regalloc"
)

// width is what one value occupies.
type width uint8

const (
	w32  width = iota // one general register
	w64               // a pair: a low half and a high half
	wf32              // one XMM register, low lane
	wf64              // the same, at double width
)

// pairs reports whether the width needs two registers.
func (w width) pairs() bool { return w == w64 }

// isFloat reports whether the width names a value in the vector file.
func (w width) isFloat() bool { return w >= wf32 }

// class is the register file a width lives in, which is what regalloc needs.
func (w width) class() regalloc.Class {
	if w.isFloat() {
		return vecClass
	}
	return regalloc.DefaultClass
}

// vecClass is the XMM register file. DefaultClass is the general one.
const vecClass regalloc.Class = 1

// widthOf is the width a VIR register type occupies, and whether this package
// can hold that type in registers at all.
//
// f80 is absent and stays absent. The Intel386 psABI's long double is the
// ten-byte x87 type, and this package's floats live in the vector unit —
// where there is no ten-byte anything. Reaching it would mean a second
// register file with a stack discipline, for one type.
func widthOf(t ir.RegType) (width, bool) {
	switch t {
	case ir.TypeI1, ir.TypeI32, ir.TypePtr:
		return w32, true
	case ir.TypeI64:
		return w64, true
	case ir.TypeF32:
		return wf32, true
	case ir.TypeF64:
		return wf64, true
	}
	return 0, false
}

// access is how many bytes of memory one §D2 sub-width instruction touches.
type access uint8

const (
	a8  access = 1
	a16 access = 2
	a32 access = 4
)

// This package's own MIR opcodes.
//
// Every one of them names single registers. A 64-bit operation is two of
// these over the two halves, decided in isel, rather than one op carrying a
// width the emitter has to expand — which keeps the register allocator
// looking at what it actually has to allocate.
type (
	// asmOp is §G4's inline assembly: a template, the references found in
	// it, and the vreg standing for each operand. It reaches emit as text
	// and leaves as instructions, through the same assembler every other op
	// here reaches through a typed helper. See asm.go.
	asmOp struct {
		template string
		refs     []asmtmpl.Ref
		ops      []asmOperand
		labels   []string
		emitted  map[string]string
		id       int
	}

	// aluOp is a two-address integer operation: Defs[0] takes
	// Uses[0] op Uses[1], with Uses[0] and Defs[0] the same register.
	aluOp struct{ verb ir.Verb }

	// carryOp is the second half of a 64-bit add or subtract: ADC and SBB,
	// which read the flag the first half left. Emitted immediately after
	// it, which is safe because the only instructions the allocator
	// inserts are moves and an x86 move does not touch the flags.
	carryOp struct{ sub bool }

	// unOp is a one-source operation on one register: NOT or NEG.
	unOp struct{ verb ir.Verb }

	// widenOp fills EDX ahead of a division: the sign of EAX for a signed
	// one, zero for an unsigned.
	widenOp struct{ signed bool }

	// divOp is DIV or IDIV. Defs are EAX and EDX, which take the quotient
	// and the remainder; Uses are those two and the divisor.
	//
	// No range check around it. x86 raises #DE for a zero divisor and for
	// the one signed quotient that does not fit, which is exactly the pair
	// §A says must trap — so on this architecture the trap is the
	// instruction's and not this package's.
	divOp struct{ signed bool }

	// wideMulOp is the one-operand MUL or IMUL: EDX:EAX takes the full
	// product of EAX and Uses[2].
	wideMulOp struct{ signed bool }

	// sbbSelfOp is SBB r, r after a compare: −1 where the borrow was set
	// and 0 where it was not. The negation of a 64-bit value needs it.
	sbbSelfOp struct{}

	// shiftOp is a shift or rotate by CL: Defs[0] takes Uses[0] shifted by
	// Uses[1], which is pinned to ECX because that is the only register
	// the variable-count forms read.
	//
	// x86 masks the count to five bits, which is §A5's "modulo the
	// namespace width" for free at thirty-two. At sixty-four the masking
	// is what the fixup below is for.
	shiftOp struct{ verb ir.Verb }

	// shiftImmOp is a shift by a literal.
	shiftImmOp struct {
		verb ir.Verb
		n    int64
	}

	// shiftDblOp is SHLD or SHRD: Defs[0] takes its own bits shifted by
	// CL with Uses[1]'s bits shifted in from the other end. One
	// instruction for the half of a 64-bit shift that crosses.
	shiftDblOp struct{ right bool }

	// bitScanOp is BSF or BSR, which find the lowest or highest set bit
	// and set ZF when there is none — which is how §A6's zero case is
	// told apart, since the destination is left undefined for it.
	bitScanOp struct{ reverse bool }

	// bswapOp reverses a register's bytes.
	bswapOp struct{}

	// movOp is a register-to-register copy.
	movOp struct{}

	// constOp materializes a literal into one register.
	constOp struct{ imm int64 }

	// cmpOp sets the flags from two registers; cmpImmOp from a register
	// and a literal.
	cmpOp    struct{}
	cmpImmOp struct{ imm int64 }

	// setccOp materializes a condition into a register as 0 or 1.
	setccOp struct{ cond condCode }
	cmovOp  struct{ cond condCode } // §F, at one register

	// testOp sets the flags from a register against itself; testImmOp
	// against a literal, which is how a shift count's bit five is read.
	testOp    struct{}
	testImmOp struct{ imm int64 }

	// jccOp is a two-way branch on the flags; jmpOp is the unconditional one.
	jccOp struct {
		cond      condCode
		then, els string
	}
	jmpOp struct{ target string }

	// retOp is the epilogue and the return. The return values are not
	// here: isel pins them into EAX and EDX and they reach this as Uses.
	retOp struct{}

	// trapOp is UD2's i386 spelling, INT3.
	trapOp struct{}

	// loadOp and storeOp are a full-width access through a pointer, at an
	// offset: the high half of a 64-bit value is the low half's address
	// plus four.
	loadOp  struct{ off int32 } // Defs[0] takes [Uses[0] + off]
	storeOp struct{ off int32 } // [Uses[1] + off] takes Uses[0]

	// extLoadOp is a load that widens, which on x86 is one instruction:
	// MOVZX and MOVSX name their source width in the mnemonic.
	extLoadOp struct {
		from   access
		signed bool
	}
	subStoreOp struct{ to access } // the low `to` bytes of Uses[0]

	// zextOp zero-extends a register in place, from a byte or a word.
	//
	// What it is for: a narrow XCHG or XADD writes only the low bytes of
	// its register and leaves the rest as it found them, so the old value
	// §H asks for arrives with whatever the operand had above it.
	zextOp struct{ from access }

	// signFillOp is the high half of a sign extension from 32 bits: the
	// source arithmetic-shifted right by 31, which is every bit of its
	// sign.
	signFillOp struct{}

	// allocaOp is §D3's dynamic allocation: ESP moved down by a value.
	// Defs[0] is the address and Defs[1] a scratch the rounding needs.
	allocaOp struct{ outArgs int32 }

	// stackSaveOp and stackRestoreOp are §D3's token: ESP into a register
	// and back.
	stackSaveOp    struct{}
	stackRestoreOp struct{}

	// blockAddrOp is a block's address as an absolute immediate, which a
	// 32-bit address fits — the one place this architecture is simpler
	// than the two 64-bit ones, where the same thing takes two
	// instructions or a PC-relative form.
	blockAddrOp struct{ label string }

	// brTableOp is §G2's br_table: a range check, then a jump through a
	// table of addresses indexed by the selector.
	brTableOp struct {
		id      string
		targets []string
		dflt    string
	}

	// brIndOp is an indirect jump through Uses[0]; callIndOp the call.
	brIndOp   struct{}
	callIndOp struct{}

	// addImmOp adds a literal to a register, for the small offsets §I's
	// list walk moves by.
	addImmOp struct{ imm int64 }

	// frameOp is an address in this function's own frame, relative to EBP.
	frameOp struct{ off int32 }

	// frameLoadOp reaches a frame slot by displacement from EBP, which is
	// what an incoming stack parameter is.
	frameLoadOp struct{ off int32 }

	// argStoreOp writes an outgoing argument, from ESP: the outgoing area
	// is at the bottom of the frame and ESP points at the bottom.
	argStoreOp struct{ off int32 }

	// spillOp and reloadOp are what regalloc asks for when it runs out;
	// fspillOp and freloadOp the same for the vector file. A spill slot is
	// four bytes and a float needs eight, so the float pair reserves two.
	spillOp   struct{ off int32 }
	reloadOp  struct{ off int32 }
	fspillOp  struct{ off int32 }
	freloadOp struct{ off int32 }

	// The §H forms. Each is the LOCK-prefixed instruction of its name;
	// XCHG carries an implicit lock and takes the prefix anyway, so that a
	// lowering that spells its atomics uniformly has no hole in the
	// pattern.
	xchgOp    struct{ a access } // Defs[0] swaps with [Uses[1]]
	xaddOp    struct{ a access } // Defs[0] takes the old value, memory the sum
	cmpxchgOp struct{ a access } // EAX against [Uses[2]], Uses[1] stored on a match

	// cmpxchg8bOp is the eight-byte compare-and-swap, and the only way
	// this architecture touches eight bytes atomically at all: EDX:EAX is
	// the expected value and what was read, ECX:EBX the value to store.
	cmpxchg8bOp struct{}

	// fenceOp is a locked read-modify-write on the stack, which is what a
	// full barrier is before SSE2 gave x86 MFENCE.
	fenceOp struct{}

	// The §A3 forms. Each writes only the low lane of its destination,
	// which is what makes a scalar float in an XMM register just its low
	// lane.
	fbinOp struct {
		verb ir.Verb
		w    width
	}

	// fmovOp is a register-to-register copy in the vector file: MOVAPS,
	// which moves all sixteen bytes. The lanes above the first hold
	// nothing, so moving them costs nothing and needs no width.
	fmovOp struct{}

	// fbitOp is one of the packed bitwise forms, used on a scalar: this is
	// how a sign bit is cleared, set or copied.
	fbitOp struct {
		op maskOp
		w  width
	}

	// fcmpOp is UCOMIS, which sets ZF, PF and CF rather than writing a
	// mask — so an ordinary SETcc reads it. PF is the unordered flag.
	fcmpOp struct{ w width }

	// fsqrtOp is SQRTSS or SQRTSD.
	fsqrtOp struct{ w width }

	// fminmaxOp is MINSS and its three siblings, which are not IEEE
	// minimum and maximum: see iselFloatMinMax for what makes up the
	// difference.
	fminmaxOp struct {
		max bool
		w   width
	}

	// floadOp and fstoreOp are a scalar access through a pointer.
	floadOp  struct{ w width }
	fstoreOp struct{ w width }

	// fframeOp and fframeStoreOp reach a frame slot, which is where a
	// float parameter arrives and where a spilled one goes.
	fframeLoadOp struct {
		off int32
		w   width
	}
	fframeStoreOp struct {
		off int32
		w   width
	}

	// fargStoreOp writes an outgoing float argument, from ESP.
	fargStoreOp struct {
		off int32
		w   width
	}

	// cvtIntToFloatOp is CVTSI2SS or CVTSI2SD, from a 32-bit integer:
	// there is no 64-bit form outside x86-64.
	cvtIntToFloatOp struct{ w width }

	// cvtFloatToIntOp is CVTTSS2SI or CVTTSD2SI, which truncate toward
	// zero and give the integer indefinite value — 0x80000000 — for a NaN
	// or a value that does not fit, rather than trapping.
	cvtFloatToIntOp struct{ from width }

	// cvtFloatOp is CVTSS2SD or CVTSD2SS.
	cvtFloatOp struct{ w width }

	// movdOp crosses between the register files, which is what a 32-bit
	// bitcast is.
	movdToXmmOp struct{}
	movdToGPOp  struct{}

	// pairToFloatOp assembles a double from two general registers through
	// a frame slot, which is the only route: MOVD crosses the files four
	// bytes at a time and there is no instruction to join two halves.
	pairToFloatOp struct{ off int32 }

	// floatToPairOp is the same in reverse.
	floatToPairOp struct{ off int32 }

	// fstReturnOp and fstpResultOp are the psABI's float return, which
	// happens in ST(0) and nowhere else: a returned value goes out through
	// memory onto the x87 stack, and a received one comes off it the same
	// way.
	fstReturnOp struct {
		off int32
		w   width
	}
	fstpResultOp struct {
		off int32
		w   width
	}

	// callOp names the callee, with arguments and clobbers pinned.
	callOp struct{ sym string }

	// symAddrOp is a symbol's address. One instruction here: a 32-bit
	// immediate holds a whole address, which is the one thing that is
	// easier on this architecture than on the other two.
	symAddrOp struct{ sym string }
)

// A maskOp is one of the packed bitwise forms.
type maskOp uint8

const (
	maskAnd  maskOp = iota // clear the bits the mask does not have
	maskAndn               // clear the bits the mask does have
	maskOr
	maskXor
)

// A condCode is an x86 condition, named as the manual names it.
type condCode uint8

const (
	condE condCode = iota
	condNE
	condL  // signed less
	condLE // signed less or equal
	condG
	condGE
	condB  // unsigned below
	condBE // unsigned below or equal
	condA
	condAE
	condO // overflow set, which is §A2's signed answer
	condP // parity, which after UCOMIS is the unordered flag
	condNP
)

// An emitter is anything isel appends an instruction to.
type emitter interface{ Emit(mir.Instr) }
