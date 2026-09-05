package arm64

import (
	"github.com/vertex-language/ir"
	"github.com/vertex-language/ir/lower/asmtmpl"
	"github.com/vertex-language/ir/lower/mir"
	"github.com/vertex-language/ir/lower/regalloc"
)

// width is the operand width one MIR op runs at.
type width uint8

const (
	w32 width = iota
	w64
	wf32
	wf64
)

// isFloat reports whether the width names a value in a vector register.
func (w width) isFloat() bool { return w >= wf32 }

// class is the register file a width lives in, which is what regalloc needs.
func (w width) class() regalloc.Class {
	if w.isFloat() {
		return vecClass
	}
	return regalloc.DefaultClass
}

// vecClass is the SIMD and floating-point register file. DefaultClass is the
// general-purpose one.
const vecClass regalloc.Class = 1

// widthOf is the width a VIR register type occupies, and whether this package
// can hold that type in a register at all.
func widthOf(t ir.RegType) (width, bool) {
	switch t {
	case ir.TypeI1, ir.TypeI32:
		return w32, true
	case ir.TypeI64, ir.TypePtr:
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
	a64 access = 8
)

// This package's own MIR opcodes. mir.Instr.Op is `any`; these are the only
// values it is ever set to here.
type (
	// aluOp is every §A verb that is one three-address instruction. Three
	// addresses is the difference from the amd64 backend's two: a
	// destination distinct from both operands is the normal shape here, so
	// nothing has to be copied into place first.
	aluOp struct {
		verb ir.Verb
		w    width
	}

	// unOp is a one-source data-processing instruction.
	unOp struct {
		verb ir.Verb
		w    width
	}

	// i1NotOp is i1.not, which emits EOR rather than ORN to invert only the low bit.
	i1NotOp struct{}

	// mulhOp is SMULH or UMULH: the half of a 64x64 product MUL discards.
	mulhOp struct{ signed bool }

	// divOp is SDIV or UDIV, which do not trap: A64 divides by zero to zero
	// and INT_MIN/-1 to INT_MIN, so §A's trapping division is a check this
	// package emits around them.
	divOp struct {
		signed bool
		w      width
	}

	// flagAluOp is ADDS or SUBS: the same arithmetic as aluOp, setting
	// NZCV. Only §A2 needs one, and only for the flags — the sum it also
	// produces is a destination nothing reads.
	flagAluOp struct {
		verb ir.Verb
		w    width
	}

	// rbitOp is RBIT, which reverses the bit order of its source.
	rbitOp struct{ w width }

	// msubOp is MSUB: Defs[0] takes Uses[2] − Uses[0]*Uses[1], which is
	// how a remainder is spelled on an architecture that has no remainder.
	msubOp struct{ w width }

	// movOp is a register-to-register copy in whichever file the width names.
	movOp struct{ w width }

	// constOp materializes an immediate. See emitConst: A64 has no
	// instruction that takes a 64-bit literal, so the wide ones are a MOVZ
	// and a run of MOVKs.
	constOp struct {
		imm int64
		w   width
	}

	// cmpOp sets NZCV from two registers; cmpImmOp from a register and a
	// literal.
	cmpOp    struct{ w width }
	cmpImmOp struct {
		imm int64
		w   width
	}

	// csetOp materializes a condition into a register as 0 or 1.
	csetOp struct{ cond condCode }

	// cselOp is §F's select.
	cselOp struct {
		cond condCode
		w    width
	}

	// brTableOp is §G2's br_table: a range check, then a jump through a
	// table of addresses in .rodata. Defs are two scratch registers the
	// allocator supplies; Uses[0] is the selector.
	brTableOp struct {
		id      string
		targets []string
		dflt    string
	}

	// brIndOp is BR through Uses[0], which is §G2's computed goto.
	brIndOp struct{}

	// blockAddrOp is a block's address: ADRP plus ADD, the same pair a
	// symbol's address is, against a label promoted to a symbol.
	blockAddrOp struct{ label string }

	// bcondOp is a two-way branch on the flags; bOp is the unconditional one.
	bcondOp struct {
		cond      condCode
		then, els string
	}
	bOp struct{ target string }

	// retOp is the epilogue and the return. The return values are not here:
	// isel pins them into their registers and they reach this as Uses.
	retOp struct{}

	// trapOp is BRK, which raises a breakpoint exception.
	trapOp struct{}

	// loadOp and storeOp are a full-width access through a pointer.
	loadOp  struct{ w width } // Defs[0] takes [Uses[0]]
	storeOp struct{ w width } // [Uses[1]] takes Uses[0]

	// extLoadOp carries the memory width and the register width, which on
	// this architecture is one instruction rather than a load and an extend:
	// LDRSB and LDRB are different mnemonics.
	extLoadOp struct {
		from   access
		signed bool
		w      width
	}
	subStoreOp struct{ to access } // the low `to` bytes of Uses[0]

	// extOp is SXTW, UXTB and their neighbours: a bitfield move that widens.
	extOp struct {
		from   access
		signed bool
	}

	// allocaOp is §D3's dynamic allocation: SP moved down by a value.
	//
	// Defs[0] is the address and Defs[1] a scratch register the rounding
	// needs — named as a destination so the allocator does not hand it to
	// something still live. outArgs is how much of the bottom of the frame
	// the outgoing argument area owns, which the new block sits above:
	// SP is where a call writes its stack arguments from, so an allocation
	// that started at SP would be underneath the next call's arguments.
	allocaOp struct{ outArgs int64 }

	// stackSaveOp and stackRestoreOp are §D3's token: SP into a register
	// and back. Both are the ADD-immediate-by-zero that MOV to and from SP
	// is, since SP is not an operand of an ordinary move.
	stackSaveOp    struct{}
	stackRestoreOp struct{}

	// addImmOp adds a literal to a register. The general path materializes
	// a constant and adds two registers; this is for the small offsets
	// §I's list walk moves by, which always fit ADD's immediate.
	addImmOp struct{ imm int64 }

	// frameOp is an address in this function's own frame, relative to FP.
	frameOp struct{ off int64 }

	// frameLoadOp and frameStoreOp reach a frame slot by displacement from
	// X29, which is what an incoming stack parameter and a spill both are.
	frameLoadOp struct {
		off int64
		w   width
	}
	frameStoreOp struct {
		off int64
		w   width
	}

	// argStoreOp writes an outgoing argument, from SP rather than X29: the
	// outgoing area is at the bottom of the frame, and SP is what points at
	// the bottom.
	argStoreOp struct {
		off int64
		w   width
	}

	// outArgAddrOp is the address of a place in the outgoing area, for an
	// aggregate the caller copies there rather than stores in one piece.
	outArgAddrOp struct{ off int64 }

	// loadAtOp reads one register of an aggregate out of the storage a byval
	// pointer names: Defs[0] takes [Uses[0] + off].
	loadAtOp struct {
		off int64
		w   width
	}

	// storeAtOp is its inverse, for a result that came back in registers:
	// [Uses[1] + off] takes Uses[0].
	storeAtOp struct {
		off int64
		w   width
	}

	// spillOp and reloadOp are what regalloc asks for when it runs out.
	spillOp struct {
		off   int64
		float bool
	}
	reloadOp struct {
		off   int64
		float bool
	}

	// callOp names the callee, with arguments and clobbers pinned.
	callOp struct{ sym string }

	// asmOp is §G4's inline assembly: a template, the references found in
	// it, and the vreg standing for each operand. It reaches emit as text
	// and leaves as instructions, through the same assembler every other op
	// in this package reaches through a typed helper. See asm.go.
	asmOp struct {
		template string
		refs     []asmtmpl.Ref
		ops      []asmOperand
		labels   []string
		emitted  map[string]string
		id       int
	}

	// callIndOp is BLR through Uses[0], which is §G's indirect call. The
	// target is a Use rather than a pinned register: every caller-saved
	// register is a destination of the call, so the one register the
	// address may not be in is any of those.
	callIndOp struct{}

	// symAddrOp is ADRP plus ADD: a symbol's address in two instructions,
	// which is how a 64-bit address reaches a register when every
	// instruction is four bytes wide.
	symAddrOp struct{ sym string }

	// tlvAddrOp is a thread-local's address in the calling thread: the
	// four instructions Mach-O's model asks for, as one mir instruction
	// because they are one indivisible sequence with a fixed register.
	//
	//	adrp x0, sym@TLVPPAGE
	//	ldr  x0, [x0, sym@TLVPPAGEOFF]   the descriptor
	//	ldr  x1, [x0]                    its thunk
	//	blr  x1                          returns the address in x0
	//
	// x0 is not a choice. The thunk's contract is that it takes the
	// descriptor there and hands the address back there, which is why
	// this cannot be built out of the ordinary address and call pieces:
	// they would let the allocator pick.
	tlvAddrOp struct{ sym string }

	// symGotAddrOp is ADRP plus LDR: a symbol's address read out of the
	// GOT, for a symbol whose address this link does not know.
	symGotAddrOp struct{ sym string }

	// fbinOp is every §A3 verb that is one three-address float
	// instruction. Which is most of them: FMIN and FMAX are IEEE-754-2019
	// minimum and maximum, FMINNM and FMAXNM are 754-2008 minNum and
	// maxNum, and §A3 asks for exactly those four. The other architecture
	// pays a fixup for all of them.
	fbinOp struct {
		verb ir.Verb
		w    width
	}

	// funOp is a one-source float instruction: FABS, FNEG, FSQRT, and the
	// four FRINTs §A3's rounding verbs name.
	funOp struct {
		verb ir.Verb
		w    width
	}

	// fmaOp is FMADD, which rounds once. Uses are the three operands of
	// a*b+c in that order.
	fmaOp struct{ w width }

	// fcmpOp sets NZCV from two float registers. The conditions that read
	// it are not the integer ones: an unordered compare sets C and V, so
	// ordered less-than is MI rather than LT and ordered less-or-equal is
	// LS rather than LE. See floatConds.
	fcmpOp struct{ w width }

	// fcselOp is §F's select in the vector file.
	fcselOp struct {
		cond condCode
		w    width
	}

	// cvtIntToFloatOp is SCVTF or UCVTF; from is the source's integer
	// width and w the destination float's.
	cvtIntToFloatOp struct {
		signed bool
		from   width
		w      width
	}

	// cvtFloatToIntOp is FCVTZS or FCVTZU, which round toward zero and
	// saturate: out of range clamps to the endpoint and a NaN gives zero.
	// That is §C2's saturating conversion exactly, so the `_sat_` verbs
	// are this instruction alone and the trapping ones are this
	// instruction behind a range check — the reverse of the other
	// architecture, where the trapping form is the cheap one.
	cvtFloatToIntOp struct {
		signed bool
		from   width
		w      width
	}

	// cvtFloatOp is FCVT between the two float widths.
	cvtFloatOp struct {
		from width
		w    width
	}

	// atomicLoadOp and atomicStoreOp are §H's plain accesses: LDR and STR
	// where the ordering asks for nothing, LDAR and STLR where it does.
	// Both are single-copy atomic at their natural alignment, which is the
	// alignment §H requires and the hardware faults without.
	atomicLoadOp struct {
		a       access
		w       width
		ordered bool
	}
	atomicStoreOp struct {
		a       access
		ordered bool
	}

	// ldxrOp and stxrOp are the exclusive pair every §H read-modify-write
	// is a loop around. stxrOp's Defs[0] is the status: zero if the store
	// happened, one if the reservation was lost and the loop runs again.
	ldxrOp struct {
		a       access
		w       width
		acquire bool
	}
	stxrOp struct {
		a       access
		release bool
	} // Defs[0] status; Uses are the value and the address

	// cbnzOp is the loop's back edge: branch when Uses[0] is not zero.
	// One instruction where a compare and a conditional branch would be
	// two, which in a retry loop is worth the opcode.
	cbnzOp struct{ then, els string }

	// clrexOp releases an exclusive reservation without storing.
	clrexOp struct{}

	// dmbOp is a data memory barrier, which is what §H's fence is when it
	// is not a compiler barrier.
	dmbOp struct{ barrier barrier }

	// floatToBitsOp and bitsToFloatOp are FMOV across the register files:
	// the same bits, read as the other kind of thing. §C3's bitcast, and
	// also how a float literal is materialized and how copysign moves a
	// sign bit.
	floatToBitsOp struct{ w width } // Defs[0] is the integer
	bitsToFloatOp struct{ w width } // Defs[0] is the float
)

// A condCode is an A64 condition, named as the ARM ARM names it.
type condCode uint8

const (
	condEQ condCode = iota
	condNE
	condLT // signed
	condLE // signed
	condGT // signed
	condGE // signed
	condLO // unsigned lower
	condLS // unsigned lower or same
	condHI // unsigned higher
	condHS // unsigned higher or same
	condMI // negative — float less-than, false for a NaN
	condPL // not negative — the negation of MI, true for a NaN
	condVS // overflow set
)

// A barrier is the shareability domain and access kind of a DMB.
type barrier uint8

const (
	barrierISH   barrier = iota // inner shareable, loads and stores
	barrierISHLD                // inner shareable, loads only
)

// An emitter is anything isel appends an instruction to.
type emitter interface{ Emit(mir.Instr) }

// A placeKind is which of AAPCS64's places a value is passed in.
type placeKind uint8

const (
	placeInt placeKind = iota
	placeFloat
	placeStack

	// placeIndirect is X8, AAPCS64 §6.9's indirect result location
	// register. A function returning something too large for X0 and X1 is
	// passed the address to write it to, and that address travels in X8
	// rather than in the argument sequence — so the first real argument is
	// still X0. Putting it in X0 instead, which this package did, shifts
	// every argument after it by one register.
	placeIndirect
)

// A regSlot is one register of an aggregate that travels in several: which
// file, which register of it, how wide, and which bytes of the aggregate it
// carries. off is not always a multiple of eight — a homogeneous aggregate of
// floats has one register every four bytes.
type regSlot struct {
	kind placeKind
	i    int
	w    width
	off  int64
}

// A place is where AAPCS64 puts one argument.
//
// A scalar uses kind, i, off and w. An aggregate classified into registers
// uses regs instead, and carries byval so that the two ends know there are
// bytes to move rather than a register to copy: the caller reads them out of
// the storage the pointer names, and the callee stores them into a slot so
// that the pointer its body reads through has something to point at.
type place struct {
	kind placeKind
	i    int
	off  int64
	w    width

	regs  []regSlot
	byval ir.FType
	size  uint64
	align uint64
}

// isAggregate reports whether this place carries bytes rather than a value.
// An aggregate passed by reference is not one: §5.4 replaced it with a
// pointer, and a pointer is an ordinary integer argument.
func (p place) isAggregate() bool { return !p.byval.IsZero() }
