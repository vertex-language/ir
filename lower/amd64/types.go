package amd64

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
	wv128 // a whole vector register: f128, and every v128
)

// isFloat reports whether the width names a value in a vector register.
// The name is the older half of the truth: every one of these lives in
// the vector file, and since v128 joined them not every one is a float.
func (w width) isFloat() bool { return w >= wf32 }

// class is the register file a width lives in, which is what regalloc needs.
func (w width) class() regalloc.Class {
	if w.isFloat() {
		return xmmClass
	}
	return regalloc.DefaultClass
}

// xmmClass is the vector register file. DefaultClass is the
// general-purpose one, which is what a vreg is in unless something says
// otherwise, and that is most of them.
const xmmClass regalloc.Class = 1

// widthOf is the width a VIR register type occupies in a general-purpose
// register, and whether this package can hold that type in one at all.
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
	case ir.TypeF128, ir.TypeV128:
		// One width for both, because they are one register. §3.2.3
		// classifies __float128 SSE and SSEUP, two eightbytes
		// travelling together in one XMM, and a v128 is the same
		// sixteen bytes read as lanes.
		return wv128, true
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

// This package's own MIR opcodes. mir.Instr.Op is `any`; these are the
// only values it is ever set to here.
type (
	// asmOp is §G4's inline assembly: a template, the references found in
	// it, and the vreg standing for each operand. It reaches emit as text
	// and leaves as instructions, through the same assembler every other
	// op here reaches through a typed helper. See asm.go.
	asmOp struct {
		template string
		refs     []asmtmpl.Ref
		ops      []asmOperand
		labels   []string
		emitted  map[string]string
		id       int

		// nouts is how many of ops are outputs, which is where the
		// instruction's Defs end and its Uses begin. The vregs in ops are
		// the ones isel built and are not the ones to allocate against:
		// a spill rewrites Defs and Uses and cannot reach inside an op's
		// payload, so an operand that was spilled would otherwise be
		// spelled with whatever register the stale vreg happened to map
		// to — register zero, which is RAX, for a vreg the allocator
		// never saw at all.
		nouts int
	}

	// aluOp is every §A verb that is one two-address instruction.
	aluOp struct {
		verb ir.Verb
		w    width
	}
	// unOp is §A's two in-place unary verbs, neg and not.
	unOp struct {
		verb ir.Verb
		w    width
	}

	// i1NotOp is i1.not, which emits XOR rather than NOT to invert only the low bit.
	i1NotOp struct{}
	trapOp  struct{}

	// divOp is a fixed-register shape for division.
	divOp struct {
		signed bool
		w      width
	}
	// mulOp is the widening multiply into RDX:RAX.
	mulOp struct {
		signed bool
		w      width
	}
	// signExtendOp is cdq/cqo: RAX's sign bit smeared through RDX.
	signExtendOp struct{ w width }
	// zeroOp clears a register.
	zeroOp struct{}
	// shiftOp is two-address like aluOp, with its count in CL.
	shiftOp struct {
		verb ir.Verb
		w    width
	}
	// cmovOp is §F's select.
	cmovOp struct {
		cond condCode
		w    width
	}
	// callOp names the callee, with arguments and clobbers pinned.
	callOp struct{ sym string }

	// callIndOp is an indirect call to the address in Uses[0]
	callIndOp struct{}

	// jmpIndOp is an indirect jump to the address in Uses[0]
	jmpIndOp struct{}

	// brTableOp is a jump table
	brTableOp struct {
		id            string
		targets       []string
		defaultTarget string
	}

	// returnOp is the epilogue and the ret. The return values are not
	// here: isel pins them into their registers and they reach this as
	// Uses, which is what keeps the copies alive to it.
	returnOp struct{}
	cmpOp    struct{ w width }
	brccOp   struct {
		cond      condCode
		then, els string // fully-qualified amd64 section labels
	}
	movOp  struct{ w width }
	swapOp struct{ w width } // exchange Defs[0] and Defs[1]; see emitParallelCopy
	// loadOp and storeOp carry an unaligned flag, which only wv128 reads:
	// a sixteen-byte vector move faults on a misaligned address and its
	// unaligned twin does not, and every narrower access on this target
	// is indifferent. The flag is set from §D4's align attribute.
	loadOp struct {
		w         width
		unaligned bool
	} // Defs[0] takes [Uses[0]]
	storeOp struct {
		w         width
		unaligned bool
	} // [Uses[1]] takes Uses[0]

	// extLoadOp carries memory width (from) and register width (w).
	extLoadOp struct {
		from   access
		signed bool
		w      width
	}
	subStoreOp struct{ to access }     // the low `to` bytes of Uses[0]
	jmpOp      struct{ target string } // fully-qualified amd64 section label
	constOp    struct {
		imm int64
		w   width
	}

	// wideConstOp materializes a 128-bit float literal, which does not
	// fit in an immediate and so goes in .rodata with a MOVAPS over it.
	// wideConstOp is sixteen bytes of literal, for the two register types
	// that are sixteen bytes: f128 and v128.
	wideConstOp struct{ lo, hi uint64 }

	// fAluOp is §A3's floating-point arithmetic.
	fAluOp struct {
		verb ir.Verb
		w    width
	}

	// fLogicOp is a packed logical instruction for manipulating sign bits.
	fLogicOp struct {
		op fLogic
		w  width
	}

	// fSqrtOp is SQRTSS or SQRTSD.
	fSqrtOp struct{ w width }

	// cvtIntToFloatOp is §C2's signed integer-to-float conversion.
	cvtIntToFloatOp struct {
		from width // w32 or w64, the general-purpose source
		to   width // wf32 or wf64, the vector destination
	}

	// cvtFloatToIntOp is §C2's truncating float-to-integer conversion.
	cvtFloatToIntOp struct {
		from width // wf32 or wf64
		to   width // w32 or w64
	}

	// cvtFloatOp is §C3's width change between the two float types.
	cvtFloatOp struct{ to width }

	// floatToBitsOp moves a bit pattern from a vector to general-purpose register.
	floatToBitsOp struct{ w width }

	// bitsToFloatOp moves a bit pattern from a general-purpose to vector register.
	bitsToFloatOp struct{ w width }

	// testOp sets the flags from Uses[0] against itself.
	testOp struct{}

	// atomicRmwOp is LOCK XADD.
	atomicRmwOp struct {
		a access
		w width
	}
	// atomicXchgOp is XCHG.
	atomicXchgOp struct {
		a access
		w width
	}
	// atomicCasOp is LOCK CMPXCHG.
	atomicCasOp struct {
		a access
		w width
	}

	// bitCountOp is POPCNT, LZCNT or TZCNT.
	bitCountOp struct {
		verb ir.Verb
		w    width
	}
	// bswapOp reverses a register in place.
	bswapOp struct{ w width }

	// mfenceOp is the full barrier.
	mfenceOp struct{}

	// setccOp materializes a compare's answer to 0 or 1.
	setccOp struct{ cond condCode }

	// zextOp zero-extends from 32 to 64 bits.
	zextOp struct{} // Defs[0] takes Uses[0]'s low 32 bits, zero-filled
	// sextOp sign-extends from 32 to 64 bits.
	sextOp struct{} // Defs[0] takes Uses[0]'s low 32 bits, sign-filled

	// argStoreOp writes an outgoing argument to the stack.
	argStoreOp struct {
		off int32
		w   width
	} // [RSP+off] takes Uses[0]
	// argLoadOp reads an incoming argument from the stack.
	argLoadOp struct {
		off int32
		w   width
	} // Defs[0] takes [RBP+off]

	// leaOp is the address of a frame slot.
	leaOp struct{ off int32 } // Defs[0] takes RBP+off, which is below RBP

	// allocaOp is §D3's dynamic allocation: RSP moves down by Uses[0],
	// Defs[0] takes the address, Defs[1] is the rounding's scratch. The
	// result is not the new RSP because the outgoing area is addressed
	// from RSP and has to stay at the bottom, so the allocation gets the
	// space above it.
	allocaOp struct{ outArgs int32 }

	// stackSaveOp and stackRestoreOp are §D3's bracket around a dynamic
	// allocation: RSP into a value, and a value back into RSP.
	stackSaveOp    struct{} // Defs[0] takes RSP
	stackRestoreOp struct{} // RSP takes Uses[0]

	// leaOutOp is an address in this function's outgoing argument area,
	// measured from RSP because that is where the area is.
	leaOutOp struct{ off int32 } // Defs[0] takes RSP+off

	// loadAtOp is a load from Uses[0] plus a constant displacement,
	// which is how a byval aggregate's eightbytes are read out of the
	// storage its pointer names.
	loadAtOp struct {
		off int32
		w   width
	} // Defs[0] takes [Uses[0]+off]

	// The ops §I's list needs: a memory operand at a constant
	// displacement from a pointer in a register. loadAtOp is above.
	storeAtOp struct { // [Uses[0]+off] takes Uses[1]
		off int32
		w   width
	}
	storeImmAtOp struct { // [Uses[0]+off] takes imm
		off int32
		imm int64
		w   width
	}
	addImmAtOp struct { // [Uses[0]+off] += imm
		off int32
		imm int64
		w   width
	}
	leaAtOp  struct{ off int32 } // Defs[0] takes Uses[0]+off
	andImmOp struct {            // Defs[0] takes Uses[0] & imm
		imm int64
		w   width
	}
	cmpImmOp struct { // flags from Uses[0] against imm
		imm int64
		w   width
	}

	// leaInOp is an address in the caller's outgoing area, above RBP.
	// The same instruction as leaOp; separate because the sign of the
	// displacement is this function's storage against the caller's, and
	// that is worth being unable to confuse.
	leaInOp struct{ off int32 } // Defs[0] takes RBP+off, which is above RBP
	// spillOp writes a vreg to the stack.
	spillOp struct {
		off int32
		w   width
	} // [RBP+off] takes Uses[0]
	// reloadOp reads a vreg from the stack.
	reloadOp struct {
		off int32
		w   width
	} // Defs[0] takes [RBP+off]
	// leaSymOp loads the address of a symbol.
	leaSymOp struct{ sym string } // Defs[0] takes the address of sym

	// movSymGotOp reads a symbol's address out of the GOT, for a symbol
	// whose address this link does not know.
	movSymGotOp struct{ sym string }

	// tlsAddrOp is a thread-local's address in the calling thread, under
	// PE's static model. Defs[0] takes it; Defs[1] is the scratch the
	// sequence needs for the module's slot index. See iselTLSAddr.
	tlsAddrOp struct{ sym string }

	// leaBlockOp is §D3's ptr.blockaddr: the address of a block in this
	// function. The label reaches the symbol table for it, the way a
	// jump table's targets do — see labeledBlocks.
	leaBlockOp struct{ label string } // Defs[0] takes the address of label
)

// fLogic is which of the four packed logical instructions an fLogicOp is.
type fLogic uint8

const (
	fAnd fLogic = iota
	fOr
	fXor
	// fAndn is (NOT dst) AND src, which is the operand order the name
	// does not suggest: the destination is the one that gets inverted.
	fAndn
)

// condCode is a condition this package knows how to branch on.
type condCode int

const (
	condE condCode = iota
	condNE
	condL  // signed
	condLE // signed
	condB  // unsigned
	condBE // unsigned
	condO  // overflow, which no comparison produces and §A2 does

	// The float readings. UCOMIS sets the flags an unsigned comparison
	// would set, so these are the unsigned conditions again — and PF,
	// which is set when either operand is NaN and which no integer
	// comparison ever produces.
	condA
	condAE
	condP
	condNP
)

// A rmwKind is which instruction answers one of §H's read-modify-writes.
type rmwKind uint8

const (
	rmwAdd  rmwKind = iota // LOCK XADD
	rmwSub                 // negated, then LOCK XADD
	rmwXchg                // XCHG, which needs no prefix to be atomic
	rmwLoop                // and, or and xor: a compare-and-swap loop
)

// A compare is what a §B row lowers to: the condition its branch tests, and the width.
type compare struct {
	cond condCode
	w    width
	swap bool
}

// condFor maps a compare to its shape.
func condFor(op ir.Op) (compare, bool) {
	switch op.Type {
	case ir.TypeI32:
		cc, ok := intConds[op.Verb]
		return compare{cond: cc, w: w32}, ok
	case ir.TypeI64:
		cc, ok := intConds[op.Verb]
		return compare{cond: cc, w: w64}, ok
	case ir.TypePtr:
		cc, ok := ptrConds[op.Verb]
		return compare{cond: cc, w: w64}, ok
	case ir.TypeF32, ir.TypeF64:
		f, ok := floatConds[op.Verb]
		w := wf32
		if op.Type == ir.TypeF64 {
			w = wf64
		}
		return compare{cond: f.cond, w: w, swap: f.swap}, ok
	}
	return compare{}, false
}

// A placeKind is which of SysV's three places a value is passed in.
type placeKind uint8

const (
	placeInt   placeKind = iota // one of the six integer registers
	placeFloat                  // one of the eight SSE registers
	placeStack                  // the caller's outgoing area
)

// A regSlot is one register an argument occupies: which file, which
// index into it, at what width. A scalar has one; a register-class byval
// has one per eightbyte, and they need not be in the same file.
type regSlot struct {
	kind placeKind
	i    int
	w    width
}

// A place is where SysV puts one argument: kind says which of the two
// shapes, with placeInt and placeFloat meaning regs and placeStack
// meaning off and size. An aggregate's kind is its first eightbyte's,
// which is only read to answer "is this on the stack".
type place struct {
	kind placeKind

	// regs is the registers this argument occupies, for a register
	// place. Length one for a scalar.
	regs []regSlot

	// off and size are where in the outgoing area a stack place sits and
	// how many bytes it takes. A scalar is always eight, whatever its
	// width: SysV gives each stack argument a whole eightbyte and leaves
	// the unused half of an i32's unspecified.
	off  int32
	size uint64

	// byval is the aggregate this argument stands for, zero for a
	// scalar: what tells a caller to copy bytes rather than move a
	// register.
	byval ir.FType

	// scalarW is a scalar's width, in a register or on the stack. Here
	// and not only on the regSlot because a stack scalar has no regSlot,
	// and the width decides whether its store is four bytes or eight.
	scalarW width

	// indirect says this aggregate travels as the address of a copy the
	// caller makes, rather than as its bytes. Only the Microsoft ABI
	// produces one: SysV either splits an aggregate into eightbytes or
	// pushes the whole thing, and neither is a pointer. copyOff is where
	// in the caller's outgoing area that copy lives — above every stack
	// argument, so that the two never share a byte.
	indirect bool
	copyOff  int32

	// dupInt says this float also travels in the integer register at the
	// same index. Only the Microsoft ABI's variadic tail produces one:
	// the callee's va_arg reads the home space, which is written from the
	// integer file, while a callee declared to take a double reads the
	// vector one, and a caller cannot know which it is talking to.
	dupInt bool
}

// w is the width of a scalar place.
func (p place) w() width { return p.scalarW }

// isAggregate reports whether this place carries a byval aggregate.
func (p place) isAggregate() bool { return !p.byval.IsZero() }

// An emitter is anything isel appends an instruction to: a MIR block, or a cursor over one.
type emitter interface{ Emit(mir.Instr) }

type hwMinMaxOp struct {
	isMax bool
	w     width
}

func (op hwMinMaxOp) String() string {
	if op.isMax {
		return "hw_max"
	}
	return "hw_min"
}

type fRoundOp struct {
	mode int64
	w    width
}

func (op fRoundOp) String() string {
	return "round"
}

type fmaOp struct {
	w width
}

func (op fmaOp) String() string {
	return "vfmadd231"
}
