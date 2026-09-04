package i386

import (
	i386asm "github.com/vertex-language/i386"
	"github.com/vertex-language/i386/operand"
	"github.com/vertex-language/i386/reg"

	"github.com/vertex-language/ir"
	"github.com/vertex-language/ir/lower/mir"
	"github.com/vertex-language/ir/lower/regalloc"
)

// emit walks mf block by block and calls the typed helper for each op.
// It returns an error only for inline assembly, whose failure is a fact about
// text the frontend wrote and has to reach the caller with the position the
// assembler gave it.
func emit(am *i386asm.Module, text *i386asm.Section, fn *ir.Func, mf *mir.Func,
	assigned map[mir.VReg]regalloc.PhysReg, fr *frame, saved []reg.R32) error {

	r := func(v mir.VReg) reg.R32 { return reg.R32(assigned[v]) }
	rm := func(v mir.VReg) operand.RM32 { return r(v) }
	// The other register file. A PhysReg means nothing without its class —
	// EAX and XMM0 are both register zero — and which one an instruction
	// names is the op's own business, decided by the width it carries.
	x := func(v mir.VReg) reg.Xmm { return reg.Xmm(assigned[v]) }

	// A block a jump table entry or a blockaddr names has to be a symbol:
	// both are relocations, and a relocation needs one.
	labeled := labeledBlocks(mf)

	for i, mb := range mf.Blocks {
		switch {
		case i == 0:
			text.Label(fn.Name(), funcBinding(fn), i386asm.Func)
			emitPrologue(text, fr, saved)
		case labeled[mb.Label]:
			text.Label(mb.Label, i386asm.Local)
		default:
			text.Label(mb.Label)
		}

		for _, in := range mb.Instrs {
			switch op := in.Op.(type) {
			case movOp:
				dst, src := in.Defs[0], in.Uses[0]
				if assigned[dst] == assigned[src] {
					break
				}
				text.MovR32RM32(r(dst), rm(src))

			case constOp:
				text.MovR32Imm32(r(in.Defs[0]), op.imm)

			case aluOp:
				emitAlu(text, op, in, r, rm)

			case carryOp:
				if op.sub {
					text.SbbRM32R32(rm(in.Defs[0]), r(in.Uses[1]))
					break
				}
				text.AdcRM32R32(rm(in.Defs[0]), r(in.Uses[1]))

			case widenOp:
				if op.signed {
					// CDQ, which fills EDX with EAX's sign. The
					// unsigned case is a zeroing move, and it is a
					// move rather than XOR because the flags are
					// nobody's here and a move says so.
					text.Cdq()
					break
				}
				text.MovR32Imm32(reg.EDX, 0)

			case divOp:
				if op.signed {
					text.IdivRM32(rm(in.Uses[2]))
					break
				}
				text.DivRM32(rm(in.Uses[2]))

			case wideMulOp:
				if op.signed {
					text.ImulRM32(rm(in.Uses[1]))
					break
				}
				text.MulRM32(rm(in.Uses[1]))

			case unOp:
				if op.verb == ir.VNot {
					text.NotRM32(rm(in.Defs[0]))
					break
				}
				text.NegRM32(rm(in.Defs[0]))

			case signFillOp:
				// Every bit of the source's sign: copy it, then
				// arithmetic-shift right by thirty-one.
				text.MovR32RM32(r(in.Defs[0]), rm(in.Uses[0]))
				text.SarRM32Imm8(rm(in.Defs[0]), 31)

			case cmpOp:
				text.CmpRM32R32(rm(in.Uses[0]), r(in.Uses[1]))

			case cmpImmOp:
				text.CmpRM32Imm32(rm(in.Uses[0]), op.imm)

			case shiftOp:
				emitShift(text, op.verb, rm(in.Defs[0]))

			case shiftImmOp:
				emitShiftImm(text, op, rm(in.Defs[0]))

			case shiftDblOp:
				if op.right {
					text.ShrdRM32R32CL(rm(in.Defs[0]), r(in.Uses[1]))
					break
				}
				text.ShldRM32R32CL(rm(in.Defs[0]), r(in.Uses[1]))

			case bitScanOp:
				if op.reverse {
					text.BsrR32RM32(r(in.Defs[0]), rm(in.Uses[0]))
					break
				}
				text.BsfR32RM32(r(in.Defs[0]), rm(in.Uses[0]))

			case zextOp:
				d := r(in.Defs[0])
				if op.from == a8 {
					text.MovzxR32RM8(d, reg.R8(d))
					break
				}
				text.MovzxR32RM16(d, reg.R16(d))

			case bswapOp:
				text.BswapR32(r(in.Defs[0]))

			case testImmOp:
				text.TestRM32Imm32(rm(in.Uses[0]), op.imm)

			case testOp:
				text.TestRM32R32(rm(in.Uses[0]), r(in.Uses[0]))

			case setccOp:
				// SETcc writes one byte, so the register has to be
				// cleared first — and the clear has to come before
				// the compare that set the flags, which is why isel
				// never has a register to clear here and this zeroes
				// with a move rather than XOR.
				emitSetcc(text, op.cond, in.Defs[0], r)

			case cmovOp:
				emitCmov(text, op.cond, r(in.Defs[0]), rm(in.Uses[1]))

			case jccOp:
				// The conditional branch, then the fallthrough as an
				// unconditional one, unconditionally emitted: which
				// block follows is the block order's business and not
				// this instruction's.
				emitJcc(text, op.cond, op.then)
				text.JmpLabel(op.els)

			case asmOp:
				if err := emitAsm(text, fn, op, assigned); err != nil {
					return err
				}

			case jmpOp:
				text.JmpLabel(op.target)

			case trapOp:
				text.Int3()

			case retOp:
				emitEpilogue(text, fr, saved)

			case loadOp:
				text.MovR32RM32(r(in.Defs[0]), operand.Mem32(r(in.Uses[0])).Disp(op.off))

			case storeOp:
				text.MovRM32R32(operand.Mem32(r(in.Uses[1])).Disp(op.off), r(in.Uses[0]))

			case extLoadOp:
				emitExtLoad(text, op, in, r)

			case subStoreOp:
				emitSubStore(text, op, in, r)

			case addImmOp:
				text.AddRM32Imm32(rm(in.Defs[0]), op.imm)

			case frameOp:
				text.LeaR32M(r(in.Defs[0]), operand.Mem32(reg.EBP).Disp(op.off))

			case frameLoadOp:
				text.MovR32RM32(r(in.Defs[0]), operand.Mem32(reg.EBP).Disp(op.off))

			case spillOp:
				text.MovRM32R32(operand.Mem32(reg.EBP).Disp(op.off), r(in.Uses[0]))

			case reloadOp:
				text.MovR32RM32(r(in.Defs[0]), operand.Mem32(reg.EBP).Disp(op.off))

			case argStoreOp:
				text.MovRM32R32(operand.Mem32(reg.ESP).Disp(op.off), r(in.Uses[0]))

			case xchgOp:
				emitLockXchg(text, op.a, in, r, rm)

			case xaddOp:
				emitLockXadd(text, op.a, in, r, rm)

			case cmpxchgOp:
				emitLockCmpxchg(text, op.a, in, r, rm)

			case cmpxchg8bOp:
				text.LockCmpxchg8bM(operand.Mem64(r(in.Uses[4])))

			case fenceOp:
				// A locked read-modify-write on the top of the stack.
				// MFENCE is SSE2 and this target predates it; the
				// lock prefix is the barrier, and the stack slot is
				// chosen because it is certainly in this thread's
				// cache and nobody else is reading it.
				text.LockAddRM32Imm8S(operand.Mem32(reg.ESP), 0)

			case fmovOp:
				dst, src := in.Defs[0], in.Uses[0]
				if assigned[dst] == assigned[src] {
					break
				}
				text.MovapsXmmXmmM128(x(dst), x(src))

			case fbinOp:
				emitFloatBinary(text, op, x(in.Defs[0]), x(in.Uses[1]))

			case fsqrtOp:
				if op.w == wf32 {
					text.SqrtssXmmXmmM32(x(in.Defs[0]), x(in.Uses[0]))
					break
				}
				text.SqrtsdXmmXmmM64(x(in.Defs[0]), x(in.Uses[0]))

			case fbitOp:
				emitFloatBit(text, op, x(in.Defs[0]), x(in.Uses[1]))

			case fminmaxOp:
				switch {
				case op.max && op.w == wf32:
					text.MaxssXmmXmmM32(x(in.Defs[0]), x(in.Uses[1]))
				case op.max:
					text.MaxsdXmmXmmM64(x(in.Defs[0]), x(in.Uses[1]))
				case op.w == wf32:
					text.MinssXmmXmmM32(x(in.Defs[0]), x(in.Uses[1]))
				default:
					text.MinsdXmmXmmM64(x(in.Defs[0]), x(in.Uses[1]))
				}

			case fcmpOp:
				if op.w == wf32 {
					text.UcomissXmmXmmM32(x(in.Uses[0]), x(in.Uses[1]))
					break
				}
				text.UcomisdXmmXmmM64(x(in.Uses[0]), x(in.Uses[1]))

			case floadOp:
				if op.w == wf32 {
					text.MovssXmmXmmM32(x(in.Defs[0]), operand.Mem32(r(in.Uses[0])))
					break
				}
				text.MovsdXmmXmmM64(x(in.Defs[0]), operand.Mem64(r(in.Uses[0])))

			case fstoreOp:
				if op.w == wf32 {
					text.MovssXmmM32Xmm(operand.Mem32(r(in.Uses[1])), x(in.Uses[0]))
					break
				}
				text.MovsdXmmM64Xmm(operand.Mem64(r(in.Uses[1])), x(in.Uses[0]))

			case fframeLoadOp:
				if op.w == wf32 {
					text.MovssXmmXmmM32(x(in.Defs[0]), operand.Mem32(reg.EBP).Disp(op.off))
					break
				}
				text.MovsdXmmXmmM64(x(in.Defs[0]), operand.Mem64(reg.EBP).Disp(op.off))

			case fargStoreOp:
				if op.w == wf32 {
					text.MovssXmmM32Xmm(operand.Mem32(reg.ESP).Disp(op.off), x(in.Uses[0]))
					break
				}
				text.MovsdXmmM64Xmm(operand.Mem64(reg.ESP).Disp(op.off), x(in.Uses[0]))

			case fspillOp:
				text.MovsdXmmM64Xmm(operand.Mem64(reg.EBP).Disp(op.off), x(in.Uses[0]))

			case freloadOp:
				text.MovsdXmmXmmM64(x(in.Defs[0]), operand.Mem64(reg.EBP).Disp(op.off))

			case cvtIntToFloatOp:
				if op.w == wf32 {
					text.Cvtsi2ssXmmRM32(x(in.Defs[0]), rm(in.Uses[0]))
					break
				}
				text.Cvtsi2sdXmmRM32(x(in.Defs[0]), rm(in.Uses[0]))

			case cvtFloatToIntOp:
				if op.from == wf32 {
					text.Cvttss2siR32XmmM32(r(in.Defs[0]), x(in.Uses[0]))
					break
				}
				text.Cvttsd2siR32XmmM64(r(in.Defs[0]), x(in.Uses[0]))

			case cvtFloatOp:
				if op.w == wf64 {
					text.Cvtss2sdXmmXmmM32(x(in.Defs[0]), x(in.Uses[0]))
					break
				}
				text.Cvtsd2ssXmmXmmM64(x(in.Defs[0]), x(in.Uses[0]))

			case movdToXmmOp:
				text.MovdXmmRM32(x(in.Defs[0]), rm(in.Uses[0]))

			case movdToGPOp:
				text.MovdRM32Xmm(rm(in.Defs[0]), x(in.Uses[0]))

			case pairToFloatOp:
				// Two dwords out and eight bytes back, through a
				// slot: MOVD crosses the files four at a time and
				// nothing joins two halves in a register.
				text.MovRM32R32(operand.Mem32(reg.EBP).Disp(op.off), r(in.Uses[0]))
				text.MovRM32R32(operand.Mem32(reg.EBP).Disp(op.off+4), r(in.Uses[1]))
				text.MovsdXmmXmmM64(x(in.Defs[0]), operand.Mem64(reg.EBP).Disp(op.off))

			case floatToPairOp:
				text.MovsdXmmM64Xmm(operand.Mem64(reg.EBP).Disp(op.off), x(in.Uses[0]))
				text.MovR32RM32(r(in.Defs[0]), operand.Mem32(reg.EBP).Disp(op.off))
				text.MovR32RM32(r(in.Defs[1]), operand.Mem32(reg.EBP).Disp(op.off+4))

			case fstReturnOp:
				// Out through memory and onto the x87 stack,
				// which is where the psABI says a float is
				// returned. FLD pushes it; the RET leaves it
				// there.
				if op.w == wf32 {
					text.MovssXmmM32Xmm(operand.Mem32(reg.EBP).Disp(op.off), x(in.Uses[0]))
					text.FldM32(operand.Mem32(reg.EBP).Disp(op.off))
					break
				}
				text.MovsdXmmM64Xmm(operand.Mem64(reg.EBP).Disp(op.off), x(in.Uses[0]))
				text.FldM64(operand.Mem64(reg.EBP).Disp(op.off))

			case fstpResultOp:
				// And off it again, which is what the caller of a
				// float-returning function has to do first.
				if op.w == wf32 {
					text.FstpM32(operand.Mem32(reg.EBP).Disp(op.off))
					text.MovssXmmXmmM32(x(in.Defs[0]), operand.Mem32(reg.EBP).Disp(op.off))
					break
				}
				text.FstpM64(operand.Mem64(reg.EBP).Disp(op.off))
				text.MovsdXmmXmmM64(x(in.Defs[0]), operand.Mem64(reg.EBP).Disp(op.off))

			case callOp:
				text.CallRef(i386asm.Ref(op.sym, i386asm.RefPC32))

			case allocaOp:
				// The rounding up, then ESP down by it, then the
				// block above whatever the outgoing area owns.
				scratch := r(in.Defs[1])
				text.MovR32RM32(scratch, rm(in.Uses[0]))
				text.AddRM32Imm32(scratch, maxAlign-1)
				text.AndRM32Imm32(scratch, ^int64(maxAlign-1)&0xffffffff)
				text.SubRM32R32(reg.ESP, scratch)
				text.LeaR32M(r(in.Defs[0]), operand.Mem32(reg.ESP).Disp(op.outArgs))

			case stackSaveOp:
				text.MovR32RM32(r(in.Defs[0]), reg.ESP)

			case stackRestoreOp:
				text.MovRM32R32(reg.ESP, r(in.Uses[0]))

			case blockAddrOp:
				// LEA of an absolute address, the same way a
				// symbol's address is taken: MOV takes an
				// immediate but not a symbolic one in this API.
				text.LeaR32M(r(in.Defs[0]),
					operand.Abs32(i386asm.Ref(op.label, i386asm.RefAbs32)))

			case brIndOp:
				text.JmpRM32(rm(in.Uses[0]))

			case callIndOp:
				text.CallRM32(rm(in.Uses[0]))

			case brTableOp:
				emitBrTable(am, text, op, in, r, rm)

			case symAddrOp:
				// LEA of an absolute address: one instruction, with
				// the whole address in the displacement. A 32-bit
				// address fits in an operand here, which is the one
				// thing easier on this architecture than on the other
				// two — arm64 needs ADRP and ADD for the same job.
				text.LeaR32M(r(in.Defs[0]), operand.Abs32(i386asm.Ref(op.sym, i386asm.RefAbs32)))
			}
		}
	}
	text.EndLabel(fn.Name())
	return nil
}

// emitBrTable is the range check and the jump through the table.
//
// One instruction for the lookup and the jump together: an indexed memory
// operand is a legal jump target on x86, and the entries are absolute
// addresses because this object is not position-independent. Both of those
// are things the 64-bit backends in this tree cannot do — arm64 needs a
// table of distances precisely because an absolute pointer into text is what
// a position-independent image may not hold.
func emitBrTable(am *i386asm.Module, text *i386asm.Section, op brTableOp, in mir.Instr,
	r func(mir.VReg) reg.R32, rm func(mir.VReg) operand.RM32) {

	sel := r(in.Uses[0])
	text.CmpRM32Imm32(sel, int64(len(op.targets)))
	text.JaeLabel(op.dflt)
	text.JmpRM32(operand.Abs32(i386asm.Ref(op.id, i386asm.RefAbs32)).Index(sel, 4))

	ro := am.Section(i386asm.ROData)
	ro.Align(4)
	ro.Label(op.id, i386asm.Local)
	for _, t := range op.targets {
		ro.Ref(i386asm.Ref(t, i386asm.RefAbs32))
	}
}

// emitPrologue pushes the frame pointer and takes the frame.
func emitPrologue(text *i386asm.Section, fr *frame, saved []reg.R32) {
	text.PushR32(reg.EBP)
	text.MovR32RM32(reg.EBP, reg.ESP)
	if n := fr.size(); n > 0 {
		text.SubRM32Imm32(reg.ESP, int64(n))
	}
	for _, s := range saved {
		text.MovRM32R32(operand.Mem32(reg.EBP).Disp(fr.saveAt[s]), s)
	}
}

// emitEpilogue undoes it. LEAVE is MOV ESP, EBP and POP EBP in one byte, and
// is right whether or not anything moved ESP in between.
func emitEpilogue(text *i386asm.Section, fr *frame, saved []reg.R32) {
	for _, s := range saved {
		text.MovR32RM32(s, operand.Mem32(reg.EBP).Disp(fr.saveAt[s]))
	}
	text.Leave()
	text.Ret()
}

// emitShift is a shift or rotate by CL. The count register is not named: the
// forms read CL and nothing else, which is why isel pins it.
func emitFloatBinary(text *i386asm.Section, op fbinOp, dst, src reg.Xmm) {
	if op.w == wf32 {
		switch op.verb {
		case ir.VAdd:
			text.AddssXmmXmmM32(dst, src)
		case ir.VSub:
			text.SubssXmmXmmM32(dst, src)
		case ir.VMul:
			text.MulssXmmXmmM32(dst, src)
		case ir.VDiv:
			text.DivssXmmXmmM32(dst, src)
		}
		return
	}
	switch op.verb {
	case ir.VAdd:
		text.AddsdXmmXmmM64(dst, src)
	case ir.VSub:
		text.SubsdXmmXmmM64(dst, src)
	case ir.VMul:
		text.MulsdXmmXmmM64(dst, src)
	case ir.VDiv:
		text.DivsdXmmXmmM64(dst, src)
	}
}

// emitFloatBit is the packed bitwise forms used on a scalar. The single- and
// double-precision spellings differ by one prefix byte and nothing else, and
// either would work on either width — the matching one is emitted anyway,
// because a listing that says andpd next to a double reads as what it is.
func emitFloatBit(text *i386asm.Section, op fbitOp, dst, src reg.Xmm) {
	if op.w == wf32 {
		switch op.op {
		case maskAnd:
			text.AndpsXmmXmmM128(dst, src)
		case maskAndn:
			text.AndnpsXmmXmmM128(dst, src)
		case maskOr:
			text.OrpsXmmXmmM128(dst, src)
		case maskXor:
			text.XorpsXmmXmmM128(dst, src)
		}
		return
	}
	switch op.op {
	case maskAnd:
		text.AndpdXmmXmmM128(dst, src)
	case maskAndn:
		text.AndnpdXmmXmmM128(dst, src)
	case maskOr:
		text.OrpdXmmXmmM128(dst, src)
	case maskXor:
		text.XorpdXmmXmmM128(dst, src)
	}
}

func emitShift(text *i386asm.Section, verb ir.Verb, dst operand.RM32) {
	switch verb {
	case ir.VShl:
		text.ShlRM32CL(dst)
	case ir.VUShr:
		text.ShrRM32CL(dst)
	case ir.VSShr:
		text.SarRM32CL(dst)
	case ir.VRotL:
		text.RolRM32CL(dst)
	case ir.VRotR:
		text.RorRM32CL(dst)
	}
}

func emitShiftImm(text *i386asm.Section, op shiftImmOp, dst operand.RM32) {
	switch op.verb {
	case ir.VShl:
		text.ShlRM32Imm8(dst, op.n)
	case ir.VUShr:
		text.ShrRM32Imm8(dst, op.n)
	case ir.VSShr:
		text.SarRM32Imm8(dst, op.n)
	}
}

func emitAlu(text *i386asm.Section, op aluOp, in mir.Instr,
	r func(mir.VReg) reg.R32, rm func(mir.VReg) operand.RM32) {

	dst, b := in.Defs[0], in.Uses[1]
	switch op.verb {
	case ir.VAdd:
		text.AddRM32R32(rm(dst), r(b))
	case ir.VSub:
		text.SubRM32R32(rm(dst), r(b))
	case ir.VAnd:
		text.AndRM32R32(rm(dst), r(b))
	case ir.VOr:
		text.OrRM32R32(rm(dst), r(b))
	case ir.VXor:
		text.XorRM32R32(rm(dst), r(b))
	case ir.VMul:
		text.ImulR32RM32(r(dst), rm(b))
	}
}

// emitSetcc materializes a condition. SETcc writes a byte, so the upper three
// are cleared afterwards with a zero-extension rather than before with a XOR
// — a XOR would set the flags SETcc is about to read.
func emitSetcc(text *i386asm.Section, cond condCode, dst mir.VReg, r func(mir.VReg) reg.R32) {
	d := r(dst)
	b := reg.R8(d)
	switch cond {
	case condE:
		text.SeteRM8(b)
	case condNE:
		text.SetneRM8(b)
	case condL:
		text.SetlRM8(b)
	case condLE:
		text.SetleRM8(b)
	case condG:
		text.SetgRM8(b)
	case condGE:
		text.SetgeRM8(b)
	case condB:
		text.SetbRM8(b)
	case condBE:
		text.SetbeRM8(b)
	case condA:
		text.SetaRM8(b)
	case condAE:
		text.SetaeRM8(b)
	case condO:
		text.SetoRM8(b)
	case condP:
		text.SetpRM8(b)
	case condNP:
		text.SetnpRM8(b)
	}
	text.MovzxR32RM8(d, b)
}

// emitCmov is a conditional move. Every condition, not just the one §F needs:
// this read CmovneR32RM32 unconditionally until §A6 asked for CMOVE and got
// the negation of what it wanted.
func emitCmov(text *i386asm.Section, cond condCode, dst reg.R32, src operand.RM32) {
	switch cond {
	case condE:
		text.CmoveR32RM32(dst, src)
	case condNE:
		text.CmovneR32RM32(dst, src)
	case condL:
		text.CmovlR32RM32(dst, src)
	case condLE:
		text.CmovleR32RM32(dst, src)
	case condG:
		text.CmovgR32RM32(dst, src)
	case condGE:
		text.CmovgeR32RM32(dst, src)
	case condB:
		text.CmovbR32RM32(dst, src)
	case condBE:
		text.CmovbeR32RM32(dst, src)
	case condA:
		text.CmovaR32RM32(dst, src)
	case condAE:
		text.CmovaeR32RM32(dst, src)
	}
}

func emitJcc(text *i386asm.Section, cond condCode, label string) {
	switch cond {
	case condE:
		text.JeLabel(label)
	case condNE:
		text.JneLabel(label)
	case condL:
		text.JlLabel(label)
	case condLE:
		text.JleLabel(label)
	case condG:
		text.JgLabel(label)
	case condGE:
		text.JgeLabel(label)
	case condB:
		text.JbLabel(label)
	case condBE:
		text.JbeLabel(label)
	case condA:
		text.JaLabel(label)
	case condAE:
		text.JaeLabel(label)
	case condP:
		text.JpLabel(label)
	case condNP:
		text.JnpLabel(label)
	}
}

func emitExtLoad(text *i386asm.Section, op extLoadOp, in mir.Instr, r func(mir.VReg) reg.R32) {
	dst, base := r(in.Defs[0]), r(in.Uses[0])
	switch {
	case op.from == a8 && op.signed:
		text.MovsxR32RM8(dst, operand.Mem8(base))
	case op.from == a8:
		text.MovzxR32RM8(dst, operand.Mem8(base))
	case op.signed:
		text.MovsxR32RM16(dst, operand.Mem16(base))
	default:
		text.MovzxR32RM16(dst, operand.Mem16(base))
	}
}

// emitSubStore writes the low bytes of a register, which is §D2's three
// truncating stores: the byte, the word, and the whole thing.
func emitSubStore(text *i386asm.Section, op subStoreOp, in mir.Instr, r func(mir.VReg) reg.R32) {
	src, base := r(in.Uses[0]), r(in.Uses[1])
	switch op.to {
	case a8:
		text.MovRM8R8(operand.Mem8(base), reg.R8(src))
	case a16:
		text.MovRM16R16(operand.Mem16(base), reg.R16(src))
	default:
		text.MovRM32R32(operand.Mem32(base), src)
	}
}

func emitLockXchg(text *i386asm.Section, a access, in mir.Instr,
	r func(mir.VReg) reg.R32, rm func(mir.VReg) operand.RM32) {

	val, base := r(in.Defs[0]), r(in.Uses[1])
	switch a {
	case a8:
		text.LockXchgRM8R8(operand.Mem8(base), reg.R8(val))
	case a16:
		text.LockXchgRM16R16(operand.Mem16(base), reg.R16(val))
	default:
		text.LockXchgRM32R32(operand.Mem32(base), val)
	}
}

func emitLockXadd(text *i386asm.Section, a access, in mir.Instr,
	r func(mir.VReg) reg.R32, rm func(mir.VReg) operand.RM32) {

	val, base := r(in.Defs[0]), r(in.Uses[1])
	switch a {
	case a8:
		text.LockXaddRM8R8(operand.Mem8(base), reg.R8(val))
	case a16:
		text.LockXaddRM16R16(operand.Mem16(base), reg.R16(val))
	default:
		text.LockXaddRM32R32(operand.Mem32(base), val)
	}
}

func emitLockCmpxchg(text *i386asm.Section, a access, in mir.Instr,
	r func(mir.VReg) reg.R32, rm func(mir.VReg) operand.RM32) {

	next, base := r(in.Uses[1]), r(in.Uses[2])
	switch a {
	case a8:
		text.LockCmpxchgRM8R8(operand.Mem8(base), reg.R8(next))
	case a16:
		text.LockCmpxchgRM16R16(operand.Mem16(base), reg.R16(next))
	default:
		text.LockCmpxchgRM32R32(operand.Mem32(base), next)
	}
}
