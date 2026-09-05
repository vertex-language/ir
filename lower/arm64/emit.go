package arm64

import (
	arm64asm "github.com/vertex-language/arm64"
	"github.com/vertex-language/arm64/reg"

	"github.com/vertex-language/ir"
	"github.com/vertex-language/ir/lower/mir"
	"github.com/vertex-language/ir/lower/regalloc"
)

// emit walks mf block by block and calls the typed helper for each op.
// It returns an error only for inline assembly, which is the one op here whose
// failure is not the module's to record: a template that will not assemble is a
// fact about text the frontend wrote, and it has to reach the caller with the
// position the assembler gave it rather than becoming a sticky module error
// with a section offset for a location.
func emit(am *arm64asm.Module, text *arm64asm.Section, fn *ir.Func, mf *mir.Func,
	assigned map[mir.VReg]regalloc.PhysReg, fr *frame, saved []reg.X, savedVec []reg.V) error {

	x := func(v mir.VReg) reg.X { return reg.X(assigned[v]) }
	w := func(v mir.VReg) reg.W { return reg.W(assigned[v]) }
	d := func(v mir.VReg) reg.D { return reg.D(assigned[v]) }
	s := func(v mir.VReg) reg.S { return reg.S(assigned[v]) }
	vreg := func(v mir.VReg) reg.V { return reg.V(assigned[v]) }

	// A block a blockaddr names has to be a symbol: the page of an address
	// is not something this layer can fold, since nothing here has decided
	// where the section loads. A jump table's targets need no such thing —
	// its entries are distances, which fold.
	labeled := labeledBlocks(mf)

	// The jump tables this function needs, emitted after its last
	// instruction rather than where the branch is: they live in .text so
	// that the distances in them are same-section and fold to constants,
	// and .text is a stream of instructions everywhere else.
	var tables []brTableOp

	for i, mb := range mf.Blocks {
		switch {
		case i == 0:
			text.Label(fn.Name(), funcBinding(fn), arm64asm.Func)
			emitPrologue(text, fr, saved, savedVec)
		case labeled[mb.Label]:
			text.Label(mb.Label, arm64asm.Local)
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
				switch {
				case op.w == wf32:
					text.FmovS(s(dst), s(src))
				case op.w == wf64:
					text.FmovD(d(dst), d(src))
				case op.w == w32:
					text.MovReg32(w(dst), w(src))
				default:
					text.MovReg64(x(dst), x(src))
				}

			case constOp:
				emitConst(text, op, x(in.Defs[0]), w(in.Defs[0]))

			case aluOp:
				emitAlu(text, op, in, x, w)

			case unOp:
				emitUnary(text, op, in, x, w)

			case i1NotOp:
				text.EorImm32(w(in.Defs[0]), w(in.Uses[0]), 1)

			case mulhOp:
				if op.signed {
					text.Smulh(x(in.Defs[0]), x(in.Uses[0]), x(in.Uses[1]))
					break
				}
				text.Umulh(x(in.Defs[0]), x(in.Uses[0]), x(in.Uses[1]))

			case extOp:
				emitExtend(text, op, x(in.Defs[0]), x(in.Uses[0]), w(in.Defs[0]), w(in.Uses[0]))

			case flagAluOp:
				emitFlagAlu(text, op, in, x, w)

			case rbitOp:
				if op.w == w32 {
					text.Rbit32(w(in.Defs[0]), w(in.Uses[0]))
					break
				}
				text.Rbit64(x(in.Defs[0]), x(in.Uses[0]))

			case cmpOp:
				if op.w == w32 {
					text.SubsShifted32(reg.WZR, w(in.Uses[0]), w(in.Uses[1]))
					break
				}
				text.SubsShifted64(reg.XZR, x(in.Uses[0]), x(in.Uses[1]))

			case cmpImmOp:
				if op.w == w32 {
					text.SubsImm32(reg.WZR, w(in.Uses[0]), op.imm)
					break
				}
				text.SubsImm64(reg.XZR, x(in.Uses[0]), op.imm)

			case csetOp:
				text.Cset64(x(in.Defs[0]), cond(op.cond))

			case cselOp:
				if op.w == w32 {
					text.Csel32(w(in.Defs[0]), w(in.Uses[0]), w(in.Uses[1]), cond(op.cond))
					break
				}
				text.Csel64(x(in.Defs[0]), x(in.Uses[0]), x(in.Uses[1]), cond(op.cond))

			case bcondOp:
				// The conditional branch, then the fallthrough as an
				// unconditional one. B.cond reaches ±1MB and B reaches
				// ±128MB, and emitting both unconditionally is what keeps
				// the range of the pair the range of the wider.
				text.BCond(cond(op.cond), arm64asm.Label(op.then))
				text.B(arm64asm.Label(op.els))

			case bOp:
				text.B(arm64asm.Label(op.target))

			case trapOp:
				text.Brk(0)

			case retOp:
				emitEpilogue(text, fr, saved, savedVec)

			case loadOp:
				emitLoad(text, op.w, in, x, w, d, s)

			case storeOp:
				emitStore(text, op.w, in, x, w, d, s)

			case extLoadOp:
				emitExtLoad(text, op, in, x, w)

			case subStoreOp:
				base := arm64asm.Mem8(x(in.Uses[1]))
				switch op.to {
				case a8:
					text.StrbImm(w(in.Uses[0]), base)
				case a16:
					text.StrhImm(w(in.Uses[0]), arm64asm.Mem16(x(in.Uses[1])))
				default:
					text.StrImm32(w(in.Uses[0]), arm64asm.Mem32(x(in.Uses[1])))
				}

			case allocaOp:
				emitAlloca(text, op, x(in.Defs[0]), x(in.Defs[1]), x(in.Uses[0]))

			case stackSaveOp:
				text.MovSp64(x(in.Defs[0]), reg.SP)

			case stackRestoreOp:
				text.MovSp64(reg.SP, x(in.Uses[0]))

			case addImmOp:
				text.AddImm64(x(in.Defs[0]), x(in.Uses[0]), op.imm)

			case frameOp:
				emitFrameAddr(text, x(in.Defs[0]), op.off)

			case frameLoadOp:
				emitFrameLoad(text, op.w, op.off, in, x, w, d, s)

			case spillOp:
				b, o := frameBase(text, op.off)
				if op.float {
					text.SturImmD(d(in.Uses[0]), arm64asm.Mem64(b).Off(o))
					break
				}
				text.SturImm64(x(in.Uses[0]), arm64asm.Mem64(b).Off(o))

			case reloadOp:
				b, o := frameBase(text, op.off)
				if op.float {
					text.LdurImmD(d(in.Defs[0]), arm64asm.Mem64(b).Off(o))
					break
				}
				text.LdurImm64(x(in.Defs[0]), arm64asm.Mem64(b).Off(o))

			case argStoreOp:
				emitArgStore(text, op, in, x, w, d, s)

			case frameStoreOp:
				emitFrameStore(text, op.w, op.off, in, x, w, d, s)

			case loadAtOp:
				emitLoadAt(text, op, in, x, w, d, s)

			case storeAtOp:
				emitStoreAt(text, op, in, x, w, d, s)

			case outArgAddrOp:
				// The outgoing area is measured from SP, and SP is not an
				// operand of an ordinary move — ADD by an immediate is how
				// its value is read.
				text.AddImm64(x(in.Defs[0]), reg.SP, op.off)

			case asmOp:
				if err := emitAsm(text, fn, op, assigned); err != nil {
					return err
				}

			case callOp:
				text.Bl(arm64asm.Ref(op.sym, arm64asm.RefCall26))

			case callIndOp:
				text.Blr(x(in.Uses[0]))

			case divOp:
				if op.w == w32 {
					if op.signed {
						text.Sdiv32(w(in.Defs[0]), w(in.Uses[0]), w(in.Uses[1]))
						break
					}
					text.Udiv32(w(in.Defs[0]), w(in.Uses[0]), w(in.Uses[1]))
					break
				}
				if op.signed {
					text.Sdiv64(x(in.Defs[0]), x(in.Uses[0]), x(in.Uses[1]))
					break
				}
				text.Udiv64(x(in.Defs[0]), x(in.Uses[0]), x(in.Uses[1]))

			case msubOp:
				if op.w == w32 {
					text.Msub32(w(in.Defs[0]), w(in.Uses[0]), w(in.Uses[1]), w(in.Uses[2]))
					break
				}
				text.Msub64(x(in.Defs[0]), x(in.Uses[0]), x(in.Uses[1]), x(in.Uses[2]))

			case fbinOp:
				emitFloatBinary(text, op, in, d, s)

			case funOp:
				emitFloatUnary(text, op, in, d, s)

			case fmaOp:
				if op.w == wf32 {
					text.FmaddS(s(in.Defs[0]), s(in.Uses[0]), s(in.Uses[1]), s(in.Uses[2]))
					break
				}
				text.FmaddD(d(in.Defs[0]), d(in.Uses[0]), d(in.Uses[1]), d(in.Uses[2]))

			case fcmpOp:
				if op.w == wf32 {
					text.FcmpS(s(in.Uses[0]), s(in.Uses[1]))
					break
				}
				text.FcmpD(d(in.Uses[0]), d(in.Uses[1]))

			case fcselOp:
				if op.w == wf32 {
					text.FcselS(s(in.Defs[0]), s(in.Uses[0]), s(in.Uses[1]), cond(op.cond))
					break
				}
				text.FcselD(d(in.Defs[0]), d(in.Uses[0]), d(in.Uses[1]), cond(op.cond))

			case cvtIntToFloatOp:
				emitIntToFloat(text, op, in, x, w, d, s)

			case cvtFloatToIntOp:
				emitFloatToInt(text, op, in, x, w, d, s)

			case cvtFloatOp:
				if op.w == wf64 {
					text.FcvtSToD(d(in.Defs[0]), s(in.Uses[0]))
					break
				}
				text.FcvtDToS(s(in.Defs[0]), d(in.Uses[0]))

			case floatToBitsOp:
				if op.w == wf32 {
					text.FmovSToW(w(in.Defs[0]), s(in.Uses[0]))
					break
				}
				text.FmovDToX(x(in.Defs[0]), d(in.Uses[0]))

			case bitsToFloatOp:
				if op.w == wf32 {
					text.FmovWToS(s(in.Defs[0]), w(in.Uses[0]))
					break
				}
				text.FmovXToD(d(in.Defs[0]), x(in.Uses[0]))

			case atomicLoadOp:
				emitAtomicLoad(text, op, in, x, w)

			case atomicStoreOp:
				emitAtomicStore(text, op, in, x, w)

			case ldxrOp:
				emitLdxr(text, op, in, x, w)

			case stxrOp:
				emitStxr(text, op, in, x, w)

			case cbnzOp:
				// The same pair bcondOp emits, for the same reason:
				// CBNZ reaches ±1MB and B reaches ±128MB, and
				// emitting both makes the pair's range the wider one.
				text.Cbnz32(w(in.Uses[0]), arm64asm.Label(op.then))
				text.B(arm64asm.Label(op.els))

			case clrexOp:
				text.Clrex()

			case dmbOp:
				if op.barrier == barrierISHLD {
					text.Dmb(arm64asm.ISHLD)
					break
				}
				text.Dmb(arm64asm.ISH)

			case brIndOp:
				text.Br(x(in.Uses[0]))

			case blockAddrOp:
				text.Adrp(x(in.Defs[0]), arm64asm.Ref(op.label, arm64asm.RefAdrPage21))
				text.AddImm64(x(in.Defs[0]), x(in.Defs[0]),
					arm64asm.PageOff(arm64asm.Ref(op.label, arm64asm.RefAddAbsLo12)))

			case brTableOp:
				emitBrTable(text, op, in, x, w)
				tables = append(tables, op)

			case symAddrOp:
				// ADRP and ADD: the page, then the offset within it. No
				// single instruction holds a 64-bit address when every
				// instruction is four bytes wide.
				text.Adrp(x(in.Defs[0]), arm64asm.Ref(op.sym, arm64asm.RefAdrPage21))
				text.AddImm64(x(in.Defs[0]), x(in.Defs[0]),
					arm64asm.PageOff(arm64asm.Ref(op.sym, arm64asm.RefAddAbsLo12)))

			case tlvAddrOp:
				// The descriptor's address, then the thunk out of it,
				// then the call. X1 is scratch and is declared a
				// destination of this instruction, so nothing the
				// allocator placed there is alive here.
				text.Adrp(reg.X0, arm64asm.Ref(op.sym, arm64asm.RefAdrTlvPage21))
				text.LdrImm64(reg.X0, arm64asm.Mem64(reg.X0).
					Off(arm64asm.TlvPageOff(arm64asm.Ref(op.sym, arm64asm.RefLdTlvLo12))))
				text.LdrImm64(reg.X1, arm64asm.Mem64(reg.X0))
				text.Blr(reg.X1)

			case symGotAddrOp:
				// ADRP and LDR: the GOT slot's page, then the pointer in it.
				text.Adrp(x(in.Defs[0]), arm64asm.Ref(op.sym, arm64asm.RefAdrGotPage21))
				text.LdrImm64(x(in.Defs[0]), arm64asm.Mem64(x(in.Defs[0])).
					Off(arm64asm.GotPageOff(arm64asm.Ref(op.sym, arm64asm.RefLd64GotLo12))))

			}
		}
		_ = vreg
	}
	text.EndLabel(fn.Name())

	for _, tb := range tables {
		text.Align(4)
		text.Label(tb.id, arm64asm.Local)
		for _, t := range tb.targets {
			text.LabelDiff(t, tb.id)
		}
	}
	return nil
}

// cond maps this package's condition names onto the assembler's.
func cond(c condCode) arm64asm.Cond {
	switch c {
	case condEQ:
		return arm64asm.EQ
	case condNE:
		return arm64asm.NE
	case condLT:
		return arm64asm.LT
	case condLE:
		return arm64asm.LE
	case condGT:
		return arm64asm.GT
	case condGE:
		return arm64asm.GE
	case condLO:
		return arm64asm.LO
	case condLS:
		return arm64asm.LS
	case condHI:
		return arm64asm.HI
	case condHS:
		return arm64asm.HS
	case condMI:
		return arm64asm.MI
	case condPL:
		return arm64asm.PL
	case condVS:
		return arm64asm.VS
	}
	return arm64asm.AL
}

// emitPrologue pushes the frame record and takes the frame.
//
// STP with pre-index is the push: it stores the pair and updates SP in one
// instruction, which is what makes a frame record cost one instruction rather
// than a subtract and two stores.
func emitPrologue(text *arm64asm.Section, fr *frame, saved []reg.X, savedVec []reg.V) {
	if !fr.needed() {
		return
	}
	text.StpPre64(reg.X29, reg.X30, arm64asm.Mem64(reg.SP).Pre(-16))
	text.MovSp64(reg.X29, reg.SP)
	if n := fr.size(); n > 0 {
		text.SubImm64(reg.SP, reg.SP, int64(n))
	}
	for _, r := range saved {
		b, o := frameBase(text, fr.saveAt[r])
		text.SturImm64(r, arm64asm.Mem64(b).Off(o))
	}
	// Sixty-four bits of each, which is all the ABI preserves and all an
	// f64 occupies.
	for _, r := range savedVec {
		b, o := frameBase(text, fr.saveAtVec[r])
		text.SturImmD(reg.D(r), arm64asm.Mem64(b).Off(o))
	}
}

// emitEpilogue undoes it. SP comes back from X29 rather than by adding the
// frame size, which is the same instruction count and is right whether or not
// anything moved SP in between.
func emitEpilogue(text *arm64asm.Section, fr *frame, saved []reg.X, savedVec []reg.V) {
	if fr.needed() {
		for _, r := range saved {
			b, o := frameBase(text, fr.saveAt[r])
			text.LdurImm64(r, arm64asm.Mem64(b).Off(o))
		}
		for _, r := range savedVec {
			b, o := frameBase(text, fr.saveAtVec[r])
			text.LdurImmD(reg.D(r), arm64asm.Mem64(b).Off(o))
		}
		text.MovSp64(reg.SP, reg.X29)
		text.LdpPost64(reg.X29, reg.X30, arm64asm.Mem64(reg.SP).Post(16))
	}
	text.Ret()
}

// usedCalleeSaved is the callee-saved registers this function's allocation
// actually named, in calleeSaved's order.
func usedCalleeSaved(pool *regalloc.Pool, assigned map[mir.VReg]regalloc.PhysReg) []reg.X {
	used := map[reg.X]bool{}
	for v, p := range assigned {
		if pool.ClassOf(v) != regalloc.DefaultClass {
			continue
		}
		used[reg.X(p)] = true
	}
	var out []reg.X
	for _, r := range calleeSaved {
		if used[r] {
			out = append(out, r)
		}
	}
	return out
}

// usedCalleeSavedVec is the same for the vector file.
//
// A call is written as clobbering all thirty-two vector registers, so nothing
// live across one is ever placed in V8 through V15 — but a value that is not
// live across a call still can be, once there are nine of them at once. That
// is a register this function has to hand back the way it found it, and this
// is the list of the ones it took.
func usedCalleeSavedVec(pool *regalloc.Pool, assigned map[mir.VReg]regalloc.PhysReg) []reg.V {
	used := map[reg.V]bool{}
	for v, p := range assigned {
		if pool.ClassOf(v) != vecClass {
			continue
		}
		used[reg.V(p)] = true
	}
	var out []reg.V
	for _, r := range calleeSavedVec {
		if used[r] {
			out = append(out, r)
		}
	}
	return out
}

// emitConst materializes an integer literal.
//
// No A64 instruction takes a literal wider than sixteen bits, so a constant
// is a move that plants one halfword and a MOVK for each further halfword
// that is not already right. Which move decides how many MOVKs follow: MOVZ
// leaves zeros everywhere else and MOVN leaves ones, so a constant is
// counted both ways and started from whichever side it already resembles.
// −1 is one MOVN and 0x7fffffffffffffff is a MOVN and three MOVKs, where
// starting from MOVZ would have cost four instructions for each.
func emitConst(text *arm64asm.Section, op constOp, xd reg.X, wd reg.W) {
	n := 4
	v := uint64(op.imm)
	if op.w == w32 {
		n, v = 2, uint64(uint32(op.imm))
	}

	half := func(i int) uint64 { return (v >> (16 * i)) & 0xffff }

	var zeros, ones int
	for i := 0; i < n; i++ {
		switch half(i) {
		case 0:
			zeros++
		case 0xffff:
			ones++
		}
	}

	// The halfword value the opening move does not have to state, and
	// whose repeats every later MOVK can skip.
	free, negate := uint64(0), false
	if ones > zeros {
		free, negate = 0xffff, true
	}

	shift := func(i int) []arm64asm.ShiftOp {
		if i == 0 {
			return nil
		}
		return []arm64asm.ShiftOp{arm64asm.Shifted(arm64asm.LSL, uint8(16*i))}
	}

	started := false
	for i := 0; i < n; i++ {
		h := half(i)
		if h == free {
			continue
		}
		switch {
		case started:
			if op.w == w32 {
				text.MovkImm32(wd, uint32(h), shift(i)...)
			} else {
				text.MovkImm64(xd, h, shift(i)...)
			}
		case negate:
			// MOVN plants the complement, so state the complement.
			if op.w == w32 {
				text.MovnImm32(wd, uint32(^h&0xffff), shift(i)...)
			} else {
				text.MovnImm64(xd, ^h&0xffff, shift(i)...)
			}
		default:
			if op.w == w32 {
				text.MovzImm32(wd, uint32(h), shift(i)...)
			} else {
				text.MovzImm64(xd, h, shift(i)...)
			}
		}
		started = true
	}

	// Every halfword was the one the opening move plants for free, which
	// is 0 or −1 and still needs the instruction that plants it.
	if !started {
		switch {
		case negate && op.w == w32:
			text.MovnImm32(wd, 0)
		case negate:
			text.MovnImm64(xd, 0)
		case op.w == w32:
			text.MovzImm32(wd, 0)
		default:
			text.MovzImm64(xd, 0)
		}
	}
}

func emitAlu(text *arm64asm.Section, op aluOp, in mir.Instr,
	x func(mir.VReg) reg.X, w func(mir.VReg) reg.W) {

	dst, a, b := in.Defs[0], in.Uses[0], in.Uses[1]
	if op.w == w32 {
		switch op.verb {
		case ir.VAdd:
			text.AddShifted32(w(dst), w(a), w(b))
		case ir.VSub:
			text.SubShifted32(w(dst), w(a), w(b))
		case ir.VMul:
			text.Mul32(w(dst), w(a), w(b))
		case ir.VAnd:
			text.AndShifted32(w(dst), w(a), w(b))
		case ir.VOr:
			text.OrrShifted32(w(dst), w(a), w(b))
		case ir.VXor:
			text.EorShifted32(w(dst), w(a), w(b))
		case ir.VShl:
			text.Lslv32(w(dst), w(a), w(b))
		case ir.VSShr:
			text.Asrv32(w(dst), w(a), w(b))
		case ir.VUShr:
			text.Lsrv32(w(dst), w(a), w(b))
		case ir.VRotR:
			text.Rorv32(w(dst), w(a), w(b))
		}
		return
	}
	switch op.verb {
	case ir.VAdd:
		text.AddShifted64(x(dst), x(a), x(b))
	case ir.VSub:
		text.SubShifted64(x(dst), x(a), x(b))
	case ir.VMul:
		text.Mul64(x(dst), x(a), x(b))
	case ir.VAnd:
		text.AndShifted64(x(dst), x(a), x(b))
	case ir.VOr:
		text.OrrShifted64(x(dst), x(a), x(b))
	case ir.VXor:
		text.EorShifted64(x(dst), x(a), x(b))
	case ir.VShl:
		text.Lslv64(x(dst), x(a), x(b))
	case ir.VSShr:
		text.Asrv64(x(dst), x(a), x(b))
	case ir.VUShr:
		text.Lsrv64(x(dst), x(a), x(b))
	case ir.VRotR:
		text.Rorv64(x(dst), x(a), x(b))
	}
}

// emitFlagAlu is ADDS and SUBS, whose destination §A2 does not read: the
// instruction is here for NZCV and the sum is a register the allocator had to
// find anyway, since an instruction that writes one has to say so.
func emitFlagAlu(text *arm64asm.Section, op flagAluOp, in mir.Instr,
	x func(mir.VReg) reg.X, w func(mir.VReg) reg.W) {

	dst, a, b := in.Defs[0], in.Uses[0], in.Uses[1]
	switch {
	case op.w == w32 && op.verb == ir.VAdd:
		text.AddsShifted32(w(dst), w(a), w(b))
	case op.w == w32:
		text.SubsShifted32(w(dst), w(a), w(b))
	case op.verb == ir.VAdd:
		text.AddsShifted64(x(dst), x(a), x(b))
	default:
		text.SubsShifted64(x(dst), x(a), x(b))
	}
}

func emitUnary(text *arm64asm.Section, op unOp, in mir.Instr,
	x func(mir.VReg) reg.X, w func(mir.VReg) reg.W) {

	dst, a := in.Defs[0], in.Uses[0]
	if op.w == w32 {
		switch op.verb {
		case ir.VNeg:
			text.SubShifted32(w(dst), reg.WZR, w(a))
		case ir.VNot:
			text.OrnShifted32(w(dst), reg.WZR, w(a))
		case ir.VClz:
			text.Clz32(w(dst), w(a))
		case ir.VBswap:
			text.Rev32(w(dst), w(a))
		}
		return
	}
	switch op.verb {
	case ir.VNeg:
		text.SubShifted64(x(dst), reg.XZR, x(a))
	case ir.VNot:
		text.OrnShifted64(x(dst), reg.XZR, x(a))
	case ir.VClz:
		text.Clz64(x(dst), x(a))
	case ir.VBswap:
		text.Rev64(x(dst), x(a))
	}
}

// emitExtend widens with a bitfield move: SXTW and UXTB are aliases of SBFM
// and UBFM, and this states the alias's own instruction.
func emitExtend(text *arm64asm.Section, op extOp, xd, xn reg.X, wd, wn reg.W) {
	bits := uint8(op.from) * 8
	if op.signed {
		text.Sbfm64(xd, xn, 0, bits-1)
		return
	}
	if op.from == a32 {
		// A write to a W register zeroes the upper half, so a 32-to-64
		// zero-extension is a move to the narrow view.
		text.MovReg32(wd, wn)
		return
	}
	text.Ubfm64(xd, xn, 0, bits-1)
}

func emitLoad(text *arm64asm.Section, wd width, in mir.Instr,
	x func(mir.VReg) reg.X, w func(mir.VReg) reg.W,
	d func(mir.VReg) reg.D, s func(mir.VReg) reg.S) {

	base := x(in.Uses[0])
	switch wd {
	case wf32:
		text.LdrImmS(s(in.Defs[0]), arm64asm.Mem32(base))
	case wf64:
		text.LdrImmD(d(in.Defs[0]), arm64asm.Mem64(base))
	case w32:
		text.LdrImm32(w(in.Defs[0]), arm64asm.Mem32(base))
	default:
		text.LdrImm64(x(in.Defs[0]), arm64asm.Mem64(base))
	}
}

func emitStore(text *arm64asm.Section, wd width, in mir.Instr,
	x func(mir.VReg) reg.X, w func(mir.VReg) reg.W,
	d func(mir.VReg) reg.D, s func(mir.VReg) reg.S) {

	base := x(in.Uses[1])
	switch wd {
	case wf32:
		text.StrImmS(s(in.Uses[0]), arm64asm.Mem32(base))
	case wf64:
		text.StrImmD(d(in.Uses[0]), arm64asm.Mem64(base))
	case w32:
		text.StrImm32(w(in.Uses[0]), arm64asm.Mem32(base))
	default:
		text.StrImm64(x(in.Uses[0]), arm64asm.Mem64(base))
	}
}

// emitExtLoad is one instruction, not a load and a widen: LDRSB and LDRB are
// different mnemonics, which is the difference from the other architecture's
// MOVSX pair.
func emitExtLoad(text *arm64asm.Section, op extLoadOp, in mir.Instr,
	x func(mir.VReg) reg.X, w func(mir.VReg) reg.W) {

	base := x(in.Uses[0])
	dst := in.Defs[0]
	switch {
	case op.from == a8 && op.signed && op.w == w64:
		text.LdrsbImm64(x(dst), arm64asm.Mem8(base))
	case op.from == a8 && op.signed:
		text.LdrsbImm32(w(dst), arm64asm.Mem8(base))
	case op.from == a8:
		text.LdrbImm(w(dst), arm64asm.Mem8(base))
	case op.from == a16 && op.signed && op.w == w64:
		text.LdrshImm64(x(dst), arm64asm.Mem16(base))
	case op.from == a16 && op.signed:
		text.LdrshImm32(w(dst), arm64asm.Mem16(base))
	case op.from == a16:
		text.LdrhImm(w(dst), arm64asm.Mem16(base))
	case op.signed:
		text.LdrswImm(x(dst), arm64asm.Mem32(base))
	default:
		text.LdrImm32(w(dst), arm64asm.Mem32(base))
	}
}

// emitBrTable is the range check, the table lookup and the jump.
//
// Each entry is the distance from the table's own start to a block, not the
// block's address. An address would be a pointer into text from data, which a
// position-independent image cannot hold in anything read-only — Mach-O
// refuses it by name — where a distance between two labels in one section is
// a constant whatever the section loads at. That is also why the table is in
// .text: the difference only folds if both ends are in the same section.
//
// The selector indexes at the full 64-bit width, which is safe because a write
// to a W register zeroes the upper half and the compare has bounded the lower.
func emitBrTable(text *arm64asm.Section, op brTableOp, in mir.Instr,
	x func(mir.VReg) reg.X, w func(mir.VReg) reg.W) {

	sel := in.Uses[0]
	base, entry, target := x(in.Defs[0]), x(in.Defs[1]), x(in.Defs[2])

	// One unsigned compare for both ends of the range: a negative selector
	// read as unsigned is a very large one, so HS catches it too.
	text.SubsImm32(reg.WZR, w(sel), int64(len(op.targets)))
	text.BCond(arm64asm.HS, arm64asm.Label(op.dflt))

	text.Adrp(base, arm64asm.Ref(op.id, arm64asm.RefAdrPage21))
	text.AddImm64(base, base, arm64asm.PageOff(arm64asm.Ref(op.id, arm64asm.RefAddAbsLo12)))
	text.AddShifted64(entry, base, x(sel), arm64asm.Shifted(arm64asm.LSL, 2))
	// LDRSW rather than a load and an extend: the distance is signed —
	// every case block precedes the table, which is emitted after the
	// function — and this is the load that already knows that.
	text.LdrswImm(entry, arm64asm.Mem32(entry))
	text.AddShifted64(target, base, entry)
	text.Br(target)
}

// emitAlloca moves SP down by a rounded size and hands back the block above
// the outgoing argument area.
//
// SUB with an extended register rather than a shifted one: the shifted form
// reads register 31 as ZR, so it cannot name SP at all. The AND that rounds
// is a logical immediate, which −16 is — a run of ones is exactly what that
// encoding holds.
func emitAlloca(text *arm64asm.Section, op allocaOp, dst, tmp, n reg.X) {
	text.AddImm64(tmp, n, maxAlign-1)
	text.AndImm64(tmp, tmp, ^uint64(maxAlign-1))
	text.SubExt64(reg.SP, reg.SP, tmp)
	text.AddImm64(dst, reg.SP, op.outArgs)
}

// emitFrameAddr is ADD Xd, X29, #off — or SUB with the magnitude, since a
// frame slot is below X29 and the immediate form is unsigned.
//
// An add-immediate states twelve bits, so a frame deeper than that puts the
// distance in the destination first and adds a register instead.
func emitFrameAddr(text *arm64asm.Section, xd reg.X, off int64) {
	mag := off
	if mag < 0 {
		mag = -mag
	}
	if mag <= 0xfff {
		if off < 0 {
			text.SubImm64(xd, reg.X29, mag)
			return
		}
		text.AddImm64(xd, reg.X29, mag)
		return
	}
	emitConst(text, constOp{w: w64, imm: mag}, xd, reg.W0)
	if off < 0 {
		text.SubShifted64(xd, reg.X29, xd)
		return
	}
	text.AddShifted64(xd, reg.X29, xd)
}

// unscaledLo and unscaledHi are STUR and LDUR's reach: a signed nine-bit byte
// offset.
const (
	unscaledLo = -256
	unscaledHi = 255
)

// frameBase is the base register and offset to address a frame slot with.
//
// A slot is at a negative offset from X29, which the scaled load and store
// forms cannot express — their immediate is unsigned — so every frame access
// is unscaled, and unscaled reaches nine signed bits. A C function with a few
// hundred bytes of locals is past that, so the address is computed into X16
// when the offset does not fit and the access uses it with no offset at all.
//
// X16 is IP0: never in the allocator's pool (see callerSaved), and clobbered
// only by a linker veneer, which happens at a call and not between these two
// instructions.
func frameBase(text *arm64asm.Section, off int64) (reg.X, int64) {
	if off >= unscaledLo && off <= unscaledHi {
		return reg.X29, off
	}
	emitFrameAddr(text, reg.X16, off)
	return reg.X16, 0
}

func emitFrameLoad(text *arm64asm.Section, wd width, off int64, in mir.Instr,
	x func(mir.VReg) reg.X, w func(mir.VReg) reg.W,
	d func(mir.VReg) reg.D, s func(mir.VReg) reg.S) {

	b, o := frameBase(text, off)
	switch wd {
	case wf32:
		text.LdurImmS(s(in.Defs[0]), arm64asm.Mem32(b).Off(o))
	case wf64:
		text.LdurImmD(d(in.Defs[0]), arm64asm.Mem64(b).Off(o))
	case w32:
		text.LdurImm32(w(in.Defs[0]), arm64asm.Mem32(b).Off(o))
	default:
		text.LdurImm64(x(in.Defs[0]), arm64asm.Mem64(b).Off(o))
	}
}

func emitFrameStore(text *arm64asm.Section, wd width, off int64, in mir.Instr,
	x func(mir.VReg) reg.X, w func(mir.VReg) reg.W,
	d func(mir.VReg) reg.D, s func(mir.VReg) reg.S) {

	b, o := frameBase(text, off)
	switch wd {
	case wf32:
		text.SturImmS(s(in.Uses[0]), arm64asm.Mem32(b).Off(o))
	case wf64:
		text.SturImmD(d(in.Uses[0]), arm64asm.Mem64(b).Off(o))
	case w32:
		text.SturImm32(w(in.Uses[0]), arm64asm.Mem32(b).Off(o))
	default:
		text.SturImm64(x(in.Uses[0]), arm64asm.Mem64(b).Off(o))
	}
}

// emitLoadAt reads one register of a byval aggregate out of the storage its
// pointer names. The offset is the aggregate's own, so it is small and
// positive and the scaled forms take it.
func emitLoadAt(text *arm64asm.Section, op loadAtOp, in mir.Instr,
	x func(mir.VReg) reg.X, w func(mir.VReg) reg.W,
	d func(mir.VReg) reg.D, s func(mir.VReg) reg.S) {

	base := x(in.Uses[0])
	switch op.w {
	case wf32:
		text.LdrImmS(s(in.Defs[0]), arm64asm.Mem32(base).Off(op.off))
	case wf64:
		text.LdrImmD(d(in.Defs[0]), arm64asm.Mem64(base).Off(op.off))
	case w32:
		text.LdrImm32(w(in.Defs[0]), arm64asm.Mem32(base).Off(op.off))
	default:
		text.LdrImm64(x(in.Defs[0]), arm64asm.Mem64(base).Off(op.off))
	}
}

// emitStoreAt writes one register of a result back into the storage the
// caller supplied for it.
func emitStoreAt(text *arm64asm.Section, op storeAtOp, in mir.Instr,
	x func(mir.VReg) reg.X, w func(mir.VReg) reg.W,
	d func(mir.VReg) reg.D, s func(mir.VReg) reg.S) {

	base := x(in.Uses[1])
	switch op.w {
	case wf32:
		text.StrImmS(s(in.Uses[0]), arm64asm.Mem32(base).Off(op.off))
	case wf64:
		text.StrImmD(d(in.Uses[0]), arm64asm.Mem64(base).Off(op.off))
	case w32:
		text.StrImm32(w(in.Uses[0]), arm64asm.Mem32(base).Off(op.off))
	default:
		text.StrImm64(x(in.Uses[0]), arm64asm.Mem64(base).Off(op.off))
	}
}

func emitFloatBinary(text *arm64asm.Section, op fbinOp, in mir.Instr,
	d func(mir.VReg) reg.D, s func(mir.VReg) reg.S) {

	dst, a, b := in.Defs[0], in.Uses[0], in.Uses[1]
	if op.w == wf32 {
		switch op.verb {
		case ir.VAdd:
			text.FaddS(s(dst), s(a), s(b))
		case ir.VSub:
			text.FsubS(s(dst), s(a), s(b))
		case ir.VMul:
			text.FmulS(s(dst), s(a), s(b))
		case ir.VDiv:
			text.FdivS(s(dst), s(a), s(b))
		case ir.VMinimum:
			text.FminS(s(dst), s(a), s(b))
		case ir.VMaximum:
			text.FmaxS(s(dst), s(a), s(b))
		case ir.VMinNum:
			text.FminnmS(s(dst), s(a), s(b))
		case ir.VMaxNum:
			text.FmaxnmS(s(dst), s(a), s(b))
		}
		return
	}
	switch op.verb {
	case ir.VAdd:
		text.FaddD(d(dst), d(a), d(b))
	case ir.VSub:
		text.FsubD(d(dst), d(a), d(b))
	case ir.VMul:
		text.FmulD(d(dst), d(a), d(b))
	case ir.VDiv:
		text.FdivD(d(dst), d(a), d(b))
	case ir.VMinimum:
		text.FminD(d(dst), d(a), d(b))
	case ir.VMaximum:
		text.FmaxD(d(dst), d(a), d(b))
	case ir.VMinNum:
		text.FminnmD(d(dst), d(a), d(b))
	case ir.VMaxNum:
		text.FmaxnmD(d(dst), d(a), d(b))
	}
}

// emitFloatUnary is §A3's one-source verbs. The four roundings are the four
// FRINTs that name their mode in the mnemonic rather than the FRINTX that
// reads FPCR: §17 says rounding is not dynamically changeable, so the mode is
// part of the instruction and not part of the machine state.
func emitFloatUnary(text *arm64asm.Section, op funOp, in mir.Instr,
	d func(mir.VReg) reg.D, s func(mir.VReg) reg.S) {

	dst, a := in.Defs[0], in.Uses[0]
	if op.w == wf32 {
		switch op.verb {
		case ir.VNeg:
			text.FnegS(s(dst), s(a))
		case ir.VAbs:
			text.FabsS(s(dst), s(a))
		case ir.VSqrt:
			text.FsqrtS(s(dst), s(a))
		case ir.VCeil:
			text.FrintpS(s(dst), s(a))
		case ir.VFloor:
			text.FrintmS(s(dst), s(a))
		case ir.VTrunc:
			text.FrintzS(s(dst), s(a))
		case ir.VNearest:
			text.FrintnS(s(dst), s(a))
		}
		return
	}
	switch op.verb {
	case ir.VNeg:
		text.FnegD(d(dst), d(a))
	case ir.VAbs:
		text.FabsD(d(dst), d(a))
	case ir.VSqrt:
		text.FsqrtD(d(dst), d(a))
	case ir.VCeil:
		text.FrintpD(d(dst), d(a))
	case ir.VFloor:
		text.FrintmD(d(dst), d(a))
	case ir.VTrunc:
		text.FrintzD(d(dst), d(a))
	case ir.VNearest:
		text.FrintnD(d(dst), d(a))
	}
}

func emitIntToFloat(text *arm64asm.Section, op cvtIntToFloatOp, in mir.Instr,
	x func(mir.VReg) reg.X, w func(mir.VReg) reg.W,
	d func(mir.VReg) reg.D, s func(mir.VReg) reg.S) {

	dst, a := in.Defs[0], in.Uses[0]
	switch {
	case op.signed && op.from == w32 && op.w == wf32:
		text.ScvtfWToS(s(dst), w(a))
	case op.signed && op.from == w32:
		text.ScvtfWToD(d(dst), w(a))
	case op.signed && op.w == wf32:
		text.ScvtfXToS(s(dst), x(a))
	case op.signed:
		text.ScvtfXToD(d(dst), x(a))
	case op.from == w32 && op.w == wf32:
		text.UcvtfWToS(s(dst), w(a))
	case op.from == w32:
		text.UcvtfWToD(d(dst), w(a))
	case op.w == wf32:
		text.UcvtfXToS(s(dst), x(a))
	default:
		text.UcvtfXToD(d(dst), x(a))
	}
}

func emitFloatToInt(text *arm64asm.Section, op cvtFloatToIntOp, in mir.Instr,
	x func(mir.VReg) reg.X, w func(mir.VReg) reg.W,
	d func(mir.VReg) reg.D, s func(mir.VReg) reg.S) {

	dst, a := in.Defs[0], in.Uses[0]
	switch {
	case op.signed && op.from == wf32 && op.w == w32:
		text.FcvtzsSToW(w(dst), s(a))
	case op.signed && op.from == wf32:
		text.FcvtzsSToX(x(dst), s(a))
	case op.signed && op.w == w32:
		text.FcvtzsDToW(w(dst), d(a))
	case op.signed:
		text.FcvtzsDToX(x(dst), d(a))
	case op.from == wf32 && op.w == w32:
		text.FcvtzuSToW(w(dst), s(a))
	case op.from == wf32:
		text.FcvtzuSToX(x(dst), s(a))
	case op.w == w32:
		text.FcvtzuDToW(w(dst), d(a))
	default:
		text.FcvtzuDToX(x(dst), d(a))
	}
}

// The atomics all address [Xn] with no offset: the exclusive and ordered
// encodings have no immediate field at all.

func emitAtomicLoad(text *arm64asm.Section, op atomicLoadOp, in mir.Instr,
	x func(mir.VReg) reg.X, w func(mir.VReg) reg.W) {

	dst, base := in.Defs[0], x(in.Uses[0])
	if !op.ordered {
		// A plain load, which is already single-copy atomic at this
		// width and alignment. Zero-extending for the narrow forms,
		// which is what §H's uload8 and uload16 ask for.
		switch op.a {
		case a8:
			text.LdrbImm(w(dst), arm64asm.Mem8(base))
		case a16:
			text.LdrhImm(w(dst), arm64asm.Mem16(base))
		case a32:
			text.LdrImm32(w(dst), arm64asm.Mem32(base))
		default:
			text.LdrImm64(x(dst), arm64asm.Mem64(base))
		}
		return
	}
	switch op.a {
	case a8:
		text.Ldarb(w(dst), arm64asm.Mem8(base))
	case a16:
		text.Ldarh(w(dst), arm64asm.Mem16(base))
	case a32:
		text.Ldar32(w(dst), arm64asm.Mem32(base))
	default:
		text.Ldar64(x(dst), arm64asm.Mem64(base))
	}
}

func emitAtomicStore(text *arm64asm.Section, op atomicStoreOp, in mir.Instr,
	x func(mir.VReg) reg.X, w func(mir.VReg) reg.W) {

	val, base := in.Uses[0], x(in.Uses[1])
	if !op.ordered {
		switch op.a {
		case a8:
			text.StrbImm(w(val), arm64asm.Mem8(base))
		case a16:
			text.StrhImm(w(val), arm64asm.Mem16(base))
		case a32:
			text.StrImm32(w(val), arm64asm.Mem32(base))
		default:
			text.StrImm64(x(val), arm64asm.Mem64(base))
		}
		return
	}
	switch op.a {
	case a8:
		text.Stlrb(w(val), arm64asm.Mem8(base))
	case a16:
		text.Stlrh(w(val), arm64asm.Mem16(base))
	case a32:
		text.Stlr32(w(val), arm64asm.Mem32(base))
	default:
		text.Stlr64(x(val), arm64asm.Mem64(base))
	}
}

func emitLdxr(text *arm64asm.Section, op ldxrOp, in mir.Instr,
	x func(mir.VReg) reg.X, w func(mir.VReg) reg.W) {

	dst, base := in.Defs[0], x(in.Uses[0])
	if op.acquire {
		switch op.a {
		case a8:
			text.Ldaxrb(w(dst), arm64asm.Mem8(base))
		case a16:
			text.Ldaxrh(w(dst), arm64asm.Mem16(base))
		case a32:
			text.Ldaxr32(w(dst), arm64asm.Mem32(base))
		default:
			text.Ldaxr64(x(dst), arm64asm.Mem64(base))
		}
		return
	}
	switch op.a {
	case a8:
		text.Ldxrb(w(dst), arm64asm.Mem8(base))
	case a16:
		text.Ldxrh(w(dst), arm64asm.Mem16(base))
	case a32:
		text.Ldxr32(w(dst), arm64asm.Mem32(base))
	default:
		text.Ldxr64(x(dst), arm64asm.Mem64(base))
	}
}

// emitStxr's status register is a W whatever the value's width is.
func emitStxr(text *arm64asm.Section, op stxrOp, in mir.Instr,
	x func(mir.VReg) reg.X, w func(mir.VReg) reg.W) {

	status, val, base := w(in.Defs[0]), in.Uses[0], x(in.Uses[1])
	if op.release {
		switch op.a {
		case a8:
			text.Stlxrb(status, w(val), arm64asm.Mem8(base))
		case a16:
			text.Stlxrh(status, w(val), arm64asm.Mem16(base))
		case a32:
			text.Stlxr32(status, w(val), arm64asm.Mem32(base))
		default:
			text.Stlxr64(status, x(val), arm64asm.Mem64(base))
		}
		return
	}
	switch op.a {
	case a8:
		text.Stxrb(status, w(val), arm64asm.Mem8(base))
	case a16:
		text.Stxrh(status, w(val), arm64asm.Mem16(base))
	case a32:
		text.Stxr32(status, w(val), arm64asm.Mem32(base))
	default:
		text.Stxr64(status, x(val), arm64asm.Mem64(base))
	}
}

func emitArgStore(text *arm64asm.Section, op argStoreOp, in mir.Instr,
	x func(mir.VReg) reg.X, w func(mir.VReg) reg.W,
	d func(mir.VReg) reg.D, s func(mir.VReg) reg.S) {

	switch op.w {
	case wf32:
		text.StrImmS(s(in.Uses[0]), arm64asm.Mem32(reg.SP).Off(op.off))
	case wf64:
		text.StrImmD(d(in.Uses[0]), arm64asm.Mem64(reg.SP).Off(op.off))
	case w32:
		text.StrImm32(w(in.Uses[0]), arm64asm.Mem32(reg.SP).Off(op.off))
	default:
		text.StrImm64(x(in.Uses[0]), arm64asm.Mem64(reg.SP).Off(op.off))
	}
}
