package amd64

import (
	"fmt"

	amd64asm "github.com/vertex-language/amd64"
	"github.com/vertex-language/amd64/operand"
	"github.com/vertex-language/amd64/reg"

	"github.com/vertex-language/ir"
	"github.com/vertex-language/ir/lower/mir"
	"github.com/vertex-language/ir/lower/regalloc"
)

// setcc writes zero or one into dst by the flags.
func setcc(text *amd64asm.Section, c condCode, dst reg.R8) {
	switch c {
	case condE:
		text.SeteRM8(dst)
	case condNE:
		text.SetneRM8(dst)
	case condL:
		text.SetlRM8(dst)
	case condLE:
		text.SetleRM8(dst)
	case condB:
		text.SetbRM8(dst)
	case condBE:
		text.SetbeRM8(dst)
	case condO:
		text.SetoRM8(dst)
	case condA:
		text.SetaRM8(dst)
	case condAE:
		text.SetaeRM8(dst)
	case condP:
		text.SetpRM8(dst)
	case condNP:
		text.SetnpRM8(dst)
	}
}

func cmov32(text *amd64asm.Section, c condCode, dst, src reg.R32) {
	switch c {
	case condE:
		text.CmoveR32RM32(dst, src)
	case condNE:
		text.CmovneR32RM32(dst, src)
	case condL:
		text.CmovlR32RM32(dst, src)
	case condLE:
		text.CmovleR32RM32(dst, src)
	case condB:
		text.CmovbR32RM32(dst, src)
	case condBE:
		text.CmovbeR32RM32(dst, src)
	case condO:
		text.CmovoR32RM32(dst, src)
	case condA:
		text.CmovaR32RM32(dst, src)
	case condAE:
		text.CmovaeR32RM32(dst, src)
	case condP:
		text.CmovpR32RM32(dst, src)
	case condNP:
		text.CmovnpR32RM32(dst, src)
	}
}

func cmov64(text *amd64asm.Section, c condCode, dst, src reg.R64) {
	switch c {
	case condE:
		text.CmoveR64RM64(dst, src)
	case condNE:
		text.CmovneR64RM64(dst, src)
	case condL:
		text.CmovlR64RM64(dst, src)
	case condLE:
		text.CmovleR64RM64(dst, src)
	case condB:
		text.CmovbR64RM64(dst, src)
	case condBE:
		text.CmovbeR64RM64(dst, src)
	case condO:
		text.CmovoR64RM64(dst, src)
	case condA:
		text.CmovaR64RM64(dst, src)
	case condAE:
		text.CmovaeR64RM64(dst, src)
	case condP:
		text.CmovpR64RM64(dst, src)
	case condNP:
		text.CmovnpR64RM64(dst, src)
	}
}

// shift32 and shift64 are shiftOps' encoding tables, where the count is always CL.
func shift32(text *amd64asm.Section, v ir.Verb, dst reg.R32) {
	switch v {
	case ir.VShl:
		text.ShlRM32CL(dst)
	case ir.VSShr:
		text.SarRM32CL(dst) // arithmetic: the sign bit fills
	case ir.VUShr:
		text.ShrRM32CL(dst) // logical: zeroes fill
	case ir.VRotL:
		text.RolRM32CL(dst)
	case ir.VRotR:
		text.RorRM32CL(dst)
	}
}

func shift64(text *amd64asm.Section, v ir.Verb, dst reg.R64) {
	switch v {
	case ir.VShl:
		text.ShlRM64CL(dst)
	case ir.VSShr:
		text.SarRM64CL(dst)
	case ir.VUShr:
		text.ShrRM64CL(dst)
	case ir.VRotL:
		text.RolRM64CL(dst)
	case ir.VRotR:
		text.RorRM64CL(dst)
	}
}

// emitPrologue opens the frame, if there is one.
// Emitted here since the frame size is not known until after register allocation.
//
// It returns what it emitted, byte offset by byte offset. Windows needs the
// prologue described in .xdata before anything can unwind through the frame,
// and the description has to come from here rather than from a reader of the
// bytes: see unwind.go.
func emitPrologue(text *amd64asm.Section, fn *ir.Func, fr *frame, saved []reg.R64) prologueShape {
	if !fr.needed() {
		return prologueShape{}
	}
	base := text.Offset()
	p := prologueShape{present: true, dynamic: fr.dynamic}

	text.PushR64(reg.RBP)
	p.pushAt = text.Offset() - base
	text.MovR64RM64(reg.RBP, reg.RSP)
	if fr.size() > 0 {
		// The 32-bit immediate form unconditionally, the way constOp
		// takes the ten-byte movabs: a frame that fits in a signed byte
		// could subtract in four bytes rather than seven, and choosing
		// between them by the value is a peephole, which this package
		// has nowhere to put.
		text.SubRM64Imm32(reg.RSP, int64(fr.size()))
		p.alloc, p.allocAt = fr.size(), text.Offset()-base
	}
	// The callee-saved registers go into frame slots rather than onto
	// the stack, so that RSP stays where the alloc offsets and the call
	// alignment both assume it is.
	for _, r := range saved {
		text.MovRM64R64(operand.Mem64(reg.RBP).Disp(fr.saveAt[r]), r)
		// The slot measured from the RSP the body runs with, which is
		// where the frame register was before the allocation minus the
		// allocation. That is the origin every unwind code counts from.
		p.saves = append(p.saves, prologueSave{
			at:  text.Offset() - base,
			r:   r,
			off: uint64(int64(fr.size()) + int64(fr.saveAt[r])),
		})
	}
	emitSaveArea(text, fn, fr)
	p.size = text.Offset() - base
	return p
}

// emitSaveArea spills a variadic function's incoming argument registers
// into §3.5.7's save area.
//
// In the prologue, which is the only place it can be: the registers
// still hold what the caller put there, classifyParams' copies being in
// the entry block. All six, not only those past the named parameters —
// va_arg indexes the area by an offset the ABI fixes, and where the list
// starts is va_start's business.
func emitSaveArea(text *amd64asm.Section, fn *ir.Func, fr *frame) {
	if !fr.saveAreaSet {
		return
	}
	// The Microsoft ABI has no save area of the callee's own. Every
	// argument, named or not, owns one eightbyte of the caller's frame,
	// and the four passed in registers own the home space the caller
	// reserved above the return address for exactly this. Homing them
	// there makes the whole argument list one contiguous array, which is
	// what a va_list on this ABI is a pointer into.
	if fn.Module().Layout().ABI == abiMS {
		for i, r := range msIntArgs {
			text.MovRM64R64(operand.Mem64(reg.RBP).Disp(msHomeOff+int32(i*8)), r)
		}
		return
	}
	for i, r := range sysvIntArgs {
		text.MovRM64R64(operand.Mem64(reg.RBP).Disp(fr.saveArea+int32(i*8)), r)
	}

	// The vector half only if there is one. AL holds the number of
	// vector registers the caller used — milestone 30 is what sets it —
	// and a caller that passed none may have left XMM registers holding
	// anything at all, including values that trap to read. Skipping is
	// not the optimisation it looks like: it is the reason AL is in the
	// ABI.
	skip := fn.Name() + ".novec"
	text.TestRM8R8(reg.AL, reg.AL)
	text.JeLabel(skip)
	for i := 0; i < saveAreaFloats; i++ {
		text.MovapsRM128Xmm(
			operand.Mem128(reg.RBP).Disp(fr.saveArea+int32(saveAreaGPSize+i*16)),
			reg.Xmm(int(reg.XMM0)+i))
	}
	text.Label(skip)
}

// usedCalleeSaved is the callee-saved registers this function's
// allocation actually named, in the ABI's own order. Which registers
// those are is the whole difference the ABI makes here: RDI and RSI are
// scratch under SysV and have to be put back under the Microsoft one.
func usedCalleeSaved(abi string, pool *regalloc.Pool, assigned map[mir.VReg]regalloc.PhysReg) []reg.R64 {
	used := map[reg.R64]bool{}
	for v, p := range assigned {
		if pool.ClassOf(v) != regalloc.DefaultClass {
			continue
		}
		used[reg.R64(p)] = true
	}
	var out []reg.R64
	for _, r := range regsFor(abi).calleeSaved {
		if used[r] {
			out = append(out, r)
		}
	}
	return out
}

// alu32 and alu64 are binOps' two encoding tables, typed by width.
func alu32(text *amd64asm.Section, v ir.Verb, dst, src reg.R32) {
	switch v {
	case ir.VAdd:
		text.AddR32RM32(dst, src)
	case ir.VSub:
		text.SubR32RM32(dst, src)
	case ir.VMul:
		text.ImulR32RM32(dst, src)
	case ir.VAnd:
		text.AndR32RM32(dst, src)
	case ir.VOr:
		text.OrR32RM32(dst, src)
	case ir.VXor:
		text.XorR32RM32(dst, src)
	}
}

func alu64(text *amd64asm.Section, v ir.Verb, dst, src reg.R64) {
	switch v {
	case ir.VAdd:
		text.AddR64RM64(dst, src)
	case ir.VSub:
		text.SubR64RM64(dst, src)
	case ir.VMul:
		text.ImulR64RM64(dst, src)
	case ir.VAnd:
		text.AndR64RM64(dst, src)
	case ir.VOr:
		text.OrR64RM64(dst, src)
	case ir.VXor:
		text.XorR64RM64(dst, src)
	}
}

// fAlu encodes the four basic floating-point operations in both scalar widths.
func fAlu(text *amd64asm.Section, v ir.Verb, w width, dst, src reg.Xmm) {
	if w == wf32 {
		switch v {
		case ir.VAdd:
			text.AddssXmmXM32(dst, src)
		case ir.VSub:
			text.SubssXmmXM32(dst, src)
		case ir.VMul:
			text.MulssXmmXM32(dst, src)
		case ir.VDiv:
			text.DivssXmmXM32(dst, src)
		}
		return
	}
	switch v {
	case ir.VAdd:
		text.AddsdXmmXM64(dst, src)
	case ir.VSub:
		text.SubsdXmmXM64(dst, src)
	case ir.VMul:
		text.MulsdXmmXM64(dst, src)
	case ir.VDiv:
		text.DivsdXmmXM64(dst, src)
	}
}

// fLogical encodes the four packed logical instructions in both widths.
func fLogical(text *amd64asm.Section, l fLogic, w width, dst, src reg.Xmm) {
	if w == wf32 {
		switch l {
		case fAnd:
			text.AndpsXmmRM128(dst, src)
		case fOr:
			text.OrpsXmmRM128(dst, src)
		case fXor:
			text.XorpsXmmRM128(dst, src)
		case fAndn:
			text.AndnpsXmmRM128(dst, src)
		}
		return
	}
	switch l {
	case fAnd:
		text.AndpdXmmRM128(dst, src)
	case fOr:
		text.OrpdXmmRM128(dst, src)
	case fXor:
		text.XorpdXmmRM128(dst, src)
	case fAndn:
		text.AndnpdXmmRM128(dst, src)
	}
}

// bitCount32 and bitCount64 are the bit counting operations in both widths.
func bitCount32(text *amd64asm.Section, v ir.Verb, dst, src reg.R32) {
	switch v {
	case ir.VPopcnt:
		text.PopcntR32RM32(dst, src)
	case ir.VClz:
		text.LzcntR32RM32(dst, src)
	case ir.VCtz:
		text.TzcntR32RM32(dst, src)
	}
}

func bitCount64(text *amd64asm.Section, v ir.Verb, dst, src reg.R64) {
	switch v {
	case ir.VPopcnt:
		text.PopcntR64RM64(dst, src)
	case ir.VClz:
		text.LzcntR64RM64(dst, src)
	case ir.VCtz:
		text.TzcntR64RM64(dst, src)
	}
}

// emit walks mf block by block and calls the amd64 typed helpers to emit instructions.
// The entry block receives the function's exported symbol, while others receive bare labels.
// labeledBlocks is every block label something in mf refers to by name
// rather than by branching to it: a jump table's entries, and the
// blockaddr of §D3.
//
// Both need the label in the symbol table, because both are relocations
// against it and a bare label leaves nothing to relocate against. Only
// these: a block a branch merely jumps to is a bare label, patched at
// Finalize and gone, which is what keeps a function's control flow out
// of the symbol table.
func labeledBlocks(mf *mir.Func) map[string]bool {
	out := map[string]bool{}
	for _, b := range mf.Blocks {
		for _, in := range b.Instrs {
			switch op := in.Op.(type) {
			case brTableOp:
				for _, t := range op.targets {
					out[t] = true
				}
			case leaBlockOp:
				out[op.label] = true
			case asmOp:
				// An asm goto's labels are branched to by text this
				// package assembled rather than emitted, so the
				// reference arrives as a symbol and needs one.
				for _, l := range op.emitted {
					out[l] = true
				}
			}
		}
	}
	return out
}

// It returns an error only for inline assembly, whose failure is a fact about
// text the frontend wrote and has to reach the caller with the position the
// assembler gave it.
func emit(am *amd64asm.Module, text *amd64asm.Section, fn *ir.Func, mf *mir.Func, assigned map[mir.VReg]regalloc.PhysReg, fr *frame, saved []reg.R64) (prologueShape, error) {
	var shape prologueShape
	r64 := func(v mir.VReg) reg.R64 { return reg.R64(assigned[v]) }
	r32 := func(v mir.VReg) reg.R32 { return reg.R32(assigned[v]) }
	r16 := func(v mir.VReg) reg.R16 { return reg.R16(assigned[v]) }
	// reg.R8's first sixteen members are the REX-form low bytes, which is
	// the view a register number means here: numbers 4 through 7 are SPL,
	// BPL, SIL and DIL, not AH through BH, and the encoder emits the bare
	// REX those four need. Naming a high-byte register is not something a
	// register number can do, which is exactly right — nothing allocates
	// one.
	r8 := func(v mir.VReg) reg.R8 { return reg.R8(assigned[v]) }
	// The other register file. A PhysReg means nothing without its class
	// — RAX and XMM0 are both register zero — and which one an
	// instruction is naming is the op's own business, decided by the
	// width it carries.
	xmm := func(v mir.VReg) reg.Xmm { return reg.Xmm(assigned[v]) }

	labeled := labeledBlocks(mf)

	// consts numbers this function's 128-bit literals, which each need a
	// label of their own in .rodata.
	consts := 0

	for i, mb := range mf.Blocks {
		switch {
		case i == 0:
			text.Label(fn.Name(), funcBinding(fn), amd64asm.Func)
			shape = emitPrologue(text, fn, fr, saved)
		case labeled[mb.Label]:
			text.Label(mb.Label, amd64asm.Local)
		default:
			text.Label(mb.Label)
		}

		for _, in := range mb.Instrs {
			switch op := in.Op.(type) {
			case aluOp:
				// Two-address: the destination is also the first
				// source, so a result in a register that is not
				// already holding the left operand needs it moved
				// there first.
				dst, a, b := in.Defs[0], in.Uses[0], in.Uses[1]
				if op.w == w32 {
					if r32(dst) != r32(a) {
						text.MovR32RM32(r32(dst), r32(a))
					}
					alu32(text, op.verb, r32(dst), r32(b))
					break
				}
				if r64(dst) != r64(a) {
					text.MovR64RM64(r64(dst), r64(a))
				}
				alu64(text, op.verb, r64(dst), r64(b))
			case unOp:
				dst, a := in.Defs[0], in.Uses[0]
				if op.w == w32 {
					if r32(dst) != r32(a) {
						text.MovR32RM32(r32(dst), r32(a))
					}
					if op.verb == ir.VNeg {
						text.NegRM32(r32(dst))
						break
					}
					text.NotRM32(r32(dst))
					break
				}
				if r64(dst) != r64(a) {
					text.MovR64RM64(r64(dst), r64(a))
				}
				if op.verb == ir.VNeg {
					text.NegRM64(r64(dst))
					break
				}
				text.NotRM64(r64(dst))
			case i1NotOp:
				dst, a := in.Defs[0], in.Uses[0]
				if r32(dst) != r32(a) {
					text.MovR32RM32(r32(dst), r32(a))
				}
				text.XorRM32Imm32(r32(dst), 1)
			case trapOp:
				text.Ud2()
			case signExtendOp:
				if op.w == w32 {
					text.Cdq()
					break
				}
				text.Cqo()
			case zeroOp:
				// The 32-bit xor whatever the width: it is two bytes,
				// and writing a 32-bit register zeroes the upper half,
				// so this clears the whole of RDX as surely as a 64-bit
				// xor would in three.
				text.XorR32RM32(r32(in.Defs[0]), r32(in.Defs[0]))
			case divOp:
				// One operand in the instruction and two implied: the
				// dividend is RDX:RAX and the answers come back there.
				// vregs.physical is what put them there.
				divisor := in.Uses[2]
				switch {
				case op.w == w32 && op.signed:
					text.IdivRM32(r32(divisor))
				case op.w == w32:
					text.DivRM32(r32(divisor))
				case op.signed:
					text.IdivRM64(r64(divisor))
				default:
					text.DivRM64(r64(divisor))
				}
			case mulOp:
				// One operand named and RAX implied, the product across
				// RDX:RAX. vregs.physical is what put the multiplicand
				// in RAX and what reserves RDX for the half nothing
				// else could name.
				src := in.Uses[1]
				switch {
				case op.w == w32 && op.signed:
					text.ImulRM32(r32(src))
				case op.w == w32:
					text.MulRM32(r32(src))
				case op.signed:
					text.ImulRM64(r64(src))
				default:
					text.MulRM64(r64(src))
				}
			case shiftOp:
				dst, a := in.Defs[0], in.Uses[0]
				if op.w == w32 {
					if r32(dst) != r32(a) {
						text.MovR32RM32(r32(dst), r32(a))
					}
					shift32(text, op.verb, r32(dst))
					break
				}
				if r64(dst) != r64(a) {
					text.MovR64RM64(r64(dst), r64(a))
				}
				shift64(text, op.verb, r64(dst))
			case returnOp:
				// Nothing about the return values here. isel copied each
				// one into a vreg pinned to the register SysV names and
				// hung them off Uses, which is the only shape that can
				// carry two of them across two register files. What is
				// left is the epilogue, which is the same either way.
				if fr.needed() {
					// Restored in the same order they were saved. The
					// slots are independent, so the order is only for a
					// reader comparing the two halves.
					for _, r := range saved {
						text.MovR64RM64(r, operand.Mem64(reg.RBP).Disp(fr.saveAt[r]))
					}
					// leave, which is mov rsp, rbp followed by pop rbp
					// in one byte. Not a peephole over two instructions
					// this package chose — it is the epilogue the ISA
					// names, and writing the pair out longhand would be
					// the deliberate choice needing a reason.
					text.Leave()
				}
				text.Ret()
			case cmpOp:
				// UCOMIS rather than CMP in the other file, and the
				// same instruction either way as far as everything
				// downstream is concerned: it writes EFLAGS, and a Jcc
				// or a SETcc reading them cannot tell which comparison
				// set them.
				switch op.w {
				case wf32:
					text.UcomissXmmXM32(xmm(in.Uses[0]), xmm(in.Uses[1]))
				case wf64:
					text.UcomisdXmmXM64(xmm(in.Uses[0]), xmm(in.Uses[1]))
				case w32:
					text.CmpR32RM32(r32(in.Uses[0]), r32(in.Uses[1]))
				default:
					text.CmpR64RM64(r64(in.Uses[0]), r64(in.Uses[1]))
				}
			case brccOp:
				switch op.cond {
				case condE:
					text.JeLabel(op.then)
				case condNE:
					text.JneLabel(op.then)
				case condL:
					text.JlLabel(op.then)
				case condLE:
					text.JleLabel(op.then)
				case condB:
					text.JbLabel(op.then)
				case condBE:
					text.JbeLabel(op.then)
				case condO:
					text.JoLabel(op.then)
				case condA:
					text.JaLabel(op.then)
				case condAE:
					text.JaeLabel(op.then)
				case condP:
					text.JpLabel(op.then)
				case condNP:
					text.JnpLabel(op.then)
				}
				text.JmpLabel(op.els)
			case movOp:
				dst, src := in.Defs[0], in.Uses[0]
				if assigned[dst] == assigned[src] {
					break
				}
				switch {
				case op.w.isFloat():
					// MOVAPS and not MOVSS, at both float widths. The
					// scalar move merges into the destination's upper
					// bits, which makes it depend on a value this copy
					// has nothing to do with; MOVAPS writes all of them,
					// is a byte shorter, and the upper bits of a scalar
					// value are not anything.
					text.MovapsXmmRM128(xmm(dst), xmm(src))
				case op.w == w32:
					text.MovR32RM32(r32(dst), r32(src))
				default:
					text.MovR64RM64(r64(dst), r64(src))
				}
			case fAluOp:
				// Two-address, like aluOp: the destination is also the
				// first source, so a result in a register that is not
				// the first operand's needs the move first.
				dst, a, b := in.Defs[0], in.Uses[0], in.Uses[1]
				if xmm(dst) != xmm(a) {
					text.MovapsXmmRM128(xmm(dst), xmm(a))
				}
				fAlu(text, op.verb, op.w, xmm(dst), xmm(b))
			case fLogicOp:
				dst, a := in.Defs[0], in.Uses[0]
				if xmm(dst) != xmm(a) {
					text.MovapsXmmRM128(xmm(dst), xmm(a))
				}
				fLogical(text, op.op, op.w, xmm(dst), xmm(in.Uses[1]))

			case hwMinMaxOp:
				dst, a, b := in.Defs[0], in.Uses[0], in.Uses[1]
				if xmm(dst) != xmm(a) {
					text.MovapsXmmRM128(xmm(dst), xmm(a))
				}
				if op.w == wf32 {
					if op.isMax {
						text.MaxssXmmXM32(xmm(dst), xmm(b))
					} else {
						text.MinssXmmXM32(xmm(dst), xmm(b))
					}
				} else {
					if op.isMax {
						text.MaxsdXmmXM64(xmm(dst), xmm(b))
					} else {
						text.MinsdXmmXM64(xmm(dst), xmm(b))
					}
				}
			case fRoundOp:
				dst, src := in.Defs[0], in.Uses[0]
				if op.w == wf32 {
					text.RoundssXmmXM32Imm8(xmm(dst), xmm(src), op.mode)
				} else {
					text.RoundsdXmmXM64Imm8(xmm(dst), xmm(src), op.mode)
				}
			case fmaOp:
				dst, a, b, c_val := in.Defs[0], in.Uses[0], in.Uses[1], in.Uses[2]
				if xmm(dst) != xmm(c_val) {
					text.MovapsXmmRM128(xmm(dst), xmm(c_val))
				}
				if op.w == wf32 {
					text.Vfmadd231ssXmmXmmXM32(xmm(dst), xmm(a), xmm(b))
				} else {
					text.Vfmadd231sdXmmXmmXM64(xmm(dst), xmm(a), xmm(b))
				}
			case fSqrtOp:
				if op.w == wf32 {
					text.SqrtssXmmXM32(xmm(in.Defs[0]), xmm(in.Uses[0]))
					break
				}
				text.SqrtsdXmmXM64(xmm(in.Defs[0]), xmm(in.Uses[0]))
			case cvtIntToFloatOp:
				switch {
				case op.to == wf32 && op.from == w32:
					text.Cvtsi2ssXmmRM32(xmm(in.Defs[0]), r32(in.Uses[0]))
				case op.to == wf32:
					text.Cvtsi2ssXmmRM64(xmm(in.Defs[0]), r64(in.Uses[0]))
				case op.from == w32:
					text.Cvtsi2sdXmmRM32(xmm(in.Defs[0]), r32(in.Uses[0]))
				default:
					text.Cvtsi2sdXmmRM64(xmm(in.Defs[0]), r64(in.Uses[0]))
				}
			case cvtFloatToIntOp:
				switch {
				case op.from == wf32 && op.to == w32:
					text.Cvttss2siR32XM32(r32(in.Defs[0]), xmm(in.Uses[0]))
				case op.from == wf32:
					text.Cvttss2siR64XM32(r64(in.Defs[0]), xmm(in.Uses[0]))
				case op.to == w32:
					text.Cvttsd2siR32XM64(r32(in.Defs[0]), xmm(in.Uses[0]))
				default:
					text.Cvttsd2siR64XM64(r64(in.Defs[0]), xmm(in.Uses[0]))
				}
			case cvtFloatOp:
				if op.to == wf64 {
					text.Cvtss2sdXmmXM32(xmm(in.Defs[0]), xmm(in.Uses[0]))
					break
				}
				text.Cvtsd2ssXmmXM64(xmm(in.Defs[0]), xmm(in.Uses[0]))
			case floatToBitsOp:
				if op.w == wf32 {
					text.MovdRM32Xmm(r32(in.Defs[0]), xmm(in.Uses[0]))
					break
				}
				text.MovqRM64Xmm(r64(in.Defs[0]), xmm(in.Uses[0]))
			case bitsToFloatOp:
				if op.w == wf32 {
					text.MovdXmmRM32(xmm(in.Defs[0]), r32(in.Uses[0]))
					break
				}
				text.MovqXmmRM64(xmm(in.Defs[0]), r64(in.Uses[0]))
			case loadOp:
				// The base register is the whole 64-bit register
				// whatever the access width: a pointer is a pointer,
				// and it is the *operand* that is four bytes wide in
				// mov eax, [rdi].
				base := r64(in.Uses[0])
				switch op.w {
				case wv128:
					if op.unaligned {
						text.MovdquXmmRM128(xmm(in.Defs[0]), operand.Mem128(base))
						break
					}
					text.MovapsXmmRM128(xmm(in.Defs[0]), operand.Mem128(base))
				case wf32:
					text.MovssXmmXM32(xmm(in.Defs[0]), operand.Mem32(base))
				case wf64:
					text.MovsdXmmXM64(xmm(in.Defs[0]), operand.Mem64(base))
				case w32:
					text.MovR32RM32(r32(in.Defs[0]), operand.Mem32(base))
				default:
					text.MovR64RM64(r64(in.Defs[0]), operand.Mem64(base))
				}
			case storeOp:
				base := r64(in.Uses[1])
				switch op.w {
				case wv128:
					if op.unaligned {
						text.MovdquRM128Xmm(operand.Mem128(base), xmm(in.Uses[0]))
						break
					}
					text.MovapsRM128Xmm(operand.Mem128(base), xmm(in.Uses[0]))
				case wf32:
					text.MovssXM32Xmm(operand.Mem32(base), xmm(in.Uses[0]))
				case wf64:
					text.MovsdXM64Xmm(operand.Mem64(base), xmm(in.Uses[0]))
				case w32:
					text.MovRM32R32(operand.Mem32(base), r32(in.Uses[0]))
				default:
					text.MovRM64R64(operand.Mem64(base), r64(in.Uses[0]))
				}
			case extLoadOp:
				dst, base := in.Defs[0], r64(in.Uses[0])
				switch op.from {
				case a8:
					switch {
					case op.signed && op.w == w32:
						text.MovsxR32RM8(r32(dst), operand.Mem8(base))
					case op.signed:
						text.MovsxR64RM8(r64(dst), operand.Mem8(base))
					case op.w == w32:
						text.MovzxR32RM8(r32(dst), operand.Mem8(base))
					default:
						text.MovzxR64RM8(r64(dst), operand.Mem8(base))
					}
				case a16:
					switch {
					case op.signed:
						// Into the whole register, at both widths. The
						// assembler declares no MovsxR32RM16, and an i32
						// result does not need one: the low 32 bits of a
						// 16-to-64 sign extension are a 16-to-32 sign
						// extension, and the bits above them are what an
						// i32 vreg does not have.
						text.MovsxR64RM16(r64(dst), operand.Mem16(base))
					case op.w == w32:
						text.MovzxR32RM16(r32(dst), operand.Mem16(base))
					default:
						text.MovzxR64RM16(r64(dst), operand.Mem16(base))
					}
				case a32:
					if op.signed {
						text.MovsxdR64RM32(r64(dst), operand.Mem32(base))
						break
					}
					// No MOVZX with a 32-bit source exists, and none is
					// needed: a write to a 32-bit register zeroes the
					// upper half, so the zero-extending four-byte load is
					// an ordinary mov whose destination happens to be
					// half of a 64-bit result.
					text.MovR32RM32(r32(dst), operand.Mem32(base))
				}
			case subStoreOp:
				base := r64(in.Uses[1])
				switch op.to {
				case a8:
					text.MovRM8R8(operand.Mem8(base), r8(in.Uses[0]))
				case a16:
					text.MovRM16R16(operand.Mem16(base), r16(in.Uses[0]))
				case a32:
					text.MovRM32R32(operand.Mem32(base), r32(in.Uses[0]))
				}
			case swapOp:
				if op.w == w32 {
					text.XchgRM32R32(r32(in.Defs[0]), r32(in.Defs[1]))
					break
				}
				text.XchgRM64R64(r64(in.Defs[0]), r64(in.Defs[1]))
			case leaOp:
				// The one instruction here that reads a memory operand
				// and touches no memory.
				text.LeaR64M(r64(in.Defs[0]), operand.Mem64(reg.RBP).Disp(op.off))
			case allocaOp:
				// Round the size up to the stack alignment, take that
				// much stack, and hand back the address just above the
				// outgoing area. Rounding is what keeps RSP 16-aligned
				// across the allocation, which every call after it
				// depends on and which no later instruction re-checks.
				n, scratch, dst := in.Uses[0], r64(in.Defs[1]), r64(in.Defs[0])
				text.LeaR64M(scratch, operand.Mem64(r64(n)).Disp(maxAlign-1))
				text.AndRM64Imm32(scratch, -maxAlign)
				text.SubR64RM64(reg.RSP, scratch)
				text.LeaR64M(dst, operand.Mem64(reg.RSP).Disp(op.outArgs))
			case stackSaveOp:
				text.MovR64RM64(r64(in.Defs[0]), reg.RSP)
			case stackRestoreOp:
				text.MovR64RM64(reg.RSP, r64(in.Uses[0]))
			case leaOutOp:
				text.LeaR64M(r64(in.Defs[0]), operand.Mem64(reg.RSP).Disp(op.off))
			case loadAtOp:
				switch op.w {
				case wv128:
					text.MovapsXmmRM128(xmm(in.Defs[0]), operand.Mem128(r64(in.Uses[0])).Disp(op.off))
				case wf32:
					text.MovssXmmXM32(xmm(in.Defs[0]), operand.Mem32(r64(in.Uses[0])).Disp(op.off))
				case wf64:
					text.MovsdXmmXM64(xmm(in.Defs[0]), operand.Mem64(r64(in.Uses[0])).Disp(op.off))
				case w32:
					text.MovR32RM32(r32(in.Defs[0]), operand.Mem32(r64(in.Uses[0])).Disp(op.off))
				default:
					text.MovR64RM64(r64(in.Defs[0]), operand.Mem64(r64(in.Uses[0])).Disp(op.off))
				}
			case storeAtOp:
				at := operand.Mem64(r64(in.Uses[0])).Disp(op.off)
				switch op.w {
				case wv128:
					text.MovapsRM128Xmm(operand.Mem128(r64(in.Uses[0])).Disp(op.off), xmm(in.Uses[1]))
				case wf32:
					text.MovssXM32Xmm(operand.Mem32(r64(in.Uses[0])).Disp(op.off), xmm(in.Uses[1]))
				case wf64:
					text.MovsdXM64Xmm(at, xmm(in.Uses[1]))
				case w32:
					text.MovRM32R32(operand.Mem32(r64(in.Uses[0])).Disp(op.off), r32(in.Uses[1]))
				default:
					text.MovRM64R64(at, r64(in.Uses[1]))
				}
			case storeImmAtOp:
				if op.w == w32 {
					text.MovRM32Imm32(operand.Mem32(r64(in.Uses[0])).Disp(op.off), op.imm)
					break
				}
				text.MovRM64Imm32(operand.Mem64(r64(in.Uses[0])).Disp(op.off), op.imm)
			case addImmAtOp:
				if op.w == w32 {
					text.AddRM32Imm32(operand.Mem32(r64(in.Uses[0])).Disp(op.off), op.imm)
					break
				}
				text.AddRM64Imm32(operand.Mem64(r64(in.Uses[0])).Disp(op.off), op.imm)
			case andImmOp:
				dst, a := in.Defs[0], in.Uses[0]
				if op.w == w32 {
					if r32(dst) != r32(a) {
						text.MovR32RM32(r32(dst), r32(a))
					}
					text.AndRM32Imm32(r32(dst), op.imm)
					break
				}
				if r64(dst) != r64(a) {
					text.MovR64RM64(r64(dst), r64(a))
				}
				text.AndRM64Imm32(r64(dst), op.imm)
			case leaAtOp:
				text.LeaR64M(r64(in.Defs[0]), operand.Mem64(r64(in.Uses[0])).Disp(op.off))
			case cmpImmOp:
				if op.w == w32 {
					text.CmpRM32Imm32(r32(in.Uses[0]), op.imm)
					break
				}
				text.CmpRM64Imm32(r64(in.Uses[0]), op.imm)
			case leaInOp:
				text.LeaR64M(r64(in.Defs[0]), operand.Mem64(reg.RBP).Disp(op.off))
			case leaSymOp:
				text.LeaR64M(r64(in.Defs[0]),
					operand.Rip(amd64asm.Ref(op.sym, amd64asm.RefPC32)))
			case tlsAddrOp:
				// See iselTLSAddr. Defs[1] is the scratch, and it is
				// dead after the third instruction — the fourth reads
				// only the block pointer and the relocation.
				dst, idx := r64(in.Defs[0]), r64(in.Defs[1])
				text.MovR64RM64(dst, operand.Addr64(tebTLSPointer).Seg(reg.GS))
				text.MovR32RM32(r32(in.Defs[1]),
					operand.Rip32(amd64asm.Ref(tlsIndexSymbol, amd64asm.RefPC32)))
				text.MovR64RM64(dst, operand.Mem64(dst).Index(idx, 8))
				text.LeaR64M(dst,
					operand.Mem64(dst).Sym(amd64asm.Ref(op.sym, amd64asm.RefSecRel32)))
			case movSymGotOp:
				// The relaxable kind, in its REX-prefixed spelling: a 64-bit
				// MOV carries a REX prefix, and the linker that rewrites the
				// load back into an LEA has to look past it. Getting this
				// wrong is a relocation the assembler refuses rather than a
				// bug that survives to run time.
				text.MovR64RM64(r64(in.Defs[0]),
					operand.Rip64(amd64asm.Ref(op.sym, amd64asm.RefRexGOTPCRELX)))
			case leaBlockOp:
				// The same instruction as leaSymOp. A block in this
				// function is still a symbol as far as the reference is
				// concerned — labeledBlocks is what made it one — and
				// RIP-relative either way, since the block is in the
				// section being written.
				text.LeaR64M(r64(in.Defs[0]),
					operand.Rip(amd64asm.Ref(op.label, amd64asm.RefPC32)))
			case atomicRmwOp:
				// Two-address: the register is the addend going in and
				// memory's old value coming out, so a result that is
				// not already in the operand's register needs the move
				// first. LOCK is what makes the read, the add and the
				// write one operation.
				dst, addr := in.Defs[0], r64(in.Uses[1])
				if assigned[dst] != assigned[in.Uses[0]] {
					if op.w == w32 {
						text.MovR32RM32(r32(dst), r32(in.Uses[0]))
					} else {
						text.MovR64RM64(r64(dst), r64(in.Uses[0]))
					}
				}
				switch op.a {
				case a8:
					text.LockXaddRM8R8(operand.Mem8(addr), r8(dst))
				case a16:
					text.LockXaddRM16R16(operand.Mem16(addr), r16(dst))
				case a32:
					text.LockXaddRM32R32(operand.Mem32(addr), r32(dst))
				default:
					text.LockXaddRM64R64(operand.Mem64(addr), r64(dst))
				}
			case atomicXchgOp:
				// No LOCK. XCHG with a memory operand is the one
				// instruction on this architecture that is locked
				// whether the prefix is written or not, and writing it
				// would be a byte saying what the opcode already says.
				dst, addr := in.Defs[0], r64(in.Uses[1])
				if assigned[dst] != assigned[in.Uses[0]] {
					if op.w == w32 {
						text.MovR32RM32(r32(dst), r32(in.Uses[0]))
					} else {
						text.MovR64RM64(r64(dst), r64(in.Uses[0]))
					}
				}
				switch op.a {
				case a8:
					text.XchgRM8R8(operand.Mem8(addr), r8(dst))
				case a16:
					text.XchgRM16R16(operand.Mem16(addr), r16(dst))
				case a32:
					text.XchgRM32R32(operand.Mem32(addr), r32(dst))
				default:
					text.XchgRM64R64(operand.Mem64(addr), r64(dst))
				}
			case atomicCasOp:
				// The expected value is already in the accumulator and
				// the read value comes back there; vregs.physical is
				// what put it there and what keeps anything else out.
				addr := r64(in.Uses[2])
				switch op.a {
				case a8:
					text.LockCmpxchgRM8R8(operand.Mem8(addr), r8(in.Uses[1]))
				case a16:
					text.LockCmpxchgRM16R16(operand.Mem16(addr), r16(in.Uses[1]))
				case a32:
					text.LockCmpxchgRM32R32(operand.Mem32(addr), r32(in.Uses[1]))
				default:
					text.LockCmpxchgRM64R64(operand.Mem64(addr), r64(in.Uses[1]))
				}
			case bitCountOp:
				dst, src := in.Defs[0], in.Uses[0]
				if op.w == w32 {
					bitCount32(text, op.verb, r32(dst), r32(src))
					break
				}
				bitCount64(text, op.verb, r64(dst), r64(src))
			case bswapOp:
				// In place, so a destination that is not already the
				// operand's register needs the move first — the ALU
				// table's shape, with one operand instead of two.
				dst, src := in.Defs[0], in.Uses[0]
				if op.w == w32 {
					if r32(dst) != r32(src) {
						text.MovR32RM32(r32(dst), r32(src))
					}
					text.BswapR32(r32(dst))
					break
				}
				if r64(dst) != r64(src) {
					text.MovR64RM64(r64(dst), r64(src))
				}
				text.BswapR64(r64(dst))
			case mfenceOp:
				text.Mfence()
			case testOp:
				text.TestRM8R8(r8(in.Uses[0]), r8(in.Uses[0]))
			case setccOp:
				// The byte, then the three above it. r8 and r32 are the
				// same register out of the same allocation, which is
				// what lets the zero-extension read its own destination.
				setcc(text, op.cond, r8(in.Defs[0]))
				text.MovzxR32RM8(r32(in.Defs[0]), r8(in.Defs[0]))
			case zextOp:
				// A write to a 32-bit register zeroes the upper half, so
				// the zero-extension is a mov — and a mov that is not
				// redundant even when its two ends are one register.
				text.MovR32RM32(r32(in.Defs[0]), r32(in.Uses[0]))
			case sextOp:
				text.MovsxdR64RM32(r64(in.Defs[0]), r32(in.Uses[0]))
			case argStoreOp:
				// An i32 argument writes four bytes into an eightbyte
				// whose upper half SysV leaves unspecified. Writing all
				// eight would be writing a register's upper half, which
				// an i32 vreg does not have.
				switch op.w {
				case wv128:
					text.MovapsRM128Xmm(operand.Mem128(reg.RSP).Disp(op.off), xmm(in.Uses[0]))
				case wf32:
					text.MovssXM32Xmm(operand.Mem32(reg.RSP).Disp(op.off), xmm(in.Uses[0]))
				case wf64:
					text.MovsdXM64Xmm(operand.Mem64(reg.RSP).Disp(op.off), xmm(in.Uses[0]))
				case w32:
					text.MovRM32R32(operand.Mem32(reg.RSP).Disp(op.off), r32(in.Uses[0]))
				default:
					text.MovRM64R64(operand.Mem64(reg.RSP).Disp(op.off), r64(in.Uses[0]))
				}
			case argLoadOp:
				switch op.w {
				case wv128:
					text.MovapsXmmRM128(xmm(in.Defs[0]), operand.Mem128(reg.RBP).Disp(op.off))
				case wf32:
					text.MovssXmmXM32(xmm(in.Defs[0]), operand.Mem32(reg.RBP).Disp(op.off))
				case wf64:
					text.MovsdXmmXM64(xmm(in.Defs[0]), operand.Mem64(reg.RBP).Disp(op.off))
				case w32:
					text.MovR32RM32(r32(in.Defs[0]), operand.Mem32(reg.RBP).Disp(op.off))
				default:
					text.MovR64RM64(r64(in.Defs[0]), operand.Mem64(reg.RBP).Disp(op.off))
				}
			case spillOp:
				switch op.w {
				case wv128:
					text.MovapsRM128Xmm(operand.Mem128(reg.RBP).Disp(op.off), xmm(in.Uses[0]))
				case wf64:
					text.MovsdXM64Xmm(operand.Mem64(reg.RBP).Disp(op.off), xmm(in.Uses[0]))
				default:
					text.MovRM64R64(operand.Mem64(reg.RBP).Disp(op.off), r64(in.Uses[0]))
				}
			case reloadOp:
				switch op.w {
				case wv128:
					text.MovapsXmmRM128(xmm(in.Defs[0]), operand.Mem128(reg.RBP).Disp(op.off))
				case wf64:
					text.MovsdXmmXM64(xmm(in.Defs[0]), operand.Mem64(reg.RBP).Disp(op.off))
				default:
					text.MovR64RM64(r64(in.Defs[0]), operand.Mem64(reg.RBP).Disp(op.off))
				}
			case cmovOp:
				// The false arm into the destination, then the true arm
				// conditionally over it. Uses[1] is the false arm and
				// Uses[0] the true one, which is §F's own order of
				// operands read the way the instruction wants them.
				dst, yes, no := in.Defs[0], in.Uses[0], in.Uses[1]
				if op.w == w32 {
					if r32(dst) != r32(no) {
						text.MovR32RM32(r32(dst), r32(no))
					}
					cmov32(text, op.cond, r32(dst), r32(yes))
					break
				}
				if r64(dst) != r64(no) {
					text.MovR64RM64(r64(dst), r64(no))
				}
				cmov64(text, op.cond, r64(dst), r64(yes))
			case callOp:
				// A relocation against the symbol, not a label: the
				// callee may be in another object, and which one is the
				// linker's question.
				//
				// RefPLT32, which is what a call is in both containers.
				// On ELF it is R_X86_64_PLT32, what every current
				// toolchain emits for a call — the linker relaxes it to
				// a direct branch when the target turns out to be local,
				// so it costs a non-PIC object nothing. On Mach-O it is
				// X86_64_RELOC_BRANCH, and there it is not optional: the
				// type is how the linker is told this is a call at all,
				// and a call to an imported symbol is what makes it
				// synthesize the stub. RefPC32 became SIGNED, an
				// ordinary RIP-relative data reference, so the import
				// was never routed through a stub and the link failed
				// with the symbol unbound.
				text.CallRef(amd64asm.Ref(op.sym, amd64asm.RefPLT32))

			case callIndOp:
				// Indirect call to the address in in.Uses[0].
				text.CallRM64(r64(in.Uses[0]))
			case jmpIndOp:
				// Indirect jump to the address in in.Uses[0].
				text.JmpRM64(r64(in.Uses[0]))
			case brTableOp:
				// One unsigned compare for both ends of the range. A
				// negative selector read as unsigned is a very large
				// one, so the JAE that catches an index past the last
				// entry catches a negative one too, and the range check
				// is one branch rather than two.
				text.CmpRM32Imm32(r32(in.Uses[0]), int64(len(op.targets)))
				text.JaeLabel(op.defaultTarget)

				// The scratch register is the allocator's, as Defs[0]:
				// which registers are free at a terminator is what
				// allocation decides. The selector indexes at the full
				// 64-bit width, safe because a 32-bit write zeroes the
				// upper half, and the compare has bounded the lower.
				base := r64(in.Defs[0])
				text.LeaR64M(base, operand.Rip(amd64asm.Ref(op.id, amd64asm.RefPC32)))
				text.JmpRM64(operand.Mem64(base).Index(r64(in.Uses[0]), 8))

				// One absolute address per entry, rather than an offset
				// from the table's own start: that would be a label
				// difference across two sections, and a Section resolves
				// differences against its own labels where a relocation
				// crosses. It also makes the jump one instruction.
				ro := am.Section(amd64asm.ROData)
				ro.Align(8)
				ro.Label(op.id, amd64asm.Local)
				for _, target := range op.targets {
					ro.Ref(amd64asm.Ref(target, amd64asm.RefAbs64))
				}
			case jmpOp:
				text.JmpLabel(op.target)
			case wideConstOp:
				// Sixteen bytes in .rodata, aligned because MOVAPS
				// requires it. One label per literal rather than a pool
				// keyed by value: folding duplicates is a pass over
				// emitted data this package has nowhere to put.
				ro := am.Section(amd64asm.ROData)
				ro.Align(16)
				label := fmt.Sprintf("%s.const16.%d", fn.Name(), consts)
				consts++
				ro.Label(label, amd64asm.Local)
				ro.Quad(op.lo)
				ro.Quad(op.hi)
				text.MovapsXmmRM128(xmm(in.Defs[0]),
					operand.Rip128(amd64asm.Ref(label, amd64asm.RefPC32)))
			case asmOp:
				if err := emitAsm(text, fn, in, op, assigned); err != nil {
					return shape, err
				}
			default:
				if !emitVec(text, in, xmm, r32, r64) {
					return shape, fmt.Errorf("%s: no emitter for %T", fn.Name(), in.Op)
				}
			case constOp:
				if op.w == w32 {
					text.MovR32Imm32(r32(in.Defs[0]), op.imm)
					break
				}
				// The full ten-byte movabs, unconditionally. A
				// value that fits in 32 bits could be five bytes
				// as a move to the 32-bit view, which zero-extends,
				// or seven as the sign-extended MovRM64Imm32 — and
				// both of those are peepholes. This package has no
				// pass to put a peephole in, and choosing an
				// encoding by the value at isel time would put one
				// here, where the reason for it could not be read.
				text.MovR64Imm64(r64(in.Defs[0]), uint64(op.imm))
			}
		}
	}
	text.EndLabel(fn.Name())
	return shape, nil
}
