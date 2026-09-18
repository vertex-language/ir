package amdgpu

// MIR to bytes: each amdOp's template filled in from the register
// assignment and handed to the assembler's Emit, which picks the form by
// mnemonic and operand classes.

import (
	"fmt"
	"math"

	amdgpuasm "github.com/vertex-language/amdgpu"
	"github.com/vertex-language/amdgpu/obj"
	"github.com/vertex-language/amdgpu/operand"
	"github.com/vertex-language/amdgpu/reg"

	"github.com/vertex-language/ir/lower/mir"
	"github.com/vertex-language/ir/lower/regalloc"
)

// emit writes one selected, allocated function into the module.
func (l *lowerer) emit(x *fnState, assigned map[mir.VReg]regalloc.PhysReg) error {
	vgprs, sgprs := registerCounts(assigned)
	// The descriptor's register counts are written before the body,
	// but the fixed VGPRs a body names are known only once it is
	// emitted: count them ahead by looking.
	for _, b := range x.mf.Blocks {
		for _, in := range b.Instrs {
			if op, ok := in.Op.(amdOp); ok {
				for _, o := range op.ops {
					if o.kind == oFixedV {
						vgprs = max(vgprs, int(o.imm)+max(o.i, 1))
					}
				}
			}
		}
	}
	k := l.am.Kernel(x.fn.Name(), x.kernelOptions(vgprs, sgprs)...)
	text := k.Section()
	e := &emitter{text: text, assigned: assigned, trapLabel: x.fn.Name() + ".trap"}
	e.forward(x.mf.Blocks)
	var blocks []*mir.Block
	for _, b := range x.mf.Blocks {
		if _, skip := e.fwd[b.Label]; !skip {
			blocks = append(blocks, b)
		}
	}
	for i, b := range blocks {
		if i > 0 {
			text.Label(b.Label)
		}
		next := ""
		if i+1 < len(blocks) {
			next = blocks[i+1].Label
		}
		for _, in := range b.Instrs {
			if err := e.instr(in, next); err != nil {
				return fmt.Errorf("lower: @%s: %s: %w", x.fn.Name(), b.Label, err)
			}
		}
	}
	if e.trapUsed {
		text.Label(e.trapLabel)
		text.Emit("s_trap", operand.Imm(2))
	}
	text.EndLabel(x.fn.Name())
	return l.am.Err()
}

// registerCounts is how many VGPRs and SGPRs the assignment reaches:
// one past the highest register any value occupies, pairs counting both
// halves. v0 and s[0:1] are pinned inputs, so a kernel that touches
// nothing still counts them.
func registerCounts(assigned map[mir.VReg]regalloc.PhysReg) (vgprs, sgprs int) {
	vgprs, sgprs = 1, 2
	for _, p := range assigned {
		switch {
		case p >= physV64:
			vgprs = max(vgprs, int(p-physV64)+2)
		case p >= physS64:
			sgprs = max(sgprs, int(p-physS64)+2)
		case p >= physV32:
			vgprs = max(vgprs, int(p-physV32)+1)
		default:
			sgprs = max(sgprs, int(p)+1)
		}
	}
	return vgprs, sgprs
}

type emitter struct {
	text     *amdgpuasm.Section
	assigned map[mir.VReg]regalloc.PhysReg

	// fwd is where a branch to an empty block goes instead: an edge block
	// whose every copy the allocator coalesced away is a branch and
	// nothing else, so the branch that reaches it takes its target.
	fwd map[string]string

	// trapLabel is the function's s_trap, emitted once if anything
	// branches to it.
	trapLabel string
	trapUsed  bool

	// fixedTop is one past the highest fixed VGPR an instruction named,
	// which the register count has to reach.
	fixedTop int
}

// forward finds the empty blocks and resolves chains of them.
func (e *emitter) forward(blocks []*mir.Block) {
	e.fwd = map[string]string{}
	for _, b := range blocks[1:] {
		if n := len(b.Instrs); n > 0 {
			if br, ok := b.Instrs[n-1].Op.(branchOp); ok && e.allElided(b.Instrs[:n-1]) {
				e.fwd[b.Label] = br.target
			}
		}
	}
	for from := range e.fwd {
		to := e.fwd[from]
		for hops := 0; hops < len(e.fwd); hops++ {
			next, ok := e.fwd[to]
			if !ok || next == from {
				break
			}
			to = next
		}
		e.fwd[from] = to
	}
}

// allElided reports whether every instruction is a copy the assignment
// made a no-op.
func (e *emitter) allElided(ins []mir.Instr) bool {
	for _, in := range ins {
		if _, ok := in.Op.(undefOp); ok {
			continue
		}
		if _, ok := in.Op.(movOp); !ok || e.reg(in.Defs[0]) != e.reg(in.Uses[0]) {
			return false
		}
	}
	return true
}

// target is a branch's label after forwarding.
func (e *emitter) target(label string) string {
	if to, ok := e.fwd[label]; ok {
		return to
	}
	return label
}

func (e *emitter) reg(v mir.VReg) reg.Reg {
	p, ok := e.assigned[v]
	if !ok {
		panic(fmt.Sprintf("amdgpu: vreg %d has no register", v))
	}
	return regOf(p)
}

func (e *emitter) half(v mir.VReg, hi bool) reg.Reg {
	p, ok := e.assigned[v]
	if !ok {
		panic(fmt.Sprintf("amdgpu: vreg %d has no register", v))
	}
	return halfOf(p, hi)
}

func (e *emitter) instr(in mir.Instr, next string) error {
	switch op := in.Op.(type) {
	case amdOp:
		ops := make([]operand.Operand, len(op.ops))
		for i, o := range op.ops {
			ops[i] = e.operand(o, in)
		}
		e.text.Emit(op.mn, ops...)
		if op.wait {
			e.text.Emit("s_waitcnt", operand.NewWaitCnt().VM(0).LGKM(0))
		}
	case movOp:
		if in.Defs[0] == in.Uses[0] {
			return nil
		}
		dst, src := e.reg(in.Defs[0]), e.reg(in.Uses[0])
		if dst == src {
			return nil
		}
		switch op.w {
		case v32:
			e.text.Emit("v_mov_b32", dst, src)
		case v64:
			e.text.Emit("v_mov_b32", e.half(in.Defs[0], false), e.half(in.Uses[0], false))
			e.text.Emit("v_mov_b32", e.half(in.Defs[0], true), e.half(in.Uses[0], true))
		case s32:
			e.text.Emit("s_mov_b32", dst, src)
		case s64:
			if op.masked {
				e.text.Emit("s_and_b64", reg.VCC, src, reg.EXEC)
				e.text.Emit("s_andn2_b64", dst, dst, reg.EXEC)
				e.text.Emit("s_or_b64", dst, dst, reg.VCC)
			} else {
				e.text.Emit("s_mov_b64", dst, src)
			}
		}
	case branchOp:
		if t := e.target(op.target); t != next {
			e.text.Emit("s_branch", operand.NewLabel(t))
		}
	case cbranchOp:
		// The mask against the active lanes, into VCC as scratch; SCC is
		// whether any bit survived.
		e.text.Emit("s_and_b64", reg.VCC, e.reg(in.Uses[0]), reg.EXEC)
		e.text.Emit("s_cbranch_scc1", operand.NewLabel(e.target(op.then)))
		if t := e.target(op.els); t != next {
			e.text.Emit("s_branch", operand.NewLabel(t))
		}
	case endpgmOp:
		e.text.Emit("s_endpgm")
	case trapOp:
		e.text.Emit("s_trap", operand.Imm(2))
	case trapIfOp:
		e.text.Emit("s_and_b64", reg.VCC, e.reg(in.Uses[0]), reg.EXEC)
		e.text.Emit("s_cbranch_scc1", operand.NewLabel(e.trapLabel))
		e.trapUsed = true
	case trapAnyOp:
		e.text.Emit("s_cmp_lg_u64", reg.EXEC, operand.Imm(0))
		e.text.Emit("s_cbranch_scc1", operand.NewLabel(e.trapLabel))
		e.trapUsed = true

	// The structurizer's ops; see structurize.go.
	case execSaveOp:
		e.text.Emit("s_mov_b64", e.reg(in.Defs[0]), reg.EXEC)
	case execRestoreOp:
		e.text.Emit("s_mov_b64", reg.EXEC, e.reg(in.Uses[0]))
	case execAndOp:
		e.text.Emit("s_and_b64", reg.EXEC, reg.EXEC, e.reg(in.Uses[0]))
	case execzOp:
		e.text.Emit("s_cbranch_execz", operand.NewLabel(e.target(op.target)))
	case maskInitOp:
		e.text.Emit("s_mov_b64", e.reg(in.Defs[0]), operand.Imm(0))
	case pendClearOp:
		p := e.reg(in.Defs[0])
		e.text.Emit("s_andn2_b64", p, p, reg.EXEC)
	case pendSetOp:
		p, m := e.reg(in.Defs[0]), e.reg(in.Uses[1])
		if op.neg {
			e.text.Emit("s_andn2_b64", reg.VCC, reg.EXEC, m)
		} else {
			e.text.Emit("s_and_b64", reg.VCC, m, reg.EXEC)
		}
		e.text.Emit("s_andn2_b64", p, p, reg.EXEC)
		e.text.Emit("s_or_b64", p, p, reg.VCC)
	case pendSetAllOp:
		p := e.reg(in.Defs[0])
		e.text.Emit("s_or_b64", p, p, reg.EXEC)
	case undefOp:
	default:
		return fmt.Errorf("%T is not an instruction this emitter knows", in.Op)
	}
	return e.text.Module().Err()
}

// operand fills one template slot.
func (e *emitter) operand(o opnd, in mir.Instr) operand.Operand {
	switch o.kind {
	case oDef:
		return e.reg(in.Defs[o.i])
	case oUse:
		return modified(o, e.reg(in.Uses[o.i]))
	case oDefLo:
		return e.half(in.Defs[o.i], false)
	case oDefHi:
		return e.half(in.Defs[o.i], true)
	case oUseLo:
		return modified(o, e.half(in.Uses[o.i], false))
	case oUseHi:
		return modified(o, e.half(in.Uses[o.i], true))
	case oImm:
		return immediate(o.imm)
	case oFImm:
		return operand.Float(math.Float32frombits(uint32(o.imm)))
	case oVCC:
		return reg.VCC
	case oExec:
		return reg.EXEC
	case oM0:
		return reg.M0
	case oFlat:
		return operand.Flat(e.reg(in.Uses[o.i])).Off(int32(o.imm)).SetGLC(o.glc).SetSC1(o.sc1)
	case oGlobal:
		return operand.Global(e.reg(in.Uses[o.i])).Off(int32(o.imm))
	case oGlobalS:
		return operand.Global(e.reg(in.Uses[o.i]), e.reg(in.Uses[o.i+1])).Off(int32(o.imm))
	case oSMEM:
		return operand.SMEM(e.reg(in.Uses[o.i])).Off(int32(o.imm))
	case oDS:
		// The LDS offset is the flat address's low dword.
		return operand.DS(e.half(in.Uses[o.i], false)).Off(int32(o.imm))
	case oSharedBase:
		return reg.SRC_SHARED_BASE
	case oCache:
		return operand.Cache(o.glc, o.sc1)
	case oFixedV:
		n := int(o.imm)
		if e.fixedTop < n+max(o.i, 1) {
			e.fixedTop = n + max(o.i, 1)
		}
		switch o.i {
		case 2:
			return reg.V2(reg.VGPR(n))
		case 4:
			return reg.V4(reg.VGPR(n))
		}
		return reg.VGPR(n)
	case oLabel:
		return operand.NewLabel(o.sym)
	case oWait:
		return operand.NewWaitCnt().VM(0).LGKM(0)
	case oSymLo:
		return operand.Ref(o.sym, obj.RefRel32Lo).WithAddend(o.imm)
	case oSymHi:
		return operand.Ref(o.sym, obj.RefRel32Hi).WithAddend(o.imm)
	}
	panic(fmt.Sprintf("amdgpu: operand kind %d", o.kind))
}

// modified wraps a source in its neg and abs modifiers.
func modified(o opnd, r reg.Reg) operand.Operand {
	var out operand.Operand = r
	if o.abs {
		out = operand.Abs(out)
	}
	if o.neg {
		out = operand.Neg(out)
	}
	return out
}

// immediate is a 32-bit constant as the assembler takes it: an inline
// constant where the value is one, a literal otherwise. The value is the
// dword's bits; isel hands a negative for a bit pattern with the top bit
// set, and either spelling of the same dword reaches the same encoding.
func immediate(v int64) operand.Operand {
	if v >= -16 && v <= 64 {
		return operand.Imm(int32(v))
	}
	return operand.Literal(uint32(v))
}
