package i386

import (
	"github.com/vertex-language/i386/reg"

	"github.com/vertex-language/ir/lower/mir"
	"github.com/vertex-language/ir/lower/regalloc"
)

// byteRegs are the registers with an eight-bit form in 32-bit code. ESI,
// EDI, EBP and ESP have none: the encodings that would name their low
// byte name AH, CH, DH and BH instead, so a byte operand allocated to ESI
// writes DH and corrupts EDX.
var byteRegs = []regalloc.PhysReg{
	regalloc.PhysReg(reg.EAX), regalloc.PhysReg(reg.ECX),
	regalloc.PhysReg(reg.EDX), regalloc.PhysReg(reg.EBX),
}

// restrictBytes tells the allocator which vregs an instruction names as a
// byte register: SETcc's destination, a zero-extension from a byte, and
// the value of a byte store or byte atomic.
func restrictBytes(mf *mir.Func, pool *regalloc.Pool) {
	for _, b := range mf.Blocks {
		for _, in := range b.Instrs {
			switch op := in.Op.(type) {
			case setccOp:
				pool.Restrict(in.Defs[0], byteRegs)
			case zextOp:
				if op.from == a8 {
					pool.Restrict(in.Defs[0], byteRegs)
				}
			case subStoreOp:
				if op.to == a8 {
					pool.Restrict(in.Uses[0], byteRegs)
				}
			case xchgOp:
				if op.a == a8 {
					pool.Restrict(in.Defs[0], byteRegs)
				}
			case xaddOp:
				if op.a == a8 {
					pool.Restrict(in.Defs[0], byteRegs)
				}
			case cmpxchgOp:
				if op.a == a8 {
					pool.Restrict(in.Uses[1], byteRegs)
				}
			}
		}
	}
}
