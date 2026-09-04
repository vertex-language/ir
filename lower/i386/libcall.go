package i386

// The calls this package makes that the module did not write.
//
// A 32-bit machine cannot divide a 64-bit value with an instruction, so §A's
// four division verbs at that width become calls to the same helpers a C
// compiler emits — __divdi3 and its neighbours, which compiler-rt and libgcc
// both provide under those names.

import (
	i386asm "github.com/vertex-language/i386"
	"github.com/vertex-language/i386/reg"

	"github.com/vertex-language/ir"
	"github.com/vertex-language/ir/lower/mir"
)

// libcalls is §E's verbs and the C library function each becomes.
var libcalls = map[ir.Verb]string{
	ir.VMemCpy:  "memcpy",
	ir.VMemMove: "memmove",
	ir.VMemSet:  "memset",
	ir.VMemCmp:  "memcmp",
}

// libcallSyms is every library symbol a module needs declared.
//
// Collected up front because an undeclared reference is an error at Finalize,
// by which point there is no instruction left to name in the message.
func libcallSyms(m *ir.Module) []string {
	seen := map[string]bool{}
	var out []string
	add := func(sym string) {
		if !seen[sym] {
			seen[sym] = true
			out = append(out, sym)
		}
	}
	for _, fn := range m.Funcs() {
		fn.WalkInsts(func(in *ir.Inst) bool {
			op := in.Op()
			if sym, ok := libcalls[op.Verb]; ok {
				add(sym)
			}
			// A 64-bit division, which only the i64 namespace has.
			if _, ok := floatHelpers[op.Verb]; ok {
				if w, ok := widthOf(op.Type); ok {
					if n, ok := helperFor(op.Verb, w); ok {
						add(n)
					}
				}
			}
			switch op.Verb {
			case ir.VSCvtI64, ir.VUCvtI64:
				if w, ok := widthOf(op.Type); ok {
					add(intToFloatHelpers[[2]bool{op.Verb == ir.VSCvtI64, w == wf64}])
				}
			}
			// A float-to-integer conversion at sixty-four bits,
			// trapping or saturating alike: SSE2 has no such
			// instruction at either width.
			if isFloatToInt(op.Verb) {
				if w, ok := widthOf(op.Type); ok && w.pairs() {
					add(floatToIntHelpers[[2]bool{
						signedConversions[op.Verb],
						f64Conversions[op.Verb],
					}])
				}
			}
			// Either allocation form: zeroed means the same on both,
			// whether the size was known when the frame was planned
			// or arrives as a value.
			switch op.Verb {
			case ir.VAlloc, ir.VAlloca:
				if in.Zeroed() {
					add(libcalls[ir.VMemSet])
				}
			}
			if op.Type == ir.TypeI64 {
				switch op.Verb {
				case ir.VSDiv:
					add(divHelper(true, false))
				case ir.VUDiv:
					add(divHelper(false, false))
				case ir.VSRem:
					add(divHelper(true, true))
				case ir.VURem:
					add(divHelper(false, true))
				}
			}
			return true
		})
	}
	return out
}

// emitLibcall calls sym with args and takes its result back.
//
// The psABI's own sequence, which is short here: every argument is a stack
// slot, so there is no classification to do and no register file to run out
// of. Deliberately less than the written-call path — no variadic tail and one
// result at most — because each of those is true of every call this package
// invents.
func emitLibcall(c *cursor, vr *vregs, fr *frame, sym string, args []value, result value, wantResult bool) error {
	var off int32
	for _, a := range args {
		if a.w.isFloat() {
			c.Emit(mir.Instr{Op: fargStoreOp{off: off, w: a.w}, Uses: []mir.VReg{a.lo}})
			off += 4
			if a.w == wf64 {
				off += 4
			}
			continue
		}
		c.Emit(mir.Instr{Op: argStoreOp{off: off}, Uses: []mir.VReg{a.lo}})
		off += 4
		if a.w.pairs() {
			c.Emit(mir.Instr{Op: argStoreOp{off: off}, Uses: []mir.VReg{a.hi}})
			off += 4
		}
	}

	eax := vr.physical(reg.EAX)
	edx := vr.physical(reg.EDX)
	ecx := vr.physical(reg.ECX)
	c.Emit(mir.Instr{Op: callOp{sym: sym}, Defs: []mir.VReg{eax, edx, ecx}})

	if !wantResult {
		return nil
	}
	if result.w.isFloat() {
		c.Emit(mir.Instr{
			Op:   fstpResultOp{off: fr.scratch(), w: result.w},
			Defs: []mir.VReg{result.lo},
		})
		return nil
	}
	emitCopy(c, result.lo, eax)
	if result.w.pairs() {
		emitCopy(c, result.hi, edx)
	}
	return nil
}

// libcallOutgoing is how much outgoing area one invented call needs, so that
// planFrame reserves it before isel asks.
//
// Generous rather than exact. Every call this package invents takes at most
// three arguments of at most eight bytes, and reserving that much for all of
// them costs a few words of frame and cannot be too small — where being too
// small would put a call's arguments on top of this function's own slots.
func libcallOutgoing(in *ir.Inst) uint64 {
	op := in.Op()
	if _, ok := libcalls[op.Verb]; ok {
		places, err := classifySysV(typesOf(in.Args()))
		if err != nil {
			return 0
		}
		return outgoingBytes(places)
	}
	if _, ok := floatHelpers[op.Verb]; ok {
		return alignUp(24, maxAlign)
	}
	switch op.Verb {
	case ir.VSDiv, ir.VUDiv, ir.VSRem, ir.VURem:
		if op.Type == ir.TypeI64 {
			return alignUp(16, maxAlign)
		}
	case ir.VSCvtI64, ir.VUCvtI64:
		return alignUp(16, maxAlign)
	}
	if isFloatToInt(op.Verb) {
		return alignUp(16, maxAlign)
	}
	// A zeroed allocation's memset: three slots, like every other call
	// this package invents.
	switch op.Verb {
	case ir.VAlloc, ir.VAlloca:
		if in.Zeroed() {
			return alignUp(12, maxAlign)
		}
	}
	return 0
}

// declareLibcalls tells the assembler about every symbol this package will
// name that the module did not.
func declareLibcalls(am *i386asm.Module, m *ir.Module) {
	for _, s := range libcallSyms(m) {
		am.Extern(s)
	}
}
