package arm64

// Compact unwind: generates __LD,__compact_unwind records for Darwin / Mach-O.
// Uses UNWIND_ARM64_MODE_FRAME encoding describing saved register pairs (X19..X28, FP/LR)
// and stack allocation depths.

import (
	arm64asm "github.com/vertex-language/arm64"

	"github.com/vertex-language/arm64/reg"
	"github.com/vertex-language/ir"
	"github.com/vertex-language/ir/lower/mir"
	"github.com/vertex-language/ir/lower/regalloc"
)

// compactUnwindSection is where the records go, in as(1)'s own spelling. The
// debug attribute is what tells a linker that does not understand the section
// to drop it rather than copy it into the image.
const compactUnwindSection = "__LD,__compact_unwind,regular,debug"

// compactUnwindRecord is the size of one record: the function, its length and
// encoding, the personality routine and the LSDA.
const compactUnwindRecord = 32

// The common bits of a compact unwind word, shared by every architecture.
const (
	// unwindHasLSDA says the record's fifth field is a language-specific
	// data area — the table @catch is compiled into.
	unwindHasLSDA = 0x40000000

	// unwindPersonalityMask holds a 1-based index into the image's
	// personality array, which the linker fills in: the compiler names the
	// routine with a relocation and does not know its number.
	unwindPersonalityMask = 0x30000000
)

// The AArch64 modes and the register-pair bits.
const (
	unwindARM64ModeFrameless = 0x02000000
	unwindARM64ModeDwarf     = 0x03000000
	unwindARM64ModeFrame     = 0x04000000

	unwindARM64X19X20 = 0x00000001
	unwindARM64X21X22 = 0x00000002
	unwindARM64X23X24 = 0x00000004
	unwindARM64X25X26 = 0x00000008
	unwindARM64X27X28 = 0x00000010
	unwindARM64D8D9   = 0x00000100
	unwindARM64D10D11 = 0x00000200
	unwindARM64D12D13 = 0x00000400
	unwindARM64D14D15 = 0x00000800

	// unwindARM64FramelessStackSize is the field a frameless function
	// states its stack in, in sixteen-byte units.
	unwindARM64FramelessStackSize = 0x00FFF000
	unwindARM64FramelessShift     = 12
)

// unwindPairs is the integer callee-saved file, in pairs, in the one order
// that matters: it is the order libunwind restores them in, which makes it
// the order the frame has to hold them in and the order savedPairs assigns.
var unwindPairs = []struct {
	bit  uint32
	regs [2]reg.X
}{
	{unwindARM64X19X20, [2]reg.X{reg.X19, reg.X20}},
	{unwindARM64X21X22, [2]reg.X{reg.X21, reg.X22}},
	{unwindARM64X23X24, [2]reg.X{reg.X23, reg.X24}},
	{unwindARM64X25X26, [2]reg.X{reg.X25, reg.X26}},
	{unwindARM64X27X28, [2]reg.X{reg.X27, reg.X28}},
}

// unwindVecPairs is the same for the vector file, which follows the integer
// one in the frame.
var unwindVecPairs = []struct {
	bit  uint32
	regs [2]reg.V
}{
	{unwindARM64D8D9, [2]reg.V{reg.V8, reg.V9}},
	{unwindARM64D10D11, [2]reg.V{reg.V10, reg.V11}},
	{unwindARM64D12D13, [2]reg.V{reg.V12, reg.V13}},
	{unwindARM64D14D15, [2]reg.V{reg.V14, reg.V15}},
}

// saves is what the prologue stores and what the epilogue puts back.
//
// Two lists rather than one, because they differ in exactly one register. A
// function that returns a swifterror leaves it in X21 for its caller to read,
// so the epilogue must not restore X21 over it — but the prologue still saves
// it, and the compact unwind record still claims the pair. Unwinding is not a
// return: there is no error result to preserve, and the caller of a frame
// being unwound past is entitled to its own X21 back.
type saves struct {
	x       []reg.X
	v       []reg.V
	restore []reg.X
}

// planSaves decides what this function's prologue stores.
func planSaves(fn *ir.Func, pool *regalloc.Pool, assigned map[mir.VReg]regalloc.PhysReg) saves {
	sv := saves{
		x: savedPairs(usedCalleeSaved(pool, assigned)),
		v: savedVecPairs(usedCalleeSavedVec(pool, assigned)),
	}
	sv.restore = sv.x
	if funcErrorResult(fn) >= 0 {
		sv.restore = without(sv.x, reg.X21)
	}
	return sv
}

// savedPairs rounds a set of used registers up to whole pairs, in frame
// order. See the pair discussion at the top of the file.
func savedPairs(used []reg.X) []reg.X {
	var out []reg.X
	for _, p := range unwindPairs {
		if containsX(used, p.regs[0]) || containsX(used, p.regs[1]) {
			out = append(out, p.regs[0], p.regs[1])
		}
	}
	return out
}

func savedVecPairs(used []reg.V) []reg.V {
	var out []reg.V
	for _, p := range unwindVecPairs {
		if containsV(used, p.regs[0]) || containsV(used, p.regs[1]) {
			out = append(out, p.regs[0], p.regs[1])
		}
	}
	return out
}

func containsX(rs []reg.X, r reg.X) bool {
	for _, have := range rs {
		if have == r {
			return true
		}
	}
	return false
}

func containsV(rs []reg.V, r reg.V) bool {
	for _, have := range rs {
		if have == r {
			return true
		}
	}
	return false
}

// unwindEncoding is the word describing fr.
func unwindEncoding(fr *frame, sv saves) uint32 {
	if !fr.needed() {
		// No frame record, no saves, and nothing subtracted from SP: the
		// caller's SP is this one and the return address is still in X30,
		// which is what a frameless entry of size zero says.
		return unwindARM64ModeFrameless
	}
	enc := uint32(unwindARM64ModeFrame)
	for _, p := range unwindPairs {
		if containsX(sv.x, p.regs[0]) {
			enc |= p.bit
		}
	}
	for _, p := range unwindVecPairs {
		if containsV(sv.v, p.regs[0]) {
			enc |= p.bit
		}
	}
	return enc
}

// emitCompactUnwind writes one record.
//
// Three of the five fields are relocations and are written as zeros: the
// function, the personality routine and the LSDA are addresses, and an
// address is not something this layer knows. The length is not — it is the
// distance between two offsets in a section being built, which is a number.
func emitCompactUnwind(am *arm64asm.Module, fn string, enc, length uint32, personality, lsda string) {
	if lsda != "" {
		enc |= unwindHasLSDA
	}
	s := am.SectionNamed(compactUnwindSection, arm64asm.Data)
	s.Align(8)
	s.Ref(fn, arm64asm.RefAbs64)
	s.Long(length)
	s.Long(enc)
	// The personality's index in the image's array of them is two bits of
	// the encoding, and the linker fills them in: it is the one that knows
	// how many distinct personalities the image ended up with.
	if personality == "" {
		s.Quad(0)
	} else {
		s.Ref(personality, arm64asm.RefAbs64)
	}
	if lsda == "" {
		s.Quad(0)
		return
	}
	s.Ref(lsda, arm64asm.RefAbs64)
}

// personalitySyms is every personality routine the module names and does not
// itself define, which have to be declared before anything references one.
func personalitySyms(m *ir.Module) []string {
	defined := make(map[string]bool)
	for _, fn := range m.Funcs() {
		defined[fn.Name()] = true
	}
	seen := map[string]bool{}
	var out []string
	for _, fn := range m.Funcs() {
		p := fn.PersonalityFn()
		if p == nil || defined[p.Name()] || seen[p.Name()] {
			continue
		}
		seen[p.Name()] = true
		out = append(out, p.Name())
	}
	return out
}
