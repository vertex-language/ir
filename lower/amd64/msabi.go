package amd64

// The Microsoft x64 calling convention, which is what a module built for
// x86_64-windows declares. It is not SysV with the registers renamed.
// Three differences reach every part of this package:
//
//   - There is one argument sequence, not two. Argument i takes register
//     i of whichever file its type belongs to, so a double in the second
//     position takes XMM1 and spends RDX without using it. Past the
//     fourth, every argument goes on the stack.
//
//   - Every argument owns an eightbyte of the caller's frame, the four
//     passed in registers included: the caller reserves 32 bytes of home
//     space above the return address for them, and the callee may write
//     them there. That is what makes a va_list here a bare pointer into
//     one contiguous array rather than §3.5.7's four-field struct.
//
//   - An aggregate travels in a register only when it is exactly one,
//     two, four, or eight bytes wide. Anything else travels as the
//     address of a copy the caller makes.
//
// RDI and RSI are callee-saved here and caller-saved under SysV, and
// XMM6 through XMM15 are callee-saved here and caller-saved under SysV,
// which is why every register this package names by its ABI role is
// reached through regsFor rather than read out of a package variable.

import (
	"fmt"

	"github.com/vertex-language/amd64/reg"

	"github.com/vertex-language/ir"
)

// The two ABI names an ir.Layout may carry here, spelled once.
const (
	abiSysV = "sysv"
	abiMS   = "ms"
)

const (
	// msHomeOff is where the caller's home space begins, measured from
	// the callee's RBP: the return address at +8 and the pushed RBP at
	// +0 are what separate them.
	msHomeOff = 16

	// msShadow is the home space every caller reserves, whether or not
	// the callee writes it and whether or not there are four arguments
	// to fill it. No outgoing area is smaller.
	msShadow = 32

	// msRegArgs is how many argument positions are passed in registers,
	// counting both files against the one sequence.
	msRegArgs = 4
)

// msIntArgs is the Microsoft AMD64 integer/pointer argument registers.
var msIntArgs = []reg.R64{reg.RCX, reg.RDX, reg.R8Q, reg.R9Q}

// msFloatArgs is the Microsoft AMD64 SSE argument registers: the same
// four positions as msIntArgs and not four more.
var msFloatArgs = []reg.Xmm{reg.XMM0, reg.XMM1, reg.XMM2, reg.XMM3}

// msIntRets is the Microsoft AMD64 integer return register. One, where
// SysV has two: a pair comes back through sret.
var msIntRets = []reg.R64{reg.RAX}

// msFloatRets is the Microsoft AMD64 SSE return register.
var msFloatRets = []reg.Xmm{reg.XMM0}

// msCalleeSaved is every Microsoft callee-saved general-purpose register
// this package will allocate. RDI and RSI belong here and not in the
// caller-saved list, which is the whole reason the two ABIs cannot share
// one table.
var msCalleeSaved = []reg.R64{reg.RBX, reg.RDI, reg.RSI, reg.R12Q, reg.R13Q, reg.R14Q, reg.R15Q}

// msCallerSaved is every Microsoft caller-saved general-purpose
// register, in the order scratchPool hands them out.
var msCallerSaved = []reg.R64{
	reg.RAX, reg.RCX, reg.RDX, reg.R8Q, reg.R9Q, reg.R10Q, reg.R11Q,
}

// msXmm is the vector registers this package allocates under the
// Microsoft ABI: the caller-saved six. XMM6 through XMM15 are
// callee-saved and nothing here saves a vector register, so nothing here
// hands one out.
var msXmm = []reg.Xmm{reg.XMM0, reg.XMM1, reg.XMM2, reg.XMM3, reg.XMM4, reg.XMM5}

// sysvXmm is the vector registers this package allocates under SysV,
// which is all sixteen: §3.2.1 makes none of them callee-saved.
var sysvXmm = func() []reg.Xmm {
	out := make([]reg.Xmm, 0, 16)
	for r := reg.XMM0; r <= reg.XMM15; r++ {
		out = append(out, r)
	}
	return out
}()

// An abiRegs is one calling convention's register tables.
type abiRegs struct {
	intArgs     []reg.R64
	floatArgs   []reg.Xmm
	intRets     []reg.R64
	floatRets   []reg.Xmm
	calleeSaved []reg.R64
	callerSaved []reg.R64
	xmm         []reg.Xmm
}

var (
	sysvRegs = &abiRegs{
		intArgs: sysvIntArgs, floatArgs: sysvFloatArgs,
		intRets: sysvIntRets, floatRets: sysvFloatRets,
		calleeSaved: calleeSaved, callerSaved: callerSaved,
		xmm: sysvXmm,
	}
	msRegs = &abiRegs{
		intArgs: msIntArgs, floatArgs: msFloatArgs,
		intRets: msIntRets, floatRets: msFloatRets,
		calleeSaved: msCalleeSaved, callerSaved: msCallerSaved,
		xmm: msXmm,
	}
)

// regsFor is the register tables of the named ABI. checkLayout has
// rejected every other name before lowering asks.
func regsFor(abi string) *abiRegs {
	if abi == abiMS {
		return msRegs
	}
	return sysvRegs
}

func intArgReg(abi string, i int) reg.R64   { return regsFor(abi).intArgs[i] }
func floatArgReg(abi string, i int) reg.Xmm { return regsFor(abi).floatArgs[i] }
func intRetReg(abi string, i int) reg.R64   { return regsFor(abi).intRets[i] }
func floatRetReg(abi string, i int) reg.Xmm { return regsFor(abi).floatRets[i] }

func numIntArgs(abi string) int   { return len(regsFor(abi).intArgs) }
func numFloatArgs(abi string) int { return len(regsFor(abi).floatArgs) }
func numIntRets(abi string) int   { return len(regsFor(abi).intRets) }
func numFloatRets(abi string) int { return len(regsFor(abi).floatRets) }

// classifyMS places a list of arguments by the Microsoft convention.
//
// One counter and not SysV's two: slot is the argument's position, it
// names a register in one file or the other until it runs past the
// fourth, and after that everything is memory. The stack area starts at
// 32 because the first four positions own the home space whether they
// used it or not, which is what keeps position and offset the same fact.
func classifyMS(args []abiArg) ([]place, error) {
	out := make([]place, len(args))
	slot := 0
	stackBytes := uint64(msShadow)

	// toStack places one argument in the outgoing area. Eightbyte-aligned
	// and eightbyte-granular: this ABI has no packed stack arguments, and
	// nothing it puts on the stack is wider than a pointer.
	toStack := func(size uint64) place {
		p := place{kind: placeStack, off: int32(stackBytes), size: size}
		stackBytes += alignUp(size, 8)
		return p
	}

	// reg1 places one value of at most eight bytes: in the register its
	// file names at the current position, or on the stack once the four
	// positions are spent.
	reg1 := func(k placeKind, w width, size uint64) place {
		if slot >= msRegArgs {
			slot++
			p := toStack(size)
			p.scalarW = w
			return p
		}
		r := regSlot{kind: k, i: slot, w: w}
		slot++
		return place{kind: k, regs: []regSlot{r}, scalarW: w}
	}

	for i, a := range args {
		if i == 0 && !a.sret.IsZero() {
			agg, inRegs, err := msSretInRegs(a.sret)
			if err != nil {
				return nil, err
			}
			if inRegs {
				// The caller supplies no address, so this parameter
				// takes no position at all. It still needs somewhere the
				// body can write through, which is a slot of the
				// callee's own — an aggregate place with nothing
				// arriving in it.
				out[i] = place{kind: placeInt, byval: a.sret, size: agg.size}
				continue
			}
			// Otherwise the hidden pointer is an ordinary first
			// argument, which is what the signature already says it is,
			// and it falls through to the scalar case below.
		}

		if a.byval.IsZero() {
			w, ok := widthOf(a.t)
			if !ok {
				return nil, fmt.Errorf("%s is not a value this package passes; only i32, i64, ptr, f32, and f64 are placed", a.t)
			}
			if w == wv128 {
				if a.t == ir.TypeV128 {
					// MSVC passes __m128 and its siblings by pointer,
					// to a copy the caller makes, and returns them in
					// XMM0 — see classifyMSRet, which does place one.
					// The copy is the frontend's, as it is for every
					// other by-pointer argument, so what arrives here
					// is a ptr and a v128 reaching this point is a
					// frontend that skipped it.
					return nil, fmt.Errorf("v128 is passed by pointer in the Microsoft calling convention; the caller makes the copy")
				}
				// long double is not a Microsoft type — MSVC makes it a
				// double — so refuse rather than invent a placement.
				return nil, fmt.Errorf("%s has no place in the Microsoft calling convention", a.t)
			}
			k := placeInt
			if w.isFloat() {
				k = placeFloat
			}
			p := reg1(k, w, 8)

			// A float in the variadic tail travels in both files. See
			// place.dupInt: which one the callee reads depends on how it
			// was declared, and the caller cannot know.
			p.dupInt = a.vararg && k == placeFloat && p.kind == placeFloat
			out[i] = p
			continue
		}

		size, _, err := sizeAlign(a.byval)
		if err != nil {
			return nil, err
		}
		switch {
		case size == 0:
			// Not an argument: an empty aggregate spends no position.
			out[i] = place{kind: placeInt, byval: a.byval, size: 0}
		case size == 1 || size == 2 || size == 4 || size == 8:
			// Small enough to travel in one integer register whatever
			// its members are: this ABI classifies aggregates by width
			// alone, with no eightbyte classes to compute.
			p := reg1(placeInt, w64, size)
			p.byval, p.size = a.byval, size
			out[i] = p
		default:
			// The address of a copy the caller makes. The place carries
			// the aggregate and its real size so that isel knows to make
			// that copy; what occupies the position is a pointer.
			p := reg1(placeInt, w64, 8)
			p.byval, p.size, p.indirect = a.byval, size, true
			out[i] = p
		}
	}

	// The copies come after every argument slot, so that a memcpy into
	// one cannot land on another. Above the home space too, which the
	// memcpy call itself is entitled to clobber.
	for i, p := range out {
		if !p.indirect {
			continue
		}
		_, align, err := sizeAlign(p.byval)
		if err != nil {
			return nil, err
		}
		if align < 8 {
			align = 8
		}
		stackBytes = alignUp(stackBytes, align)
		out[i].copyOff = int32(stackBytes)
		stackBytes += alignUp(p.size, 8)
	}
	return out, nil
}

// msOutgoingBytes is outgoingBytes plus the room the indirect copies
// take, which is above every stack argument and so is not covered by the
// stack places alone.
func msOutgoingBytes(places []place) uint64 {
	n := outgoingBytes(places)
	if n < msShadow {
		n = msShadow
	}
	for _, p := range places {
		if !p.indirect {
			continue
		}
		if end := uint64(p.copyOff) + alignUp(p.size, 8); end > n {
			n = end
		}
	}
	return alignUp(n, 8)
}

// msSretInRegs is sretInRegs for this ABI: an aggregate comes back in
// RAX when it is one, two, four, or eight bytes wide, and through memory
// otherwise. No eightbyte classes and no XMM0 case — a two-double struct
// is sixteen bytes and goes through memory here, where SysV would return
// it in XMM0 and XMM1.
func msSretInRegs(t ir.FType) (aggregate, bool, error) {
	if t.IsZero() {
		return aggregate{}, false, nil
	}
	size, _, err := sizeAlign(t)
	if err != nil {
		return aggregate{}, false, fmt.Errorf("sret %s: %w", t, err)
	}
	agg := aggregate{size: size}
	switch size {
	case 1, 2, 4, 8:
		return agg, true, nil
	}
	return agg, false, nil
}

// msSretRetSlots is sretRetSlots for this ABI: the one register a small
// aggregate comes back in.
func msSretRetSlots(aggregate) []regSlot {
	return []regSlot{{kind: placeInt, i: 0, w: w64}}
}

// classifyMSRet places a call's results. This ABI returns one value, in
// RAX or XMM0; anything wider comes back through sret, which is a
// parameter and so is classifyMS's business rather than this one's.
func classifyMSRet(types []ir.RegType) ([]place, error) {
	out := make([]place, len(types))
	for i, t := range types {
		if i > 0 {
			return nil, fmt.Errorf("the Microsoft calling convention returns one value; a second comes back in memory, which is sret and is not written yet")
		}
		w, ok := widthOf(t)
		if !ok {
			return nil, fmt.Errorf("%s is not a value this package returns; only i32, i64, ptr, f32, and f64 are placed", t)
		}
		k := placeInt
		if w.isFloat() {
			k = placeFloat
		}
		out[i] = place{kind: k, regs: []regSlot{{kind: k, i: 0, w: w}}, scalarW: w}
	}
	return out, nil
}
