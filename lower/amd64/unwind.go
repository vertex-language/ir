package amd64

// Windows x64 unwind data: the .pdata and .xdata a function needs before
// anything can unwind through it.
//
// On this platform a frame is not walkable from the code. There is no frame
// pointer convention the operating system trusts and no prologue it will
// guess at; instead every function that is not a leaf has a RUNTIME_FUNCTION
// in .pdata naming its bounds and an UNWIND_INFO in .xdata describing, code by
// code, what its prologue did to the stack. RtlLookupFunctionEntry finds the
// first and RtlVirtualUnwind replays the second backwards.
//
// Without it, unwinding does not degrade — it fails. longjmp on x64 is
// implemented as an unwind, so a program that calls it through a frame with no
// entry dies with STATUS_BAD_FUNCTION_TABLE before longjmp's target is
// reached; so does any structured exception, and so does every stack trace a
// debugger or a crash reporter tries to take.
//
// The prologue this describes is emitPrologue's, and it is described from what
// that function reports rather than from a re-derivation of it: an unwind
// record that disagrees with the code by one byte is worse than none, and the
// only way to keep the two in step is for the code to say what it emitted.

import (
	"fmt"

	amd64asm "github.com/vertex-language/amd64"
	"github.com/vertex-language/amd64/reg"
)

// The unwind operations, from winnt.h.
const (
	uwopPushNonVol    = 0 // push of a nonvolatile register
	uwopAllocLarge    = 1 // stack allocation, 136 bytes or more
	uwopAllocSmall    = 2 // stack allocation, 8 to 128 bytes
	uwopSetFPReg      = 3 // the frame register is established here
	uwopSaveNonVol    = 4 // a nonvolatile register stored with a MOV, 16-bit offset
	uwopSaveNonVolFar = 5 // the same, 32-bit offset
)

// unwindVersion is UNWIND_INFO's version field. There is only one.
const unwindVersion = 1

// The two sections the records live in. They are named the Windows way
// because that is what the linker and the loader look for: .pdata is the
// exception directory the image header points at, and .xdata is ordinary
// read-only data that only .pdata references.
const (
	pdataSection = ".pdata"
	xdataSection = ".xdata"
)

// A prologueShape is what emitPrologue did, recorded as it did it.
//
// Byte offsets rather than instruction counts, because an unwind code is
// keyed on the offset of the byte *after* the instruction it describes —
// "the offset in the prologue at which this operation has completed".
type prologueShape struct {
	// present is false for a function with no frame at all. Such a
	// function is a leaf by Windows' definition — it neither calls, nor
	// allocates, nor saves a nonvolatile — and a leaf needs no entry.
	present bool

	// size is the whole prologue's length in bytes, which is what
	// UNWIND_INFO.SizeOfProlog carries.
	size int

	// pushAt is the offset after "push rbp".
	pushAt int

	// allocAt is the offset after "sub rsp, N", and alloc is N. Zero alloc
	// means the instruction was not emitted.
	allocAt int
	alloc   uint64

	// saves are the callee-saved registers stored into frame slots, in the
	// order the prologue stored them.
	saves []prologueSave

	// dynamic marks a function whose RSP moves inside the body — an alloca
	// — so that its frame can only be found through RBP.
	dynamic bool
}

// A prologueSave is one "mov [rbp+disp], reg" in the prologue: the offset
// after it, the register, and where the slot sits relative to the RSP the
// body runs with, which is the frame the unwind codes measure from.
type prologueSave struct {
	at  int
	r   reg.R64
	off uint64
}

// unwindRegNum is the register number the unwind codes use, which is the
// encoding's own numbering: RAX 0 through RDI 7, then R8 through R15.
func unwindRegNum(r reg.R64) uint8 { return uint8(r) }

// emitUnwind writes one function's .pdata and .xdata records.
//
// It is called only for the Microsoft ABI: SysV walks a frame with DWARF CFI,
// which this package does not emit either, and putting a .pdata section in an
// ELF object would be a section nothing reads.
func emitUnwind(am *amd64asm.Module, fnName string, textLen int, p prologueShape) error {
	if !p.present {
		// A leaf. The specification says outright that a leaf function
		// needs no entry, because the unwinder can recover the caller
		// from RSP and the return address alone.
		return nil
	}

	codes, frameReg, frameOff, err := unwindCodes(p)
	if err != nil {
		return fmt.Errorf("unwind %s: %w", fnName, err)
	}
	if codes == nil {
		return nil
	}
	if p.size > 0xff {
		return fmt.Errorf("unwind %s: a %d-byte prologue does not fit SizeOfProlog", fnName, p.size)
	}
	if len(codes)/2 > 0xff {
		return fmt.Errorf("unwind %s: %d unwind codes do not fit CountOfCodes", fnName, len(codes)/2)
	}

	xdata := am.SectionNamed(xdataSection, amd64asm.ROData)
	xdata.Align(4)
	label := "$unwind$" + fnName
	xdata.Label(label, amd64asm.Local)
	xdata.Byte(unwindVersion) // Version 1, no flags: no exception handler
	xdata.Byte(byte(p.size))
	xdata.Byte(byte(len(codes) / 2))
	xdata.Byte(frameReg | frameOff<<4)
	xdata.Data(codes)
	if (len(codes)/2)%2 == 1 {
		// The code array is padded to an even number of entries, so that
		// what follows an UNWIND_INFO is DWORD-aligned.
		xdata.Long(0)
	}

	// The RUNTIME_FUNCTION: three image-relative addresses, which is the
	// only form the exception directory has. The end address is the start
	// plus the function's length, written as an addend rather than a
	// second symbol because there is no symbol at a function's end.
	pdata := am.SectionNamed(pdataSection, amd64asm.ROData)
	pdata.Align(4)
	pdata.Ref(amd64asm.Ref(fnName, amd64asm.RefImageRel32))
	pdata.Ref(amd64asm.Ref(fnName, amd64asm.RefImageRel32).Add(int64(textLen)))
	pdata.Ref(amd64asm.Ref(label, amd64asm.RefImageRel32))
	return nil
}

// unwindCodes builds the code array, and the frame-register nibbles that go
// with it, from a prologue's shape.
//
// The array is ordered by descending prologue offset, which is the order the
// unwinder replays it in: the last thing the prologue did is the first thing
// to undo. A nil array with a nil error means this function cannot be
// described and gets no entry at all — see the frame-register case.
func unwindCodes(p prologueShape) (codes []byte, frameReg, frameOff byte, err error) {
	// The saves first: they are the highest offsets in the prologue, and
	// they are emitted in prologue order, so the array walks them
	// backwards.
	for i := len(p.saves) - 1; i >= 0; i-- {
		s := p.saves[i]
		if s.off%8 != 0 {
			return nil, 0, 0, fmt.Errorf("a callee-saved slot at %d is not eightbyte-aligned", s.off)
		}
		slot := s.off / 8
		switch {
		case slot <= 0xffff:
			codes = append(codes, byte(s.at), uwopSaveNonVol|unwindRegNum(s.r)<<4)
			codes = append(codes, byte(slot), byte(slot>>8))
		default:
			codes = append(codes, byte(s.at), uwopSaveNonVolFar|unwindRegNum(s.r)<<4)
			codes = append(codes, byte(slot*8), byte(slot*8>>8), byte(slot*8>>16), byte(slot*8>>24))
		}
	}

	if p.alloc > 0 {
		n := p.alloc
		switch {
		case n%8 != 0:
			return nil, 0, 0, fmt.Errorf("a frame of %d bytes is not eightbyte-aligned", n)
		case n <= 128:
			codes = append(codes, byte(p.allocAt), uwopAllocSmall|byte(n/8-1)<<4)
		case n <= 512*1024-8:
			codes = append(codes, byte(p.allocAt), uwopAllocLarge|0<<4)
			codes = append(codes, byte(n/8), byte(n/8>>8))
		default:
			codes = append(codes, byte(p.allocAt), uwopAllocLarge|1<<4)
			codes = append(codes, byte(n), byte(n>>8), byte(n>>16), byte(n>>24))
		}
	}

	// The frame register, for a function whose RSP moves inside the body.
	// Only an alloca does that; every other frame is a constant size, and
	// a constant size is exactly what the codes above already say.
	//
	// The encoding is the constraint here and it is a real one. FrameOffset
	// is four bits scaled by sixteen, and it has to satisfy
	// RBP = RSP_body + 16*FrameOffset — this package's prologue takes RBP
	// before it allocates, so 16*FrameOffset is the frame size, and a
	// dynamic frame larger than 240 bytes cannot be expressed. Such a
	// function gets no entry rather than a wrong one: unwinding through it
	// fails, which is where it started, instead of unwinding to a
	// fabricated caller.
	if p.dynamic {
		if p.alloc == 0 || p.alloc%16 != 0 || p.alloc/16 > 15 {
			return nil, 0, 0, nil
		}
		frameReg, frameOff = unwindRegNum(reg.RBP), byte(p.alloc/16)
		// SET_FPREG completes at the end of "mov rbp, rsp", which is the
		// instruction after the push.
		codes = append(codes, byte(p.pushAt+3), uwopSetFPReg)
	}

	codes = append(codes, byte(p.pushAt), uwopPushNonVol|unwindRegNum(reg.RBP)<<4)
	return codes, frameReg, frameOff, nil
}
