package i386

// §H, on a machine whose widest atomic is eight bytes only because CMPXCHG8B
// exists.
//
// The 32-bit forms are nearly free: x86's memory model already orders loads
// and stores the way acquire and release ask for, so an atomic load is a MOV
// and an atomic store is a MOV — except a sequentially consistent store,
// which needs the store and the following load kept apart and gets XCHG,
// whose implicit lock is the barrier. The read-modify-writes are the LOCK
// forms, and where there is no single instruction — and, or, xor — a
// CMPXCHG retry loop.
//
// The 64-bit forms have no instruction at all except CMPXCHG8B, so every one
// of them is a loop around it. Including the load: reading eight bytes
// atomically means comparing them against themselves and taking what comes
// back, which is a write to memory that stores the value already there.
// LLVM does the same thing for the same reason, and it is why a 64-bit
// atomic load here needs writable memory.

import (
	"fmt"

	"github.com/vertex-language/i386/reg"

	"github.com/vertex-language/ir"
	"github.com/vertex-language/ir/lower/mir"
)

// atomicRmwAlus is §H's read-modify-write family.
var atomicRmwAlus = map[ir.Verb]struct {
	alu ir.Verb
	a   access
}{
	ir.VAtomicRmwAdd:  {alu: ir.VAdd},
	ir.VAtomicRmwSub:  {alu: ir.VSub},
	ir.VAtomicRmwAnd:  {alu: ir.VAnd},
	ir.VAtomicRmwOr:   {alu: ir.VOr},
	ir.VAtomicRmwXor:  {alu: ir.VXor},
	ir.VAtomicRmwXchg: {},

	ir.VAtomicRmwAdd8:  {alu: ir.VAdd, a: a8},
	ir.VAtomicRmwSub8:  {alu: ir.VSub, a: a8},
	ir.VAtomicRmwAnd8:  {alu: ir.VAnd, a: a8},
	ir.VAtomicRmwOr8:   {alu: ir.VOr, a: a8},
	ir.VAtomicRmwXor8:  {alu: ir.VXor, a: a8},
	ir.VAtomicRmwXchg8: {a: a8},

	ir.VAtomicRmwAdd16:  {alu: ir.VAdd, a: a16},
	ir.VAtomicRmwSub16:  {alu: ir.VSub, a: a16},
	ir.VAtomicRmwAnd16:  {alu: ir.VAnd, a: a16},
	ir.VAtomicRmwOr16:   {alu: ir.VOr, a: a16},
	ir.VAtomicRmwXor16:  {alu: ir.VXor, a: a16},
	ir.VAtomicRmwXchg16: {a: a16},
}

var (
	atomicLoads = map[ir.Verb]access{
		ir.VAtomicLoad:    0,
		ir.VAtomicULoad8:  a8,
		ir.VAtomicULoad16: a16,
	}
	atomicStores = map[ir.Verb]access{
		ir.VAtomicStore:   0,
		ir.VAtomicStore8:  a8,
		ir.VAtomicStore16: a16,
	}
	atomicCases = map[ir.Verb]access{
		ir.VAtomicCas:   0,
		ir.VAtomicCas8:  a8,
		ir.VAtomicCas16: a16,
	}
)

// atomicWidths is one §H instruction's memory access width and the register
// view its value lives in.
func atomicWidths(op ir.Op, narrow access) (access, width, error) {
	w, ok := widthOf(op.Type)
	if !ok {
		return 0, 0, fmt.Errorf("%s: only i32, i64 and ptr atomics are emitted", op)
	}
	if narrow != 0 {
		return narrow, w, nil
	}
	if w.pairs() {
		return 0, w64, nil // the eight-byte case, which has no access constant
	}
	return a32, w32, nil
}

func seqCst(o ir.Ordering) bool { return o == ir.SeqCst }

// iselAtomicLoad lowers §H's loads.
func iselAtomicLoad(c *cursor, vr *vregs, in *ir.Inst, narrow access) error {
	op := in.Op()
	a, w, err := atomicWidths(op, narrow)
	if err != nil {
		return err
	}
	addr, ok := vr.lookup(in.Arg(0))
	if !ok {
		return fmt.Errorf("%s: address defined outside the function", op)
	}
	dst, err := vr.define(in.Result(0))
	if err != nil {
		return fmt.Errorf("%s: %w", op, err)
	}

	if w.pairs() {
		return cmpxchg8bLoad(c, vr, dst, addr)
	}
	// An ordinary load. x86 orders every load after every earlier load and
	// store, so acquire costs nothing and sequential consistency costs
	// nothing on the loading side either.
	switch a {
	case a8, a16:
		c.Emit(mir.Instr{
			Op:   extLoadOp{from: a, signed: false},
			Defs: []mir.VReg{dst.lo}, Uses: []mir.VReg{addr.lo},
		})
	default:
		c.Emit(mir.Instr{Op: loadOp{}, Defs: []mir.VReg{dst.lo}, Uses: []mir.VReg{addr.lo}})
	}
	return nil
}

// iselAtomicStore lowers §H's stores.
func iselAtomicStore(c *cursor, vr *vregs, in *ir.Inst, narrow access) error {
	op := in.Op()
	a, w, err := atomicWidths(op, narrow)
	if err != nil {
		return err
	}
	ords := in.Orderings()
	if len(ords) == 0 {
		return fmt.Errorf("%s: no ordering", op)
	}
	val, ok := vr.lookup(in.Arg(0))
	if !ok {
		return fmt.Errorf("%s: value defined outside the function", op)
	}
	addr, ok := vr.lookup(in.Arg(1))
	if !ok {
		return fmt.Errorf("%s: address defined outside the function", op)
	}

	if w.pairs() {
		_, err := cmpxchg8bLoop(c, vr, addr, func(old value) (value, error) {
			return val, nil
		})
		return err
	}

	// A release store is an ordinary store: x86 orders it after everything
	// before it. A sequentially consistent one is not, because a later
	// load may be reordered ahead of it — XCHG's implicit lock is the
	// barrier that stops that, and it is cheaper than a store and a fence.
	if seqCst(ords[0]) {
		scratch := vr.reg32()
		emitCopy(c, scratch, val.lo)
		c.Emit(mir.Instr{
			Op:   xchgOp{a: a},
			Defs: []mir.VReg{scratch},
			Uses: []mir.VReg{scratch, addr.lo},
		})
		return nil
	}
	if a == a32 {
		c.Emit(mir.Instr{Op: storeOp{}, Uses: []mir.VReg{val.lo, addr.lo}})
		return nil
	}
	c.Emit(mir.Instr{Op: subStoreOp{to: a}, Uses: []mir.VReg{val.lo, addr.lo}})
	return nil
}

// iselAtomicRmw lowers §H's read-modify-writes.
func iselAtomicRmw(c *cursor, vr *vregs, in *ir.Inst, narrow access, alu ir.Verb) error {
	op := in.Op()
	a, w, err := atomicWidths(op, narrow)
	if err != nil {
		return err
	}
	if o := in.Orderings(); len(o) > 0 && o[0] == ir.Unordered {
		return fmt.Errorf("%s: a read-modify-write is not unordered", op)
	}
	val, ok := vr.lookup(in.Arg(0))
	if !ok {
		return fmt.Errorf("%s: value defined outside the function", op)
	}
	addr, ok := vr.lookup(in.Arg(1))
	if !ok {
		return fmt.Errorf("%s: address defined outside the function", op)
	}
	dst, err := vr.define(in.Result(0))
	if err != nil {
		return fmt.Errorf("%s: %w", op, err)
	}

	if w.pairs() {
		got, err := cmpxchg8bLoop(c, vr, addr, func(old value) (value, error) {
			next := vr.temp(w64)
			if alu == "" {
				return val, nil
			}
			emitCopy(c, next.lo, old.lo)
			c.Emit(mir.Instr{Op: aluOp{verb: alu}, Defs: []mir.VReg{next.lo}, Uses: []mir.VReg{next.lo, val.lo}})
			emitCopy(c, next.hi, old.hi)
			switch alu {
			case ir.VAdd, ir.VSub:
				c.Emit(mir.Instr{
					Op:   carryOp{sub: alu == ir.VSub},
					Defs: []mir.VReg{next.hi}, Uses: []mir.VReg{next.hi, val.hi},
				})
			default:
				c.Emit(mir.Instr{Op: aluOp{verb: alu}, Defs: []mir.VReg{next.hi}, Uses: []mir.VReg{next.hi, val.hi}})
			}
			return next, nil
		})
		if err != nil {
			return err
		}
		emitCopy(c, dst.lo, got.lo)
		emitCopy(c, dst.hi, got.hi)
		return nil
	}

	// XCHG and XADD are single instructions and answer three of the six
	// verbs directly: exchange, add, and subtract as the add of a
	// negation.
	switch alu {
	case "":
		emitCopy(c, dst.lo, val.lo)
		c.Emit(mir.Instr{
			Op:   xchgOp{a: a},
			Defs: []mir.VReg{dst.lo},
			Uses: []mir.VReg{dst.lo, addr.lo},
		})
		return zeroExtend(c, dst.lo, a)
	case ir.VAdd, ir.VSub:
		emitCopy(c, dst.lo, val.lo)
		if alu == ir.VSub {
			c.Emit(mir.Instr{Op: unOp{verb: ir.VNeg}, Defs: []mir.VReg{dst.lo}, Uses: []mir.VReg{dst.lo}})
		}
		c.Emit(mir.Instr{
			Op:   xaddOp{a: a},
			Defs: []mir.VReg{dst.lo},
			Uses: []mir.VReg{dst.lo, addr.lo},
		})
		return zeroExtend(c, dst.lo, a)
	}

	// and, or and xor have LOCK forms that do not return the old value,
	// which §H requires — so they are a compare-and-swap loop like any
	// operation the hardware does not have.
	return cmpxchgLoop(c, vr, a, addr, dst, func(old mir.VReg) mir.VReg {
		next := vr.reg32()
		emitCopy(c, next, old)
		c.Emit(mir.Instr{Op: aluOp{verb: alu}, Defs: []mir.VReg{next}, Uses: []mir.VReg{next, val.lo}})
		return next
	})
}

// iselAtomicCas lowers §H's compare-and-swap, which is the instruction
// CMPXCHG is: EAX holds the expected value going in and the value read
// coming out, either way.
func iselAtomicCas(c *cursor, vr *vregs, in *ir.Inst, narrow access) error {
	op := in.Op()
	a, w, err := atomicWidths(op, narrow)
	if err != nil {
		return err
	}
	if ords := in.Orderings(); len(ords) != 2 {
		return fmt.Errorf("%s: %d orderings, want a success and a failure", op, len(ords))
	}
	expect, ok := vr.lookup(in.Arg(0))
	if !ok {
		return fmt.Errorf("%s: expected value defined outside the function", op)
	}
	newVal, ok := vr.lookup(in.Arg(1))
	if !ok {
		return fmt.Errorf("%s: new value defined outside the function", op)
	}
	addr, ok := vr.lookup(in.Arg(2))
	if !ok {
		return fmt.Errorf("%s: address defined outside the function", op)
	}
	dst, err := vr.define(in.Result(0))
	if err != nil {
		return fmt.Errorf("%s: %w", op, err)
	}

	if w.pairs() {
		eax := vr.physical(reg.EAX)
		edx := vr.physical(reg.EDX)
		ebx := vr.physical(reg.EBX)
		ecx := vr.physical(reg.ECX)
		emitCopy(c, eax, expect.lo)
		emitCopy(c, edx, expect.hi)
		emitCopy(c, ebx, newVal.lo)
		emitCopy(c, ecx, newVal.hi)
		c.Emit(mir.Instr{
			Op:   cmpxchg8bOp{},
			Defs: []mir.VReg{eax, edx},
			Uses: []mir.VReg{eax, edx, ebx, ecx, addr.lo},
		})
		emitCopy(c, dst.lo, eax)
		emitCopy(c, dst.hi, edx)
		return nil
	}

	eax := vr.physical(reg.EAX)
	emitCopy(c, eax, expect.lo)
	// A narrow compare-and-swap compares only the bytes it accesses and
	// writes only those back, so the expected value is cut to the access
	// width first — which is also what leaves the result zero-extended,
	// the way §H says the old value comes back.
	if err := zeroExtend(c, eax, a); err != nil {
		return err
	}
	next := vr.reg32()
	emitCopy(c, next, newVal.lo)
	c.Emit(mir.Instr{
		Op:   cmpxchgOp{a: a},
		Defs: []mir.VReg{eax},
		Uses: []mir.VReg{eax, next, addr.lo},
	})
	emitCopy(c, dst.lo, eax)
	return nil
}

// cmpxchgLoop is the retry a verb without its own instruction becomes.
//
//	     load  old, [addr]
//	retry:
//	     next = f(old)
//	     lock cmpxchg [addr], next    ; EAX is old going in and read coming out
//	     jne  retry
//
// The old value is what §H asks the verb to return, and it is already in EAX
// when the loop exits, because that is where CMPXCHG leaves what it read.
func cmpxchgLoop(c *cursor, vr *vregs, a access, addr, dst value, f func(old mir.VReg) mir.VReg) error {
	eax := vr.physical(reg.EAX)
	if a == a32 {
		c.Emit(mir.Instr{Op: loadOp{}, Defs: []mir.VReg{eax}, Uses: []mir.VReg{addr.lo}})
	} else {
		c.Emit(mir.Instr{
			Op:   extLoadOp{from: a, signed: false},
			Defs: []mir.VReg{eax}, Uses: []mir.VReg{addr.lo},
		})
	}

	retry, done := c.open("retry"), c.open("done")
	c.to(retry)

	next := f(eax)
	c.Emit(mir.Instr{
		Op:   cmpxchgOp{a: a},
		Defs: []mir.VReg{eax},
		Uses: []mir.VReg{eax, next, addr.lo},
	})
	c.branch(condNE, retry, done)

	emitCopy(c, dst.lo, eax)
	return nil
}

// cmpxchg8bLoad reads eight bytes atomically by comparing them against
// themselves: whatever the comparison decides, EDX:EAX holds what was there.
//
// It stores. Comparing zero against memory and swapping in zero writes back
// the value it found when they matched, which is why an eight-byte atomic
// load on this target needs memory it is allowed to write.
func cmpxchg8bLoad(c *cursor, vr *vregs, dst, addr value) error {
	eax := vr.physical(reg.EAX)
	edx := vr.physical(reg.EDX)
	ebx := vr.physical(reg.EBX)
	ecx := vr.physical(reg.ECX)
	for _, r := range []mir.VReg{eax, edx, ebx, ecx} {
		c.Emit(mir.Instr{Op: constOp{imm: 0}, Defs: []mir.VReg{r}})
	}
	c.Emit(mir.Instr{
		Op:   cmpxchg8bOp{},
		Defs: []mir.VReg{eax, edx},
		Uses: []mir.VReg{eax, edx, ebx, ecx, addr.lo},
	})
	emitCopy(c, dst.lo, eax)
	emitCopy(c, dst.hi, edx)
	return nil
}

// cmpxchg8bLoop is the retry every 64-bit read-modify-write becomes.
func cmpxchg8bLoop(c *cursor, vr *vregs, addr value, f func(old value) (value, error)) (value, error) {
	eax := vr.physical(reg.EAX)
	edx := vr.physical(reg.EDX)
	c.Emit(mir.Instr{Op: loadOp{}, Defs: []mir.VReg{eax}, Uses: []mir.VReg{addr.lo}})
	c.Emit(mir.Instr{Op: loadOp{off: 4}, Defs: []mir.VReg{edx}, Uses: []mir.VReg{addr.lo}})

	retry, done := c.open("retry"), c.open("done")
	c.to(retry)

	old := value{lo: eax, hi: edx, w: w64}
	next, err := f(old)
	if err != nil {
		return value{}, err
	}
	ebx := vr.physical(reg.EBX)
	ecx := vr.physical(reg.ECX)
	emitCopy(c, ebx, next.lo)
	emitCopy(c, ecx, next.hi)
	c.Emit(mir.Instr{
		Op:   cmpxchg8bOp{},
		Defs: []mir.VReg{eax, edx},
		Uses: []mir.VReg{eax, edx, ebx, ecx, addr.lo},
	})
	c.branch(condNE, retry, done)

	out := vr.temp(w64)
	emitCopy(c, out.lo, eax)
	emitCopy(c, out.hi, edx)
	return out, nil
}

// iselFence lowers §H's fence.
//
// A single-thread fence emits nothing. Everything weaker than sequential
// consistency emits nothing either: x86 already orders load-load, load-store
// and store-store, so acquire and release are free and only store-load needs
// a barrier. That barrier is a locked read-modify-write on the stack rather
// than MFENCE, which is an SSE2 instruction the 386 does not have.
func iselFence(c *cursor, in *ir.Inst) error {
	if in.SingleThread() {
		return nil
	}
	ords := in.Orderings()
	if len(ords) == 0 {
		return fmt.Errorf("%s: no ordering", in.Op())
	}
	if !seqCst(ords[0]) {
		return nil
	}
	c.Emit(mir.Instr{Op: fenceOp{}})
	return nil
}

// zeroExtend cuts a register to an access width, where that width is narrower
// than the register.
func zeroExtend(c *cursor, r mir.VReg, a access) error {
	if a != a8 && a != a16 {
		return nil
	}
	c.Emit(mir.Instr{Op: zextOp{from: a}, Defs: []mir.VReg{r}, Uses: []mir.VReg{r}})
	return nil
}
