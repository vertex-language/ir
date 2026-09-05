// Package amd64 lowers VIR to a finished, immutable *amd64/obj.Object.
//
// # What it lowers, and how it got here
//
// Thirty-seven milestones, each of which either made something new lowerable
// or changed what a milestone before it had assumed. In order:
//
//   - 1-5, the shape of a function: a straight line, a two-way branch, a
//     block parameter, a loop, and block arguments on every edge that can
//     carry them. Milestone 5 is where a back edge that permutes the
//     parameters it assigns to forced emitParallelCopy, since that is the
//     first shape whose moves are parallel rather than merely sequential.
//   - 6, 64-bit values. i64 and ptr join i32, and a physical register
//     stops being one width — see width.
//   - 7 and 8, memory: §D's full-width load and store and §D2's
//     sub-width family, which together are every access this IR has that
//     is not atomic. A pointer becomes something you can follow.
//   - 9, the frame: ptr.alloc, a SysV prologue, and leave before every
//     ret — emitted only by a function that has an allocation.
//   - 10, everything outside a function body: §5's globals as bytes in
//     the section their domain names, and ptr.getaddr as this repo's
//     first relocation. See global.go.
//   - 11, liveness. The first milestone that made already-working code
//     shorter rather than making new code possible: two vregs never live
//     at once share a register, so the pool bounds how many values are
//     live at a time rather than how many a function names.
//   - 12, the arithmetic: every §A verb that is a single two-address
//     instruction here, plus the trap terminator. See binOps and unOps.
//   - 13, instructions that name a register. §A's divisions read and
//     write RAX and RDX and its shifts count in CL, none of which a MIR
//     whose operands are vregs could say. vregs.physical is how it says
//     it now, and mir.Instr.Copy is what keeps it from costing a move it
//     does not need.
//   - 14, calls, which is where all of the above has to hold at once. A
//     parameter stopped being pinned to the register it arrived in and
//     became an ordinary value copied out of it; the pool gained the
//     callee-saved registers, and the frame a place to save them; and a
//     call names every caller-saved register as a destination, which is
//     what forces a value live across it somewhere safe without a
//     spiller having to exist.
//   - 15, §F's select, as a compare and a conditional move. It fuses its
//     condition the way a brif does and more strictly, since a select is
//     not a terminator and anything between the compare and the move
//     could write flags.
//   - 16, spilling. More live values than registers is a rewrite now
//     rather than a refusal, which also finished the frame: a function's
//     stack size is not known until allocation has run, so the prologue
//     became something emit writes rather than an instruction isel
//     builds. The callee-saved save area is sized the same way, to the
//     registers a function actually used.
//   - 17, the ABI layout table: what §5's types occupy and align to.
//     Two refusals were waiting on it and are now answers — ptr.alloc of
//     a named type, which states a type where the other form states a
//     size, and an aggregate global initializer, whose padding is the
//     difference between where the next field starts and where the last
//     one ended. A global also aligns to its type now rather than to
//     wherever the previous one happened to end. See layout.go, which is
//     also the table §19.18's last clause defers to a target for.
//   - 18, the seventh argument. A value in memory at a constant offset
//     from a frame register is the operand this MIR could not express —
//     spilling aside, which regalloc wrote for itself — and a stack
//     argument is both ends of it: an outgoing one stored into an area
//     at the bottom of the frame, an incoming one loaded from above RBP
//     where the caller left it. See argStoreOp and argLoadOp.
//   - 19, §C's integer conversions, and the i1 that stops being only a
//     flag. A compare read by anything other than a branch or a select
//     is materialized now — a setcc and the zero-extension that makes
//     its byte a whole value — rather than refused, which is what
//     zext_i1 needed to have something to widen. See fusesEveryUse,
//     which is what decides whether a compare exists as an instruction
//     at all.
//   - 20, a branch that tests a value. A brif and a select fuse a
//     compare where there is one to fuse and test the condition in a
//     register where there is not, which is what makes an i1 parameter,
//     an i1 a call returned, and an i1 carried along an edge into a
//     block parameter into conditions this package can branch on. The
//     test is a byte wide; see iselBrIfValue.
//   - 21, what an arithmetic instruction produces besides its result.
//     §A's smul_hi and umul_hi want the half of a product a two-address
//     multiply throws away, and §A2's predicates want a flag no
//     comparison can reconstruct — signed overflow is a fact about an
//     addition and not a relation between its operands. The one-operand
//     multiply answers the first and the setcc of milestone 19 answers
//     the second. See mulOp and ovfOps.
//   - 22, the second register file. Everything before this held every
//     value in a general-purpose register; a float lives in an XMM
//     register, which is not a wider register or a different width of
//     the same one but a disjoint set no integer instruction can name.
//     regalloc gained classes for it, and the fact that makes them
//     necessary is that RAX and XMM0 are both register number zero. §A3
//     arithmetic, float constants as their own bit patterns, §D at float
//     widths, and SysV's two-file classification. See classifySysV,
//     width.class, and fAluOp.
//   - 23, §B's float rows. UCOMIS sets the flags an unsigned comparison
//     would set and sets PF when either operand is NaN, so lt and le are
//     the operands compared the other way round — above and not below,
//     which is what makes a NaN answer false. eq and ne are the two rows
//     no single condition answers and are built as two setccs and a
//     combine; being unfusable costs them nothing, because milestone 20
//     made an i1 in a register a condition. See floatConds and
//     floatPairConds.
//   - 24, the conversions and the sign bit. §C's crossings between the
//     two register files, and §A3's neg, abs, sqrt and copysign — which
//     are a mask and one logical instruction, because a float's sign is
//     a bit and not an operation. It also fixed the one thing milestone
//     22 got silently wrong: a float neg reached the integer NEG, which
//     is a subtraction from zero and is not what §A3 means on a zero or
//     a NaN. See iselFloatConvert and emitSignMask.
//   - 25, §H. Most of the section costs nothing here: this
//     architecture's memory model reorders a store followed by a load
//     and nothing else, so an atomic load is a load and an atomic store
//     below sequential consistency is a store. What costs an
//     instruction is a sequentially consistent store, which is an
//     exchange, and the read-modify-writes, which are LOCK XADD and
//     XCHG. See atomicRmws.
//   - 26, an isel that can branch. Everything before it emitted into the
//     block it was handed and could take for granted that the same block
//     was still the one at the end of its instruction. §H's atomic and,
//     or and xor cannot: each is a compare-and-swap that retries, and a
//     retry is a loop. isel names the block it is filling now — see
//     cursor — and an instruction that needs more than one says so.
//   - 27, §C2's trapping conversions, which is what the blocks were for.
//     CVTTSD2SI does not trap: given a NaN or a value out of range it
//     writes the destination's most negative value and carries on, which
//     is the silent wrong answer §C exists to refuse. So the conversion
//     is a range check on the source and then the instruction. See
//     f2iRange, where the bounds are and why three of the four rows
//     close the interval the fourth leaves open.
//   - 28, §A6, and the Options field three quarters of it needed.
//     popcnt, clz and ctz are ordinary integer verbs whose instructions
//     carry CPUID bits, so lowering one is a question about the
//     processor and not only about the module — and Options.Features is
//     where the caller answers it. Refused rather than expanded where
//     the answer is no; see checkFeatures.
//   - 29, the float verbs that were left. §C2's unsigned and saturating
//     conversions, which are sequences and not instructions — there is
//     no unsigned row in the silicon before AVX-512, and a saturating
//     conversion is the trapping one with a clamp where the trap is.
//     The min and max families, whose IEEE definitions MINSD does not
//     have, so each is a compare and a fixup. The rounding verbs, which
//     are SSE4.1's ROUNDSD, and FMA, which is VEX — both gated in
//     gatedVerbs the way milestone 28 gated the bit verbs.
//   - Then, unnumbered, the indirect control flow §G2 and §G name:
//     callind, brind, and br_table. None was blocked on anything here;
//     each is its own shape, and br_table is the only one that needed a
//     section to put a jump table in.
//   - 30, the second return register, and the variadic count. §3.2.3
//     gives two registers of each file, so a call stopped having at
//     most one result. The move that puts a value in its return
//     register left emit for isel, which is what makes two possible at
//     all: returnOp could carry one width and one register file, and a
//     pinned copy carries its own. A variadic call sets AL to the
//     vector-register count, which is §I's caller half. callSite is
//     what keeps a register named twice to one vreg.
//   - 31, the jump table, made honest. The unnumbered work above took
//     two hard-coded scratch registers and did not declare the
//     instruction wrote them, so the allocator was free to leave a
//     value live across the terminator in one — a wrong answer, not a
//     slow one. Naming the scratch as a Def is the whole fix: a
//     destination interferes with everything live after it, which at a
//     terminator is the successors' live-in. The table moved to .rodata
//     as absolute addresses, a label difference not crossing sections
//     where a relocation does.
//   - 32, §D3's addresses, and the ninth float. ptr.blockaddr is the
//     one worth naming: brind could jump to an address and nothing
//     here could make one. It is a lea of a block label, promoted to a
//     symbol the way milestone 31 promoted a table's targets.
//     frameaddr, returnaddr and ptr.diff are one instruction each. And
//     a float past the eighth takes a stack slot at last, out of the
//     same queue an integer past the sixth has used since milestone 18.
//   - 33, §E, as the calls it is. The four bulk-memory verbs are calls
//     to the C library function of the same name; REP MOVSB is the
//     wrong answer for every size not known to be large. Their IR
//     signatures match the C ones, so a bulk operation on parameters
//     lowers to the call alone. That settled planFrame's refusal of a
//     zeroed ptr.alloc, carried since milestone 9. See libcall.go.
//   - 34, byval and sret. ir.RegType has no aggregate in it, so one
//     reaches a signature as a pointer carrying byval — and that
//     attribute had been ignored, which is a wrong answer rather than a
//     refusal. Honouring it is §3.2.3's classification: eightbytes,
//     each field classified into the one it lands in, and the classes
//     that share one merged, INTEGER over SSE. classifySysV became a
//     fold for it, the registers and the stack being one running state.
//     A result is classified the same way, since §3.2.3 draws no
//     distinction: what is not MEMORY comes back in RAX and RDX or in
//     XMM0 and XMM1, and the sret parameter the front end wrote names a
//     slot of the callee's own rather than storage the caller supplied.
//     See abi.go.
//   - 35, §D3's dynamic frame, and §C4. ptr.alloca's size arrives as a
//     value, so its only home is below RSP — and what has to survive
//     RSP moving is the outgoing argument area, which is addressed from
//     it. So the allocation is handed the space above that area rather
//     than the new RSP, and every offset a call was planned with still
//     names the same place. §C4 is the identity, ptrbits being 64.
//   - 36, §I's callee half. §3.5.7's va_list is four fields rather than
//     a pointer because the arguments are in two places: a register
//     save area the prologue writes, and the caller's outgoing area.
//     va_arg is the branch between them, which milestone 26's
//     block-splitting isel is what makes expressible. The vector half
//     of the save area is written only when AL says the caller used
//     one. See variadic.go.
//   - 37, f128, and the soft-float table. §0 is explicit that this is
//     not optional: lowering supplies a runtime call where the target
//     has no instruction. The value needs none — §3.2.3 classifies
//     __float128 SSE and SSEUP, so it is an f64's work done sixteen
//     bytes wide — and neither do the sign verbs, ANDPD and XORPD
//     having always operated on all sixteen. The rest is compiler-rt,
//     and that table is what libcall.go declined to be at 33: those
//     names were their own verb's, these are chosen by verb and width.
//     See softfloat.go.
//
// # Where the register width lives
//
// regalloc hands out a register, not a name for one — it says RAX, and
// whether the instruction reading it spells that operand rax or eax is
// the instruction's own business. So the pools below are R64, the whole
// register, which is what is actually being handed out; every MIR op
// this package builds carries the width it runs at; and emit narrows to
// the 32-bit view at the point of encoding.
//
// ptr rides along at that width with no work of its own. §3's layout
// block makes a pointer 64 bits here — checkLayout is where that stops
// being an assumption — ptr.const is null, ptr.add and ptr.sub take an
// i64 byte count, and §B's four pointer comparisons are unsigned. None
// of that asks this package to know anything about pointers: a ptr is a
// 64-bit value and lowering treats it as one.
//
// # What is still missing
//
// f80 is not a value here at all. It is the x87 stack, which is a third
// register file with a different shape from either of the two this
// package allocates — and the assembler it would be written against
// declares no x87 instruction at all, so it is that repository's work
// before it is this one's.
//
// §A3's remaining f128 rows: sqrt, fma, the min and max families and
// the rounding verbs. Each is a libm name rather than a compiler-rt one
// — sqrtl, fmal, fminl — which is a different table and a dependency on
// a different library, and worth being a decision rather than an
// afterthought.
//
// A third result of either class, which comes back through memory. The
// mechanism is sret and milestone 34 has it, but as something a
// signature states rather than something a lowering may introduce: a
// caller that needed one would have to allocate the storage and rewrite
// the call, and rewriting a call is a legalization pass and not a
// selection rule.
//
// ptr.tlsaddr, which is missing for the reason §5's tls domain is: a
// TLS model and its relocations, which is a question about the target
// rather than about the verb.
//
// §G3's unwinding: invoke, invokeind and resume. ir/verify already
// checks the pads and the arity, so the IR is ahead of the lowering
// here.
//
// One corner of §I: ptr.va_arg_ref of an aggregate small enough to have
// been passed in registers. Its eightbytes are scattered through the
// save area, so producing one address for it means gathering them into
// a slot — a different shape of work from advancing a pointer, and the
// only §I row milestone 36 refuses.
//
// Options and Target are still each backend's own. What three
// architectures turned out to share is the machine-independent middle —
// mir, regalloc, and globals, each of which is now its own package — and
// not the entry points, which differ in what they take: this one names a
// feature set, arm64 names a variadic convention, and i386 names neither.
// Hoisting those into one root package would be inventing a union rather
// than finding one.
//
// # Where to find it
//
// The pipeline is: checkLayout and checkFeatures validate the module
// against the target, lowerGlobals (global.go) emits the data sections,
// and lowerFunc runs isel, register allocation, and emission for each
// function.
//
// See types.go for the MIR opcode types, vregs.go for virtual register
// management, frame.go for stack frame planning and SysV
// classification, control.go for block/branch lowering, isel.go for
// instruction selection, isel_float_ext.go for milestone 29's float
// sequences, atomic.go for §H, and emit.go for machine code emission.
package amd64

import (
	"fmt"

	amd64asm "github.com/vertex-language/amd64"
	"github.com/vertex-language/amd64/feature"
	amd64obj "github.com/vertex-language/amd64/obj"

	"github.com/vertex-language/ir"
	"github.com/vertex-language/ir/lower/mir"
	"github.com/vertex-language/ir/lower/regalloc"
)

// OptLevel is how hard Lower tries. Only O0 exists today.
type OptLevel uint8

const O0 OptLevel = iota

// Options are the things Lower has no opinion about.
type Options struct {
	OptLevel OptLevel

	// Features is the processor being compiled for. The zero value is
	// the baseline, which every AMD64 processor implements.
	Features feature.Set

	// LibcallPrefix is prepended to the name of every function this
	// package calls that the module did not declare — §E's memcpy, and
	// the soft-float routines §C's f128 cases become.
	//
	// Stated rather than derived. The names a module writes are the names
	// it gets; a name this package invents has no such author, and the
	// two platforms disagree — Mach-O prefixes a C symbol with an
	// underscore and ELF does not.
	LibcallPrefix string
}

// features is the processor Options names, or the baseline.
func (o Options) features() feature.Set {
	var unset feature.Set
	if o.Features == unset {
		return feature.Default()
	}
	return o.Features
}

// gatedVerbs is the §A6 verbs whose instruction carries a CPUID bit.
var gatedVerbs = map[ir.Verb]feature.Feature{
	ir.VPopcnt:  feature.POPCNT,
	ir.VClz:     feature.LZCNT,
	ir.VCtz:     feature.BMI1,
	ir.VFMA:     feature.FMA,
	ir.VCeil:    feature.SSE41,
	ir.VFloor:   feature.SSE41,
	ir.VTrunc:   feature.SSE41,
	ir.VNearest: feature.SSE41,
}

// Lower builds an AMD64 object from m.
func Lower(m *ir.Module, opts Options) (*amd64obj.Object, error) {
	if err := checkLayout(m); err != nil {
		return nil, err
	}
	if err := checkFeatures(m, opts.features()); err != nil {
		return nil, err
	}

	am := amd64asm.NewModule(amd64asm.WithFeatures(opts.features()))

	for _, f := range m.FuncImports() {
		am.Extern(f.Name())
	}
	for _, g := range m.GlobalImports() {
		am.Extern(g.Name())
	}
	for _, sym := range libcallSyms(m, opts.LibcallPrefix) {
		am.Extern(sym)
	}
	for _, sym := range softFloatSyms(m, opts.LibcallPrefix) {
		am.Extern(sym)
	}
	if needsTLSIndex(m) {
		// The CRT's, and a reference rather than a definition: the loader
		// fills it with the slot this image was given. It is declared
		// here rather than where it is used because a section's
		// references resolve against the module, and the module is what
		// knows whether anything is thread-local at all.
		am.Extern(tlsIndexSymbol)
	}

	if err := lowerGlobals(am, m); err != nil {
		return nil, err
	}

	// Items rather than Funcs, because a module-level asm block has a
	// position among them: it is emitted where it was declared, and that
	// is what decides where its bytes fall.
	text := am.Section(amd64asm.Text)
	nasm := 0
	for _, it := range m.Items() {
		switch x := it.(type) {
		case *ir.Func:
			if err := lowerFunc(am, text, x, opts); err != nil {
				return nil, err
			}
		case *ir.ModuleAsm:
			if err := emitModuleAsm(text, x, nasm); err != nil {
				return nil, err
			}
			nasm++
		}
	}

	return am.Finalize()
}

// checkLayout refuses a module whose layout block is not one this
// package can lower for (64-bit, little-endian, SysV ABI).
func checkLayout(m *ir.Module) error {
	l := m.Layout()
	switch {
	case l.PtrBits != 64:
		return fmt.Errorf("lower: module %q declares ptrbits %d; amd64 pointers are 64 bits", m.Name(), l.PtrBits)
	case l.Endian != ir.LittleEndian:
		return fmt.Errorf("lower: module %q declares %s-endian; amd64 is little-endian", m.Name(), l.Endian)
	case l.ABI != "sysv" && l.ABI != "ms":
		return fmt.Errorf("lower: module %q declares abi %q; only sysv and ms are supported", m.Name(), l.ABI)
	}
	return nil
}

// checkFeatures refuses a module that uses an instruction the target
// processor does not have.
func checkFeatures(m *ir.Module, set feature.Set) error {
	var bad error
	for _, fn := range m.Funcs() {
		fn.WalkInsts(func(in *ir.Inst) bool {
			g, gated := gatedVerbs[in.Op().Verb]
			if !gated || set.Has(g) {
				return true
			}
			bad = fmt.Errorf("lower: %s: %s needs %v, which Options.Features does not have (the set is %s)",
				fn.Name(), in.Op(), g, set)
			return false
		})
		if bad != nil {
			return bad
		}
	}
	return nil
}

// lowerFunc runs the full pipeline for one function: frame planning,
// isel, register allocation, and emission.
func lowerFunc(am *amd64asm.Module, text *amd64asm.Section, fn *ir.Func, opts Options) error {
	if body, ok := fn.AsmBodyText(); ok {
		return emitAsmBody(text, fn, body)
	}

	fr, err := planFrame(fn)
	if err != nil {
		return err
	}

	abi := fn.Module().Layout().ABI
	mf := mir.NewFunc()
	pool := regalloc.NewPool(allocatable(abi))
	pool.AddClass(xmmClass, allocatableXmm(abi))
	vr := newVRegs(mf, pool, len(fn.Params()))

	blocks := fn.Blocks()
	mbs := make([]*mir.Block, len(blocks))
	var entry *mir.Block
	for i, blk := range blocks {
		mbs[i] = mf.NewBlock(blockLabel(fn, blk))
		if blk.IsEntry() {
			entry = mbs[i]
		}
	}
	if err := classifyParams(fn, entry, vr, fr); err != nil {
		return err
	}
	if err := classifyBlockParams(fn, vr); err != nil {
		return err
	}

	// Selection order is not emission order. An instruction's result
	// gets its vreg where the instruction is selected, so a block
	// using a value has to come after the block defining it — which
	// is reverse postorder, and not necessarily the order the blocks
	// were built in. See the same comment in lower/arm64. The mir
	// blocks stay indexed by the function's own order, so what is
	// emitted is unchanged.
	uses := indexUses(fn)
	at := make(map[*ir.Block]int, len(blocks))
	for i, blk := range blocks {
		at[blk] = i
	}
	done := make([]bool, len(blocks))
	sel := func(blk *ir.Block) error {
		i, ok := at[blk]
		if !ok || done[i] {
			return nil
		}
		done[i] = true
		if err := iselBlock(fn, mf, newCursor(fn, mf, mbs[i], opts.LibcallPrefix), vr, fr, blk, uses); err != nil {
			return fmt.Errorf("lower: %s: %w", fn.Name(), err)
		}
		return nil
	}
	for _, blk := range fn.RPO() {
		if err := sel(blk); err != nil {
			return err
		}
	}
	for _, blk := range blocks {
		if err := sel(blk); err != nil {
			return err
		}
	}

	assigned, err := regalloc.Spilling(mf, pool, &spiller{fr: fr})
	if err != nil {
		return fmt.Errorf("lower: %s: %w", fn.Name(), err)
	}

	saved := usedCalleeSaved(abi, pool, assigned)
	fr.reserveSaves(saved)

	start := text.Offset()
	shape, err := emit(am, text, fn, mf, assigned, fr, saved)
	if err != nil {
		return err
	}
	if abi == abiMS {
		// Windows walks a frame through .pdata and .xdata rather than
		// through the code, and a frame with no record is one nothing can
		// unwind through. See unwind.go.
		return emitUnwind(am, fn.Name(), text.Offset()-start, shape)
	}
	return nil
}
