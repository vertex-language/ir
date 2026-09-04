# Assembly support

LLVM's arrangement, shaped to this tree. The load-bearing idea is one sentence:
**one encoder, two front doors.** A text assembler and a compiler back end are
not two implementations of an instruction set; they are two ways of naming one,
and they converge before anything is encoded.

## The finding

The second door already exists.

`arm64.Section.Emit(mnem string, ops ...any)` and
`amd64.Section.Emit(mnem string, ops ...operand.Operand)` resolve a form from
the ISA table at run time, from a mnemonic that is data, and hand it to the same
`internal/encode` the typed helpers use. That is exactly LLVM's `MCInst →
MCCodeEmitter` path, and it is already written, already tested, and already
documented as the escape hatch for table-driven emission.

`internal/encode.lower` even says so out loud: an `int` is accepted "because
assembly source writes constants," a bare string is refused "because a bare name
could be a label or a register." The operand universe was designed against a
parser that does not exist yet.

So this is not a request to build an assembler alongside the compiler. It is a
request to build a *front end* onto an encoder that is finished:

```
                                    ┌──────────────────────────┐
    .s text ──→ gas ──→ <arch>/asm ─┤                          │
                                    │  Section.Emit(mnem, ops) │──┐
    ir ──→ isel ──→ mir ──→ regalloc ──→ typed helpers ────────┘  │
                                                                  ▼
                                                        internal/encode
                                                                  │
                                                                  ▼
                                                        bytes + fixups + refs
```

## Three features, not one

Separating these is most of the design.

| | What it needs |
|---|---|
| **Standalone assembler** — `.s` file in, object out | The parser. Nothing else. |
| **Module-level asm** — file-scope `__asm__`, naked functions | The parser, applied to a string. No operands, no regalloc. |
| **Inline asm** — `ir.Asm`, the thing `asm.go` models | The parser, plus operand substitution and constraint-driven allocation. |

The second two bottom out in the first. There is no path to inline asm that does
not pass through a text parser, because after substitution you are holding text
that has to become bytes, and this tree has no `.s` output and no external
assembler to hand it to.

Which means: **build the assembler, and inline asm is plumbing.**

## Module layout

**`github.com/vertex-language/asm` — new module, `gas` its first package.** The
dialect layer. It owns statements and knows nothing about instructions: lines,
labels, comment conventions, `.`-prefixed directives, integer and symbol
expressions, numeric local labels (`1:` / `2f` / `1b`), the section stack. It is
a sibling of `elf` and `macho`, not a subpackage of an architecture, because
none of what it knows is architectural. And it is `asm/gas` rather than a module
named for the dialect, because GNU as is not the only syntax that fits this
shape: Intel syntax and the `.macro` preprocessor are both deferred below, and
both want to be packages beside `gas` rather than more files inside it.

**`arm64/asm`, `amd64/asm`, `i386/asm` — new leaf packages.** Each imports its
own architecture module and `gas`. Each owns everything after the mnemonic:
operand grammar, register names, addressing forms, relocation modifiers. Leaf,
like `obj/elf`, so a build that does not assemble text does not pay for a parser.

**`ir/lower/{arm64,amd64,i386}`** gain an inline-asm path that drives
`<arch>/asm` in template mode.

`ir` itself grows nothing. It already carries the template and constraints
(`asm.go`), and it must not learn what an architecture is.

## Where the split falls

At the statement, not the operand. `gas` reads up to and including the mnemonic;
the arch package reads the rest of the line.

```go
// package gas

// An Arch parses what gas will not.
type Arch interface {
	// Inst parses one instruction statement and emits it. The mnemonic is
	// lexed; ops is the remainder of the line, unconsumed.
	Inst(sec Section, mnem string, ops *Cursor) error

	// Reloc places a symbol reference carrying a modifier — ":lo12:",
	// "@GOTPCREL". Spelling and meaning are both the architecture's.
	Reloc(sec Section, mod, sym string, addend int64, width int) error
}

// A Section is the emission surface gas needs, and it is very nearly the
// surface both architecture modules already expose.
type Section interface {
	Byte(byte)
	Long(uint32)
	Quad(uint64)
	Ascii(string)
	Asciz(string)
	Zero(int)
	Data([]byte)
	Align(int)
	Label(string, ...any)
	EndLabel(string)
	LabelRef(string)
	Offset() int
}
```

`gas` never touches relocations, and that line falls where it does on its own:
`obj.RefKind` is declared once per architecture module, so a shared package
could not name it — and modifier syntax is architectural anyway (`:lo12:` is
not a thing on x86, `@GOTPCREL` is not a thing on AArch64). The constraint and
the correct design agree, which is usually a sign the boundary is real.

## Directives are almost entirely already written

| GAS | Call |
|---|---|
| `.byte` | `Section.Byte` |
| `.word` / `.4byte` / `.long` | `Section.Long` |
| `.quad` / `.8byte` | `Section.Quad` |
| `.ascii` | `Section.Ascii` |
| `.asciz` / `.string` | `Section.Asciz` |
| `.zero` / `.space` / `.skip` | `Section.Zero` |
| `.balign` / `.p2align` / `.align` | `Section.Align` (byte count; text sections already pad with NOPs) |
| label + `.globl` / `.weak` / `.local` | `Section.Label(name, Binding, SymbolType)` |
| `.size sym, .-sym` | `Section.EndLabel` |
| `.hidden` / `.protected` | `Module.SetVisibility` |
| `.extern` | `Module.Extern` |
| `.set` / `.equ` naming a symbol | `Module.Alias` |
| `.section` / `.pushsection` / `.previous` | `Module.SectionNamed`, with the stack in `gas` |
| `.inst` (AArch64) | `Section.Inst` |

Genuinely absent: `.set sym, <arbitrary expression>` has no home — `Alias` binds
a name to a symbol's offset, not to a computed value. Defer it; it is rare
outside hand-written startup code.

Deferred on purpose: `.macro`, `.rept`, `.if`. These are a preprocessor, not an
assembler, and they are the difference between building musl and building the
kernel. Build them when a header actually demands it, not before — and measure
which headers do rather than assuming.

## Dialect

**GAS, and it is not a preference.** `asm.go` already accepts GCC constraint
letters (`CStr` for the target-specific ones), GCC clobber semantics including
`"memory"` and `"cc"`, and GCC's `asm goto`. The corpus that motivates the
feature at all — glibc, musl, kernels, exactly what the doc comment cites — is
GNU as text written against those constraints. Having accepted the constraint
vocabulary, the syntax came with it.

**x86's Intel mode is a flag, not a package.** Two operand parsers over one
mnemonic matcher, selected by an option and by `.intel_syntax` / `.att_syntax`
mid-file. This is what LLVM does with `AsmVariants` and it is the only place in
the whole design where operand grammar genuinely forks.

**AArch64 does not fork.** GAS, LLVM, and armasm all speak UAL; `add x0, x1, #4`
is the same text everywhere. The differences are directives and relocation
modifiers, which are layers 2 and 4, already placed above.

## Inline asm: parse early, encode late

This is the one place to diverge from LLVM, and the reason is that the
constraint that forced LLVM's hand does not apply here.

LLVM re-parses the substituted template at `AsmPrinter` time, feeding it through
`MCAsmParser` to get MCInsts. It does that because Clang hands it an opaque
string and there is no earlier point in the pipeline where a target parser is
reachable. Both ends belong to this tree, so the parse can move to the front:

1. **`lower/<arch>` parses `Template` on entry**, in template mode, where `%0`
   and `%w0` are legal operand tokens. Output is a checked statement list
   with holes. A syntax error surfaces here, at lowering, naming the IR
   instruction — not as an unlocated assembler failure at the end.
2. **Constraints plus the parsed holes give the operand descriptors**: which
   holes are defs, which uses, which class, which fixed register.
3. **isel emits one `mir.Instr`** whose `Op` carries the statement list. `Op` is
   `any` and target-defined, so this costs `mir` nothing.
4. **regalloc colours it like anything else.**
5. **emit substitutes assigned physregs into the holes** and calls
   `Section.Emit(mnem, ops...)` once per statement. No second parse.

The payoff over LLVM's ordering: arity of `%N` references is checked before
allocation, register classes are known rather than inferred from constraint
letters, and the template is parsed exactly once.

## The operand model costs nothing

This was the surprise. Every GCC constraint feature maps onto machinery
`lower/regalloc` already has.

- **Register class** (`r` vs `w`/`x` on AArch64, `x` on x86) → `Pool.Classify`.
  Exists.
- **Fixed register** (`a`, `c`, `S`, `D` on x86) → `Pool.Pin`. Exists, and its
  doc already frames it as "a fact about where a value has to be."
- **Clobber list** → one fresh vreg per clobbered physreg, pinned to it, listed
  among the instruction's `Defs`. `interference` then adds an edge from it to
  everything live after, which is precisely the meaning of a clobber. No new
  concept.
- **Early clobber `&`** → already the default. `interference` adds
  `edge(d, u)` for every def/use pair of an instruction unless `Copy` is set, so
  `"=&r"` is correct today, unmodified.
- **Plain `"=r"`** → correct but pessimistic: denied the register sharing GCC
  permits. That costs a register, never correctness. Ship it.
- **Tied `"+r"` and `"0"`** → **one vreg, appearing in both `Uses` and `Defs`.**
  `edge` drops the `a == b` case, liveness is already computed over both roles,
  and the register is shared by construction rather than by a coalescing pass.
  isel copies the input into a fresh vreg first when the input is live after the
  asm — which is the same thing LLVM does for tied operands, arrived at without
  LLVM's flag words.

So `mir.Instr` does not grow a third time, and `regalloc` does not grow at all.
LLVM needs `InlineAsm::Flag` — a packed word per operand group encoding kind,
count, and register class — because `MachineInstr` operands are untyped and the
allocator has no other channel. This pipeline has typed channels already.

## Two hazards, to check rather than assume

**`"cc"`.** Neither isel models flags as a vreg. If a compare and its consumer
can straddle an instruction today, a `cc` clobber silently breaks the pair.
Verify how NZCV and EFLAGS are kept before claiming `cc` is supported; the fix,
if needed, is a flags vreg in a class of its own, which `Pool.AddClass` already
admits.

**`"memory"`.** A barrier for a scheduler that does not exist. While isel is a
linear walk it is a no-op — but record it rather than dropping it, because it
stops being a no-op the day anything reorders, and a dropped barrier is not a
failure that announces itself.

## Two asymmetries between the arch modules

Both will be felt by whoever writes the `gas` interface, and neither should be
fixed by unifying the modules.

**`Emit` signatures differ.** `arm64` takes `...any` and accepts a bare `int`;
`amd64` takes `...operand.Operand`, sealed, so a bare int cannot leak in. The
sealed version is the better door for a parser. The cost of the difference is
one construction helper per arch, in a package that is arch-specific by
construction — pay it, do not unify.

**`LabelDelta(from, to)` on arm64 is `LabelDiff(to, from)` on amd64.** Different
name, reversed arguments, same operation. This one is a real trap: an adapter
written from one side and applied to the other produces a sign error, not a
compile error. Fix the naming before anything depends on it, or keep it out of
the shared interface entirely.

## Order

1. **`gas` + `arm64/asm`, standalone only.** Most of the work, and none of it
   architectural. Prove it against the difftest harness `arm64` already runs
   against clang, and end-to-end through `Finalize` → `obj/macho` → host link
   and execute, which that module already does natively on this machine.
2. **Module-level asm in `ir`.** No operands, no regalloc — the first real user,
   and it isolates every remaining risk to the parser.
3. **Inline asm on arm64**, via the five-step flow above.
4. **`amd64/asm`**, AT&T first, Intel mode after the pipeline is proven.
5. **`asm goto`** — `%l[name]` substitution against `mir` block labels. The
   terminator shape in `asm.go` already matches LLVM's `callbr`.
6. **`.macro` / `.if` / `.rept`**, when a header forces it and not before.

One decision worth taking deliberately at step 1, because it is a one-way door:
whether `<arch>/asm` calls `Section.Emit` directly, or produces an intermediate
instruction value first. Direct is simpler and matches the tree. An intermediate
is what would make textual `.s` *output* and a disassembler nearly free later —
LLVM chose that and got its `.s` printer for it. Given that "no objectfile
layer" is already a stated position here, direct is the consistent choice; make
it knowingly rather than by default.

---

# Status

Every step of the order below is done except the last, which waits on a header
that forces it. What follows is what was built, what it cost, and what the
building of it found.

## Built

**`github.com/vertex-language/asm/gas`** — the dialect layer, in a new sibling
module. Lexer, GNU as expression grammar, directives, numeric local labels,
section stack, and the `Target`/`Emitter` split. ~2100 lines with tests.

**`arm64/asm`** — the AArch64 front end. `Assemble`, `AssembleInto`, and
`AssembleFragment` for text landing mid-function. **94 lines of ordinary
AArch64 assemble byte-identically to clang** (`asm/difftest_test.go`); twelve
more parse correctly and are refused by the ISA table, listed in `tableGaps`
with what each needs.

**`ir/lower/asmtmpl`** — reads the `%`-references in a GCC template and checks
each one names an operand that exists.

**`ir/lower/arm64` §G4** — `asm` and `asm goto` lower, allocate, assemble, link
and run. Ten native tests cover the operand cases, a tied operand whose input
is still live afterwards, a callee-saved clobber observed from the C caller,
one template expanded twice with the same local label in both, and two
templates that are supposed to be refused.

## The cost estimate held

`mir.Instr` and `regalloc` did not change. Every constraint feature mapped onto
`Pool.Classify`, `Pool.Pin`, and one vreg in two roles, exactly as the section
above predicted. No `InlineAsm::Flag` equivalent was needed.

The parse-early ordering held too, and turned out to be simpler than described:
the template is *checked* at isel and *expanded and parsed once* at emit, rather
than parsed into a hole-carrying form early and re-materialised later. The
holes never needed to exist, because the register classes come from the
operands' VIR types rather than from the template.

## Four bugs, three of them older than this work

Pointing a parser at an encoder that only had one caller found things that
caller could not reach.

**`shiftCount` and `extendCount` were each one short.** Both const blocks state
explicit values rather than counting with `iota`, and in such a block an
omitted expression *repeats the previous one* instead of continuing. So
`shiftCount` was 3 and `extendCount` 7 — making `ROR.Valid()` and
`SXTX.Valid()` return false, so neither could be encoded through any path,
typed or textual. Nothing noticed for the same reason nothing ever notices
this: an operand that reports itself invalid is refused, and a refusal reads
like a row the table does not carry.

**`cset` inverted its condition in the typed helper rather than on the row.**
`Emit` reached the same row and encoded the condition uninverted — two doors,
one mnemonic, different instructions, which is precisely what sharing an
encoder is supposed to make impossible. The inversion is now `AttrInvertCond`
on the form.

**`Resolve` zipped arguments to slots one for one**, but a memory operand is
one argument filling two — a base and the displacement beside it. `encodeForm`
always knew that; `Resolve` did not, so no memory form could resolve through
`Emit` at all. It now walks the two indices the way `encodeForm` does, and
checks the writeback modes in *both* directions, which the typed surface never
needed because its caller cannot get `[sp, #16]` and `[sp, #16]!` crossed.

**A direct reference to a local label in its own section became a relocation**
for a distance that cannot change. Now folded at Finalize. Local is the whole
condition: a branch to a global must stay relocatable so a linker can still
redirect it.

## Two hazards resolved

**`"cc"` is safe on this backend**, and it is worth writing down why rather than
assuming it. `iselCompare` emits a compare and its `cset` adjacently, and
`iselBrIf` re-tests a *value* rather than reading flags a predecessor set. So
no flag is ever live across another instruction for an asm to destroy. A
peephole fusing a compare into a branch would end that, and would have to model
flags to do it.

**`"memory"` is a recorded no-op.** Instructions are emitted in the order isel
produced them and nothing reorders across anything, so the guarantee already
holds. It stops being free the day a scheduler exists.

## Two decisions made along the way

**Every label the assembler emits becomes a symbol**, including plain ones the
parent package would fold and forget. A parser cannot know which labels are
address-taken — `adrp x0, msg` may be hundreds of lines below `msg:` — so the
choice is to promote every label or to walk the token stream twice. Promoting
costs a local symbol a linker may drop; guessing costs a build that fails at
Finalize for a reason nothing in the source suggests.

**A name referred to and never defined becomes an implicit `Extern`.** Every
hand-written `.s` relies on it; nothing writes `.extern puts` before calling
puts. The parent package refuses undefined references on purpose, so the two
are reconciled at the end of the parse, when what was never defined is finally
known.

## §G4 is complete

`asm` and `asm goto` lower on all three backends.

| | Assembler | Agrees with clang | §G4 verified by |
|---|---|---|---|
| **arm64** | `arm64/asm` | 94 lines | linking and running natively |
| **amd64** | `amd64/asm` | 78 lines | disassembly (no user-mode qemu on this host) |
| **i386** | `i386/asm` | 57 lines | booting a kernel under qemu |

Two things came out of the second and third backends.

**`"=a"` and `"a"` are the same register.** A syscall names RAX twice — once
as the result and once as the number — and giving each its own pinned vreg is
two live values in one place, which the allocator refused. They are one value,
read and then written. So an asm now keeps one vreg per physical register it
names, whether the name came from a constraint or from the clobber list, which
is the rule a call site already followed. arm64 needed the same map for the
smaller case of `Clobber("x9", "w9")`.

**A 64-bit operand is refused on i386.** An i64 is a register pair there, so
one `%`-reference would be naming two registers. GCC does not solve this either
— it offers `"A"` for EDX:EAX specifically and otherwise expects the frontend
to have split the value. Substituting the low half and hoping would assemble,
run, and be wrong about the top half.

## The two forms that are not instructions

Step 2 of the order, done after §G4 rather than before it: `m.Asm` is a
module-scope block of assembly, and `fn.AsmBody` is a naked function's body.
Both are §3b and §7 in the grammar, both print, both verify, and all three
backends emit them. Neither goes near `mir`, `regalloc`, or `asmtmpl` — there
is nothing to allocate and nothing to substitute — so the whole of each
backend's share is a call to `AssembleFragment` and, for the body, the symbol
around it.

**Naked implies an asm body, and that is now a rule.** The two are the same
fact: a function with no prologue, no epilogue and no frame has nothing an
instruction could lower against. So `AsmBody` sets `naked`, a function has one
kind of body or the other and never both (`ir.ErrPlacement` at whichever came
second), and a `naked` function with blocks is §19.19, the verifier's
`ErrNakedBody`. Before this, `naked` printed and did nothing else: every
backend emitted a prologue for one anyway, which was silently wrong code rather
than a diagnostic.

**Each module-level block is its own assembly.** The section stack, the numeric
local labels and the absolute symbols of one do not reach the next, so a
`.pushsection` a block leaves open is a fault local to that block. This was
found by writing a test for the opposite — one block opening a section and a
later one closing it, which is what GCC's file-scope concatenation would allow
— and deciding against making it work. Order between two blocks still decides
where the second's bytes fall relative to the first's, which is the part real
corpus depends on.

**A naked function's `.text` is its body exactly.** This is the one place in
the tree where an expected-bytes test is the right test rather than a lazy one:
§G4's bytes depend on what the allocator chose, so an expected-bytes test there
would encode the allocator's answer in order to check the assembler's, but here
there are no operands and the bytes are a function of the text alone. The amd64
test compares against clang's for the same three instructions and checks the
symbol's size, which is what catches a prologue nobody asked for.

Verified the way §G4 was, per target: arm64 links and runs five cases natively
(a naked body, one with a numeric local label, module asm defining a function
between two lowered ones, a block that switches sections and switches back, and
two blocks appending to one section in order); i386 boots the same shapes under
qemu, including a lowered function calling a naked one; amd64 reads the object.

## The gap lists are empty

Each difftest carried a list of lines it parsed correctly and the ISA table
behind it could not encode — twelve on arm64, one on amd64, one on i386. All
three are empty, and how they emptied is worth more than the fact:

**Twelve on arm64, and most of them were width.** `CMP`, `CMN`, `TST`, `NEG`
and the shifts by an immediate were declared at 64 bits and not at 32. `MVN`
was absent at both, as were the four shifts under their register-form names.
`SXTB` through `UXTH` are `SBFM`/`UBFM` with both immediates fixed, so they
carry no immediate operand at all. `UBFX` and `SBFX` are the same two
instructions under the operands a programmer has — where the field starts and
how wide it is — which made one immediate rule read the operand before it, the
only one in that table that does. `ROR` by an immediate is `EXTR` with one
source named twice, which is a fact about the encoding and now lives on the
row beside the condition inversion `CSET` needed, for the same reason: a rule
in a typed helper is a rule the assembler reaches around. `CCMP` and `PRFM`
were simply missing.

**One on amd64, and it was a category.** `hlt`, with a note saying the
privileged instructions were largely absent. They are rows now — `HLT`, `CLI`,
`STI`, `PAUSE`, `RDTSC`, `RDMSR`, `WRMSR`, `WBINVD` — and nothing in this tree
selects any of them. That is why they were missing and why the assembler is
what made them worth having: a kernel's idle loop, critical section and spin
loop arrive as inline assembly or not at all.

**One on i386, and it was a byte.** `movl %gs:0, %eax` assembled correctly
through the r/m row and one byte longer than clang, because the `moffs` form —
opcode, then a bare address, no ModRM — had no row. It is the encoding every
i386 thread-local access uses. The row is the accumulator's alone, which is
the part the difftest now pins: `movl 0x1234, %ebx` still resolves to the r/m
form.

Agreement with clang, per assembler: arm64 136 lines (was 94), amd64 86 (was
78), i386 63 (was 57), and 243 typed-helper cases on arm64 (was 193).

## What a corpus probe found next

With the gap lists empty, the next question is what the gap lists could not
ask: they were built from lines someone thought to write down. So 185 shapes
taken from real inline assembly — a libc's syscall stubs and atomics, a
kernel's barriers, cache maintenance, port I/O and interrupt masking — were
put through all three front ends, and 94 of them were refused.

Twenty-five are refused now, and what closed the other sixty-nine splits into
two kinds.

**Four parser bugs, one of which was silent.** `bts %eax, %ecx` assembled as
`BT` on i386: the suffix rule read the trailing `s` as this architecture's
x87 width letter and peeled it off. Not an error — a different instruction.
The suffix rule now asks the table in both directions, which is what tells BTS
from BT and INT from IN, and what makes MULL work without MULL being written
down anywhere. `movzbl` did not assemble on either x86 target, because the
mnemonic names two widths and the rule assumed one. `movb $0xff, %al` did not
assemble on amd64, because the resolver asked whether the value fit a *signed*
byte while the encoder underneath already accepted a mask — and fixing that
alone would have made `addl $200` silently mean `addl $-56`, so the
sign-extended imm8 became a class of its own. And a memory operand whose width
no suffix can state — `clflush (%rdi)`, `lgdt`, `fnstcw` — is now sized by
asking the table, which replaced the front end's growing list of exceptions
with the rule those exceptions were instances of.

**Four tranches of missing rows, all of them the same kind of missing.**
Nothing in this tree selects any of them: an instruction selector emits
arithmetic, loads, branches and calls, and it does not emit HLT, or BT, or ADC.
So x86 grew the bit-test group, the double-precision shifts, port I/O, the
descriptor tables, the cache and state-management instructions and the
privileged no-operand ones; arm64 grew add-and-subtract-with-carry, the
widening multiply, the conditional and bitfield-insert aliases, the hint
space, ERET, register-offset addressing — `[Xn, Xm, LSL #3]`, the mode a
subscripted array uses — and MSR with an immediate, which is how a kernel
masks interrupts.

Agreement with clang, per assembler: arm64 198 lines and 301 typed-helper
cases, amd64 174, i386 119.

## The constraint list §8b promised

§8b ends by saying what this layer does check: "the constraint list and the
arity of the operand *lists*, not the body of the string." Nothing did. The
CFG half of `asm goto` was solid — a label is a real edge, so an entry-block
label is caught by §19.17's rule and a block reachable only through one is
reachable — but the operand lists were carried to the backends unexamined.

Three faults could reach a backend, and one of them was silent.

**A matching constraint naming an output that does not exist.** `"2"` against
one output, or the `"0"` someone writes before deciding to have an output at
all. Every backend reads the tie, finds the index out of range, and falls back
to treating the input as an ordinary operand in a register of its own — so the
template's `%1` names a register the author did not mean, and the object is
wrong with nothing said. GCC refuses the same constraint. This is the rule that
earned the other two.

**An output constrained `imm`.** An immediate is a literal in the instruction
stream; there is nowhere for a result to be written. It was accepted, printed,
and lowered as though it said `reg`.

**The empty string, as a constraint or as a clobber.** It names no register
class, no tie and no target letter, and there is no reading under which the
operand has a place to live.

`verify/asm.go` reports all three as `ErrAsmConstraint`, positioned at the
instruction — `asm goto` reaches the same rules as its block's terminator. An
output whose own constraint is a matching one is the fourth: the numbering runs
the other way, and an entry in the output list cannot defer to a later one.

The tie is read by **`ir.Constraint.Tied`**, which is new and is the point of
the exercise. All three backends had a verbatim copy of the same digit parse,
each ending in the same silent fallback; they now call the one accessor, and
the verifier calls it too, so what counts as a tie is decided in one place
rather than four. A run of digits too long to be an index saturates instead of
wrapping, which turns the one input that could have wrapped into an index the
verifier rejects.

The tests that were missing came with it. `ir`'s own asm tests covered §3b and
§7 — the two forms that are not instructions — and nothing at all of the
instruction: `Builder.Asm`, `AsmGoto`, and `Constraint` were at zero coverage,
as were the printer's `asm` and `asmArgs`. They are covered now, against the
grammar in §8b and §14 rather than against themselves: the payload a backend
reads, the terminator's targets and labels and the edges they make, the zero
Value and nil label that are refused, and golden `.vir` for both forms.

## §14 gains the outputs it was refusing

`asm goto` had no outputs, on the reasoning that "a call has one edge on which
it did complete, and an `asm goto` has none it can name." The premise is
wrong: the fallthrough edge is exactly the one on which the assembled text ran
to the end without branching, and it is where gcc defines the outputs. A C
frontend meets this on its first kernel-shaped header, since clang and gcc
have both carried the feature for years.

So the rule is now `invoke`'s, applied to the same shape. An output declares a
type and a constraint, names no register, and arrives as a trailing parameter
of the fallthrough target after whatever arguments that edge carries. §19.16's
arity rule covers it, the target may have no second predecessor, and the
labels still take nothing — the branch to one is in the assembled text and
carries no argument list this IR could write.

Nothing new was needed below the builder. The backends already had the two
pieces: block parameters have vregs, so the asm writes straight into the
output's, and the edge copies for the fallthrough now cover the leading
arguments alone. It runs natively on arm64 and under qemu on i386.

## Remaining

- The `rep` prefix and the string instructions, on both x86 targets. It is the
  largest thing the probe still refuses and the only one a libc writes often:
  `rep movsb` is a memcpy.
- The LSE atomics at orderings other than acquire-release, which is the
  limitation `arm64`'s README already states.
- AArch64's `AT`, `DC`, `IC` and `TLBI`, whose operands are named system
  operations rather than registers.
- An extended-register `add` whose second source is a W — the one case where a
  slot's accepted class depends on a later operand.
- An `asm goto` output read on a *label* path rather than on the fallthrough
  path. §14 binds the outputs to the fallthrough target, which is gcc 11's
  rule and the one this IR can state as an edge; gcc 12 and clang extend it to
  every edge, which needs the output copied on each — and a label block is an
  ordinary block with other predecessors, so the copies want split edges
  rather than block parameters.
- A 32-bit base register in a 64-bit memory operand on amd64: `leal 1(%edx),
  %eax` is refused, and GNU as assembles it with the 0x67 address-size
  prefix. A C frontend meets this where a template addresses through an `int`
  operand without asking for its 64-bit view — gcc's own headers write `%q0`
  to avoid it, which is why it took a corpus to find.
- `.macro` / `.if` / `.rept`, when a header forces it.
- Intel syntax as a mode on `amd64/asm`'s operand parser.
- The extension tranches each table names: BMI and MMX on x86, CRC32 and
  pointer authentication beyond the hints on AArch64.

---

# End usage

Everything in this section was built against the current API, compiled, and
printed by `ir/text`. The VIR shown under each is real output, not a sketch.

## What exists today

### A barrier — no operands at all

```go
m := ir.NewModule("barrier", ir.AArch64Linux)
fn := m.Func("barrier").Export().NoUnwind()
e := fn.Entry()

e.Asm("").Volatile().Clobber("memory").Emit()

e.Return()
```

```vertex-ir
export func @barrier() nounwind {
@entry:
  asm volatile "" () clobber "memory"
  return
}
```

### One output — read the thread pointer

`Emit` returns `Results`; read it in the Go type of its reg-type, the same way
a call's results are read.

```go
r := e.Asm("mrs %0, tpidr_el0").
	Volatile().
	Out(ir.TypeI64, ir.CStr("=r")).
	Emit()

e.Return(r.I64(0))
```

```vertex-ir
export func @read_tp() i64 nounwind {
@entry:
  (%0 i64 "=r") = asm volatile "mrs %0, tpidr_el0" ()
  return %0
}
```

### A tied operand — one register in and out

`"0"` on the input ties it to output zero. This is the case that lowers to one
vreg appearing in both `Uses` and `Defs`.

```go
r := e.Asm("add %0, %0, #1").
	Out(ir.TypeI64, ir.CStr("=r")).
	In(x, ir.CStr("0")).
	Emit()
```

```vertex-ir
export func @addone(%x i64) i64 nounwind {
@entry:
  (%0 i64 "=r") = asm "add %0, %0, #1" (%x "0")
  return %0
}
```

### Fixed registers and clobbers — a Linux syscall

The shape that actually appears in a libc. Every constraint here is a pin, and
the whole thing lowers with no new allocator machinery.

```go
r := e.Asm("syscall").
	Volatile().
	Out(ir.TypeI64, ir.CStr("=a")).
	In(nr, ir.CStr("a")).
	In(a1, ir.CStr("D")).
	In(a2, ir.CStr("S")).
	In(a3, ir.CStr("d")).
	Clobber("rcx", "r11", "memory").
	Emit()
```

```vertex-ir
export func @syscall3(%nr i64, %a1 i64, %a2 i64, %a3 i64) i64 nounwind {
@entry:
  (%0 i64 "=a") = asm volatile "syscall" (%nr "a", %a1 "D", %a2 "S", %a3 "d") clobber "rcx", "r11", "memory"
  return %0
}
```

### Several outputs

Outputs are declared in order and indexed in order. An output nothing reads is
still allocated a register — it has to be, since the instruction writes it.

```go
r := e.Asm("cpuid").
	Out(ir.TypeI32, ir.CStr("=a")).
	Out(ir.TypeI32, ir.CStr("=b")).
	Out(ir.TypeI32, ir.CStr("=c")).
	Out(ir.TypeI32, ir.CStr("=d")).
	In(leaf, ir.CStr("a")).
	Emit()

e.Return(r.I32(1))
```

```vertex-ir
export func @cpuid_lo(%leaf i32) i32 nounwind {
@entry:
  (%0 i32 "=a", %1 i32 "=b", %2 i32 "=c", %3 i32 "=d") = asm "cpuid" (%leaf "a")
  return %1
}
```

### A multi-statement template with a local label

The keyword constraints are the target-independent spelling; `CReg` on an
output means `=r`, since direction comes from `Out` versus `In` rather than
from the letter.

```go
r := e.Asm("1: ldxr %0, [%1]\n"+
	"   add  x9, %0, %2\n"+
	"   stxr w10, x9, [%1]\n"+
	"   cbnz w10, 1b").
	Volatile().
	Out(ir.TypeI64, ir.CStr("=&r")).
	In(p, ir.CReg).
	In(v, ir.CReg).
	Clobber("x9", "w10", "memory").
	Emit()
```

```vertex-ir
export func @fetch_add(%p ptr, %v i64) i64 nounwind {
@entry:
  (%0 i64 "=&r") = asm volatile "1: ldxr %0, [%1]\n   add  x9, %0, %2\n   stxr w10, x9, [%1]\n   cbnz w10, 1b" (%p reg, %v reg) clobber "x9", "w10", "memory"
  return %0
}
```

`1:` / `1b` is why the `gas` layer owns numeric local labels: they are scoped to
the template, not to the function, and two expansions of the same template must
not collide.

### `asm goto` — the terminator

`To` takes the fallthrough target and then the label list. The labels take no
arguments, which is the rule `brind` follows too.

```go
e := fn.Entry()
fall := fn.Block("fall")
taken := fn.Block("taken")

e.AsmGoto("test %0, %0\n\tjnz %l[taken]").
	In(x, ir.CReg).
	Clobber("cc").
	To(fall.To(), taken)

fall.Return(fall.I32.Const(0))
taken.Return(taken.I32.Const(1))
```

```vertex-ir
export func @maybe(%x i64) i32 nounwind {
@entry:
  asm goto "test %0, %0\n\tjnz %l[taken]" (%x reg) clobber "cc" to @fall, [@taken]

@fall:
  %0 = i32.const 0
  return %0

@taken:
  %1 = i32.const 1
  return %1
}
```

## Two things writing these examples turned up

**Nothing checks hole numbering.** The `fetch_add` template above was written
first with `%2` and `%3` for its two inputs. With one output and two inputs the
inputs are `%1` and `%2`, so `%3` named nothing — and the module built, verified,
and printed without complaint. Today the template is an opaque string; an
off-by-one in it produces a bad object with no diagnostic anywhere in the
pipeline. That is precisely what step 1 of *parse early, encode late* buys back,
and it is a better argument for the ordering than anything about performance.

**The template's spelling was unspecified; it is now GCC's.** Nothing in
`asm.go` or the spec said whether a template referred to its operands as GCC
does (`%0`, `%l[label]`, `%%`) or as LLVM's internal form does (`$0`, `${0:w}`,
`$$`). §8b of `spec/grammar.md` now states GCC's, and `Asm.Template` documents
it.

The first pass of this document recommended `$N`, on the grounds that the text
format already spells registers `%0` and a GCC template collides with that on
the printed line. That argument is weaker than it looked — the template is
inside a quoted string, so the collision is visual and never grammatical — and
it loses to two costs on the other side. `$` is the immediate sigil in AT&T
syntax, so `$N` forces `movl $$1, %eax` for every x86 immediate, which is what
LLVM has to do. And the corpus this instruction exists for is written in GCC's
spelling, so any other convention makes every frontend implement Clang's
rewriter (`%` → `$`, `$` → `$$`, `%%` → `%`) before it can lower a single
`__asm__`.

Corpus compatibility is the whole reason `asm` is in this IR. It decides this.

## The two forms with no operands

Both exist now. They are `Module.Asm` and `Func.AsmBody`, and the sketches this
section carried are what they became, near enough to leave standing.

**Module-level asm.** Feature (c) — no operands, no allocation, straight into
the section it was declared in:

```go
m.Asm(".pushsection .init_array,\"aw\"\n" +
      ".quad my_ctor\n" +
      ".popsection")
```

```vertex-ir
asm ".pushsection .init_array,\"aw\"\n.quad my_ctor\n.popsection"
```

**A naked function's body.** The module-level form scoped to a function, with
the function's own symbol around it:

```go
fn := m.Func("_start").Export().NoReturn()
fn.AsmBody("mov x0, sp\n" +
           "bl  __libc_start_main")
```

```vertex-ir
export func @_start() naked noreturn {
  asm "mov x0, sp\nbl  __libc_start_main"
}
```

`Naked()` is no longer written at the call site because `AsmBody` implies it —
see the status section above for why that is one fact rather than two.
