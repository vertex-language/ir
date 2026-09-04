# ir

Vertex IR (VIR) in Go: a typed SSA intermediate representation with a
builder API, a verifier, a text printer, and a lowering path that ends in
a linkable object file.

```
go get github.com/vertex-language/ir
```

Four things live here and nothing else:

- **`ir`** — the IR itself. Build modules by calling methods; walk and
  rewrite them. Zero dependencies.
- **`ir/verify`** — §19, whole-module. A module that passes is one the
  builder could have produced and lowering may assume.
- **`ir/text`** — the `.vir` syntax, print-only. Print a module. The
  only package in the repo that knows what VIR *looks like*.
- **`ir/lower`** — VIR to a finished, immutable `obj.Object`, per
  architecture. Its own `go.mod`, because it depends on one of `i386`,
  `amd64`, or `arm64` and the other three must not.

The normative specification is [`spec/`](spec/): the
[overview](spec/overview.md), the [instruction
index](spec/instruct_index.md), and the [grammar](spec/grammar.md).
This README is the Go surface.

---

## Quick start

Build a function, check it, print it.

```go
package main

import (
	"log"
	"os"

	"github.com/vertex-language/ir"
	"github.com/vertex-language/ir/text"
	"github.com/vertex-language/ir/verify"
)

func main() {
	m := ir.NewModule("demo", ir.X86_64Linux)

	fn := m.Func("sum").Export().NoUnwind()
	p := fn.ParamPtr("p", ir.NoAlias)
	n := fn.ParamI64("n")
	fn.ReturnsI32()

	entry := fn.Entry()
	loop := fn.Block("loop")
	i := loop.ParamI64("i")
	acc := loop.ParamI32("acc")
	body := fn.Block("body")
	done := fn.Block("done")
	out := done.ParamI32("r")

	entry.Br(loop.To(entry.I64.Const(0), entry.I32.Const(0)))
	loop.BrIf(loop.I64.SLt(i, n), body.To(), done.To(acc))

	addr := body.Ptr.Add(p, body.I64.Shl(i, body.I64.Const(2))).Named("addr")
	body.Br(loop.To(
		body.I64.Add(i, body.I64.Const(1)),
		body.I32.Add(acc, body.I32.Load(addr)),
	))

	done.Return(out)

	if err := verify.Module(m); err != nil {
		log.Fatal(err)
	}
	text.Print(os.Stdout, m)
}
```

```vertex-ir
module demo

use "x86_64/linux"

layout {
  abi        sysv,
  endian     little,
  ptrbits    64,
  stackalign 16,
  extfloat   f80, f128,
}

export func @sum(%p ptr noalias, %n i64) i32 nounwind {
@entry:
  %0 = i64.const 0
  %1 = i32.const 0
  br @loop(%0, %1)

@loop(%i i64, %acc i32):
  %2 = i64.slt %i, %n
  brif %2, @body, @done(%acc)

@body:
  %3 = i64.const 2
  %4 = i64.shl %i, %3
  %addr = ptr.add %p, %4
  %5 = i64.const 1
  %6 = i64.add %i, %5
  %7 = i32.load %addr
  %8 = i32.add %acc, %7
  br @loop(%6, %8)

@done(%r i32):
  return %r
}
```

Parameter and block names you supply survive. A result is a temporary
unless you name it — Go cannot see the name of the variable a value is
assigned to, so `.Named("addr")` is how `%addr` gets its name and
everything else gets `%0…`. `body.To()` with no arguments emits the bare
`@body` form.

---

## All the way down

The same module, lowered and written as a relocatable ELF object. This is
the whole pipeline; there is no driver in between.

```go
package main

import (
	"log"
	"os"

	amd64elf "github.com/vertex-language/amd64/obj/elf"

	"github.com/vertex-language/ir"
	lower "github.com/vertex-language/ir/lower/amd64"
	"github.com/vertex-language/ir/verify"
)

func main() {
	m := build() // as above

	// 1. §19, whole-module. Lowering assumes what this checks.
	if err := verify.Module(m); err != nil {
		log.Fatal(err)
	}

	// 2. VIR to this target's own finished object. Options.Features says
	//    what may be emitted; the zero value is the baseline every AMD64
	//    processor implements.
	obj, err := lower.Lower(m, lower.Options{})
	if err != nil {
		log.Fatal(err)
	}

	// 3. Bytes. Choosing the container format is the target module's
	//    own subpackage, called directly — obj/elf, obj/pe, obj/macho.
	f, err := os.Create("sum.o")
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()
	if err := amd64elf.Write(f, obj); err != nil {
		log.Fatal(err)
	}
}
```

```console
$ objdump -d sum.o

sum.o:	file format elf64-x86-64

Disassembly of section .text:

0000000000000000 <sum>:
       0: 48 b8 00 00 00 00 00 00 00 00	movabsq	$0x0, %rax
       a: ba 00 00 00 00               	movl	$0x0, %edx
       f: e9 00 00 00 00               	jmp	0x14 <sum+0x14>
      14: 48 3b c6                     	cmpq	%rsi, %rax
      17: 0f 8c 05 00 00 00            	jl	0x22 <sum+0x22>
      1d: e9 37 00 00 00               	jmp	0x59 <sum+0x59>
      22: 48 b9 02 00 00 00 00 00 00 00	movabsq	$0x2, %rcx
      ...
```

No prologue, because nothing in `sum` allocates or calls. At `O0` — the
only optimization level there is — a constant is materialized where it is
used and a block argument is a move; what the allocator guarantees is
that the moves are right, not that they are few.

Three lines of it are the API: `verify.Module`, `<arch>.Lower`, and that
target's `obj/<format>.Write`. The middle one is the only one that is
per-architecture, and it is per-architecture on purpose — see
[`lower`](#lower).

For another target, swap the two imports: `ir/lower/i386` with
`i386/obj/elf`, or `ir/lower/arm64` with `arm64/obj/macho`. The IR
between them does not change shape — what has to change is the module's
`layout` block, which `NewModule` takes and which every backend checks
first. Lowering a module built for one target against another is refused
by name rather than lowered anyway.

---

## Where this sits

There is no `asm` package and no `objectfile` layer. What this repo
lowers to is three sibling, independently-versioned repos, one per
architecture:

| repo | encodes | writes |
|---|---|---|
| [`i386`](https://github.com/vertex-language/i386) | Intel 386 | ELF, COFF |
| [`amd64`](https://github.com/vertex-language/amd64) | AMD64 | ELF, COFF, Mach-O |
| [`arm64`](https://github.com/vertex-language/arm64) | AArch64 | ELF, COFF, Mach-O |

Each is a complete, standalone instruction builder — registers, an ISA
table, an encoder, and an in-memory `obj.Object` — with its own
`obj/elf`, `obj/pe`, and (on `amd64`/`arm64`) `obj/macho`
subpackages to turn that object into bytes. Those, in turn, are thin
adapters over the standalone
[`elf`](https://github.com/vertex-language/elf),
[`pe`](https://github.com/vertex-language/pe), and
[`macho`](https://github.com/vertex-language/macho) container-format
libraries — the same modules that *read* the objects back, which is what
makes each target's own round-trip tests a round trip and not two
independent guesses. Container-format writing is not a fourth layer
below the assemblers; it is a subpackage of whichever one produced the
bytes.

```
your frontend               vcc, v++, the Vertex language — changes constantly
────────────────────────────────────────────────────────────────────────
ir · ir/verify · ir/text    this repo: a specified IR, a specified syntax
ir/lower                    isel · regalloc · frame — moves when the IR moves
────────────────────────────────────────────────────────────────────────
i386 · amd64 · arm64        an ISA does not change
elf · pe · macho            a container format does not change
```

---

## What lowers today

The IR is complete: every mnemonic in §17 has a builder method, and
`verify` checks every §19 rule. Lowering is not, and this table is the
honest state of it rather than the intended one. A row that is not ✅ is
refused by name at `Lower`, with the reason — never lowered as something
else.

| | amd64 | arm64 | i386 |
|---|---|---|---|
| §A · §A2 arithmetic, wide multiply, overflow predicates | ✅ | ✅ | ✅ |
| §A3 float arithmetic, at `f32`/`f64` | ✅ | ✅ | ✅ |
| §A4–§A7 bitwise, shifts, bit counting, constants | ✅ | ✅ | ✅ |
| §B comparisons | ✅ | ✅ | ✅ |
| §C–§C4 conversions | ✅ | ✅ | ✅ |
| §D · §D2 memory, sub-width memory | ✅ | ✅ | ✅ |
| §D3 pointer ops | all but `tlsaddr` | all but `tlsaddr` | all but `tlsaddr` |
| §E bulk memory | ✅ non-`volatile` | ✅ non-`volatile` | ✅ non-`volatile` |
| §F select | ✅ | ✅ | ✅ |
| §G · §G2 calls, terminators, computed branches | ✅ | ✅ | ✅ |
| §G3 unwinding — `invoke`, `invokeind`, `resume` | — | — | — |
| §G4 inline assembly | — | — | — |
| §H atomics | ✅ | ✅ | ✅ |
| §I variadics | ✅ ¹ | Apple's variant only ² | ✅ |
| ext-float | `f128` ✅, `f80` — ³ | `f128` — ⁴ | `f80` — ³ |

1. Except `ptr.va_arg_ref` of an aggregate small enough to have been
   passed in registers, whose eightbytes are scattered through the save
   area.
2. AAPCS64 has two variadic conventions and they are not compatible, so
   `Options.Variadic` states which. The base standard needs a register
   save area in the prologue and a two-region walk in `va_arg`, and is
   refused by name rather than lowered as Apple's — which would be a
   wrong call at every use rather than a slow one.
3. `f80` is the x87 stack: a third register file with a different shape
   from the two these backends allocate, and one the sibling assemblers
   declare no instruction for. It is that repository's work before it is
   this one's. AArch64's `long double` is binary128, so `f80` is not a
   type that target has at all.
4. `f128` on amd64 is compiler-rt — §0 is explicit that a namespace the
   layout admits is usable whether or not silicon implements it, and
   that lowering supplies the call. Its §A3 rows are the arithmetic and
   the sign verbs, not yet `sqrt`, `fma`, the min and max families or
   the rounding four: those are libm names rather than compiler-rt ones,
   which is a different table and a dependency on a different library.
   §B, §C and §C2 out of `f128` are complete. arm64 needs the same table
   and has none of it; i386's layout does not admit `f128` at all.

Also unwritten on all three: `sret` as something *lowering introduces*.
A signature that states one is honoured; a return of more values than
the ABI's registers hold would have to allocate the storage and rewrite
the call, which is a legalization pass and not a selection rule.

Only `O0` exists. `Options.OptLevel` is there so the day there is a
second level is not a signature change.

---

## Repository layout

```
ir/
├── go.mod                    github.com/vertex-language/ir — no dependencies
├── doc.go
├── README.md
├── spec/
│   ├── overview.md           the design commitments, non-normative
│   ├── instruct_index.md     normative: shape, operand types, result types
│   └── grammar.md            normative: the complete EBNF
│
│   ── the model: what a module is ──
├── module.go                 Module, NewModule, Target, Layout, module items
├── type.go                   RegType, StoreType, named types, arrays,
│                             sizeof/alignof/offsetof as values
├── func.go                   Func, Sig, params, callconv, attrs, placements,
│                             function imports                    (§6, whole)
├── global.go                 globals, domains, initializers, reloc, aliases,
│                             global imports                  (§5, §5b, §6)
├── block.go                  Block, Pad, block params, Target()      (§7)
├── inst.go                   Value, Instruction, operands, results
├── op.go                     the mnemonic table (§17); Op.String()
│
│   ── the emission surface: how you write one ──
├── builder.go                entry points, register naming, sticky err
├── ns_int.go                 I1/I32/I64   (§A, §A2, §A4–§A7, §B int rows)
├── ns_float.go               F32/F64/ext              (§A3, §B float rows)
├── ns_ptr.go                 ptr arithmetic and compares   (§D3, §B ptr)
├── convert.go                every conversion verb   (§C, §C2, §C3, §C4)
├── mem.go                    load/store, sub-width, bulk, access attrs
│                                                 (§D, §D2, §D4, §E)
├── atomic.go                 the atomic verb sets                   (§H)
├── select.go                 the select family                      (§F)
├── bare.go                   terminators, calls, fence, va_*
│                                             (§G2, §G3, §I, §17)
├── asm.go                    the inline-assembly escape hatch      (§G4)
├── meta.go                   metadata nodes and attachment         (§16)
│
│   ── the rest of the root ──
├── error.go                  Error + builder sentinels; sticky, first-wins
├── walk.go                   visitors, value/use iteration, replacement,
│                             Preds/Succs/RPO — CFG shape as traversal
│
├── verify/                   §19, whole-module
│   ├── verify.go             Module, Func, Options
│   ├── dom.go                dominance; unexported, §19.1 and §19.5
│   ├── ssa.go                §19.1–2, §19.17   dominance, terminators,
│   │                         reachability, the entry block's own inputs
│   ├── unwind.go             §19.4–5, §19.16   pads, resume, invoke arity
│   ├── memory.go             §19.6–9   alloc placement, blockaddr, align, CAS
│   ├── module.go             §19.10–13, §19.18   initializers, linkage, structs
│   └── error.go              verifier sentinels, positioned by fn/block/inst
│
├── text/
│   └── print.go              Printer, Print, Format
│
├── cmd/vir/                  vir list · vir cat · vir verify — a debugging
│                             tool, not a compiler driver
│
└── lower/                    ← its own go.mod; depends on i386, amd64, arm64
    ├── go.mod                github.com/vertex-language/ir/lower
    │
    │   ── what three backends turned out to share ──
    ├── mir/                  machine IR: vregs, physregs, operand roles,
    │                         parallel copies, liveness, still rewritable
    ├── regalloc/             allocation over mir, parameterized by register
    │                         classes; interference, coalescing, spilling
    ├── globals/              §5's initializer walk — what a brace means,
    │                         which fields a partial initializer leaves zero,
    │                         how much padding sits between two members — and
    │                         §2's sizeof/alignof/offsetof resolved against a
    │                         target's layout table
    │
    │   ── and what they do not ──
    ├── i386/                 against github.com/vertex-language/i386
    ├── amd64/                against github.com/vertex-language/amd64
    └── arm64/                against github.com/vertex-language/arm64
```

Each `lower/<arch>` is the same dozen files, because the job decomposes
the same way three times: `lower.go` (the entry point and its
`Options`), `layout.go` (that psABI's size and alignment table),
`frame.go` (frame planning and signature classification), `isel*.go`
(instruction selection, split by spec section where one file got long),
`control.go` (blocks, branches, edges), `atomic.go` (§H), `variadic.go`
(§I), `vregs.go` (the value environment and register classes),
`types.go` (that backend's MIR opcodes), `libcall.go` (the symbols it
invents), `global.go` (its `globals.Target` adapter), and `emit.go` (MIR
to that assembler's typed helpers).

There is no `pass/`, `abi/`, `frame/`, `unwind/`, or `debug/` package —
not yet, and the distinction matters. `abi/` and `frame/` are not absent
because classification and frame layout are simple; they are absent
because all three backends turned out to want *different* shapes for
them, and hoisting a common one before that is settled would be
inventing a union rather than finding one. `mir`, `regalloc` and
`globals` are what genuinely converged, and each was hoisted when the
second or third backend needed it — `globals` after two backends had
written the initializer walk independently and agreed line for line
except where one of them had a bug.

**Import discipline, as everywhere in this project: nothing imports
upward.**

```
ir              imports nothing outside the standard library
ir/verify       imports ir
ir/text         imports ir
ir/lower        imports ir, ir/verify, plus one of i386, amd64, arm64
```

Nothing imports `lower`. The nested `go.mod` makes that
machine-checkable rather than a review convention: a frontend that only
wants to *build* IR does not pull the assembler tree. `ir/text` needs
nothing from `ir/verify` either — it has no `Parse` to hand back a
verified module from, so its only dependency is the surface it prints.

### Why the root looks like this

Go's one-package-per-directory rule means everything touching unexported
IR internals is flat, so the file count is forced and the only real
question is where the cuts fall. They fall on **spec families, not
receivers.** Arithmetic, comparisons, shifts, and bit counting are
namespace-shaped and live in `ns_*.go`. Conversions, memory access,
atomics, and select are not — `i64.scvt_f32` belongs to neither the int
nor the float namespace, and §D's `i32.load` / `f64.store` / `ptr.load`
span three — so each gets its own file.

Conversions earn theirs twice over. The overview names them the standing
exception to additive extension: *each new register type adds a verb per
existing type it converts with, in both directions*. That is the one
combinatorial axis in the IR, and when §K's lane-width conversions
arrive it should be one file's diff.

The two files named for keywords rather than concepts are gone. Imports
sit with what they declare — function imports in `func.go`, global
imports in `global.go` — the way `alias-decl` always sat in `global.go`,
because §6 already shares `abs-signature` between an import and a
definition.

---

## The model

A `*ir.Module` is **everything a frontend decided and nothing a backend
will decide.** Instructions are named, types are fixed, control flow is
explicit — and no register is physical, no address is known, no
instruction has been selected.

The contrast with an `<arch>.Module` — `i386.Module`, `amd64.Module`,
`arm64.Module`, whichever `ir/lower` is targeting — is exact, and the
two APIs differ because of it:

|  | `ir.Module` | `<arch>.Module` |
|---|---|---|
| registers | virtual, infinite, SSA | physical only |
| after construction | rewritten by passes | frozen by `Finalize` |
| terminal call | `verify.Module` — may run any number of times | `Finalize()` — once, then immutable |
| what a pass may do | everything | nothing; there is nothing to walk |

`ir` has no `Finalize` because an IR exists to be rewritten. None of
`i386`, `amd64`, or `arm64` has a verifier because there is nothing left
to be wrong about once every register is physical. Reaching for the
other one's terminal call is the signal that a layer leaked.

### What the builder guarantees

**One Go type per `reg-type`.** `I1`, `I32`, `I64`, `F32`, `F64`, `Ptr`,
and `F80`/`F128`. There is no `I8` or `I16` Go value type, because §2
says they are storage-only — those widths are reachable exactly through
the sub-width load and store verbs in `mem.go` and nowhere else.

**`Type.Verb` is namespace value, method name.** `i32.add` is
`b.I32.Add`; `i64.sext_i32` is `b.I64.SExtI32`; `f64.minnum` is
`b.F64.MinNum`. A mnemonic the spec does not have has no method: there
is no `Gt`, no `Ge`, no `I32.USubO`, no `F32.VaArg`, no register-width
`I8.Mul`. Every §L omission is a compile error rather than a runtime
refusal.

**Bare mnemonics are builder methods.** `b.Br`, `b.BrTable`, `b.MemCpy`,
`b.Fence`, `b.Call` — matching §17's bare set exactly.

**`module`, `use`, and `layout` are constructor arguments.** They appear
exactly once each, in order, ahead of every module item, so they are
`NewModule`'s parameters and not builder methods. §19.15 stops being a
check the verifier can fail.

**Blocks are declared, parameterized, then filled.** Params come back as
typed values; a block freezes its parameter list at its first
instruction. There are no phi nodes to construct and no
predecessor-indexed operand lists to keep in sync — which is also why
predecessors are computed by `walk.go` rather than stored.

### Where each class of error surfaces

Errors are **sticky and first-wins**, as in `i386`, `amd64`, and
`arm64`, and for the same reason: a frontend emitting IR is the same
call pattern as a lowering emitting instructions, and neither should be
a run of `if err != nil`.
Every builder call after a failure is a no-op. `Module.Err()` bails
early; `verify.Module` checks it first and returns it, so soundness is
one call and not two.

| class | where it surfaces |
|---|---|
| wrong operand type, missing verb, `i8` in a register, non-`i1` condition, store operand order | `go build` |
| branch argument arity and type vs. block params, `ptr.alloc` outside the entry block, `align N` exceeding the access width, CAS ordering rules (§19.9), `sret` not first, an `ext-float` absent from `layout` | sticky, at `Err()` |
| dominance, missing or duplicate terminator, pad reachability, `resume` without a dominating pad, `blockaddr` with no matching `brind`, initializer structure vs. declared type | `verify.Module` |
| a value from another function, a block from another function | panic — a Go bug, not IR data |

The middle two rows are two sentinel sets in two packages, split by
where the fault is *detected*: `ir/error.go` holds what the builder
catches as you emit, `verify/error.go` holds what only a finished module
reveals. `ErrLayout` is the one worth knowing early — `b.F80()` on a
module whose `layout` omits it. Availability is a run-time property of
the `layout` block, so it is the one thing Go's type system cannot carry:
the Go-level analogue of §19.12's *rejected, not emulated*.

---

## `verify`

```go
func Module(m *ir.Module) error
func Func(f *ir.Func) error

type Options struct{ MaxErrors int }

func (Options) Module(m *ir.Module) error
func (Options) Func(f *ir.Func) error
```

The two package-level functions are the zero `Options`. A cap on how many
faults come back is a driver's concern, not a rule's, so it sits on a
value you call *through* rather than on a parameter every call has to
pass. Both entry points return `ir.Module.Err()` if there is one, and
otherwise every §19 fault they found: an `Errors` list, not the first one.
The builder is first-wins because it is a writer and every call after a
failure is a no-op; a verifier reads a finished module where the faults
are independent and all still there.

A package rather than a method, for three reasons. §19 is eighteen rules
and wants more than one file. Its failure corpus — one case per rule,
each expected to fail with a named sentinel at a named position —
belongs next to the sentinels that name it. Those cases are Go functions,
not a `testdata/` directory: there is nothing here to load a `.vir` file
with, and a module is built by calling methods.

And the verifier is the most demanding reader the IR has, so making it a
*client* of the public surface is a standing proof that surface is
complete. If a §19 rule can only be checked from inside the package, the
fix is an exported iterator, not an exported field — and if no such
method is defensible, that rule belongs in the sticky tier instead.

Dominance lives here, unexported, because §19.1 and §19.5 are its only
consumers. There is no `analysis` package: call graphs and loop nesting
had no consumer in a repo with no inliner, and liveness belongs to
`lower/mir`, since register allocation is a function of MIR — vregs *and*
physregs — not of VIR. CFG shape is traversal, so it is `walk.go`'s.

---

## `text`

```go
func Print(w io.Writer, m *ir.Module) error
func Format(m *ir.Module) ([]byte, error)

type Printer struct {
	Indent     string
	NameTemps  bool  // %0.. vs. keeping builder-supplied names
	Metadata   bool  // omit !nodes for a readable diff
	SortModule bool  // canonical item order, for golden files
}
```

`text` is a printer and nothing else. There is no `Parse`. A
`*ir.Module` is built by calling methods, full stop — reading one back in
from `.vir` would make this package a second front door for
constructing a module, with its own invariants to defend, and no
consumer here needs that door. `.vir` exists for humans and for tests:
`-emit-vir`, a diff between two passes, a golden file a change should
visibly move.

`Format` refuses to print a module carrying a sticky builder error
(`m.Err()`) — there is nothing faithful to print — but it does not run
the verifier itself. Whether a module is *sound* is `verify.Module`'s
question; `text` only answers whether it can be shown.

The distinction from `i386`, `amd64`, and `arm64`, none of which has a
text layer at all: there, Go *is* the input language, so a parser would
be a second, competing one. Here there was room for a `.vir` reader —
this repo isn't the input language itself — but nothing in the pipeline
consumes one, so it stayed unbuilt rather than added on spec.

---

## `lower`

```go
import (
	"github.com/vertex-language/amd64/feature"
	"github.com/vertex-language/ir"
	lower "github.com/vertex-language/ir/lower/amd64"
)

obj, err := lower.Lower(m, lower.Options{
	Features: feature.Default().Add(feature.SSE41, feature.POPCNT),
})
// obj is *amd64obj.Object — github.com/vertex-language/amd64/obj —
// the target's own finished, immutable artifact. There is no
// intermediate *amd64.Module left lying around to mutate: Lower builds
// one internally, calls its Finalize, and hands back only what that
// returned.
```

Each backend's `Options` is its own type and holds only what that target
actually has a decision to make about:

| | `amd64` | `arm64` | `i386` |
|---|---|---|---|
| `OptLevel` | ✅ | ✅ | ✅ |
| `Features` | `feature.Set` | `arm64.FeatureSet` | — ¹ |
| `Variadic` | — | AAPCS64 or Darwin | — |
| `LibcallPrefix` | — | ✅ ² | — |

1. `i386`'s baseline is fixed at i686 plus SSE2 — SSE2 is where this
   backend's floats live, since x87's registers are a stack that no
   allocator here can model. There is nothing left to gate on.
2. Prepended to the name of every function the backend calls that the
   module did not declare — §E's `memcpy` and its neighbours. Stated
   rather than derived: nothing else here mangles a symbol, and Mach-O
   prefixes a C symbol with an underscore where ELF does not.

`Options.Features` is deliberately not one shared vocabulary. `amd64`
gates on `feature.Level`s named `V1`..`V4` plus orthogonal extensions
like `AVX2`; `arm64` gates on levels named `Armv8A` upward plus its own
orthogonal set. Two instruction sets do not share one notion of "what
can I emit," and `lower/<arch>` takes that target's own feature type
rather than inventing a third vocabulary that would just be translated
back into one of the other two at the call. Where the answer is no, the
verb is refused by name — never expanded into a slower sequence behind
your back.

**There is no cross-architecture entry point,** because there is no
cross-architecture object type to return. Each of `i386`, `amd64`, and
`arm64` is its own module with its own `obj.Object` and no type in
common: a shared operand type across architectures would be an IR, the
thing those packages are defined by not being. So `Lower` is per-arch
and its return type is that arch's own object, and a driver that carries
the architecture as data carries a closure or a small interface of its
own rather than one this repo guessed at.

### The path, and why `mir` exists

```
VIR                      SSA, block params, virtual registers, typed
  │  <arch>/ isel        pattern match on VIR nodes, one block at a time —
  │                      and sometimes into several, since a retry loop and
  ▼                      a range check are branches isel itself introduces
MIR                      machine opcodes, vregs + physregs, still rewritable
  │  regalloc/           liveness, interference, coalescing, spilling
  │  <arch>/ emit        prologue, epilogue, block order, typed helper calls
  ▼
<arch>.Section            i386.Section, amd64.Section, or arm64.Section —
                          decided by typed helper calls
  │  Finalize()
  ▼
<arch>/obj.Object         everything decided; immutable, ready for that
                          target's own obj/elf, obj/pe, or obj/macho
```

Register allocation cannot run against an `<arch>.Section` past
`Finalize`, and it cannot run against `<arch>/obj.Object` at all: both
are append-only or immutable, with every register physical and nothing
symbolic-and-rewritable. That is the model each of the three targets
states for itself, not a limitation this repo works around. So the
representation regalloc is a function *of* has to live here, in `mir` —
and so does the liveness it consumes, which is over machine opcodes
after isel, not over VIR.

There is no target-independent legalization pass, and VIR's own
commitments say there should eventually be one. "No undefined behaviour"
means every `sdiv` carries an implied `/0` and `INT_MIN/-1` check, every
non-`_sat_` float→int conversion a range test, every block argument a
parallel assignment needing cycle-breaking. Today each backend does its
own — `x86` gets the division checks free from `#DE` where `arm64` has to
emit them, and the three range checks agree on the interval because they
were made to agree rather than because one piece of code computes it.
That is the clearest candidate for the next thing to hoist, and it is
not hoisted yet.

### Full stop at the target

`Lower` returns that architecture's own `obj.Object` and the story ends.
This repo does not choose a container format, does not know a
relocation number, does not link, does not take a `-o` flag, and does
not open a file. Picking ELF, COFF, or Mach-O — and writing it — is the
target module's own subpackage, called directly, with nothing of
`ir/lower`'s in the way:

```go
import (
	elfcore "github.com/vertex-language/elf"
	amd64elf "github.com/vertex-language/amd64/obj/elf"
)

err := amd64elf.Write(w, obj, amd64elf.Options{GNUStack: elfcore.StackOmit})
```

Linking is further still: `elf`, `pe`, and `macho` each ship their own
`link` package, none of it wired to anything here, and every target's
own round-trip tests still prove correctness against an outside tool —
`objdump`, or a real `clang`-driven link-and-run — rather than a linker
in this tree. `ir/lower` hands back bytes and references; what resolves
those references, in-tree or out, is a different repo's README.

---

## What is deliberately absent, and why

| absent | why |
|---|---|
| **an `analysis` package** | Dominance has one consumer and is unexported in `verify`. Liveness is MIR's, where regalloc runs. CFG shape is traversal, so it is `walk.go`'s. Call graphs and loop nesting had no consumer in a repo with no inliner and no interprocedural pass. |
| **`Module.Verify()`** | A method would make `ir` import `ir/verify`. `ir` imports nothing, so the verifier is a function over the public surface — which is also what keeps that surface honest. |
| **a `.vir` parser** | Reading text back in would make `text` a second way to construct a `*ir.Module`, with its own invariants to defend, for no consumer in this repo. Everything that builds IR does it by calling methods; `.vir` is an output format, not an input one. This is also why `cmd/vir` has no `fmt`: formatting means reading, so the tool takes the *name* of a module it builds rather than a file. |
| **object file writing, relocation numbers** | `Lower` returns that target's own `obj.Object`. `R_X86_64_PLT32` and `IMAGE_REL_ARM64_BRANCH26` are `amd64/obj/elf`'s and `arm64/obj/pe`'s vocabulary, reached by naming that target's own semantic `RefKind` — `amd64.RefPLT32`, `arm64.RefCall26` — at the typed-helper call; this repo never states a relocation number itself. |
| **linking, archives, LTO** | Nothing here has an address, and nothing here has a second translation unit. |
| **a driver, `-o`, file I/O** | `Lower` takes a value and returns a value; `text.Print` takes an `io.Writer`. Whoever wants a file writes one. `cmd/vir` is a debugging tool and imports the same public API you do. |
| **an encoder** | If `lower/<arch>` ever computes a byte, the wall moved. It calls that target's typed helpers and lets `i386`, `amd64`, or `arm64` decide encodings. |
| **host CPU detection** | `Options.Features` says what you *may emit*. Detecting the host is a runtime library's job; conflating them makes cross-compilation a special case instead of the default. |
| **a C front end, a type system with signedness, declarations** | VIR's types are signless and its verbs carry interpretation. Anything that knows what a `struct` *means* rather than how it is laid out belongs above this repo. |
| **a dynamic floating-point environment** | §L. Rounding is round-to-nearest-even; exception flags are unobservable. Modelling otherwise puts an implicit operand and result on every float instruction in the IR. |
| **a constant expression grammar in initializers** | `reloc` admits what relocation records admit: one symbol, optionally minus a second, plus one displacement. That displacement may be `sizeof`, `alignof` or `offsetof`, which `lower/globals` resolves against the target's layout table — `&arr[3]` is the array's symbol plus an `offsetof` and needs no arithmetic. Admitting an *operator* would mean owning a constant evaluator, and `Init` has no Go type that could accept an expression. |
| **an interpreter, a JIT execution engine** | A consumer of this IR, not a member of it. The lowering path is the execution story. |
| **text as a compile-path format** | `text` exists for humans and tests. A compiler lowers by calling methods; there is no reader here to round-trip through. |

---

## Testing

Every boundary is a plain value you can stop at, and each layer is
checked against something that is not itself:

- **builder** — `text.Format(m)` against the `.vir` the test states
  inline. Not a `testdata/` file: there is no reader here to load one
  with, which is the same reason the verifier's cases are Go functions.
  The printer is the only thing that can show what the builder built, so
  `text`'s own tests check it the other way round, from modules whose
  shape the test states.
- **verifier** — one case per §19 rule, each expected to fail with a
  named sentinel at a named position.
- **`lower/globals`** — the initializer walk over a layout that answers
  by rule rather than by any psABI, so that what is under test is the
  walk — which field, which element, what a union does — and not a
  target's table.
- **isel, on `amd64`** — pinned byte strings for the short sequences,
  and `objdump -d`/`-r` for the shapes where the block layout is the
  point. Nothing links, because nothing on the host runs x86-64.
- **isel, on `arm64`** — lowered to Mach-O, linked against a C `main`
  with the native toolchain, and run. Which is the strongest check
  available: disassembly proves the bytes decode, and this proves they
  compute — that the ABI is right on both sides of a real call boundary,
  that the frame balances, that a callee-saved register really is saved.
  Skipped rather than failed off Apple Silicon.
- **isel, on `i386`** — also run, which took some arranging. There is no
  way to execute a 32-bit x86 process on an Apple Silicon host, so the
  harness links the output against a freestanding runtime into a
  multiboot kernel, boots it under `qemu-system-i386`, and reads the
  answers off the emulated serial port. Skipped without `clang`,
  `ld.lld` and `qemu`.

The oracle is chosen to be a different argument from the one under test,
not the same argument written twice. `arm64`'s wide multiply goes
against C's own cast; `i386`'s goes against a shift-and-add loop,
because a reference that multiplied in 32-bit columns would be the
lowering restated. `amd64`'s unsigned 64-bit conversion is checked
against Go's own — and against the assertion that the naive form
*disagrees* somewhere, so the test fails if the careful version is doing
nothing.

Compiler middles are miserable to debug because these seams are usually
smeared: a pass that half-lowers, a "module" that is also a buffer, a
verifier that only runs in one mode. Here each one is a value with a
name, and every layer's failures are its own.
