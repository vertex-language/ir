# Vertex IR — Overview

## What it is

Vertex IR (VIR) is a typed SSA intermediate representation for native AOT and JIT compilation. It is the common backend of the Vertex toolchain: `vcc` (C), `v++` (C++), and the Vertex language's own native lowering path. It targets physical CPUs directly and is not a sandboxed VM.

**Naming.** The IR is *Vertex IR* in prose, `VIR` in short form, `.vir` as a file extension.

## Core commitments

**SSA with block parameters.** Infinite virtual registers, each assigned once. Basic blocks take typed parameters and receive arguments at every branch — there are no phi nodes and no predecessor-indexed operand lists.

**Namespace.Operation grammar.** Instructions in a type-parameterized family are spelled `Type.Verb`. A dot means exactly one thing: this opcode is parameterized over a machine type. Instructions with no per-type sibling are bare, and module scope uses plain keywords with no dots at all. No mnemonic carries a `vir.` prefix — the IR's name lives in the file extension and the documentation, not in the instruction stream.

**Strict monomorphism, with one stated exception.** No instruction is overloaded. The mnemonic alone is a complete contract — one operand type combination, one result type. Signed and unsigned are separate verbs, since the hardware uses separate circuits. `asm` (inline assembly) is the sole instruction the grammar admits without this contract: it has no fixed operand shape and no semantics this IR can state. It exists because glibc, musl, and every kernel are unbuildable without it, and it is spelled as an escape hatch, not folded into the general rule.

**Hardware workflow.** The IR neither hides machine realities nor invents machines that do not exist. There is no `i8.mul`, because no target has an 8-bit multiplier. A namespace the `layout` block admits is available to the instruction stream whether or not the target implements it in hardware; where it does not, lowering supplies a runtime call.

**No undefined behaviour.** Every operation produces a defined value or traps. This is a compilation position rather than a safety one: a trap specification is shorter than a UB specification, and an optimizer needs one or the other.

## Type system

Types are signless — interpretation lives in the verb. The set splits by where a value can live. Register types are `i1`, `i32`, `i64`, `f32`, `f64`, `ptr`, zero, one, or both of `f80` and `f128`, and `v128` where the `layout` block provides them; storage types are `i8`, `i16`, `i32`, `i64`, `f32`, `f64`, `ptr`, `f80`/`f128`/`v128` where present, named types, and fixed arrays.

`v128` is the whole of the vector type system, and one type is enough because the lane shape is not a property of the register — sixteen bytes are eight words to one instruction and four doublewords to the next. The shape rides in the verb (`v128.i16x8_add`), which is the same place signedness already rides and for the same reason. Nothing is bitcast between vector shapes because there is nothing to bitcast between, which is also what lets a frontend map C's `__m128i`, `__m128` and `__m128d` — one register spelled three ways — onto one type and lose nothing.

`i8` and `i16` are storage-only widths, reached through sign- or zero-extending sub-width loads and truncating stores. `i1` is register-only and has no storage width; widen it to put it in memory. Pointers are opaque — the IR tracks addresses, not pointee types — and aggregates are never register values, never cross a function boundary as such, and never parameterize a namespace. (A signature may still return several registers as an ABI-level multi-value result; that's a call-boundary shape, not an aggregate register type.).

Named types are layout descriptions the assembler consults, not values the instruction stream manipulates. They exist so that a frontend's flattening into byte offsets becomes checkable through `sizeof`, `alignof`, and `offsetof` rather than hand-computed and silently target-dependent.

## Instruction set shape

Integer and float arithmetic on the register widths, including whichever extended-float namespace the target has. Comparisons always yield `i1`, with no `gt` or `ge` in any namespace, and a distinct `uno` because float `ne` is not the negation of ordered `eq`. Conversions name the destination in the namespace and the source in the verb, so a conversion and its inverse share a verb. Memory access is value-first, address-last, assumes natural alignment, and traps otherwise unless an `align` attribute overrides downward. Control flow is bare-verb, including computed goto (`brind`) over block addresses. Atomics cover `i32` and `i64` with explicit orderings.

Where a form is derivable from another at no cost, it is omitted rather than given a second opcode.

## Frontend coverage

`vcc` reaches the whole of C through this instruction set, including `__int128` (via the `mulhi` and overflow-predicate pairs), computed goto, and `__attribute__((cleanup))` on unwind paths. `v++` needs no additional instructions: the `catch` and `filter` pad clauses cost C nothing and leave C++ exceptions reachable, and virtual dispatch is `ptr.load` plus `callind` against a `func` typedef.

## Module scope

Five declaration forms, all shaped `[modifier] keyword @name`: types, globals, function imports, function definitions, and aliases. A sixth form has no name and is deliberate about it: a module-level `asm` block is text handed to the assembler where it was declared, and whatever it defines it defines to the linker rather than to this module — nothing can call it, alias it, or collide with it. Linkage, visibility, and binding are independent modifiers. Global initializers are literals and symbol addresses only, with structure matching the declared type exactly — admitting one operator would mean owning a constant evaluator. An alias binds a second name to an existing definition and initializes nothing of its own.

## Extension

Three axes are left open: sub-word atomics, pointer atomics, and new metadata kinds — none needs a grammar change. New arithmetic, comparison, memory, and select types mostly arrive as new namespaces over the existing verb set rather than as new verbs, which is the payoff for keeping narrow integers out of the arithmetic namespaces in the first place. Conversions are the standing exception: because a conversion verb names the *other* type as well as the namespace, each new register type adds a verb per existing type it converts with, in both directions. `f80`/`f128` already paid this cost once.

SIMD was the fourth, and arrived as `v128` rather than as the `f32x4`/`i32x8` family this section once anticipated. Putting the lane shape in the verb rather than in the type is what avoided the conversion cost: a lane-width change is `v128.i8x16_narrow_s`, one verb, and not a verb per pair of vector namespaces. Wider registers — a `v256`, a `v512` — would be new types under the same rule, and the shape-suffixed verbs generalize to them unchanged.