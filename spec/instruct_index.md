# Vertex IR — Instruction Index

Normative on instruction shape, operand types, and result types.

Sections run §0, §A–§I, §V, §K and §L. There is no §J; the letter is unassigned and
reserved, so that a citation to `§K` in an older document still means what it
meant.

## 0. Global Rules

These hold everywhere and are not repeated per row.

**Integer arithmetic wraps.** `add`, `sub`, `mul`, `neg`, and `shl` are
two's-complement modulo the namespace width, signed and unsigned alike.
`i32.neg` of `INT_MIN` is `INT_MIN`. Overflow is detected by the §A2 predicates,
never by the arithmetic itself.

**Shift amounts are masked.** For an `iN` namespace, `shl`, `sshr`, `ushr`,
`rotl`, and `rotr` take the shift amount modulo `N`. No shift traps and no shift
is undefined.

**Division traps.** `sdiv`, `udiv`, `srem`, `urem` trap on a zero divisor;
the signed forms additionally trap on `INT_MIN / -1`.

**Float rounding is round-to-nearest-even** and is not dynamically changeable.
Exception flags are not observable. See §L.

**Memory access assumes natural alignment** and traps otherwise, unless an
`align` attribute overrides it downward.

**Comparisons yield `i1`.** There is no `gt` or `ge` in any namespace; swap the
operands.

This is a rule about the scalar namespaces. A vector comparison has one
answer per lane and yields a mask instead — see §V.

**Extended float.** `f80` and `f128` rows apply only where the module's `layout`
block's `extfloat` attribute lists that namespace. A `layout` may list zero,
one, or both — x86-64 System V lists both, since `long double` is `f80` and
`__float128` is `f128`.

**Vector.** §V applies only where the `layout` block's `vector` attribute admits
`v128`. x86-64 and AArch64 both do, since SSE2 and Advanced SIMD are part of
the definition of those architectures; a plain i386 target does not.

**Namespace availability is not a hardware claim.** A namespace the `layout`
block admits is usable by the instruction stream whether or not the target
implements it in silicon; where it does not, lowering supplies a runtime call.
No instruction changes shape, no operand becomes implicit, and this document
names no runtime symbol. `f128.add` on x86-64 is the same instruction it is
anywhere; only its lowering differs.

**Pointer width.** `ptr` is `ptrbits` wide, from the `layout` block. `ptr.add`
and `ptr.sub` wrap modulo `2^ptrbits`.

## A. Integer Arithmetic

| Instruction | Operands | Result |
| --- | --- | --- |
| `i32.add` | `i32, i32` | `i32` — wraps |
| `i64.add` | `i64, i64` | `i64` — wraps |
| `i32.sub` | `i32, i32` | `i32` — wraps |
| `i64.sub` | `i64, i64` | `i64` — wraps |
| `i32.mul` | `i32, i32` | `i32` — low half, wraps |
| `i64.mul` | `i64, i64` | `i64` — low half, wraps |
| `i32.smulhi` | `i32, i32` | `i32` — high half of the signed product |
| `i64.smulhi` | `i64, i64` | `i64` — high half of the signed product |
| `i32.umulhi` | `i32, i32` | `i32` — high half of the unsigned product |
| `i64.umulhi` | `i64, i64` | `i64` — high half of the unsigned product |
| `i32.sdiv` | `i32, i32` | `i32` — traps on `/0` and `INT_MIN/-1` |
| `i64.sdiv` | `i64, i64` | `i64` — traps on `/0` and `INT_MIN/-1` |
| `i32.udiv` | `i32, i32` | `i32` — traps on `/0` |
| `i64.udiv` | `i64, i64` | `i64` — traps on `/0` |
| `i32.srem` | `i32, i32` | `i32` — traps on `/0` and `INT_MIN/-1` |
| `i64.srem` | `i64, i64` | `i64` — traps on `/0` and `INT_MIN/-1` |
| `i32.urem` | `i32, i32` | `i32` — traps on `/0` |
| `i64.urem` | `i64, i64` | `i64` — traps on `/0` |
| `i32.neg` | `i32` | `i32` — wraps at `INT_MIN` |
| `i64.neg` | `i64` | `i64` — wraps at `INT_MIN` |

## A2. Overflow Predicates

| Instruction | Operands | Result |
| --- | --- | --- |
| `i32.saddo` | `i32, i32` | `i1` — true iff signed `a+b` overflows |
| `i64.saddo` | `i64, i64` | `i1` |
| `i32.uaddo` | `i32, i32` | `i1` — true iff unsigned `a+b` carries out |
| `i64.uaddo` | `i64, i64` | `i1` |
| `i32.ssubo` | `i32, i32` | `i1` — true iff signed `a-b` overflows |
| `i64.ssubo` | `i64, i64` | `i1` |
| `i32.smulo` | `i32, i32` | `i1` — true iff signed `a*b` overflows |
| `i64.smulo` | `i64, i64` | `i1` |
| `i32.umulo` | `i32, i32` | `i1` — true iff unsigned `a*b` carries out |
| `i64.umulo` | `i64, i64` | `i1` |

Each predicate pairs with the corresponding wrapping verb in §A: the flag and the
truncated result together are the full widened answer.

**There is deliberately no `usubo`.** Unsigned subtract borrow is `a <u b`,
computable from the inputs alone via `ult` — unlike unsigned add carry, which
needs the result. The asymmetry is real hardware asymmetry, not an omission.

## A3. Float Arithmetic

`float-ns` below is any of `f32`, `f64`, or an available `ext-float`. Rows are
given for `f32`/`f64`; the extended namespaces carry the identical verb set.

| Instruction | Operands | Result |
| --- | --- | --- |
| `f32.add` | `f32, f32` | `f32` |
| `f64.add` | `f64, f64` | `f64` |
| `f32.sub` | `f32, f32` | `f32` |
| `f64.sub` | `f64, f64` | `f64` |
| `f32.mul` | `f32, f32` | `f32` |
| `f64.mul` | `f64, f64` | `f64` |
| `f32.div` | `f32, f32` | `f32` |
| `f64.div` | `f64, f64` | `f64` |
| `f32.fma` | `f32, f32, f32` | `f32` — `a*b+c`, one rounding |
| `f64.fma` | `f64, f64, f64` | `f64` — `a*b+c`, one rounding |
| `f32.neg` | `f32` | `f32` — sign flip, NaN payload preserved |
| `f64.neg` | `f64` | `f64` |
| `f32.abs` | `f32` | `f32` — sign clear, NaN payload preserved |
| `f64.abs` | `f64` | `f64` |
| `f32.sqrt` | `f32` | `f32` |
| `f64.sqrt` | `f64` | `f64` |
| `f32.minimum` | `f32, f32` | `f32` — NaN propagates; `−0 < +0` |
| `f64.minimum` | `f64, f64` | `f64` |
| `f32.maximum` | `f32, f32` | `f32` — NaN propagates; `−0 < +0` |
| `f64.maximum` | `f64, f64` | `f64` |
| `f32.minnum` | `f32, f32` | `f32` — returns the non-NaN operand |
| `f64.minnum` | `f64, f64` | `f64` |
| `f32.maxnum` | `f32, f32` | `f32` — returns the non-NaN operand |
| `f64.maxnum` | `f64, f64` | `f64` |
| `f32.copysign` | `f32, f32` | `f32` |
| `f64.copysign` | `f64, f64` | `f64` |
| `f32.ceil` | `f32` | `f32` — round toward +∞ |
| `f64.ceil` | `f64` | `f64` |
| `f32.floor` | `f32` | `f32` — round toward −∞ |
| `f64.floor` | `f64` | `f64` |
| `f32.trunc` | `f32` | `f32` — round toward zero |
| `f64.trunc` | `f64` | `f64` |
| `f32.nearest` | `f32` | `f32` — round to nearest, ties to even |
| `f64.nearest` | `f64` | `f64` |

**`minimum`/`maximum` vs `minnum`/`maxnum`.** The first pair is IEEE-754-2019
`minimum`/`maximum`: any NaN operand yields NaN. The second is
IEEE-754-2008 `minNum`/`maxNum`, which discards a NaN operand in favour of the
other — this is what C's `fmin`/`fmax` require, and lowering `fmin` to `minimum`
is silently wrong. Both pairs are single instructions on common hardware.

Signed zero is fully specified rather than left to the target: `minimum` and
`minnum` of `(+0, −0)` return `−0`; `maximum` and `maxnum` return `+0`. A target
whose instruction disagrees pays a fixup.

There is no float remainder verb; see §L.

## A4. Bitwise

| Instruction | Operands | Result |
| --- | --- | --- |
| `i1.not` | `i1` | `i1` |
| `i32.not` | `i32` | `i32` |
| `i64.not` | `i64` | `i64` |
| `i1.and` | `i1, i1` | `i1` |
| `i32.and` | `i32, i32` | `i32` |
| `i64.and` | `i64, i64` | `i64` |
| `i1.or` | `i1, i1` | `i1` |
| `i32.or` | `i32, i32` | `i32` |
| `i64.or` | `i64, i64` | `i64` |
| `i1.xor` | `i1, i1` | `i1` |
| `i32.xor` | `i32, i32` | `i32` |
| `i64.xor` | `i64, i64` | `i64` |

The `i1` namespace has these five verbs, `const`, and `select`, and no
comparisons: `i1.ne` is `i1.xor` and `i1.eq` is `i1.not` of it. See §L.

## A5. Shifts / Rotates

Shift amount is taken modulo the namespace width. No form traps.

| Instruction | Operands | Result |
| --- | --- | --- |
| `i32.shl` | `i32, i32` | `i32` — amount mod 32 |
| `i64.shl` | `i64, i64` | `i64` — amount mod 64 |
| `i32.sshr` | `i32, i32` | `i32` — sign-propagating, amount mod 32 |
| `i64.sshr` | `i64, i64` | `i64` — sign-propagating, amount mod 64 |
| `i32.ushr` | `i32, i32` | `i32` — zero-filling, amount mod 32 |
| `i64.ushr` | `i64, i64` | `i64` — zero-filling, amount mod 64 |
| `i32.rotl` | `i32, i32` | `i32` |
| `i64.rotl` | `i64, i64` | `i64` |
| `i32.rotr` | `i32, i32` | `i32` |
| `i64.rotr` | `i64, i64` | `i64` |

## A6. Bit Counting / Byte Swap

| Instruction | Operands | Result |
| --- | --- | --- |
| `i32.clz` | `i32` | `i32` — 32 for a zero input |
| `i64.clz` | `i64` | `i64` — 64 for a zero input |
| `i32.ctz` | `i32` | `i32` — 32 for a zero input |
| `i64.ctz` | `i64` | `i64` — 64 for a zero input |
| `i32.popcnt` | `i32` | `i32` |
| `i64.popcnt` | `i64` | `i64` |
| `i32.bswap` | `i32` | `i32` — reverse byte order |
| `i64.bswap` | `i64` | `i64` — reverse byte order |

The zero-input results are specified, not target-defined. C's `__builtin_clz`
leaves this undefined; the IR does not.

## A7. Constants

| Instruction | Operands | Result |
| --- | --- | --- |
| `i1.const` | literal (`0`/`1`) | `i1` |
| `i32.const` | literal | `i32` |
| `i64.const` | literal | `i64` |
| `f32.const` | literal | `f32` |
| `f64.const` | literal | `f64` |
| `f80.const` / `f128.const` | literal | per target |
| `ptr.const` | `0` | `ptr` — null only |

A non-null absolute address is `ptr.from_i64` of an `i64.const`.

## B. Comparisons

| Instruction | Operands | Result |
| --- | --- | --- |
| `i32.eq` | `i32, i32` | `i1` |
| `i64.eq` | `i64, i64` | `i1` |
| `i32.ne` | `i32, i32` | `i1` |
| `i64.ne` | `i64, i64` | `i1` |
| `i32.slt` | `i32, i32` | `i1` |
| `i64.slt` | `i64, i64` | `i1` |
| `i32.ult` | `i32, i32` | `i1` |
| `i64.ult` | `i64, i64` | `i1` |
| `i32.sle` | `i32, i32` | `i1` |
| `i64.sle` | `i64, i64` | `i1` |
| `i32.ule` | `i32, i32` | `i1` |
| `i64.ule` | `i64, i64` | `i1` |
| `f32.eq` | `f32, f32` | `i1` — ordered; false if either is NaN |
| `f64.eq` | `f64, f64` | `i1` — ordered; false if either is NaN |
| `f32.ne` | `f32, f32` | `i1` — exact negation of `eq`; true if either is NaN |
| `f64.ne` | `f64, f64` | `i1` — exact negation of `eq`; true if either is NaN |
| `f32.lt` | `f32, f32` | `i1` — ordered; false if either is NaN |
| `f64.lt` | `f64, f64` | `i1` — ordered; false if either is NaN |
| `f32.le` | `f32, f32` | `i1` — ordered; false if either is NaN |
| `f64.le` | `f64, f64` | `i1` — ordered; false if either is NaN |
| `f32.uno` | `f32, f32` | `i1` — true iff either operand is NaN |
| `f64.uno` | `f64, f64` | `i1` — true iff either operand is NaN |
| `f80` / `f128` | same five verbs | `i1` |
| `ptr.eq` | `ptr, ptr` | `i1` |
| `ptr.ne` | `ptr, ptr` | `i1` |
| `ptr.lt` | `ptr, ptr` | `i1` — unsigned address comparison |
| `ptr.le` | `ptr, ptr` | `i1` — unsigned address comparison |

`ne` is the exact negation of `eq`, which is what C's `!=` means. `uno` exists
because unordered-ness is not reachable from `eq`/`ne` alone: `isnan(x)` is
`uno(x, x)`, and an *ordered* `!=` is `and(not(uno), ne)`. Ordered and unordered
variants of `lt`/`le` are likewise built from `uno`, so only one of each is given.

There are no `i1` comparisons; see §A4 and §L.

## C. Conversions — Integer

| Instruction | Operands | Result |
| --- | --- | --- |
| `i32.wrap_i64` | `i64` | `i32` — discard high 32 bits |
| `i64.sext_i32` | `i32` | `i64` — sign-extend |
| `i64.zext_i32` | `i32` | `i64` — zero-extend |
| `i32.zext_i1` | `i1` | `i32` — `0` or `1` |
| `i64.zext_i1` | `i1` | `i64` — `0` or `1` |

Narrowing to `i8`/`i16` is a store; widening from them is a sub-width load. There
is no register-to-register form, because `i8` and `i16` are not register types.

## C2. Conversions — Int ↔ Float

| Instruction | Operands | Result |
| --- | --- | --- |
| `f32.scvt_i32` | `i32` | `f32` |
| `f32.scvt_i64` | `i64` | `f32` |
| `f64.scvt_i32` | `i32` | `f64` |
| `f64.scvt_i64` | `i64` | `f64` |
| `f32.ucvt_i32` | `i32` | `f32` |
| `f32.ucvt_i64` | `i64` | `f32` |
| `f64.ucvt_i32` | `i32` | `f64` |
| `f64.ucvt_i64` | `i64` | `f64` |
| `f80.{scvt,ucvt}_{i32,i64}` | `i32`/`i64` | `f80` |
| `f128.{scvt,ucvt}_{i32,i64}` | `i32`/`i64` | `f128` |
| `i32.scvt_f32` | `f32` | `i32` — traps out-of-range/NaN |
| `i32.scvt_f64` | `f64` | `i32` — traps out-of-range/NaN |
| `i64.scvt_f32` | `f32` | `i64` — traps out-of-range/NaN |
| `i64.scvt_f64` | `f64` | `i64` — traps out-of-range/NaN |
| `i32.ucvt_f32` | `f32` | `i32` — traps out-of-range/NaN |
| `i32.ucvt_f64` | `f64` | `i32` — traps out-of-range/NaN |
| `i64.ucvt_f32` | `f32` | `i64` — traps out-of-range/NaN |
| `i64.ucvt_f64` | `f64` | `i64` — traps out-of-range/NaN |
| `{i32,i64}.{scvt,ucvt}_{f80,f128}` | ext | `i32`/`i64` — traps |
| `i32.scvt_sat_f32` | `f32` | `i32` — clamps, NaN→0 |
| `i32.scvt_sat_f64` | `f64` | `i32` — clamps, NaN→0 |
| `i64.scvt_sat_f32` | `f32` | `i64` — clamps, NaN→0 |
| `i64.scvt_sat_f64` | `f64` | `i64` — clamps, NaN→0 |
| `i32.ucvt_sat_f32` | `f32` | `i32` — clamps, NaN→0 |
| `i32.ucvt_sat_f64` | `f64` | `i32` — clamps, NaN→0 |
| `i64.ucvt_sat_f32` | `f32` | `i64` — clamps, NaN→0 |
| `i64.ucvt_sat_f64` | `f64` | `i64` — clamps, NaN→0 |
| `{i32,i64}.{scvt,ucvt}_sat_{f80,f128}` | ext | `i32`/`i64` — clamps, NaN→0 |

A C frontend emits the `_sat_` forms only where it has proven the conversion
in range or has chosen to define the out-of-range case; the trapping forms are
the default, per the no-UB commitment.

## C3. Conversions — Float Width / Bitcast

| Instruction | Operands | Result |
| --- | --- | --- |
| `f64.fcvt_f32` | `f32` | `f64` — widen, exact |
| `f32.fcvt_f64` | `f64` | `f32` — narrow, round-to-nearest |
| `f80.fcvt_f32` | `f32` | `f80` — widen, exact |
| `f80.fcvt_f64` | `f64` | `f80` — widen, exact |
| `f32.fcvt_f80` | `f80` | `f32` — narrow, round-to-nearest |
| `f64.fcvt_f80` | `f80` | `f64` — narrow, round-to-nearest |
| `f128.fcvt_f32` / `f128.fcvt_f64` | `f32`/`f64` | `f128` — widen, exact |
| `f32.fcvt_f128` / `f64.fcvt_f128` | `f128` | `f32`/`f64` — narrow, round |
| `f128.fcvt_f80` | `f80` | `f128` — widen, exact; both namespaces required |
| `f80.fcvt_f128` | `f128` | `f80` — narrow, round; both namespaces required |
| `i32.bitcast_f32` | `f32` | `i32` — reinterpret, no conversion |
| `f32.bitcast_i32` | `i32` | `f32` — reinterpret, no conversion |
| `i64.bitcast_f64` | `f64` | `i64` — reinterpret, no conversion |
| `f64.bitcast_i64` | `i64` | `f64` — reinterpret, no conversion |

There is no bitcast for `f80` or `f128`: neither has an integer register type of
matching width. Reach their representation through memory.

## C4. Conversions — Pointer ↔ Integer

| Instruction | Operands | Result |
| --- | --- | --- |
| `ptr.from_i64` | `i64` | `ptr` — truncates where `ptrbits` < 64 |
| `i64.from_ptr` | `ptr` | `i64` — zero-extends where `ptrbits` < 64 |

There is no `i32` pair; see §L.

## D. Memory — Full-Width Load / Store

| Instruction | Operands | Result |
| --- | --- | --- |
| `i32.load` | `ptr` | `i32` |
| `i64.load` | `ptr` | `i64` |
| `f32.load` | `ptr` | `f32` |
| `f64.load` | `ptr` | `f64` |
| `f80.load` / `f128.load` | `ptr` | per target |
| `ptr.load` | `ptr` | `ptr` |
| `i32.store` | `i32, ptr` | — |
| `i64.store` | `i64, ptr` | — |
| `f32.store` | `f32, ptr` | — |
| `f64.store` | `f64, ptr` | — |
| `f80.store` / `f128.store` | ext, `ptr` | — |
| `ptr.store` | `ptr, ptr` | — value first, destination second |

## D2. Memory — Sub-Width Load / Store

| Instruction | Operands | Result |
| --- | --- | --- |
| `i32.sload8` | `ptr` | `i32` — load 1 byte, sign-extend |
| `i32.sload16` | `ptr` | `i32` — load 2 bytes, sign-extend |
| `i64.sload8` | `ptr` | `i64` — load 1 byte, sign-extend |
| `i64.sload16` | `ptr` | `i64` — load 2 bytes, sign-extend |
| `i64.sload32` | `ptr` | `i64` — load 4 bytes, sign-extend |
| `i32.uload8` | `ptr` | `i32` — load 1 byte, zero-extend |
| `i32.uload16` | `ptr` | `i32` — load 2 bytes, zero-extend |
| `i64.uload8` | `ptr` | `i64` — load 1 byte, zero-extend |
| `i64.uload16` | `ptr` | `i64` — load 2 bytes, zero-extend |
| `i64.uload32` | `ptr` | `i64` — load 4 bytes, zero-extend |
| `i32.store8` | `i32, ptr` | — truncate to 1 byte, write |
| `i32.store16` | `i32, ptr` | — truncate to 2 bytes, write |
| `i64.store8` | `i64, ptr` | — truncate to 1 byte, write |
| `i64.store16` | `i64, ptr` | — truncate to 2 bytes, write |
| `i64.store32` | `i64, ptr` | — truncate to 4 bytes, write |

## D3. Memory — Pointer Ops

| Instruction | Operands | Result |
| --- | --- | --- |
| `ptr.alloc` | `unsigned size, align unsigned`, `zeroed`? | `ptr` — entry block only |
| `ptr.alloc` | `@Type`, `zeroed`? | `ptr` — size and alignment from the named type |
| `ptr.alloca` | `i64 size, align unsigned`, `zeroed`? | `ptr` — any block, dynamic frame |
| `ptr.stacksave` | — | `ptr` — opaque token |
| `ptr.stackrestore` | `ptr` token | — |
| `ptr.getaddr` | `@global` or `@func` | `ptr` |
| `ptr.tlsaddr` | `@global` (domain `tls`) | `ptr` — address in the calling thread |
| `ptr.blockaddr` | `@label` | `ptr` — for `brind` only |
| `ptr.frameaddr` | — | `ptr` — current frame, level 0 only |
| `ptr.returnaddr` | — | `ptr` — current frame's return address, level 0 only |
| `ptr.add` | `ptr, i64` | `ptr` — wraps mod `2^ptrbits` |
| `ptr.sub` | `ptr, i64` | `ptr` — wraps mod `2^ptrbits` |
| `ptr.diff` | `ptr, ptr` | `i64` — signed, sign-extended from `ptrbits` |

`zeroed` on either allocation form guarantees the storage reads as all-zero
bytes on entry to the allocation's live range. Absent it, the contents are
unspecified but defined — arbitrary bytes, never a trap and never UB.

`ptr.frameaddr` and `ptr.returnaddr` describe the current frame only. Walking
outward is a runtime's job, not an instruction's; a frame-pointer-omitting target
has no reliable answer for level > 0.

## D4. Access Attributes

| Attribute | Meaning |
| --- | --- |
| `align N` | `N` is a power of two no greater than the access width. Absence asserts natural alignment. |
| `volatile` | The access is observable: not elidable, duplicable, reorderable across other volatiles, or widenable. |

## E. Bulk Memory

| Instruction | Operands | Result |
| --- | --- | --- |
| `memcpy` | `ptr dst, ptr src, i64 len` | — non-overlapping |
| `memmove` | `ptr dst, ptr src, i64 len` | — overlap-safe |
| `memset` | `ptr dst, i32 val, i64 len` | — low byte of `val` written |
| `memcmp` | `ptr a, ptr b, i64 len` | `i32` — 0 if equal; <0/>0 by first differing byte, unsigned |

A zero `len` is well-defined and touches nothing, including when a pointer is
null. `memcpy` with overlapping ranges traps or produces an unspecified-but-
defined result at the implementation's choice; it does not have unbounded effect.

## F. Select

| Instruction | Operands | Result |
| --- | --- | --- |
| `i1.select` | `i1, i1, i1` | `i1` |
| `i32.select` | `i1, i32, i32` | `i32` |
| `i64.select` | `i1, i64, i64` | `i64` |
| `f32.select` | `i1, f32, f32` | `f32` |
| `f64.select` | `i1, f64, f64` | `f64` |
| `f80.select` / `f128.select` | `i1`, ext, ext | ext |
| `ptr.select` | `i1, ptr, ptr` | `ptr` |

Both arms are evaluated. `select` is not a short-circuit; C's `&&`, `||`, and
`?:` lower to branches unless the frontend has proven both arms safe.

`select` plus a comparison is also how integer min and max are spelled; see §L.

## G. Calls

| Instruction | Operands | Result |
| --- | --- | --- |
| `call` | `@func(args...)` | per signature |
| `callind` | `ptr : @Type(args...)` | per the named function type |

The calling convention comes from the signature — from the declaration for
`call`, from the named `func` type for `callind`. A `callind` whose named type
disagrees with the callee's actual convention is a program error the IR cannot
detect, exactly as in C.

Absent an explicit `callconv`, a signature's convention is `ccc`, which is the
`abi` named in the module's `layout` block.

## G2. Control Flow — Terminators

| Instruction | Operands | Result |
| --- | --- | --- |
| `br` | `@label(args...)` | — |
| `brif` | `i1, @label(args...), @label(args...)` | — |
| `br_table` | `i32 selector, [@label(args...), ...], @default(args...)` | — |
| `brind` | `ptr, [@label, ...]` | — targets take no parameters |
| `return` | `%register, ...`? | — one per `ret-item` |
| `trap` | — | — |

`br_table`'s selector is `i32` and indexes the table from zero; an out-of-range
selector takes the default edge. A `switch` on a wider or offset type is a
frontend-emitted subtract and range check — work the frontend already does to
find the default edge.

The entry block is not a target of any of these; see the grammar, §19.17.

There is no `unreachable`; see §L.

## G3. Unwinding

| Instruction | Operands | Result |
| --- | --- | --- |
| `invoke` | `@func(args...) to @normal(args...) unwind @pad` | — terminator; results bind on the normal edge |
| `invokeind` | `ptr : @Type(args...) to @normal(args...) unwind @pad` | — terminator; results bind on the normal edge |
| `resume` | `ptr exn` | — terminator |

**An `invoke`'s results are the trailing parameters of its normal target.** The
`to` target's argument list carries what the branch supplies; the callee's
results are appended after them, one per `ret-item`, and the target block
declares a parameter for each. A void call appends nothing.

`invoke` has no result list of its own, and must not have one: a register it
defined would have to dominate both edges, and on the unwind edge the call did
not complete. Binding results as parameters puts them exactly where they are
live. This is the same reasoning that gives `asm goto` no outputs (§G4) — the
difference being that an `invoke` has one edge on which the call did complete
and can therefore name a place for its results, and an `asm goto` has none.

```vertex-ir
@try:
  invoke @get_int(%a) to @ok(%tag) unwind @pad

@ok(%tag i32, %r i32):        // %tag from the branch, %r from the call
  return %r
```

The unwind edge names a **pad block**, declared as:

```text
@pad pad (%exn ptr, %sel i32) cleanup:
```

Its two parameters are supplied by the personality routine — the exception object
pointer and the personality's selector — not by any branch. This is why the
unwind edge carries no argument list while the normal edge does.

| Pad clause | Meaning |
| --- | --- |
| `cleanup` | The pad runs destructors and re-raises. Sufficient for all of C. |
| `catch @ti` | The pad handles the type described by type-info global `@ti`. |
| `filter [@ti, ...]` | The pad handles anything *not* in the listed set. |

A function containing `invoke`, `invokeind`, or a pad block declares
`personality @fn`. `resume` takes the `%exn` parameter of a dominating pad and
returns control to the unwinder.

C reaches all of this through `__attribute__((cleanup))` and through
`-fexceptions` code that must run cleanups on an unwind path crossing it; the
`catch` and `filter` clauses cost nothing for C and leave C++ reachable.

## G4. Inline Assembly

| Instruction | Operands | Result |
| --- | --- | --- |
| `asm` | template string, constrained operands, clobber list | zero or more, per `asm-outs` |
| `asm goto` | template string, constrained inputs, clobber list, fallthrough target, label list | — terminator, no outputs |

`asm` is the one instruction with no fixed operand shape and no semantics this IR
can state. It exists because glibc, musl, and every kernel are unbuildable
without it. `asm goto` is its terminator form, implicitly volatile; it has no
outputs, because an output register would need a definition dominating edges on
which no definition occurred. Results travel through memory.

Clobbers admit target register names plus `"memory"` and `"cc"`.

## H. Atomics

All forms take an explicit `ordering` and an optional `volatile`.

| Instruction | Operands | Result |
| --- | --- | --- |
| `i32.atomic_load` | `ptr, ordering` | `i32` |
| `i64.atomic_load` | `ptr, ordering` | `i64` |
| `ptr.atomic_load` | `ptr, ordering` | `ptr` |
| `i32.atomic_uload8` | `ptr, ordering` | `i32` — 1 byte, zero-extend |
| `i32.atomic_uload16` | `ptr, ordering` | `i32` — 2 bytes, zero-extend |
| `i32.atomic_store` | `i32, ptr, ordering` | — |
| `i64.atomic_store` | `i64, ptr, ordering` | — |
| `ptr.atomic_store` | `ptr, ptr, ordering` | — |
| `i32.atomic_store8` | `i32, ptr, ordering` | — low byte |
| `i32.atomic_store16` | `i32, ptr, ordering` | — low 2 bytes |
| `i32.atomic_rmwadd` | `i32, ptr, ordering` | `i32` — returns old value |
| `i64.atomic_rmwadd` | `i64, ptr, ordering` | `i64` — returns old value |
| `i32.atomic_rmwsub` | `i32, ptr, ordering` | `i32` |
| `i64.atomic_rmwsub` | `i64, ptr, ordering` | `i64` |
| `i32.atomic_rmwand` | `i32, ptr, ordering` | `i32` |
| `i64.atomic_rmwand` | `i64, ptr, ordering` | `i64` |
| `i32.atomic_rmwor` | `i32, ptr, ordering` | `i32` |
| `i64.atomic_rmwor` | `i64, ptr, ordering` | `i64` |
| `i32.atomic_rmwxor` | `i32, ptr, ordering` | `i32` |
| `i64.atomic_rmwxor` | `i64, ptr, ordering` | `i64` |
| `i32.atomic_rmwxchg` | `i32, ptr, ordering` | `i32` |
| `i64.atomic_rmwxchg` | `i64, ptr, ordering` | `i64` |
| `i32.atomic_rmw{op}8` | `i32, ptr, ordering` | `i32` — old value, zero-extended |
| `i32.atomic_rmw{op}16` | `i32, ptr, ordering` | `i32` — old value, zero-extended |
| `ptr.atomic_rmwadd` | `i64 delta, ptr, ordering` | `ptr` — returns old value |
| `ptr.atomic_rmwsub` | `i64 delta, ptr, ordering` | `ptr` |
| `ptr.atomic_rmwxchg` | `ptr, ptr, ordering` | `ptr` |
| `i32.atomic_cas` | `i32 expect, i32 new, ptr, ordering, ordering` | `i32` |
| `i64.atomic_cas` | `i64 expect, i64 new, ptr, ordering, ordering` | `i64` |
| `ptr.atomic_cas` | `ptr expect, ptr new, ptr, ordering, ordering` | `ptr` |
| `i32.atomic_cas8` | `i32 expect, i32 new, ptr, ordering, ordering` | `i32` |
| `i32.atomic_cas16` | `i32 expect, i32 new, ptr, ordering, ordering` | `i32` |
| `fence` | `ordering`, `singlethread`? | — bare, no type involved |

`{op}` ranges over `add`, `sub`, `and`, `or`, `xor`, `xchg`.

Narrow atomics live in the `i32` namespace only. An `i64` set would be reachable
by zero-extension with no hardware distinction, so it is omitted. `_Atomic(_Bool)`
uses the 8-bit forms, since `i1` has no storage width.

Ordering constraints: read-modify-write and compare-and-swap orderings are not
`unordered`; a compare-and-swap's failure ordering is no stronger than its
success ordering and is neither `release` nor `acq_rel`. `atomic_cas` returns
the value read; success is `eq` against `expect`.

**`fence ... singlethread` is a compiler barrier.** It orders this thread's
accesses against this thread's own interrupted execution — a signal handler, a
scheduler on the same core — and emits no machine barrier. It is C11's
`atomic_signal_fence`, which is otherwise inexpressible: lowering it as an
ordinary `fence` is a correctness-preserving pessimization at every use, and the
alternative frontends reach for instead, `asm volatile ("" ::: "memory")`, is
opaque to the optimizer in ways a fence is not. No access carries the token;
`fence` alone does.

Atomic accesses assume natural alignment for their width and trap otherwise. An
`align` attribute does not weaken this — a misaligned atomic is not atomic on any
target the IR addresses.

## I. Variadics

| Instruction | Operands | Result |
| --- | --- | --- |
| `va_start` | `ptr` | — bare |
| `va_end` | `ptr` | — bare |
| `va_copy` | `ptr dst, ptr src` | — bare |
| `i32.va_arg` | `ptr` | `i32` |
| `i64.va_arg` | `ptr` | `i64` |
| `f64.va_arg` | `ptr` | `f64` |
| `f80.va_arg` / `f128.va_arg` | `ptr` | per target |
| `ptr.va_arg` | `ptr` | `ptr` |
| `ptr.va_arg_ref` | `ptr, @Type` | `ptr` — address of the next argument |

There is no `f32.va_arg` and no `i8`/`i16` form: the default argument promotions
mean no such argument can be present.

`ptr.va_arg_ref` advances the list past one argument of the named type and yields
its address, which is how `va_arg(ap, struct S)` is expressed. The ABI knowledge
it requires is the same knowledge `byval` already demands.

## V. Vector

The `v128` namespace. Every row applies only where the module's `layout`
block's `vector` attribute admits `v128`.

`v128` is a 128-bit register and the lane shape is not part of it — the same
sixteen bytes are eight words to one instruction and four doublewords to the
next. The shape is therefore in the verb, prefixed, so that the half a reader
needs first comes first: `v128.i16x8_add` is "add, as eight words". Verbs with
no shape prefix are the ones the shape cannot change: sixteen bytes ANDed are
sixteen bytes ANDed however the lanes are drawn.

Two global rules of §0 are stated differently here, and the difference is the
hardware's:

**Vector comparisons yield a mask, not `i1`.** A lane of the result is
all-ones where the comparison holds and all-zero where it does not, so the
result is a `v128` and feeds the bitwise verbs. There are sixteen answers, or
eight, or four; `i1` can carry one. `v128.i8x16_bitmask` is how they leave the
register file as a single integer.

**Lane shifts saturate rather than wrap the count.** A count at or past the
lane width yields zero lanes, and all-sign-bit lanes for `shr_s`. §0's masking
rule is the scalar namespaces' and does not reach here.

There is no bitcast in this namespace and nothing to bitcast between. There is
also no `gt_u` — an unsigned lane compare is `min_u` against `eq`, and no
target has the instruction.

### V1. Shape-free

| Instruction | Operands | Result |
| --- | --- | --- |
| `v128.const` | 16-byte literal, memory order | `v128` |
| `v128.and` / `or` / `xor` | `v128, v128` | `v128` |
| `v128.not` | `v128` | `v128` |
| `v128.andnot` | `v128 a, v128 b` | `v128` — `a AND NOT b` |
| `v128.shl_bytes` / `shr_bytes` | `v128`, literal | `v128` — whole register, in bytes |
| `v128.zext_i32` | `i32` | `v128` — low lane, rest zero |
| `v128.zext_i64` | `i64` | `v128` — low lane, rest zero |
| `v128.load` / `v128.store` | §D's shapes | `v128` / — |
| `v128.select` | `i1, v128, v128` | `v128` — §F, one condition for all lanes |

`v128.const`'s literal is bytes and not a number because a number would mean
choosing a lane width and an endianness to read it at. A frontend that has
lane values converts them once, here.

`v128.zext_i32` is not a splat, and the difference is load-bearing: it builds
a vector from one scalar without filling the lanes a splat would fill.

### V2. Lane arithmetic

`S` below ranges over the shapes each row lists.

| Instruction | Shapes | Operands | Result |
| --- | --- | --- | --- |
| `v128.S_add` / `S_sub` | `i8x16 i16x8 i32x4 i64x2` | `v128, v128` | `v128` |
| `v128.i16x8_mul` | — | `v128, v128` | `v128` — low halves |
| `v128.i16x8_mulhi_s` / `_u` | — | `v128, v128` | `v128` — high halves |
| `v128.i32x4_mul_even_u` | — | `v128, v128` | `v128` — even lanes widened to `i64x2` |
| `v128.S_add_sat_s` / `_u` | `i8x16 i16x8` | `v128, v128` | `v128` |
| `v128.S_sub_sat_s` / `_u` | `i8x16 i16x8` | `v128, v128` | `v128` |
| `v128.i8x16_min_u` / `_max_u` | — | `v128, v128` | `v128` |
| `v128.i16x8_min_s` / `_max_s` | — | `v128, v128` | `v128` |
| `v128.S_avgr_u` | `i8x16 i16x8` | `v128, v128` | `v128` — rounded up |
| `v128.i8x16_sad_u` | — | `v128, v128` | `v128` — per half, into the low word of each quadword |
| `v128.i16x8_madd_s` | — | `v128, v128` | `v128` — adjacent pairs, into `i32x4` |

The min/max and saturating rows are the shapes the baseline hardware provides
and not the shapes an orthogonal table would predict. A signed byte minimum is
absent here because it is absent from SSE2 and from the set of things every
target in scope can do in one instruction; the caller who wants one writes the
two instructions rather than having a verb hide them.

### V3. Lane comparison, shift, and mask

| Instruction | Shapes | Operands | Result |
| --- | --- | --- | --- |
| `v128.S_eq` / `S_gt_s` | `i8x16 i16x8 i32x4` | `v128, v128` | `v128` mask |
| `v128.S_shl` / `S_shr_u` | `i16x8 i32x4 i64x2` | `v128, i32` | `v128` |
| `v128.S_shr_s` | `i16x8 i32x4` | `v128, i32` | `v128` |
| `v128.i8x16_bitmask` | — | `v128` | `i32` — top bit per lane, lane 0 in bit 0 |

There is no `i64x2_shr_s` and no `i64x2_eq`: neither is a baseline
instruction, and a verb that lowered to a sequence would hide the cost from
the one caller who cares about it.

### V4. Lane movement

| Instruction | Shapes | Operands | Result |
| --- | --- | --- | --- |
| `v128.S_splat` | `i8x16 i16x8 i32x4` | `i32` | `v128` |
| `v128.i64x2_splat` | — | `i64` | `v128` |
| `v128.i16x8_extract_lane_u` | — | `v128`, literal | `i32` — zero-extended |
| `v128.i16x8_replace_lane` | — | `v128, i32`, literal | `v128` |
| `v128.i32x4_shuffle` | — | `v128`, literal | `v128` — four 2-bit selectors |
| `v128.i16x8_shuffle_low` / `_high` | — | `v128`, literal | `v128` — one half permuted, the other copied |
| `v128.S_unpack_low` / `_unpack_high` | `i8x16 i16x8 i32x4 i64x2` | `v128, v128` | `v128` |
| `v128.i8x16_narrow_s` / `_u` | — | `v128, v128` of `i16x8` | `v128` — saturating |
| `v128.i16x8_narrow_s` | — | `v128, v128` of `i32x4` | `v128` — saturating |

Unpacking interleaves the named half of each operand, the first operand's lane
first. Widening a vector is unpacking it against zero; duplicating a lane is
unpacking it against itself. Narrowing is the other direction and saturates.

The lane index and shuffle pattern are literals because the instructions'
are. A variable index is a store and a scalar load, which the frontend writes
where it means it.

## K. Reserved

| Axis | How it arrives |
| --- | --- |
| Wider vectors | A `v256` or `v512` `reg-type`, admitted by the `layout` block. §V's shape-prefixed verbs generalize unchanged. |
| Atomic min/max | `i32.atomic_rmwsmin` and siblings — §H's shape, new verbs. |
| Atomic nand | `i32.atomic_rmwnand` and siblings — §H's shape, new verbs. |
| Sub-word and pointer atomic coverage | New rows in §H's existing families. |
| Wider extended floats | A new `ext-float` member, admitted by the `layout` block. |
| Function memory effects | New function placements (`readnone`, `readonly`, `argmemonly`). |
| Pointer parameter facts | New parameter attributes (`nonnull`, `dereferenceable`, `align`). |
| Tail calls | A modifier on `call`. |
| Named sync scopes | A generalization of `singlethread` on `fence`. |
| Metadata kinds | New `!ident` names; debug information is the first consumer. |

Each row is additive: a new namespace, a new verb in an existing family, or a
new member of an existing modifier list. None changes the meaning of any
conforming module.

## L. Deliberate Omissions

Stated so that their absence reads as a decision rather than an oversight.

**`usubo`.** Unsigned subtract borrow is `a <u b`. See §A2.

**`gt` / `ge`.** Swap the operands.

**Ordered `ne`, unordered `lt`/`le`.** Built from `uno`. See §B.

**`i1.eq` / `i1.ne`.** `i1.ne` is `i1.xor`; `i1.eq` is `i1.not` of that. The
`i1` namespace is deliberately five bitwise verbs, `const`, and `select` — a
one-bit comparison is a one-bit bitwise operation, and giving it a second
spelling would put two mnemonics on one machine instruction.

**Integer `smin` / `smax` / `umin` / `umax`.** A comparison and a `select`,
which every target has a conditional move for. Note that §K reserves the
*atomic* min/max verbs: there the operation is indivisible and a select-based
expansion is not equivalent, which is the whole reason the atomic forms need
verbs and these do not.

**`unreachable`.** It is undefined behaviour under a friendlier name, and this
IR has none. A path a frontend believes cannot be taken ends in `trap`, which
costs one instruction and defines what happens when the belief is wrong.

**Float remainder (`fmod`, `frem`).** No hardware computes it in one
instruction on any target this IR addresses; C's is a libm call and remains one.
Giving it a verb would put a namespace-parameterized mnemonic on something that
is always a call.

**`debugtrap`.** `trap` is the abnormal-termination instruction. A breakpoint
you resume from is a different thing — `int3` against `ud2`, and not
interchangeable — but it is a debugger protocol rather than a program semantic,
so `__builtin_debugtrap` lowers through `asm volatile`, not through a second
trapping verb.

**`ptr` ↔ `i32`.** On a `ptrbits 32` module, `(intptr_t)p` is `i64.from_ptr`
then `i32.wrap_i64`, and back is `i64.zext_i32` then `ptr.from_i64`. Both are
lossless — `i64.from_ptr` zero-extends and `ptr.from_i64` truncates — and both
fold to a register move in lowering. A second conversion pair would be two
verbs whose only justification is saving an instruction that costs nothing.

**Register-width `i8` / `i16` arithmetic.** No target has an 8-bit multiplier.
Keeping narrow integers out of the arithmetic namespaces is what makes a new
namespace cheap to add.

**A dynamic floating-point environment.** Rounding is round-to-nearest-even and
exception flags are not observable, so `#pragma STDC FENV_ACCESS ON` is not
supported. A frontend needing it must go through `asm volatile` and accept that
the optimizer will not respect the dependency. This is a stated non-goal, not a
gap to be filled later: modelling it would put an implicit operand and an
implicit result on every float instruction in the IR.

**`__builtin_return_address(n)` for n > 0.** See §D3.

**A general constant expression grammar in initializers.** `reloc` admits what
relocation records admit and no more.

### Changes from the previous revision

| § | change |
| --- | --- |
| head | section-letter note: §J unassigned |
| 0 | "target record" → `layout` block, in both the extended-float and `ptrbits` rules; namespace-availability paragraph added |
| A3, A4, B, C4, F, G2 | cross-references to new §L entries |
| D3 | `zeroed` added to the `ptr.alloc`/`ptr.alloca` rows, with its guarantee stated |
| G | `ccc` tied to `layout`'s `abi` |
| G3 | `invoke` result binding stated, with example and rationale |
| H | `fence` gains `singlethread`, with rationale |
| K | reserved list extended; additivity sentence added |
| L | six entries added: `i1` comparisons, integer min/max, `unreachable`, float remainder, `debugtrap`, `ptr` ↔ `i32` |
