# Vertex IR (VIR) — Grammar Specification

The complete EBNF grammar for the Vertex IR (VIR).

## 0. Notation

```text
X*      zero or more
X?      optional
X | Y   alternation
X+      one or more
( )     grouping
"lit"   literal token
(* *)   non-normative comment
```

Two conventions apply throughout:

- **Mnemonic concatenation.** Inside a mnemonic production, juxtaposed literals
  form a single token with no intervening whitespace. `( "i32" | "i64" ) "."
  ( "add" | "sub" )` denotes the four tokens `i32.add`, `i32.sub`, `i64.add`,
  `i64.sub` — not a `.` separated from its neighbours.
- **Whitespace and comments** are insignificant between tokens and are not shown
  in the productions.

## 1. Lexical Grammar

```ebnf
letter       ::= "A".."Z" | "a".."z"
digit        ::= "0".."9"
hexdigit     ::= digit | "a".."f" | "A".."F"
ident        ::= ( letter | "_" ) ( letter | digit | "_" )*
name         ::= ident
decimal      ::= digit+
unsigned     ::= decimal | "0x" hexdigit+
int          ::= "-"? unsigned
exponent     ::= ( "e" | "E" ) ( "+" | "-" )? decimal
hexexponent  ::= ( "p" | "P" ) ( "+" | "-" )? decimal
float        ::= "-"? decimal "." decimal* exponent?
               | "-"? decimal exponent
               | "-"? "0x" hexdigit+ ( "." hexdigit* )? hexexponent
               | "-"? "inf"
               | "-"? "nan" ( ":" "0x" hexdigit+ )?
register     ::= "%" ( ident | decimal )
symbol       ::= "@" ident
metaname     ::= "!" ident
ellipsis     ::= "..."
string       ::= '"' schar* '"'
schar        ::= (* any source byte except '"' and '\' *) | escape
escape       ::= "\n" | "\r" | "\t" | "\a" | "\b" | "\f" | "\v"
               | "\0" | "\\" | '\"' | "\x" hexdigit hexdigit
comment      ::= "//" (* to end of line *)
```

### 1b. Symbol Classes

`TypeName`, `GlobalName`, `FuncName`, and `Label` are the same lexical class as
`symbol`. They are distinguished by position and by namespace, not by spelling.

```ebnf
TypeName     ::= symbol
GlobalName   ::= symbol
FuncName     ::= symbol
Label        ::= symbol
```

Three namespaces exist:

- **Types.** `TypeName` — module scope, disjoint from the value namespace.
- **Values.** `GlobalName` and `FuncName` share one module-scope namespace, since
  they share one at link time.
- **Labels.** `Label` — function-local, disjoint from both.

A module's own name (see §3) is a fourth, even smaller namespace: it names
nothing else and nothing else can reference it.

## 2. Types

```ebnf
reg-type     ::= "i1" | "i32" | "i64" | "f32" | "f64" | "ptr" | ext-float
               | vector-type
ext-float    ::= "f80" | "f128"
vector-type  ::= "v128"
store-type   ::= "i8" | "i16" | "i32" | "i64" | "f32" | "f64" | "ptr"
               | ext-float | vector-type
ftype        ::= store-type | TypeName | "[" unsigned "]" ftype
```

An `ext-float` namespace is available only if the module's `layout` block lists
it. A `layout` block may provide zero, one, or both. `v128` is gated the same
way, by `layout`'s `vector` attribute.

`v128` is a 128-bit vector register and there is exactly one of it. The lane
shape is not part of the type: the same sixteen bytes are eight words to one
instruction and four doublewords to the next, so the shape rides in the verb
(`v128.i16x8_add`, `v128.i32x4_add`), which is where §1 already puts
signedness for the same reason. There is consequently nothing to bitcast
between, and no bitcast in the namespace. See §V of the instruction index.

A namespace the `layout` block admits is available to the instruction stream
whether or not the target implements it in hardware; where it does not, lowering
supplies a runtime call. This is a lowering fact and not a grammar one — no
instruction changes shape, and this document names no runtime symbol.

## 3. Module Structure

```ebnf
module        ::= module-decl use-decl layout-decl module-item*
module-decl   ::= "module" ident
use-decl      ::= "use" string
module-item   ::= type-decl | global-decl | import-decl | func-def
                | alias-decl | meta-decl | asm-decl
linkage       ::= "export" | "internal"
visibility    ::= "hidden" | "protected" | "dllimport" | "dllexport"
binding       ::= "weak" | "common"

layout-decl      ::= "layout" "{" layout-attr-list? "}"
layout-attr-list ::= layout-attr ( "," layout-attr )* ","?
layout-attr      ::= "abi"        ident
                   | "endian"     ( "little" | "big" )
                   | "ptrbits"    unsigned
                   | "stackalign" unsigned
                   | "extfloat"   ( "none" | ext-float ( "," ext-float )* )
                   | "vector"     ( "none" | vector-type )
```

`module-decl`, `use-decl`, and `layout-decl` each appear exactly once, in that
order, ahead of every `module-item`. `module main` names the module for
tooling and diagnostics — it is a bare identifier, not a `symbol`, and
nothing else in the file can reference it. `use` takes a single path-shaped
string identifying the target triple-ish pair (`"x86_64/linux"`); everything
about calling convention, register width, and float support beyond that pair
lives in `layout`. Together the three declarations are what makes `sizeof`,
`alignof`, and `offsetof` determinate, and what admits or rejects an
`ext-float` namespace.

`layout`'s six attributes are unordered within the braces and each appears
once; a trailing comma after the last attribute is permitted but not
required, matching every other brace-delimited list in this grammar.

**`abi` names the module's default calling convention**, and `ccc` in a
`callconv` position (§6) means exactly that convention. A signature naming a
`callconv` explicitly overrides the default for itself and for nothing else. So
`ccc` on an `abi ms` module is Microsoft x64, and `sysv_abi` on the same module
is System V for that one signature — the two spellings are not
interchangeable, and a module has no convention-free default.

`layout` is also the sole source of `ptrbits` and of extended-float
availability; §0 of the instruction index reads both from here.

```vertex-ir
module main

use "x86_64/linux"

layout {
  abi        sysv,
  endian     little,
  ptrbits    64,
  stackalign 16,
  extfloat   none,
  vector     none,
}
```

### 3b. Module-Level Assembly

```ebnf
asm-decl      ::= "asm" string
```

Assembly at module scope, emitted where it is declared and otherwise
uninterpreted. It has no operands — there is no register in scope to constrain
— and so no template substitution: `%` and `$` in it mean whatever the target
assembler says they mean. This is `__asm__` at file scope in C, and the shape
it appears in is a directive with data behind it:

```vertex-ir
asm ".pushsection .init_array,\"aw\"\n.quad my_ctor\n.popsection"
```

**It declares no symbol.** A name it defines is defined to the linker and not
to this module: nothing can `call` it, alias it, or take its address, and a
`module-item` may declare that same name without collision. That is the whole
of the containment this form has, and it is why an `asm-decl` has no
`GlobalName` — one would be a claim about text this document does not read.

**Each block is its own assembly.** A section it opens it closes, and neither
the section stack nor the numeric local labels nor the absolute symbols of one
block reach the next — a `.pushsection` left open is a fault local to the block
that left it. Order between two `asm-decl`s therefore decides one thing only:
where the bytes of the second fall relative to the first in a section they both
append to. Order against any other `module-item` decides the same and nothing
more.

## 4. Type Declarations

```ebnf
type-decl      ::= linkage? "type" TypeName typedef meta*
typedef        ::= struct-layout? "struct" "{" fields? "}"
                 | struct-layout? "union"  "{" fields? "}"
                 | ftype
                 | "func" abs-signature
struct-layout  ::= "packed" | "align" unsigned
fields         ::= field ( "," field )*
field          ::= name ftype ( "at" unsigned )?
```

`struct-layout` is named to avoid colliding with the module-level `layout`
block in §3 — the two are unrelated grammar points that happen to both
describe memory shape, one per-aggregate and one per-module.

`at` states a field's byte offset explicitly. The grammar admits it on any
subset of a struct's fields; §19.18 admits it on all or none, because a struct
with some offsets stated and some computed has no determinate `offsetof` and so
defeats the reason §4 exists. A union's fields all begin at zero and `at` is not
permitted on them.

## 5. Globals

```ebnf
global-decl        ::= linkage? visibility? binding? "global" domain GlobalName
                        ftype global-placement* "=" init meta*
domain             ::= "ro" | "rw" | "tls"
global-placement   ::= "section"  string
                     | "comdat"   string?
                     | "align"    unsigned
                     | "tlsmodel" tls-model
tls-model          ::= "global-dynamic" | "local-dynamic"
                     | "initial-exec"   | "local-exec"

init               ::= literal | string | "zeroed" | reloc
                     | "{" init-list? "}"
init-list          ::= init ( "," init )*
                     | field-init ( "," field-init )*
field-init         ::= name "=" init

reloc              ::= sym-ref ( "-" sym-ref )? ( "+" addend )?
sym-ref            ::= GlobalName | FuncName
addend             ::= int | symconst
```

The `ftype` is **required**. §19.10 checks an initializer's structure against
the declared type, which is unenforceable without one, and `zeroed` has no
width to take from an initializer that states none. There is no inference from
the initializer, because `= 0` would have to pick between `i32` and `i64` and
neither answer is derivable.

`reloc` is a relocation, not an expression. It admits exactly what an object
file's relocation records admit: a symbol, optionally minus a second symbol, plus
one assemble-time-known displacement. `&arr[3]` is written
`@arr + offsetof @ArrTy [3]`; no multiplication is required or provided.

A `comdat` with no key defaults to the declared symbol's own name.

## 5b. Aliases

```ebnf
alias-decl   ::= linkage? visibility? "weak"? "alias"
                 ( "func"   FuncName   FuncName
                 | "global" GlobalName GlobalName )
```

## 6. Functions

```ebnf
import-decl        ::= visibility? "weak"? "import"
                        ( "func"   FuncName   abs-signature
                                              import-placement*
                        | "global" GlobalName ftype
                                              global-placement* ) meta*
func-def           ::= linkage? visibility? "weak"? "func" FuncName signature
                        func-placement* meta* func-body
func-placement     ::= "section" string | "comdat" string? | "nounwind"
                     | "personality" FuncName | "returns_twice"
                     | "naked" | "noreturn" | "align" unsigned
import-placement   ::= "nounwind" | "returns_twice" | "noreturn"

signature          ::= callconv? "(" def-param-list? var-tail? ")" ret?
abs-signature      ::= callconv? "(" abs-param-list? var-tail? ")" ret?
var-tail           ::= ( "," )? ellipsis

def-param-list     ::= def-param ( "," def-param )*
abs-param-list     ::= abs-param ( "," abs-param )*
def-param          ::= register reg-type param-attr*
abs-param          ::= register? reg-type param-attr*
param-attr         ::= "byval" TypeName | "sret" TypeName
                     | "zext" | "sext" | "noalias"

ret                ::= reg-type ret-attr*
                     | "(" ret-item ( "," ret-item )* ")"
ret-item           ::= reg-type ret-attr*
ret-attr           ::= "zext" | "sext"

callconv           ::= "ccc" | "fastcc" | "preserve_most" | "preserve_all"
                     | "stdcall" | "fastcall" | "thiscall" | "vectorcall"
                     | "ms_abi"  | "sysv_abi"
```

An import carries `visibility` and `weak` because both are facts about the
reference, not the definition: `dllimport` describes how this module reaches a
symbol another module defines, and a weak import is a reference that may go
unresolved, which is `&f != NULL` in C and a live idiom in every C runtime.
`linkage` is absent because an import defines nothing to give linkage to, and
`common` is absent for the same reason — it is a definition's tentative
allocation, not a reference's property.

A definition names every parameter register; an import or a function *type* need
not, since neither has a body to reference them. `callconv` sits on the
signature so that a `func` typedef carries it and `callind` remains well-typed.
Absent `callconv` means `ccc`, which is the `abi` named in §3's `layout` block.

`var-tail` admits `(...)`, `(i32 %n, ...)`, and the leading-comma spelling
uniformly, replacing the three-way alternation of earlier drafts.

## 7. Blocks

```ebnf
func-body     ::= "{" entry-block ( block | pad-block )* "}"
                | "{" asm-body "}"
asm-body      ::= "asm" string
entry-block   ::= Label ":" meta* instruction* terminator
block         ::= Label ( "(" block-params ")" )? ":" meta*
                    instruction* terminator
pad-block     ::= Label "pad" "(" register "ptr" "," register "i32" ")"
                    pad-clause+ ":" meta* instruction* terminator
block-params  ::= block-param ( "," block-param )*
block-param   ::= register reg-type
target        ::= Label ( "(" ( register ( "," register )* )? ")" )?
label-list    ::= Label ( "," Label )*

pad-clause    ::= "cleanup"
                | "catch"  GlobalName
                | "filter" "[" ( GlobalName ( "," GlobalName )* )? "]"
```

A pad block's two parameters are supplied by the personality routine, not by any
branch: the exception object pointer and the personality's selector value. This
is why the unwind edge of `invoke` is a bare `Label` and not a `target` — there
are no arguments for a branch to pass.

**A block parameter has four possible sources, and the block header does not
distinguish them.** A parameter is filled by a branch's argument list, by the
personality routine (a pad block's two), by an `invoke`'s results on its normal
edge, or by an `asm goto`'s outputs on its fallthrough edge (§14, §19.16). The
last two are why neither instruction has a result list of its own: a register
defined by an `invoke` would have to dominate both the normal and the unwind
edge, and on the unwind edge no call completed and no definition occurred.
Binding results as parameters of the target puts them exactly where they are
live and nowhere else.

The entry block takes no parameters; its inputs are the signature's parameter
registers. The entry block is never a pad block, and is never a branch target
(§19.17) — a `br` to it would re-enter a block whose inputs no branch can
supply.

**An `asm-body` is the other kind of body, and it is `naked` (§6).** The two
are the same fact stated twice, so the body implies the placement: a function
whose body is assembly has no prologue to emit, no epilogue to reach, and no
frame to lay out, which is all `naked` says. A `func-def` therefore has an
`asm-body` or blocks and never both, and a `naked` one always has the former.

An `asm-body` takes no operands, exactly as an `asm-decl` does, and for the
same reason: there is no register in scope to constrain. A parameter the
signature declares is reachable only where the calling convention put it,
which is what a naked function is written to know.

```vertex-ir
export func @_start() naked noreturn {
  asm "mov x0, sp\n\tbl __libc_start_main"
}
```

## 8. Instruction Shape

```ebnf
instruction  ::= ( inst-unary   | inst-binary  | inst-ternary | inst-const
                 | inst-load    | inst-store   | inst-subload | inst-substore
                 | inst-alloc   | inst-alloca
                 | inst-stacksave | inst-stackrestore
                 | inst-getaddr | inst-tlsaddr | inst-blockaddr
                 | inst-frameaddr
                 | inst-bulk
                 | inst-atomic-load  | inst-atomic-store
                 | inst-atomic-rmw   | inst-atomic-cas | inst-fence
                 | inst-call    | inst-callind
                 | inst-asm
                 | inst-vaarg   | inst-vaargref | inst-vamanage ) meta*

mem-attr     ::= "align" unsigned | "volatile"
ordering     ::= "unordered" | "monotonic" | "acquire" | "release"
               | "acq_rel"   | "seq_cst"
literal      ::= int | float | symconst
symconst     ::= "sizeof"   ( TypeName | GlobalName )
               | "alignof"  ( TypeName | GlobalName )
               | "offsetof" TypeName path
path         ::= ( "." name | "[" unsigned "]" )+
```

## 8b. Inline Assembly

```ebnf
inst-asm       ::= ( asm-outs "=" )? "asm" "volatile"? string
                    "(" asm-arg-list? ")" ( "clobber" string-list )?
asm-outs       ::= "(" out-param ( "," out-param )* ")"
out-param      ::= register reg-type constraint
asm-arg-list   ::= asm-arg ( "," asm-arg )*
asm-arg        ::= register constraint
constraint     ::= "reg" | "mem" | "imm" | string
string-list    ::= string ( "," string )*
```

The `string` form of `constraint` is the escape hatch for target-specific
constraint letters and tied operands. The clobber list admits the two
pseudo-registers `"memory"` and `"cc"` alongside target register names.

`asm` is the sole instruction without a fixed operand shape. Its terminator form
is `term-asm-goto` in §14.

**The template is GCC's.** An operand is referenced `%0`, `%1`, …, numbering the
outputs first and then the inputs, each in declaration order. `%%` is a literal
`%`. `%l[label]` names one of an `asm goto`'s labels by its block label. A letter
between the sigil and the digits — `%w0` on AArch64, `%b0` on x86 — is a
target-specific modifier selecting a view of the operand, admitted for the same
reason `constraint` admits a string. There are no named operands: an `out-param`
and an `asm-arg` have positions, not names.

This is GCC's spelling rather than a new one because the assembly this
instruction exists to carry is already written in it. A frontend lowering C's
`__asm__` passes the template through unchanged; any other convention would make
every frontend implement the same rewriter first.

Nothing at this layer parses the template. A backend that assembles it does, and
that is where a reference to an operand that does not exist is caught — this IR
checks the constraint list and the arity of the operand *lists*, not the body of
the string.

## 9. Value-Producing Arithmetic, Logic, and Conversion

```ebnf
inst-unary    ::= register "=" unary-verb register
inst-binary   ::= register "=" binary-verb register "," register
inst-ternary  ::= register "=" ternary-verb register "," register "," register
inst-const    ::= register "=" const-verb literal

unary-verb    ::= int-unary | float-unary | not-verb | bitcount-verb | conv-verb
binary-verb   ::= int-arith | mulhi-verb | overflow-verb | float-arith
                | bitwise-verb | shift-verb | cmp-verb | ptr-arith
ternary-verb  ::= float-ternary | select-verb

int-unary     ::= "i32.neg" | "i64.neg"
float-unary   ::= float-ns "." ( "neg" | "abs" | "sqrt"
                                | "ceil" | "floor" | "trunc" | "nearest" )
float-ns      ::= "f32" | "f64" | ext-float
not-verb      ::= "i1.not" | "i32.not" | "i64.not"
bitcount-verb ::= ( "i32" | "i64" ) "." ( "clz" | "ctz" | "popcnt" | "bswap" )

int-arith     ::= ( "i32" | "i64" ) "." ( "add" | "sub" | "mul"
                                        | "sdiv" | "udiv" | "srem" | "urem" )
mulhi-verb    ::= ( "i32" | "i64" ) "." ( "smulhi" | "umulhi" )
overflow-verb ::= ( "i32" | "i64" ) "." ( "saddo" | "uaddo" | "ssubo"
                                        | "smulo" | "umulo" )
float-arith   ::= float-ns "." ( "add" | "sub" | "mul" | "div"
                                | "minimum" | "maximum"
                                | "minnum"  | "maxnum"
                                | "copysign" )
float-ternary ::= float-ns ".fma"
bitwise-verb  ::= ( "i1" | "i32" | "i64" ) "." ( "and" | "or" | "xor" )
shift-verb    ::= ( "i32" | "i64" ) "." ( "shl" | "sshr" | "ushr"
                                        | "rotl" | "rotr" )
cmp-verb      ::= ( "i32" | "i64" ) "." ( "eq" | "ne" | "slt" | "ult"
                                        | "sle" | "ule" )
                | float-ns "." ( "eq" | "ne" | "lt" | "le" | "uno" )
                | "ptr" "." ( "eq" | "ne" | "lt" | "le" )
ptr-arith     ::= "ptr.add" | "ptr.sub" | "ptr.diff"

const-verb    ::= "i1.const" | "i32.const" | "i64.const"
                | float-ns ".const" | "ptr.const"
select-verb   ::= ( "i1" | "i32" | "i64" | "ptr" ) ".select"
                | float-ns ".select"

conv-verb     ::= int-conv | intfloat-conv | float-conv | ptr-conv
int-conv      ::= "i32.wrap_i64" | "i64.sext_i32" | "i64.zext_i32"
                | "i32.zext_i1"  | "i64.zext_i1"
intfloat-conv ::= float-ns "." ( "scvt_i32" | "scvt_i64"
                                | "ucvt_i32" | "ucvt_i64" )
                | ( "i32" | "i64" ) "." ( "scvt_" float-src
                                        | "ucvt_" float-src
                                        | "scvt_sat_" float-src
                                        | "ucvt_sat_" float-src )
float-src     ::= "f32" | "f64" | ext-float
float-conv    ::= "f64.fcvt_f32" | "f32.fcvt_f64"
                | ext-float ".fcvt_f32" | ext-float ".fcvt_f64"
                | "f32.fcvt_" ext-float | "f64.fcvt_" ext-float
                | "f128.fcvt_f80" | "f80.fcvt_f128"
                | "i32.bitcast_f32" | "f32.bitcast_i32"
                | "i64.bitcast_f64" | "f64.bitcast_i64"
ptr-conv      ::= "ptr.from_i64" | "i64.from_ptr"
```

`f128.fcvt_f80` and its inverse exist only on modules whose `layout` block's
`extfloat` list names both namespaces.

`cmp-verb` has no `i1` alternative: `i1.ne` is `i1.xor` and `i1.eq` is
`i1.not` of that. See the instruction index, §L.

## 10. Memory

```ebnf
inst-load          ::= register "=" load-verb register mem-attr*
inst-store         ::= store-verb register "," register mem-attr*
inst-subload       ::= register "=" subload-verb register mem-attr*
inst-substore      ::= substore-verb register "," register mem-attr*
inst-alloc         ::= register "=" "ptr.alloc" ( unsigned "align" unsigned
                                                | TypeName ) "zeroed"?
inst-alloca        ::= register "=" "ptr.alloca" register
                        "align" unsigned "zeroed"?
inst-stacksave     ::= register "=" "ptr.stacksave"
inst-stackrestore  ::= "ptr.stackrestore" register
inst-getaddr       ::= register "=" "ptr.getaddr" ( GlobalName | FuncName )
inst-tlsaddr       ::= register "=" "ptr.tlsaddr" GlobalName
inst-blockaddr     ::= register "=" "ptr.blockaddr" Label
inst-frameaddr     ::= register "=" ( "ptr.frameaddr" | "ptr.returnaddr" )

load-verb          ::= ( "i32" | "i64" | "f32" | "f64" | "ptr" ) ".load"
                     | ext-float ".load"
store-verb         ::= ( "i32" | "i64" | "f32" | "f64" | "ptr" ) ".store"
                     | ext-float ".store"
subload-verb       ::= "i32" "." ( "sload8" | "sload16" | "uload8" | "uload16" )
                     | "i64" "." ( "sload8" | "sload16" | "sload32"
                                 | "uload8" | "uload16" | "uload32" )
substore-verb      ::= "i32" "." ( "store8" | "store16" )
                     | "i64" "." ( "store8" | "store16" | "store32" )
```

## 11. Bulk Memory

```ebnf
inst-bulk   ::= ( "memcpy"  register "," register "," register
                | "memmove" register "," register "," register
                | "memset"  register "," register "," register
                | register "=" "memcmp" register "," register ","
                  register ) mem-attr*
```

## 12. Atomics

```ebnf
inst-atomic-load   ::= register "=" atomic-load-verb register ordering
                        "volatile"?
inst-atomic-store  ::= atomic-store-verb register "," register ordering
                        "volatile"?
inst-atomic-rmw    ::= register "=" rmw-verb register "," register ordering
                        "volatile"?
inst-atomic-cas    ::= register "=" cas-verb register "," register ","
                        register ordering ordering "volatile"?
inst-fence         ::= "fence" ordering "singlethread"?

narrow             ::= "8" | "16"
rmw-op             ::= "add" | "sub" | "and" | "or" | "xor" | "xchg"

atomic-load-verb   ::= ( "i32" | "i64" | "ptr" ) ".atomic_load"
                     | "i32" ".atomic_uload" narrow
atomic-store-verb  ::= ( "i32" | "i64" | "ptr" ) ".atomic_store"
                     | "i32" ".atomic_store" narrow
rmw-verb           ::= ( "i32" | "i64" ) ".atomic_rmw" rmw-op
                     | "i32" ".atomic_rmw" rmw-op narrow
                     | "ptr" ".atomic_rmw" ( "add" | "sub" | "xchg" )
cas-verb           ::= ( "i32" | "i64" | "ptr" ) ".atomic_cas"
                     | "i32" ".atomic_cas" narrow
```

Narrow atomics live in the `i32` namespace alone. An `i64` set would be reachable
by zero-extension with no hardware distinction, and is omitted under the
derivability rule.

**`singlethread` is a compiler barrier and nothing else.** It orders this
thread's accesses against this thread's own interrupted execution — a signal
handler, a scheduler on the same core — and emits no machine barrier. It exists
because C11's `atomic_signal_fence` is otherwise inexpressible: lowering it as
an ordinary `fence` is a correctness-preserving pessimization on every use, and
the alternative every frontend reaches for, `asm volatile ("" ::: "memory")`,
is opaque to the optimizer in ways a fence is not. The token is optional and
appears on `fence` alone; no access carries it.

## 13. Calls

```ebnf
inst-call     ::= ( result-list "=" )? "call" FuncName "(" arg-list? ")"
inst-callind  ::= ( result-list "=" )? "callind" register ":" TypeName
                   "(" arg-list? ")"
result-list   ::= register ( "," register )*
arg-list      ::= register ( "," register )*
```

`invoke` and `invokeind` (§14) have no `result-list`. Their results arrive as
parameters of the normal target block; see §7 and §19.16.

## 14. Terminators

```ebnf
terminator   ::= ( "br" target
                 | "brif" register "," target "," target
                 | "br_table" register "," "[" target-list? "]" "," target
                 | "brind" register "," "[" label-list "]"
                 | "return" ( register ( "," register )* )?
                 | "trap"
                 | "invoke" FuncName "(" arg-list? ")"
                   "to" target "unwind" Label
                 | "invokeind" register ":" TypeName "(" arg-list? ")"
                   "to" target "unwind" Label
                 | "resume" register
                 | term-asm-goto ) meta*
target-list  ::= target ( "," target )*

term-asm-goto ::= ( asm-goto-outs "=" )? "asm" "goto" string
                   "(" asm-arg-list? ")" ( "clobber" string-list )?
                   "to" target "," "[" label-list "]"
asm-goto-outs ::= "(" asm-goto-out ( "," asm-goto-out )* ")"
asm-goto-out  ::= reg-type constraint
```

**An `invoke`'s results are the trailing parameters of its normal target.** The
`to` target's argument list carries values the branch supplies, in the ordinary
way; the callee's results are appended after them, one per `ret-item`, and the
target block declares a parameter for each. A call returning nothing appends
nothing, and the normal edge is then an ordinary branch. §19.16 states the
arity rule. This is why `invoke` needs no result list of its own — and why it
must not have one, since a register it defined would not dominate the unwind
edge.

```vertex-ir
@try:
  invoke @get_int(%a) to @ok(%tag) unwind @pad

@ok(%tag i32, %r i32):        // %tag from the branch, %r from the call
  return %r

@pad pad (%exn ptr, %sel i32) cleanup:
  resume %exn
```

**`asm goto` is implicitly volatile, and its outputs are the trailing
parameters of its fallthrough target.** This is `invoke`'s rule for the same
shape and the same reason: a register the terminator defined would have to
dominate every edge, and on an edge the assembled text branched along, the
text did not reach the end that writes the output. The fallthrough edge is the
one on which it did — which is what GCC means when it says an asm goto's
outputs are valid on the fallthrough path.

So an `asm-goto-out` declares a type and a constraint and names no register:
the value is the target block's parameter. §19.16's arity rule covers it, and
the fallthrough target may have no other predecessor, since the outputs arrive
on that edge and on no other.

`br_table`'s selector is `i32`. A wider switch requires a frontend-emitted range
check, which the frontend must emit regardless in order to select the default
edge.

A `brind`'s targets take no parameters, and so do the labels of an `asm goto`:
an edge the assembled text branches along carries no argument list this IR
could write. The fallthrough target is the exception, and the outputs are what
it takes.

## 15. Variadics

```ebnf
inst-vaarg     ::= register "=" vaarg-verb register
inst-vaargref  ::= register "=" "ptr.va_arg_ref" register "," TypeName
inst-vamanage  ::= "va_start" register
                 | "va_end"   register
                 | "va_copy"  register "," register
vaarg-verb     ::= ( "i32" | "i64" | "f64" | "ptr" ) ".va_arg"
                 | ext-float ".va_arg"
```

There is no `f32.va_arg`, and no `i8`/`i16` form, because the default argument
promotions mean no such argument can be present.

## 16. Metadata

```ebnf
meta-decl     ::= metaname "=" meta-node
meta          ::= metaname meta-arg*
meta-node     ::= "{" meta-arg-list? "}"
meta-arg-list ::= meta-arg ( "," meta-arg )*
meta-arg      ::= unsigned | int | float | string | symbol | ident
                | metaname | meta-node
```

Metadata nodes may reference one another. This is what makes a debug-information
graph — scope trees, type DAGs, location chains — expressible without a further
grammar change.

Metadata attaches to instructions, terminators, block headers, type
declarations, global declarations, imports, and function definitions.

## 17. Mnemonic Index (non-normative)

A mnemonic is either `Type.Verb`, parameterized over a machine type, or bare. The
bare set, for reference:

```text
br  brif  br_table  brind  return  trap  resume
call  callind  invoke  invokeind
asm
fence
memcpy  memmove  memset  memcmp
va_start  va_end  va_copy
```

Everything else is namespaced. Module scope uses plain keywords with no dots —
including `module`, `use`, and `layout` themselves.

## 18. Reserved Space

None of the following requires a grammar change beyond the noted production.

| Axis | Arrival |
| --- | --- |
| Atomic min/max | `rmw-op` gains `smin`, `smax`, `umin`, `umax`. |
| Atomic nand | `rmw-op` gains `nand`. |
| Sub-word / pointer atomic coverage | New `rmw-verb` and `cas-verb` rows; no production shape changes. |
| Function memory effects | `func-placement` and `import-placement` gain members (`readnone`, `readonly`, `argmemonly`). |
| Pointer parameter facts | `param-attr` gains members (`nonnull`, `dereferenceable`, `align`). |
| Tail calls | `inst-call` gains a `"tail"?`. |
| Named sync scopes | `"singlethread"` generalizes to a named-scope token on `inst-fence`. |
| New metadata kinds | New `!ident` names; no production changes. |
| Wider extended floats | `ext-float` gains a member; `layout`'s `extfloat` attribute admits it. |

```ebnf
```

Every row above is additive to a list. None changes how any text conforming to
this document is read, which is why none of them needed to land before it
froze.

## 19. Well-Formedness Beyond the Grammar

The grammar admits these; a verifier rejects them.

1. Every register is assigned exactly once and its definition dominates every use.
2. Every block ends in exactly one terminator; only the entry block may be
   unreachable-from-nothing.
3. A branch's argument list matches its target block's parameter list in arity
   and type. For an `invoke`'s normal edge, see rule 16.
4. `invoke`/`invokeind`'s unwind edge names a pad block; a pad block is reachable
   only by unwind edges; a function containing either declares a `personality`.
5. `resume` takes the `ptr` parameter of a pad block that dominates it.
6. `ptr.alloc` appears in the entry block only. `ptr.blockaddr`'s label is a
   target of some `brind` in the same function.
7. `ptr.tlsaddr` names a global in domain `tls`; `tlsmodel` appears only on such
   globals.
8. `align N` on an access is a power of two not exceeding the access width.
9. `atomic_cas`'s failure ordering is no stronger than its success ordering and is
   neither `release` nor `acq_rel`. Read-modify-write and compare-and-swap
   orderings are not `unordered`.
10. A global initializer's structure matches its declared type exactly — arity,
    nesting, and element widths.
11. `dllimport` appears only on imports and `dllexport` only on definitions;
    `common` appears only on globals in domain `rw`.
12. A module naming an `ext-float` namespace absent from its `layout` block is
    rejected, not emulated.
13. `sret` appears on at most one parameter, which is the first.
14. A `callind`'s `TypeName` resolves to a `func` typedef; its argument arity and
    types match that signature.
15. A file's `module`, `use`, and `layout` declarations each appear exactly
    once and precede every `module-item`; a module missing any of the three is
    rejected before any other check runs.
16. An `invoke`'s normal target declares exactly *k + n* parameters, where *k* is
    the arity of the edge's argument list and *n* is the number of `ret-item`s in
    the callee's signature. The first *k* match the arguments in type; the last
    *n* match the results in type and order. A block may be the normal target of
    at most one `invoke`, and a block that is an `invoke`'s normal target has no
    other predecessor — otherwise a parameter would be supplied by a call on one
    edge and by a branch on another.

    An `asm goto`'s fallthrough target is the same rule with *n* the number of
    declared outputs, and has the same one-predecessor restriction when there
    are any. The labels are not targets in this sense: they take no parameters,
    because the branch to one is in the assembled text and carries nothing.
17. The entry block is not the target of any `br`, `brif`, `br_table`, `brind`,
    `invoke`, `invokeind`, or `asm goto`, and is not named by `ptr.blockaddr`.
    Its inputs are the signature's parameter registers, which no branch can
    supply.
18. Within one `struct`, either every field carries `at` or none does. Where
    they do, offsets are strictly increasing, no field overlaps its successor,
    and each offset satisfies its field type's alignment unless the struct is
    `packed`. `at` does not appear on a `union`'s fields.
19. A `naked` function's body is an `asm-body` (§7). The grammar cannot state
    this, since `naked` is a `func-placement` and the body is a separate
    production; a `naked` function with blocks would be asking a backend to
    lower instructions into a function with no frame to lower them against.
