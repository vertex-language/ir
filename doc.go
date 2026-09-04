// Package ir implements Vertex IR (VIR): a typed SSA intermediate
// representation for native AOT and JIT compilation.
//
// A *Module is everything a frontend decided and nothing a backend will
// decide. Instructions are named, types are fixed, control flow is explicit —
// and no register is physical, no address is known, no instruction has been
// selected. The normative specification is spec/; this package is the Go
// surface over it.
//
// # Building
//
// Modules are built by calling methods. module, use and layout are constructor
// arguments, since the grammar admits each exactly once ahead of every module
// item:
//
//	m := ir.NewModule("demo", ir.X86_64Linux)
//	fn := m.Func("sum").Export().NoUnwind()
//	p := fn.ParamPtr("p", ir.NoAlias)
//	n := fn.ParamI64("n")
//	fn.ReturnsI32()
//
//	entry := fn.Entry()
//	entry.Return(entry.I32.Const(0))
//
// Blocks are declared, parameterized, then filled. A block freezes its
// parameter list at its first instruction. There are no phi nodes and no
// predecessor-indexed operand lists; predecessors are computed by walk.go
// rather than stored.
//
// # One Go type per reg-type
//
// I1, I32, I64, F32, F64, Ptr, and F80/F128 are the SSA value types. There is
// no I8 or I16: §2 makes those storage-only widths, reachable exactly through
// the sub-width load and store verbs and nowhere else.
//
// Type.Verb is namespace value, method name: i32.add is b.I32.Add, i64.sext_i32
// is b.I64.SExtI32, f64.minnum is b.F64.MinNum. A mnemonic the spec does not
// have has no method — there is no Gt, no Ge, no I32.USubO, no register-width
// I8.Mul. Every §L omission is a compile error rather than a runtime refusal.
// Bare mnemonics are builder methods: Br, BrTable, MemCpy, Fence, Call.
//
// Extended-float availability is a run-time property of the layout block, so
// F80 and F128 are methods rather than fields: b.F80() on a module whose layout
// omits it records ErrLayout. That is the Go-level analogue of §19.12's
// rejected, not emulated.
//
// # Errors
//
// Errors are sticky and first-wins. Every builder call after a failure is a
// no-op, so a frontend emitting IR is not a run of if err != nil. Module.Err
// reports the first failure, running any deferred checks (branch argument arity
// and type against block parameters) before it answers.
//
// Wrong operand types, missing verbs, and store operand order are compile
// errors. Everything a finished module reveals — dominance, terminators, pad
// reachability, initializer structure — belongs to ir/verify, which is a client
// of this package's public surface.
//
// This package imports nothing.
package ir
