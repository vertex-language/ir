package air

import (
	"fmt"
	"strings"

	am "github.com/vertex-language/air"
	"github.com/vertex-language/ir"
)

// fn is one kernel being lowered.
type fn struct {
	l *lowerer
	f *ir.Func
	k *am.Func

	space  map[*ir.Def]am.Space
	vals   map[*ir.Def]am.Value
	blocks map[*ir.Block]*am.Block
	phis   map[*ir.Def]*am.Inst
	cur    *am.Block // where instructions go: a VIR block's, or a block a lowering split it into

	builtin map[ir.Verb]*am.Param // the built-in arguments, by the verb they answer
	shared  map[ir.Symbol]*am.Param
	// casSlot is each compare-and-swap's expected value's thread memory,
	// allocated in the entry block so that a loop does not grow the frame.
	casSlot map[*ir.Inst]*am.Inst
}

func byValOf(p ir.Param) (*ir.Type, bool) {
	for _, a := range p.Attrs {
		if a.IsByVal() {
			return a.Type(), true
		}
	}
	return nil, false
}

// builtins is which built-in argument answers each work-item verb, in the
// order they are added.
var builtins = []struct {
	verb ir.Verb
	name string
	b    am.Builtin
	vec  bool // a uint3, read by axis
}{
	{ir.VWorkitemID, "thread_position_in_threadgroup", am.ThreadPositionInThreadgroup, true},
	{ir.VWorkgroupID, "threadgroup_position_in_grid", am.ThreadgroupPositionInGrid, true},
	// The dispatched size, not the thread's own group's: VIR's workgroup
	// size is the same in every group, as CUDA's blockDim is, and Metal's
	// last group of a grid the groups do not divide is smaller.
	{ir.VWorkgroupSize, "dispatch_threads_per_threadgroup", am.DispatchThreadsPerThreadgroup, true},
	{ir.VNumWorkgroups, "threadgroups_per_grid", am.ThreadgroupsPerGrid, true},
	{ir.VLaneID, "thread_index_in_simdgroup", am.ThreadIndexInSimdgroup, false},
	{ir.VWaveSize, "threads_per_simdgroup", am.ThreadsPerSimdgroup, false},
}

func (l *lowerer) kernel(f *ir.Func) error {
	if _, asm := f.AsmBodyText(); asm {
		return fmt.Errorf("a kernel whose body is assembly has nothing to lower")
	}
	// The frontend's locals are frame slots; as SSA values, a pointer kept
	// in one has the space of what was stored, which inference can see.
	// A struct's slot is split into its fields first, so a pointer kept
	// in a struct is promoted too.
	f.SplitSlots()
	f.PromoteSlots()
	bind, err := bindingsOf(f)
	if err != nil {
		return err
	}
	space, err := l.inferSpaces(f, bind)
	if err != nil {
		return err
	}
	x := &fn{l: l, f: f, k: l.out.Kernel(kernelName(f.Name())), space: space,
		vals: map[*ir.Def]am.Value{}, blocks: map[*ir.Block]*am.Block{}, phis: map[*ir.Def]*am.Inst{},
		builtin: map[ir.Verb]*am.Param{}, shared: map[ir.Symbol]*am.Param{}, casSlot: map[*ir.Inst]*am.Inst{}}
	if n, ok := launchBound(f, "max_workgroup_size"); ok {
		x.k.MaxTotalThreadsPerThreadgroup(int(n))
	}

	// Arguments: every parameter, then the built-ins and threadgroup
	// buffers the body uses. AIR wants them all before the body.
	type scalar struct {
		p *am.Param
		d *ir.Def
	}
	var scalars []scalar
	var byval []scalar
	var builtinVecs []scalar // vector built-ins, which the body reads through thread memory
	sig := f.Signature().Params()
	for i, d := range f.Params() {
		name := d.Name()
		if name == "" {
			name = fmt.Sprintf("arg%d", i)
		}
		at := bind[i].index
		if bi := bind[i]; bi.builtin != 0 {
			if bi.n == 1 {
				x.vals[d] = x.k.Builtin(name, am.UInt, bi.builtin)
				continue
			}
			builtinVecs = append(builtinVecs, scalar{x.k.Builtin(name, am.Vec(bi.elem, bi.n), bi.builtin), d})
			continue
		}
		switch d.Type() {
		case ir.TypePtr:
			if t, ok := byValOf(sig[i]); ok {
				size, _, err := sizeAlign(t.FType())
				if err != nil {
					return fmt.Errorf("parameter %d: %w", i, err)
				}
				byval = append(byval, scalar{x.k.ConstantBuffer(name, am.Array(am.Char, int(size)), at), d})
				continue
			}
			if bind[i].space == am.Constant {
				x.vals[d] = x.k.ConstantBuffer(name, am.Char, at)
				continue
			}
			x.vals[d] = x.k.Buffer(name, am.Char, at, am.ReadWrite)
		case ir.TypeI1:
			scalars = append(scalars, scalar{x.k.ConstantRef(name, am.UChar, at), d})
		case ir.TypeI32:
			scalars = append(scalars, scalar{x.k.ConstantRef(name, am.UInt, at), d})
		case ir.TypeI64:
			scalars = append(scalars, scalar{x.k.ConstantRef(name, am.ULong, at), d})
		case ir.TypeF32:
			scalars = append(scalars, scalar{x.k.ConstantRef(name, am.Float, at), d})
		default:
			return fmt.Errorf("parameter %d is %s, which an Apple GPU does not have", i, d.Type())
		}
	}
	used := map[ir.Verb]bool{}
	usedShared := map[ir.Symbol]bool{}
	f.WalkInsts(func(in *ir.Inst) bool {
		used[in.Op().Verb] = true
		if in.Op().Verb == ir.VGetAddr {
			if g, ok := in.Symbol().(*ir.GlobalImport); ok && g.Domain() == ir.Shared {
				usedShared[g] = true
			}
		}
		return true
	})
	for _, bi := range builtins {
		if !used[bi.verb] {
			continue
		}
		t := am.Type(am.UInt)
		if bi.vec {
			t = am.Vec(am.UInt, 3)
		}
		x.builtin[bi.verb] = x.k.Builtin(bi.name, t, bi.b)
	}
	for _, g := range l.m.GlobalImports() {
		if usedShared[g] {
			x.shared[g] = x.k.ThreadgroupBuffer(g.Name(), am.Char, l.shared[g])
		}
	}

	// Blocks, in reverse postorder, so that a value is lowered before any
	// use of it outside a phi; a block no path reaches is left out. And a
	// phi for every block parameter.
	order := f.RPO()
	for _, b := range order {
		if b.IsEntry() {
			x.blocks[b] = x.k.Entry()
			continue
		}
		if b.IsPad() {
			return fmt.Errorf("@%s is a landing pad: there is no unwinding on the GPU", b.Label())
		}
		x.blocks[b] = x.k.Block(b.Label())
	}
	for _, b := range order {
		for _, p := range b.Params() {
			t, err := x.typ(p)
			if err != nil {
				return fmt.Errorf("@%s: %w", b.Label(), err)
			}
			phi := x.blocks[b].Phi(t)
			if p.Name() != "" {
				phi.SetName(p.Name())
			}
			x.phis[p] = phi
			x.vals[p] = phi
		}
	}

	// The prologue: scalar arguments read from their constant buffers, and
	// by-value aggregates' constant bytes as the pointer every other is.
	entry := x.k.Entry()
	for _, s := range byval {
		x.vals[s.d] = entry.Bitcast(s.p, am.Ptr(am.Constant, am.Char))
	}
	for _, s := range builtinVecs {
		slot := entry.Alloca(s.p.Type())
		entry.Store(s.p, slot)
		x.vals[s.d] = entry.Bitcast(slot, am.Ptr(am.Thread, am.Char))
	}
	f.WalkInsts(func(in *ir.Inst) bool {
		if in.Op().Verb == ir.VAtomicCas {
			x.casSlot[in] = entry.Alloca(am.UInt)
		}
		return true
	})
	for _, s := range scalars {
		v := am.Value(entry.Load(s.p))
		if s.d.Type() == ir.TypeI1 {
			v = entry.ICmp(am.NE, v, am.ConstUint(am.UChar, 0))
		}
		x.vals[s.d] = v
	}

	for _, b := range order {
		x.cur = x.blocks[b]
		for _, in := range b.Insts() {
			if err := x.inst(in); err != nil {
				return fmt.Errorf("@%s: %s: %w", b.Label(), in.Op(), err)
			}
		}
		if err := x.term(b.Term()); err != nil {
			return fmt.Errorf("@%s: %s: %w", b.Label(), b.Term().Op(), err)
		}
	}
	return nil
}

// A binding is where a kernel parameter is bound: its [[buffer(n)]]
// index, and for a pointer the space it points into.
type binding struct {
	index int
	space am.Space

	// builtin is a parameter that is not bound but filled by Metal: one
	// of MSL's built-in arguments, [[thread_position_in_grid]] and the
	// rest, declared as the type it has (uint, ushort2, uint3). A scalar
	// is an i32 parameter; a vector is a by-value one, its components in
	// consecutive words or halfwords.
	builtin am.Builtin
	elem    *am.IntType
	n       int
}

// mslBuiltins are the built-in arguments a frontend may name, by their
// MSL attribute.
var mslBuiltins = map[string]am.Builtin{
	"thread_position_in_grid":        am.ThreadPositionInGrid,
	"thread_position_in_threadgroup": am.ThreadPositionInThreadgroup,
	"threadgroup_position_in_grid":   am.ThreadgroupPositionInGrid,
	"threads_per_threadgroup":        am.ThreadsPerThreadgroup,
	"threadgroups_per_grid":          am.ThreadgroupsPerGrid,
	"threads_per_grid":               am.ThreadsPerGrid,
	"thread_index_in_threadgroup":    am.ThreadIndexInThreadgroup,
	"thread_index_in_simdgroup":      am.ThreadIndexInSimdgroup,
	"simdgroup_index_in_threadgroup": am.SimdgroupIndexInThreadgroup,
	"simdgroups_per_threadgroup":     am.SimdgroupsPerThreadgroup,
	"threads_per_simdgroup":          am.ThreadsPerSimdgroup,
	"thread_execution_width":         am.ThreadsPerSimdgroup,

	"dispatch_threads_per_threadgroup": am.DispatchThreadsPerThreadgroup,
}

// bindingsOf is each parameter's binding. With no attachments, parameter
// i is [[buffer(i)]], and a pointer is into device memory -- a by-value
// aggregate's is constant. A frontend that knows better says so:
//
//	!binding 0, 2, -                  each parameter's [[buffer(n)]]
//	!param_space device, constant, -  where each pointer points
//	!param_builtin -, -, "thread_position_in_grid uint2"
//	                                  a built-in argument, and its MSL type
//
// which is what vcx writes for a .metal kernel, whose source names all
// three. A - is a parameter the attachment says nothing about.
func bindingsOf(f *ir.Func) ([]binding, error) {
	params := f.Params()
	out := make([]binding, len(params))
	for i, d := range params {
		out[i] = binding{index: i, space: am.Device}
		if d.Type() == ir.TypePtr {
			if _, byval := byValOf(f.Signature().Params()[i]); byval {
				out[i].space = am.Constant
			}
		}
	}
	for _, a := range f.Attached() {
		switch a.Name {
		case "binding", "param_space", "param_builtin":
		default:
			continue
		}
		if len(a.Args) != len(params) {
			return nil, fmt.Errorf("!%s has %d entries for %d parameters", a.Name, len(a.Args), len(params))
		}
		seen := map[int]int{}
		for i, arg := range a.Args {
			if (arg.Kind() == ir.MetaIdent || arg.Kind() == ir.MetaString) && arg.String_() == "-" {
				continue
			}
			if a.Name == "param_builtin" {
				if err := out[i].setBuiltin(arg, params[i], f.Signature().Params()[i]); err != nil {
					return nil, fmt.Errorf("parameter %d: %w", i, err)
				}
				continue
			}
			if a.Name == "binding" {
				if arg.Kind() != ir.MetaInt || arg.Int() < 0 || arg.Int() > 30 {
					return nil, fmt.Errorf("!binding entry %d is not a buffer index (0 to 30)", i)
				}
				n := int(arg.Int())
				if j, dup := seen[n]; dup {
					return nil, fmt.Errorf("parameters %d and %d are both [[buffer(%d)]]", j, i, n)
				}
				seen[n] = i
				out[i].index = n
				continue
			}
			switch arg.Kind() {
			case ir.MetaIdent, ir.MetaString:
			default:
				return nil, fmt.Errorf("!param_space entry %d is not a space name", i)
			}
			switch arg.String_() {
			case "device":
				out[i].space = am.Device
			case "constant":
				out[i].space = am.Constant
			default:
				return nil, fmt.Errorf("!param_space entry %d is %q: a buffer is device or constant", i, arg.String_())
			}
		}
	}
	return out, nil
}

// setBuiltin reads a !param_builtin entry: the attribute and the type,
// "thread_position_in_grid uint3".
func (b *binding) setBuiltin(arg ir.MetaArg, d *ir.Def, p ir.Param) error {
	if arg.Kind() != ir.MetaString && arg.Kind() != ir.MetaIdent {
		return fmt.Errorf("!param_builtin names a built-in and its type")
	}
	name, typ, _ := strings.Cut(arg.String_(), " ")
	which, ok := mslBuiltins[name]
	if !ok {
		return fmt.Errorf("[[%s]] is not a built-in argument this backend binds", name)
	}
	b.builtin = which
	b.space = am.Thread
	b.elem, b.n = am.UInt, 1
	base := strings.TrimRight(typ, "234")
	switch base {
	case "uint", "":
	case "ushort":
		b.elem = am.UShort
	default:
		return fmt.Errorf("[[%s]] is uint or ushort, or a vector of them, not %s", name, typ)
	}
	if len(base) < len(typ) {
		fmt.Sscanf(typ[len(base):], "%d", &b.n)
	}
	_, byval := byValOf(p)
	switch {
	case b.n == 1 && d.Type() != ir.TypeI32:
		return fmt.Errorf("[[%s]] %s is an i32 parameter, not %s", name, typ, d.Type())
	case b.n > 1 && !byval:
		return fmt.Errorf("[[%s]] %s is a by-value parameter holding its components", name, typ)
	}
	return nil
}

// kernelName is a kernel's name as Metal looks it up: VIR's, where it is
// already an identifier.
func kernelName(s string) string { return s }

// launchBound is a launch-bound attachment's value, as vcx writes
// __launch_bounds__: !max_workgroup_size 256.
func launchBound(f *ir.Func, name string) (uint64, bool) {
	for _, a := range f.Attached() {
		if a.Name != name || len(a.Args) != 1 {
			continue
		}
		if n := a.Args[0].Int(); a.Args[0].Kind() == ir.MetaInt && n > 0 {
			return uint64(n), true
		}
	}
	return 0, false
}

// typ is a def's AIR type. A pointer is an i8 in its inferred space.
func (x *fn) typ(d *ir.Def) (am.Type, error) {
	return x.regType(d.Type(), x.space[d])
}

func (x *fn) regType(t ir.RegType, sp am.Space) (am.Type, error) {
	switch t {
	case ir.TypeI1:
		return am.I1, nil
	case ir.TypeI32:
		return am.UInt, nil
	case ir.TypeI64:
		return am.ULong, nil
	case ir.TypeF32:
		return am.Float, nil
	case ir.TypePtr:
		return am.Ptr(sp, am.Char), nil
	case ir.TypeF64, ir.TypeF80, ir.TypeF128:
		return nil, fmt.Errorf("%s: Apple GPUs have no floating point wider than 32 bits", t)
	case ir.TypeV128:
		return nil, fmt.Errorf("v128 is not lowered for AIR yet")
	}
	return nil, fmt.Errorf("%s has no AIR type", t)
}

// v is a def's AIR value.
func (x *fn) v(d *ir.Def) am.Value {
	v, ok := x.vals[d]
	if !ok {
		panic(fmt.Sprintf("air: %v used before it is lowered", d))
	}
	return v
}

func (x *fn) arg(in *ir.Inst, i int) am.Value { return x.v(in.Arg(i)) }

// def records an instruction's result.
func (x *fn) def(in *ir.Inst, v am.Value) {
	r := in.Result(0)
	x.vals[r] = v
	if n := r.Name(); n != "" {
		if i, ok := v.(*am.Inst); ok && i.Name == "" {
			i.SetName(n)
		}
	}
}
