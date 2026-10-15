package ir

// Strength reduction of division, remainder and multiplication by a power
// of two.
//
// A CPU backend selects shifts for these itself. Apple's GPU compiler,
// reading AIR, does not: an i64 sdiv by 2 is its 64-bit division routine,
// paid every time it runs -- in a kernel's index arithmetic, in the loop
// a wave reduction halves its stride by. ReducePowersOfTwo rewrites them
// before the AIR is written:
//
//	udiv x, 2^k  ->  ushr x, k          urem x, 2^k  ->  and x, 2^k-1
//	mul  x, 2^k  ->  shl x, k
//	sdiv x, 2^k  ->  sshr x, k          when x is known not negative,
//	                 sshr (x + (sshr x, w-1 ushr w-k)), k  otherwise
//	srem x, 2^k  ->  and x, 2^k-1       when x is known not negative,
//	                 x - shl (sdiv x, 2^k), k  otherwise
//
// A value is known not negative when its bits are bounded below its width:
// a zero-extension, a constant, and the arithmetic of such values whose
// bound stays under the width (bitBound). A division by a non-zero
// constant cannot trap, so nothing is lost by the rewrite.

// ReducePowersOfTwo rewrites f's integer division, remainder and
// multiplication by powers of two, and reports how many it rewrote.
func (f *Func) ReducePowersOfTwo() int {
	if f == nil || len(f.blocks) == 0 || f.m == nil || f.m.err != nil {
		return 0
	}
	n := 0
	repl := map[*Def]*Def{}
	for _, b := range f.blocks {
		out := make([]*Inst, 0, len(b.insts))
		emit := func(t RegType, v Verb, im *imm, args ...*Def) *Def {
			in := &Inst{op: Op{t, v}, blk: b, args: args, im: im}
			d := f.newDef(t, "", in, 0)
			in.results = []*Def{d}
			out = append(out, in)
			return d
		}
		con := func(t RegType, v int64) *Def {
			return emit(t, VConst, &imm{lit: Int(v), hasLit: true})
		}
		for _, in := range b.insts {
			if r := reducePow2(in, emit, con); r != nil {
				repl[in.results[0]] = r
				n++
				continue
			}
			out = append(out, in)
		}
		b.insts = out
	}
	f.replaceDefs(repl)
	return n
}

func reducePow2(in *Inst, emit func(RegType, Verb, *imm, ...*Def) *Def, con func(RegType, int64) *Def) *Def {
	t := in.op.Type
	if (t != TypeI32 && t != TypeI64) || len(in.args) != 2 || len(in.results) != 1 {
		return nil
	}
	w := 32
	if t == TypeI64 {
		w = 64
	}
	x, y := in.args[0], in.args[1]
	c, ok := constOf(y)
	if !ok && in.op.Verb == VMul {
		if c2, ok2 := constOf(x); ok2 {
			x, c, ok = y, c2, true
		}
	}
	if !ok || c <= 1 || c&(c-1) != 0 {
		return nil
	}
	k := int64(0)
	for v := c; v > 1; v >>= 1 {
		k++
	}
	if int(k) >= w {
		return nil
	}
	nonNeg := bitBound(x, w, 8) < w
	switch in.op.Verb {
	case VUDiv:
		return emit(t, VUShr, nil, x, con(t, k))
	case VURem:
		return emit(t, VAnd, nil, x, con(t, c-1))
	case VMul:
		return emit(t, VShl, nil, x, con(t, k))
	case VSDiv:
		if nonNeg {
			return emit(t, VSShr, nil, x, con(t, k))
		}
		return sdivPow2(t, w, k, x, emit, con)
	case VSRem:
		if nonNeg {
			return emit(t, VAnd, nil, x, con(t, c-1))
		}
		q := sdivPow2(t, w, k, x, emit, con)
		return emit(t, VSub, nil, x, emit(t, VShl, nil, q, con(t, k)))
	}
	return nil
}

// sdivPow2 is x / 2^k rounded towards zero: a negative x is biased by
// 2^k - 1 first.
func sdivPow2(t RegType, w int, k int64, x *Def, emit func(RegType, Verb, *imm, ...*Def) *Def, con func(RegType, int64) *Def) *Def {
	sign := emit(t, VSShr, nil, x, con(t, int64(w-1)))
	bias := emit(t, VUShr, nil, sign, con(t, int64(w)-k))
	return emit(t, VSShr, nil, emit(t, VAdd, nil, x, bias), con(t, k))
}

// bitBound is a bound on how many low bits of d can be set, treating d as
// unsigned, or w when nothing is known: a result under w is a value that
// is not negative as a w-bit signed integer. depth limits the search.
func bitBound(d *Def, w, depth int) int {
	if d == nil {
		return w
	}
	if c, ok := constOf(d); ok {
		if c < 0 {
			return w
		}
		n := 0
		for v := c; v > 0; v >>= 1 {
			n++
		}
		return n
	}
	if depth == 0 || d.inst == nil || len(d.inst.results) != 1 {
		return w
	}
	in := d.inst
	arg := func(i int) int {
		if i >= len(in.args) {
			return w
		}
		return bitBound(in.args[i], w, depth-1)
	}
	clamp := func(n int) int {
		if n > w {
			return w
		}
		return n
	}
	switch in.op.Verb {
	case VZExtI32:
		return 32
	case VZExtI1:
		return 1
	case VAnd:
		a, b := arg(0), arg(1)
		if a < b {
			return a
		}
		return b
	case VOr, VXor:
		a, b := arg(0), arg(1)
		if a > b {
			return a
		}
		return b
	case VAdd:
		a, b := arg(0), arg(1)
		if a > b {
			return clamp(a + 1)
		}
		return clamp(b + 1)
	case VMul:
		return clamp(arg(0) + arg(1))
	case VUShr, VSShr:
		a := arg(0)
		if in.op.Verb == VSShr && a >= w {
			return w
		}
		if s, ok := constOf(in.args[1]); ok && s >= 0 {
			if a-int(s) < 0 {
				return 0
			}
			return a - int(s)
		}
		return a
	case VShl:
		if s, ok := constOf(in.args[1]); ok && s >= 0 {
			return clamp(arg(0) + int(s))
		}
		return w
	case VUDiv:
		return arg(0)
	case VURem:
		a, b := arg(0), arg(1)
		if a < b {
			return a
		}
		return b
	case VSelect:
		a, b := arg(1), arg(2)
		if a > b {
			return a
		}
		return b
	}
	return w
}
