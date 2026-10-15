package ir

// Folding instructions whose operands are all constants.
//
// A frontend lowers `x << 4` as the language defines it: a shift by a
// negative amount, or by the width or more, has its own answer, and each
// case is a branch on a comparison of the amount with the width. The
// amount is usually a literal, so every comparison is of two constants,
// but nothing downstream of the frontend evaluates them: Apple's GPU
// compiler, reading AIR, keeps all four arms of every shift. FoldConstants
// evaluates such instructions -- integer arithmetic, bitwise operations,
// shifts in range, comparisons, extensions and wraps -- and a select whose
// condition is constant, so NumberValues can settle the branches on them
// and MergeBlocks straighten what is left.

// FoldConstants replaces each instruction it can evaluate with a constant,
// or with the operand a constant select picks, and reports how many.
func (f *Func) FoldConstants() int {
	if f == nil || len(f.blocks) == 0 || f.m == nil || f.m.err != nil {
		return 0
	}
	n := 0
	for {
		repl := map[*Def]*Def{}
		resolve := func(d *Def) *Def {
			for k := 0; k <= len(repl); k++ {
				r, ok := repl[d]
				if !ok {
					return d
				}
				d = r
			}
			return d
		}
		for _, b := range f.blocks {
			out := make([]*Inst, 0, len(b.insts))
			for _, in := range b.insts {
				for i, a := range in.args {
					in.args[i] = resolve(a)
				}
				if len(in.results) == 1 {
					if in.op.Verb == VSelect && len(in.args) == 3 {
						if c, ok := constOf(in.args[0]); ok {
							pick := in.args[2]
							if c != 0 {
								pick = in.args[1]
							}
							repl[in.results[0]] = pick
							continue
						}
					}
					if v, ok := foldInst(in); ok {
						c := &Inst{op: Op{in.results[0].typ, VConst}, blk: b, im: &imm{lit: Int(v), hasLit: true}}
						d := f.newDef(in.results[0].typ, "", c, 0)
						c.results = []*Def{d}
						out = append(out, c)
						repl[in.results[0]] = d
						continue
					}
				}
				out = append(out, in)
			}
			b.insts = out
		}
		if len(repl) == 0 {
			return n
		}
		n += len(repl)
		f.replaceDefs(repl)
	}
}

// width is the bits of an integer register type, or 0.
func width(t RegType) int {
	switch t {
	case TypeI1:
		return 1
	case TypeI32:
		return 32
	case TypeI64:
		return 64
	}
	return 0
}

// norm is v as a w-bit integer is kept in a constant: sign-extended.
func norm(v int64, w int) int64 {
	switch w {
	case 1:
		return v & 1
	case 32:
		return int64(int32(v))
	}
	return v
}

// foldInst evaluates in when its operands are integer constants.
func foldInst(in *Inst) (int64, bool) {
	if len(in.results) != 1 || len(in.args) == 0 || len(in.args) > 2 {
		return 0, false
	}
	w := width(in.op.Type)
	rw := width(in.results[0].typ)
	if w == 0 || rw == 0 {
		return 0, false
	}
	var a, b int64
	var ok bool
	if a, ok = constOf(in.args[0]); !ok {
		return 0, false
	}
	a = norm(a, width(in.args[0].typ))
	if len(in.args) == 2 {
		if b, ok = constOf(in.args[1]); !ok {
			return 0, false
		}
		b = norm(b, width(in.args[1].typ))
	}
	mask := uint64(1)<<uint(w) - 1
	if w == 64 {
		mask = ^uint64(0)
	}
	ua, ub := uint64(a)&mask, uint64(b)&mask
	bit := func(c bool) (int64, bool) {
		if c {
			return 1, true
		}
		return 0, true
	}
	switch in.op.Verb {
	case VAdd:
		return norm(a+b, w), true
	case VSub:
		return norm(a-b, w), true
	case VMul:
		return norm(a*b, w), true
	case VAnd:
		return norm(a&b, w), true
	case VOr:
		return norm(a|b, w), true
	case VXor:
		return norm(a^b, w), true
	case VShl:
		if b < 0 || b >= int64(w) {
			return 0, false
		}
		return norm(int64(ua<<uint(b)), w), true
	case VUShr:
		if b < 0 || b >= int64(w) {
			return 0, false
		}
		return norm(int64(ua>>uint(b)), w), true
	case VSShr:
		if b < 0 || b >= int64(w) {
			return 0, false
		}
		return norm(a>>uint(b), w), true
	case VEq:
		return bit(ua == ub)
	case VNe:
		return bit(ua != ub)
	case VSLt:
		return bit(a < b)
	case VSLe:
		return bit(a <= b)
	case VULt:
		return bit(ua < ub)
	case VULe:
		return bit(ua <= ub)
	case VSExtI32:
		if len(in.args) == 1 {
			return int64(int32(a)), true
		}
	case VZExtI32:
		if len(in.args) == 1 {
			return int64(uint32(a)), true
		}
	case VZExtI1:
		if len(in.args) == 1 {
			return a & 1, true
		}
	case VWrapI64:
		if len(in.args) == 1 {
			return int64(int32(a)), true
		}
	}
	return 0, false
}
