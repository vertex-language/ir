package arm64

// AAPCS64 §5.4's classification of a composite argument.
//
// ir.RegType has no aggregate in it, so an aggregate reaches a signature the
// way the ABI passes one: a pointer carrying byval, naming the type. Honouring
// that attribute is the algorithm here, and it is not SysV's. There are no
// eightbyte classes and no per-field merge; there are three questions asked in
// order, and the first one that answers decides.
//
//  1. Is it a homogeneous floating-point aggregate — one to four members, all
//     of the same fundamental float type, however deeply they are nested or
//     how many arrays they are spread across? Then one SIMD register each.
//     §5.4 puts no size limit on this: three doubles is twenty-four bytes and
//     still travels in D0, D1 and D2.
//  2. Otherwise, is it sixteen bytes or less? Then ceil(size/8) consecutive
//     general-purpose registers, which is one or two.
//  3. Otherwise the caller copies it and passes the address of the copy in
//     one general-purpose register.
//
// The third case is what this backend did for every aggregate, which is why a
// program compiled entirely by this tree worked and nothing else could call
// it: a private convention is self-consistent.

import (
	"fmt"

	"github.com/vertex-language/ir"
)

// An aggKind is which of §5.4's three answers a composite got.
type aggKind uint8

const (
	// aggEmpty is an aggregate with no bytes. It occupies no register and
	// no stack slot — there is nothing to copy, so it is not memory either.
	aggEmpty aggKind = iota
	aggGPR
	aggHFA
	aggIndirect
)

// maxHFAMembers is §5.4's limit on a homogeneous aggregate. A fifth member
// makes it an ordinary composite, decided by size like any other.
const maxHFAMembers = 4

// maxRegisterAggregate is the largest non-homogeneous composite that travels
// in registers rather than by reference.
const maxRegisterAggregate = 16

// An aggregate is how §5.4 passes one byval parameter.
type aggregate struct {
	kind  aggKind
	size  uint64
	align uint64

	// n is how many registers it occupies, w is the width of each, and step
	// is the distance between the bytes each one carries. For a homogeneous
	// aggregate those are the member's, so four floats are four S registers
	// four bytes apart; for the general case they are a doubleword.
	n    int
	w    width
	step uint64
}

// classifyAggregate runs §5.4 over one type.
func classifyAggregate(t ir.FType) (aggregate, error) {
	size, align, err := sizeAlign(t)
	if err != nil {
		return aggregate{}, err
	}
	agg := aggregate{size: size, align: align}
	if size == 0 {
		return agg, nil // aggEmpty
	}

	// f128 is the one member type this package cannot put in a register:
	// its widths are S and D, and a Q register is not among them. An
	// aggregate containing one keeps the by-reference passing it has always
	// had here rather than being classified into registers that cannot hold
	// it — wrong against clang either way, and this way it is wrong the way
	// it already was. See lower.go's list.
	if has, err := hasF128(t); err != nil {
		return aggregate{}, err
	} else if has {
		agg.kind, agg.n, agg.w, agg.step = aggIndirect, 1, w64, 8
		return agg, nil
	}

	if s, n, ok, err := hfaOf(t); err != nil {
		return aggregate{}, err
	} else if ok && n >= 1 && n <= maxHFAMembers {
		w, step, err := hfaWidth(s)
		if err != nil {
			return aggregate{}, err
		}
		agg.kind, agg.n, agg.w, agg.step = aggHFA, n, w, step
		return agg, nil
	}

	if size > maxRegisterAggregate {
		agg.kind, agg.n, agg.w, agg.step = aggIndirect, 1, w64, 8
		return agg, nil
	}

	agg.kind = aggGPR
	agg.n = int((size + 7) / 8)
	agg.w, agg.step = w64, 8
	return agg, nil
}

// hfaOf reports whether t is homogeneous, and in what: every leaf the same
// fundamental float type, counted across nesting and arrays.
//
// A union counts as its widest member rather than the sum, because that is
// what a union is — its members share the storage, so a union of two floats
// is one member's worth and travels in one register.
func hfaOf(t ir.FType) (ir.StoreType, int, bool, error) { return hfaAt(t, 0) }

func hfaAt(t ir.FType, depth int) (ir.StoreType, int, bool, error) {
	if depth > maxNesting {
		return 0, 0, false, fmt.Errorf("type nests more than %d deep", maxNesting)
	}
	t = resolveAlias(t)

	switch t.Kind() {
	case ir.FTypeScalar:
		switch s := t.Scalar(); s {
		case ir.StoreF32, ir.StoreF64, ir.StoreF128:
			return s, 1, true, nil
		}
		return 0, 0, false, nil

	case ir.FTypeArray:
		if t.Len() == 0 {
			return 0, 0, false, nil
		}
		s, n, ok, err := hfaAt(t.Elem(), depth+1)
		if err != nil || !ok {
			return 0, 0, false, err
		}
		return s, n * int(t.Len()), true, nil

	case ir.FTypeNamed:
		named := t.Named()
		if named == nil {
			return 0, 0, false, fmt.Errorf("a named type with no definition")
		}
		union := false
		switch named.Kind() {
		case ir.KindStruct:
		case ir.KindUnion:
			union = true
		default:
			return 0, 0, false, nil
		}
		fields := named.Fields()
		if len(fields) == 0 {
			return 0, 0, false, nil
		}
		var base ir.StoreType
		total := 0
		for i, f := range fields {
			s, n, ok, err := hfaAt(f.Type, depth+1)
			if err != nil || !ok {
				return 0, 0, false, err
			}
			if i == 0 {
				base = s
			} else if s != base {
				return 0, 0, false, nil
			}
			if union {
				if n > total {
					total = n
				}
				continue
			}
			total += n
		}
		return base, total, true, nil
	}
	return 0, 0, false, nil
}

// hfaWidth is the register one member of a homogeneous aggregate occupies,
// and the distance to the next.
func hfaWidth(s ir.StoreType) (width, uint64, error) {
	switch s {
	case ir.StoreF32:
		return wf32, 4, nil
	case ir.StoreF64:
		return wf64, 8, nil
	}
	return 0, 0, fmt.Errorf("a homogeneous aggregate of %s needs a register wider than D, which this package does not have", s)
}

// hasF128 reports whether any leaf of t is a quadword float.
func hasF128(t ir.FType) (bool, error) { return hasF128At(t, 0) }

func hasF128At(t ir.FType, depth int) (bool, error) {
	if depth > maxNesting {
		return false, fmt.Errorf("type nests more than %d deep", maxNesting)
	}
	t = resolveAlias(t)

	switch t.Kind() {
	case ir.FTypeScalar:
		return t.Scalar() == ir.StoreF128, nil
	case ir.FTypeArray:
		return hasF128At(t.Elem(), depth+1)
	case ir.FTypeNamed:
		named := t.Named()
		if named == nil {
			return false, fmt.Errorf("a named type with no definition")
		}
		for _, f := range named.Fields() {
			has, err := hasF128At(f.Type, depth+1)
			if err != nil || has {
				return has, err
			}
		}
	}
	return false, nil
}
