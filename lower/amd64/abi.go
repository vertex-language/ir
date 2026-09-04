package amd64

// §3.2.3's classification of an aggregate.
//
// ir.RegType has no aggregate in it, so one reaches a signature the way
// the ABI passes it: a pointer carrying byval. Honouring that attribute
// is the algorithm here — cut the aggregate into eightbytes, classify
// every field into the one it lands in, merge the classes that share
// one, and the answer is a short register list or MEMORY.

import (
	"fmt"

	"github.com/vertex-language/ir"
)

// An abiClass is §3.2.3's classification of one eightbyte — the four
// that can come out of the types this package lays out. X87 and SSEUP
// are refused before classification rather than represented, neither
// being a value this package holds in a register.
type abiClass uint8

const (
	classNone    abiClass = iota // no field has reached this eightbyte yet
	classInteger                 // a general-purpose register
	classSSE                     // a vector register
	classMemory                  // the whole aggregate goes on the stack
)

// maxEightbytes is how large an aggregate can be and still be passed in
// registers. §3.2.3 says anything larger is MEMORY, and for the field
// types this package admits that threshold is two eightbytes.
const maxEightbytes = 2

// merge is §3.2.3's rule for two fields sharing one eightbyte. The
// asymmetry is the point: INTEGER wins over SSE, because a value that is
// half integer cannot be half in each file and the ABI picks the one
// that holds arbitrary bits.
func (c abiClass) merge(d abiClass) abiClass {
	switch {
	case c == d:
		return c
	case c == classNone:
		return d
	case d == classNone:
		return c
	case c == classMemory || d == classMemory:
		return classMemory
	case c == classInteger || d == classInteger:
		return classInteger
	}
	return classSSE
}

// An aggregate is how §3.2.3 passes one byval parameter: either a short
// list of eightbyte classes, or memory.
type aggregate struct {
	size  uint64
	align uint64

	// classes is one entry per eightbyte when the aggregate travels in
	// registers, and empty when it does not.
	classes []abiClass
}

// inMemory reports whether the whole aggregate goes on the stack.
func (a aggregate) inMemory() bool { return len(a.classes) == 0 }

// classifyAggregate runs §3.2.3 over one type.
func classifyAggregate(t ir.FType) (aggregate, error) {
	size, align, err := sizeAlign(t)
	if err != nil {
		return aggregate{}, err
	}
	agg := aggregate{size: size, align: align}

	// An empty aggregate has nothing to pass and no eightbyte to pass it
	// in. It is not MEMORY — there are no bytes to copy — so it occupies
	// no register and no stack slot at all.
	if size == 0 {
		agg.classes = []abiClass{}
		return agg, nil
	}
	if size > maxEightbytes*8 {
		return agg, nil // MEMORY, by size alone
	}

	classes := make([]abiClass, (size+7)/8)
	if err := classifyInto(classes, t, 0, 0); err != nil {
		return aggregate{}, err
	}
	for _, c := range classes {
		if c == classMemory {
			return agg, nil
		}
	}
	agg.classes = classes
	return agg, nil
}

// classifyInto walks t's scalar leaves and folds each into the eightbyte
// its offset lands in.
//
// base is where t begins inside the aggregate being classified, which is
// what makes this recursive: a nested struct's fields are classified at
// their absolute offsets, not their offsets within the nest. §3.2.3 has
// no notion of a field being inside something — it has bytes, and which
// eightbyte each byte is in.
func classifyInto(classes []abiClass, t ir.FType, base uint64, depth int) error {
	if depth > maxNesting {
		return fmt.Errorf("type nests more than %d deep", maxNesting)
	}
	t = resolveAlias(t)

	switch t.Kind() {
	case ir.FTypeScalar:
		return classifyScalar(classes, t.Scalar(), base)

	case ir.FTypeArray:
		elem := t.Elem()
		size, _, err := sizeAlign(elem)
		if err != nil {
			return err
		}
		for i := uint64(0); i < t.Len(); i++ {
			if err := classifyInto(classes, elem, base+i*size, depth+1); err != nil {
				return err
			}
		}
		return nil

	case ir.FTypeNamed:
		named := t.Named()
		if named == nil {
			return fmt.Errorf("a named type with no definition")
		}
		switch named.Kind() {
		case ir.KindStruct:
			offs, err := fieldOffsets(named)
			if err != nil {
				return err
			}
			for i, f := range named.Fields() {
				if err := classifyInto(classes, f.Type, base+offs[i], depth+1); err != nil {
					return err
				}
			}
			return nil
		case ir.KindUnion:
			// Every member at zero, and all of them classified: a union
			// is whichever was last written, so it travels somewhere
			// that would hold any.
			for _, f := range named.Fields() {
				if err := classifyInto(classes, f.Type, base, depth+1); err != nil {
					return err
				}
			}
			return nil
		}
		return fmt.Errorf("@%s is a %v, which has no storage layout", named.Name(), named.Kind())
	}
	return fmt.Errorf("%s has no layout", t)
}

// classifyScalar folds one leaf field into the eightbytes it covers.
func classifyScalar(classes []abiClass, s ir.StoreType, off uint64) error {
	size, align, err := scalarSizeAlign(s)
	if err != nil {
		return err
	}

	switch s {
	case ir.StoreF80, ir.StoreF128:
		// X87 and SSEUP. Both are representable in the ABI and neither
		// is a value this package holds in a register, so an aggregate
		// containing one cannot be passed in registers by a package that
		// could not then read it back out.
		return fmt.Errorf("an aggregate field of type %s is not classified yet", s)
	}

	// A field straddling an eightbyte boundary makes the whole aggregate
	// MEMORY: a register pair has nowhere to put the half in each. Only
	// a stated field offset can arrange it.
	if off%align != 0 || off/8 != (off+size-1)/8 {
		for i := range classes {
			classes[i] = classMemory
		}
		return nil
	}

	c := classInteger
	if s == ir.StoreF32 || s == ir.StoreF64 {
		c = classSSE
	}
	i := off / 8
	if i >= uint64(len(classes)) {
		return fmt.Errorf("a field at offset %d is outside the %d bytes the aggregate occupies", off, len(classes)*8)
	}
	classes[i] = classes[i].merge(c)
	return nil
}
