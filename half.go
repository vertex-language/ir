package ir

import "math"

// HalfBits is v rounded to nearest even in the half format t (TypeF16 or
// TypeBF16), as that format's sixteen bits. It is the rounding
// LegalizeHalf's instructions perform, done here for the literals it
// folds and for a backend encoding a half initializer.
func HalfBits(t RegType, v float64) uint16 {
	f := math.Float32bits(toOdd32(v))
	if t == TypeBF16 {
		if math.IsNaN(v) {
			return uint16(f>>16) | 0x40
		}
		return uint16((f + 0x7fff + (f>>16)&1) >> 16)
	}
	sign := f & 0x80000000
	f ^= sign
	var o uint32
	switch {
	case f >= 0x47800000: // Inf, NaN, and everything that rounds past 65504
		o = 0x7c00
		if f > 0x7f800000 {
			o = 0x7e00 | (f>>13)&0x1ff
		}
	case f < 0x38800000: // f16's subnormals and zero
		o = math.Float32bits(math.Float32frombits(f)+0.5) - 0x3f000000
	default:
		o = (f + 0xc8000fff + (f>>13)&1) >> 13
	}
	return uint16(o | sign>>16)
}

// HalfValue is the value of a half format's bits, exactly.
func HalfValue(t RegType, h uint16) float64 {
	if t == TypeBF16 {
		return float64(math.Float32frombits(uint32(h) << 16))
	}
	sign := 1.0
	if h&0x8000 != 0 {
		sign = -1
	}
	e, m := int(h>>10)&0x1f, float64(h&0x3ff)
	switch e {
	case 0:
		return sign * math.Ldexp(m, -24)
	case 0x1f:
		if m != 0 {
			return math.NaN()
		}
		return math.Inf(int(sign))
	}
	return sign * math.Ldexp(1024+m, e-25)
}

// toOdd32 rounds v to float32 towards zero, setting the lowest bit when
// anything was lost: the step that lets a second rounding be the only one.
func toOdd32(v float64) float32 {
	r := float32(v)
	if math.IsNaN(v) || float64(r) == v {
		return r
	}
	b := math.Float32bits(r)
	if math.Abs(float64(r)) > math.Abs(v) {
		b--
	}
	return math.Float32frombits(b | 1)
}

func inf64() float64 { return math.Inf(1) }
