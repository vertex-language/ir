package globals

import (
	"math"
	"testing"
)

// The widening to binary128, against encodings worked out from the format
// rather than from the function: a 15-bit exponent biased by 16383, and a
// 112-bit mantissa whose top 52 bits are the double's.
func TestFloat128Bits(t *testing.T) {
	for _, tc := range []struct {
		name   string
		v      float64
		lo, hi uint64
	}{
		{"zero", 0, 0, 0},
		{"negative zero", math.Copysign(0, -1), 0, 1 << 63},
		{"one", 1, 0, 0x3fff000000000000},
		{"two", 2, 0, 0x4000000000000000},
		{"minus one and a half", -1.5, 0, 0xbfff800000000000},
		{"half", 0.5, 0, 0x3ffe000000000000},
		{"infinity", math.Inf(1), 0, 0x7fff000000000000},
		{"negative infinity", math.Inf(-1), 0, 0xffff000000000000},
		// A quiet NaN: the exponent stays all ones and the payload's
		// leading bit lands at the top of the wider mantissa.
		{"quiet nan", math.Float64frombits(0x7ff8_0000_0000_0000), 0, 0x7fff800000000000},
		// A payload rides across with it, four bits into the low word.
		{"nan payload", math.Float64frombits(0x7ff8_0000_0000_0001),
			1 << 60, 0x7fff800000000000},
		// The smallest normal double, which is an ordinary normal here.
		{"smallest normal", math.Float64frombits(1 << 52), 0, 0x3c01000000000000},
		// And the smallest subnormal, which is too: 2^-1074 is well
		// inside binary128's range, so normalizing it costs 52 of the
		// exponent's headroom and nothing else.
		{"smallest subnormal", math.Float64frombits(1), 0, 0x3bcd000000000000},
		// A mantissa that straddles the two words: its low four bits
		// are the top four of the low one.
		{"straddling mantissa", math.Float64frombits(0x3ff0_0000_0000_000f),
			0xf000000000000000, 0x3fff000000000000},
	} {
		lo, hi := float128Bits(tc.v)
		if lo != tc.lo || hi != tc.hi {
			t.Errorf("%s: float128Bits(%v) = %016x_%016x, want %016x_%016x",
				tc.name, tc.v, hi, lo, tc.hi, tc.lo)
		}
	}
}
