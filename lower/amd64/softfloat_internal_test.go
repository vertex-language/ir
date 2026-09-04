package amd64

import "testing"

// The binary128 encodings the §C2 range check compares against, worked out
// from the format rather than from the function: a 15-bit exponent biased by
// 16383, and the leading one of the value dropped into the exponent with what
// is under it at the top of the 112-bit mantissa.
//
// Two of these are why the integer path exists at all. 2^63+1 is not a double
// — it needs 64 significant bits and a double keeps 53 — and 2^64 is not a
// uint64, so neither bound can be reached by widening a float64 or by naming
// a magnitude. The first is a low bound and the second a high one.
func TestF128IntegerEncodings(t *testing.T) {
	for _, tc := range []struct {
		name           string
		gotLo, gotHi   uint64
		wantLo, wantHi uint64
	}{
		{"zero", lo(f128FromUint64(false, 0)), hi(f128FromUint64(false, 0)), 0, 0},
		{"negative zero", lo(f128FromUint64(true, 0)), hi(f128FromUint64(true, 0)), 0, 1 << 63},
		{"one", lo(f128FromUint64(false, 1)), hi(f128FromUint64(false, 1)), 0, 0x3fff000000000000},
		{"minus one", lo(f128FromUint64(true, 1)), hi(f128FromUint64(true, 1)), 0, 0xbfff000000000000},
		{"two", lo(f128FromUint64(false, 2)), hi(f128FromUint64(false, 2)), 0, 0x4000000000000000},
		{"three", lo(f128FromUint64(false, 3)), hi(f128FromUint64(false, 3)), 0, 0x4000800000000000},
		{"2^31", lo(f128FromUint64(false, 1<<31)), hi(f128FromUint64(false, 1<<31)),
			0, 0x401e000000000000},
		// 2^31+1: the mantissa's one bit lands at 112−31 = 81, which is
		// bit 17 of the high word.
		{"-(2^31+1)", lo(f128FromUint64(true, 1<<31+1)), hi(f128FromUint64(true, 1<<31+1)),
			0, 0xc01e000000020000},
		{"2^63", lo(f128FromUint64(false, 1<<63)), hi(f128FromUint64(false, 1<<63)),
			0, 0x403e000000000000},
		// 2^63+1: bit 112−63 = 49, which is in the low word.
		{"-(2^63+1)", lo(f128FromUint64(true, 1<<63+1)), hi(f128FromUint64(true, 1<<63+1)),
			1 << 49, 0xc03e000000000000},
		{"pow2 31", lo(f128Pow2(31)), hi(f128Pow2(31)), 0, 0x401e000000000000},
		{"pow2 32", lo(f128Pow2(32)), hi(f128Pow2(32)), 0, 0x401f000000000000},
		{"pow2 63", lo(f128Pow2(63)), hi(f128Pow2(63)), 0, 0x403e000000000000},
		{"pow2 64", lo(f128Pow2(64)), hi(f128Pow2(64)), 0, 0x403f000000000000},
	} {
		if tc.gotLo != tc.wantLo || tc.gotHi != tc.wantHi {
			t.Errorf("%s = %016x_%016x, want %016x_%016x",
				tc.name, tc.gotHi, tc.gotLo, tc.wantHi, tc.wantLo)
		}
	}

	// The two paths into binary128 have to agree wherever both can go: a
	// value a double expresses exactly must encode the same either way.
	for _, v := range []uint64{0, 1, 2, 3, 7, 1 << 31, 1 << 52, 1<<53 - 1, 1 << 62} {
		aLo, aHi := f128FromUint64(false, v)
		bLo, bHi := f64ToF128Bits(float64(v))
		if aLo != bLo || aHi != bHi {
			t.Errorf("%d: from the integer %016x_%016x, from the double %016x_%016x",
				v, aHi, aLo, bHi, bLo)
		}
	}
}

// The intervals themselves, as the four rows §C2 has here.
func TestF128FixRange(t *testing.T) {
	for _, tc := range []struct {
		name           string
		to             width
		signed         bool
		wantLo, wantHi uint64 // the high words, which is where the exponent is
	}{
		{"i32 signed", w32, true, 0xc01e000000020000, 0x401e000000000000},
		{"i64 signed", w64, true, 0xc03e000000000000, 0x403e000000000000},
		{"i32 unsigned", w32, false, 0xbfff000000000000, 0x401f000000000000},
		{"i64 unsigned", w64, false, 0xbfff000000000000, 0x403f000000000000},
	} {
		_, loHi, _, hiHi := f128FixRange(tc.to, tc.signed)
		if loHi != tc.wantLo || hiHi != tc.wantHi {
			t.Errorf("%s = (%016x, %016x), want (%016x, %016x)",
				tc.name, loHi, hiHi, tc.wantLo, tc.wantHi)
		}
	}
}

func lo(lo, _ uint64) uint64 { return lo }
func hi(_, hi uint64) uint64 { return hi }
