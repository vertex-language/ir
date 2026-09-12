package arm64_test

import (
	"testing"

	"github.com/vertex-language/ir"
)

// §B's pointer lt and le compare addresses, which have no sign. The
// pointers here straddle the top bit so that a signed compare would get
// every row backwards.
func TestRunPointerCompare(t *testing.T) {
	m := ir.NewModule("t", ir.AArch64Linux)
	for _, row := range []struct {
		name string
		le   bool
	}{{"_below", false}, {"_at_or_below", true}} {
		fn := m.Func(row.name).Export()
		p := fn.ParamPtr("p")
		q := fn.ParamPtr("q")
		fn.ReturnsI32()
		e := fn.Entry()
		c := e.Ptr.Lt(p, q)
		if row.le {
			c = e.Ptr.Le(p, q)
		}
		e.Return(e.I32.ZExtI1(c))
	}

	got := runNative(t, m, `
#include <stdio.h>
#include <stdint.h>
int below(void *, void *);
int at_or_below(void *, void *);
int main(void) {
	void *lo = (void *)(uintptr_t)0x10, *hi = (void *)(uintptr_t)0x8000000000000010ull;
	printf("%d%d%d %d%d%d\n",
		below(lo, hi), below(hi, lo), below(lo, lo),
		at_or_below(lo, hi), at_or_below(hi, lo), at_or_below(lo, lo));
	return 0;
}
`)
	if want := "100 101\n"; got != want {
		t.Errorf("printed %q, want %q", got, want)
	}
}
