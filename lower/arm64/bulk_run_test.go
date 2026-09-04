package arm64_test

// §E, which is four calls to the C library function of the same name.
//
// The IR's signatures were chosen to match the C ones, so what these check is
// really the call: that AAPCS64 puts three arguments where the library expects
// them, that the frame record is saved for a call the module never wrote, and
// that memcmp's result comes back.

import (
	"testing"

	"github.com/vertex-language/ir"
	arm64lower "github.com/vertex-language/ir/lower/arm64"
)

func TestRunBulkMemory(t *testing.T) {
	m := ir.NewModule("t", ir.AArch64Linux)

	cp := m.Func("_bcpy").Export()
	cpD := cp.ParamPtr("d")
	cpS := cp.ParamPtr("s")
	cpN := cp.ParamI64("n")
	e := cp.Entry()
	e.MemCpy(cpD, cpS, cpN)
	e.Return()

	mv := m.Func("_bmove").Export()
	mvD := mv.ParamPtr("d")
	mvS := mv.ParamPtr("s")
	mvN := mv.ParamI64("n")
	e2 := mv.Entry()
	e2.MemMove(mvD, mvS, mvN)
	e2.Return()

	st := m.Func("_bset").Export()
	stD := st.ParamPtr("d")
	stV := st.ParamI32("v")
	stN := st.ParamI64("n")
	e3 := st.Entry()
	e3.MemSet(stD, stV, stN)
	e3.Return()

	cm := m.Func("_bdiff").Export()
	cmA := cm.ParamPtr("a")
	cmB := cm.ParamPtr("b")
	cmN := cm.ParamI64("n")
	cm.ReturnsI32()
	e4 := cm.Entry()
	e4.Return(e4.MemCmp(cmA, cmB, cmN))

	// A bulk operation with a live value around it, which is what makes
	// the call a call rather than a tail of the function.
	mix := m.Func("_bmix").Export()
	mixD := mix.ParamPtr("d")
	mixS := mix.ParamPtr("s")
	mixK := mix.ParamI64("k")
	mix.ReturnsI64()
	e5 := mix.Entry()
	e5.MemCpy(mixD, mixS, e5.I64.Const(16))
	e5.Return(e5.I64.Add(mixK, e5.I64.Const(1)))

	got := runNative(t, m, `
#include <stdio.h>
#include <string.h>
void bcpy(void*,void*,long), bmove(void*,void*,long), bset(void*,int,long);
int bdiff(void*,void*,long);
long bmix(void*,void*,long);
static int fail = 0;
static void chk(const char *what, long got, long want) {
    if (got != want) { printf("%s: got %ld want %ld\n", what, got, want); fail = 1; }
}
int main(void) {
    char src[32], dst[32];
    for (int i = 0; i < 32; i++) src[i] = (char)(i + 1);
    memset(dst, 0, sizeof dst);
    bcpy(dst, src, 20);
    chk("memcpy", memcmp(dst, src, 20), 0);
    chk("memcpy stopped", dst[20], 0);

    // Overlapping, forwards and backwards, which is the whole point of
    // memmove being a different verb.
    char ov[16];
    for (int i = 0; i < 16; i++) ov[i] = (char)i;
    bmove(ov + 4, ov, 8);
    for (int i = 0; i < 8; i++) chk("memmove up", ov[4 + i], i);
    for (int i = 0; i < 16; i++) ov[i] = (char)i;
    bmove(ov, ov + 4, 8);
    for (int i = 0; i < 8; i++) chk("memmove down", ov[i], i + 4);

    char fill[16];
    memset(fill, 7, sizeof fill);
    // Only the low byte of the value is written, which is C's rule too.
    bset(fill, 0x1234ab, 10);
    for (int i = 0; i < 10; i++) chk("memset", (unsigned char)fill[i], 0xab);
    chk("memset stopped", (unsigned char)fill[10], 7);

    char a[8] = "abcdefg", b[8] = "abcdefg", c[8] = "abcdefx";
    chk("memcmp eq", bdiff(a, b, 7), 0);
    if (bdiff(a, c, 7) >= 0) { printf("memcmp lt: got %d\n", bdiff(a, c, 7)); fail = 1; }
    if (bdiff(c, a, 7) <= 0) { printf("memcmp gt: got %d\n", bdiff(c, a, 7)); fail = 1; }
    // A zero length touches nothing and compares equal.
    chk("memcmp zero", bdiff(a, c, 0), 0);

    memset(dst, 0, sizeof dst);
    chk("mix result", bmix(dst, src, 41), 42);
    chk("mix copied", memcmp(dst, src, 16), 0);
    printf("%s\n", fail ? "MISMATCH" : "ok");
    return 0;
}
`)
	if got != "ok\n" {
		t.Errorf("printed %q, want %q", got, "ok\n")
	}
}

// A volatile bulk operation is refused rather than lowered: volatile is a
// promise about how the bytes are touched and memcpy makes no such promise,
// so the call is the wrong lowering rather than a slow one.
func TestLowerRefusesVolatileBulk(t *testing.T) {
	m := ir.NewModule("t", ir.AArch64Linux)
	fn := m.Func("f").Export()
	d := fn.ParamPtr("d")
	s := fn.ParamPtr("s")
	n := fn.ParamI64("n")
	entry := fn.Entry()
	entry.MemCpy(d, s, n, ir.Volatile)
	entry.Return()

	if _, err := arm64lower.Lower(m, arm64lower.Options{}); err == nil {
		t.Error("Lower should refuse a volatile memcpy")
	}
}
