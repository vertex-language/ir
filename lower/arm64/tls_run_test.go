package arm64_test

import (
	"testing"

	"github.com/vertex-language/ir"
	arm64lower "github.com/vertex-language/ir/lower/arm64"
)

// Thread-local storage, under Mach-O's model.
//
// A thread-local is two symbols there: a template, which no instruction
// names, and a three-word descriptor, which every access goes through.
// The access is a call — the descriptor's thunk is the loader's, and it
// hands back the address of this thread's copy — so what these check is
// not only the number that comes out but that the call was made with the
// register the model names and the frame a call needs.

// TestRunThreadLocal reads and writes one from the C side, which is the
// whole chain: the descriptor is emitted, the linker resolves the offset
// field against the template, and dyld's thunk finds the block.
func TestRunThreadLocal(t *testing.T) {
	m := ir.NewModule("t", ir.AArch64MacOS)
	g := m.Global("_counter", ir.TLS, ir.StoreI32.FType()).Init(ir.Lit(ir.Int(7))).Export()

	get := m.Func("_get").Export()
	get.ReturnsI32()
	e := get.Entry()
	e.Return(e.I32.Load(e.Ptr.TLSAddr(g)))

	add := m.Func("_add").Export()
	n := add.ParamI32("n")
	add.ReturnsI32()
	ae := add.Entry()
	p := ae.Ptr.TLSAddr(g)
	sum := ae.I32.Add(ae.I32.Load(p), n)
	ae.I32.Store(sum, p)
	ae.Return(sum)

	got := runNative(t, m, `
#include <stdio.h>
int get(void); int add(int);
int main(void) { int a = get(); int b = add(35); printf("%d %d\n", a, b); return 0; }
`)
	if want := "7 42\n"; got != want {
		t.Errorf("printed %q, want %q", got, want)
	}
}

// TestRunThreadLocalZeroed: a template with no initializer goes to
// __thread_bss rather than __thread_data, and still has a descriptor.
// Emitting it as an ordinary zerofill global would leave every thread
// reading the same word.
func TestRunThreadLocalZeroed(t *testing.T) {
	m := ir.NewModule("t", ir.AArch64MacOS)
	g := m.Global("_blank", ir.TLS, ir.StoreI64.FType()).Export()

	fn := m.Func("_bump").Export()
	fn.ReturnsI64()
	e := fn.Entry()
	p := e.Ptr.TLSAddr(g)
	v := e.I64.Add(e.I64.Load(p), e.I64.Const(21))
	e.I64.Store(v, p)
	e.Return(v)

	got := runNative(t, m, `
#include <stdio.h>
long bump(void);
int main(void) { long a = bump(); long b = bump(); printf("%ld %ld\n", a, b); return 0; }
`)
	if want := "21 42\n"; got != want {
		t.Errorf("printed %q, want %q", got, want)
	}
}

// TestRunThreadLocalIsPerThread is the property the whole model exists
// for, and the one a single-threaded test cannot see: a descriptor whose
// offset field held an address rather than an offset still reads and
// writes memory, and it is one location shared by every thread.
func TestRunThreadLocalIsPerThread(t *testing.T) {
	m := ir.NewModule("t", ir.AArch64MacOS)
	g := m.Global("_slot", ir.TLS, ir.StoreI32.FType()).Init(ir.Lit(ir.Int(7))).Export()

	set := m.Func("_set").Export()
	n := set.ParamI32("n")
	se := set.Entry()
	se.I32.Store(n, se.Ptr.TLSAddr(g))
	se.Return()

	get := m.Func("_get").Export()
	get.ReturnsI32()
	ge := get.Entry()
	ge.Return(ge.I32.Load(ge.Ptr.TLSAddr(g)))

	got := runNative(t, m, `
#include <stdio.h>
#include <pthread.h>
void set(int); int get(void);

static void *worker(void *arg) {
    (void)arg;
    /* The thread starts from the template, not from what main stored. */
    if (get() != 7) return (void *)1;
    set(99);
    if (get() != 99) return (void *)2;
    return (void *)0;
}

int main(void) {
    set(1000);
    pthread_t th; void *r;
    pthread_create(&th, 0, worker, 0);
    pthread_join(th, &r);
    /* main's own copy is untouched by the thread. */
    printf("%d %d\n", (int)(long)r, get());
    return 0;
}
`)
	if want := "0 1000\n"; got != want {
		t.Errorf("printed %q, want %q", got, want)
	}
}

// TestThreadLocalNeedsAModel: a target this backend has no thread-local
// model for is refused rather than given storage nothing can reach. The
// bytes and the sequence that finds them are one feature.
func TestThreadLocalNeedsAModel(t *testing.T) {
	m := ir.NewModule("t", ir.AArch64Linux)
	g := m.Global("_counter", ir.TLS, ir.StoreI32.FType()).Init(ir.Lit(ir.Int(7))).Export()
	fn := m.Func("_get").Export()
	fn.ReturnsI32()
	e := fn.Entry()
	e.Return(e.I32.Load(e.Ptr.TLSAddr(g)))

	// The base variadic convention, which is to say not Darwin's: the one
	// thing this backend reads to know which container it is emitting for.
	_, err := arm64lower.Lower(m, arm64lower.Options{})
	if err == nil {
		t.Error("a thread-local was lowered for a target with no model for one")
	}
}
