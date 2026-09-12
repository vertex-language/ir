package ir_test

// The builder-tier faults §W and the scopes add: each is caught as you
// emit, sticky, first-wins, exactly like the rest of ir/error.go.

import (
	"errors"
	"testing"

	"github.com/vertex-language/ir"
)

func wantSticky(t *testing.T, m *ir.Module, sentinel error) {
	t.Helper()
	if err := m.Err(); !errors.Is(err, sentinel) {
		t.Fatalf("Err = %v, want %v", err, sentinel)
	}
}

// A scope is §H's attribute and means nothing on a plain load.
func TestScopeOnPlainLoad(t *testing.T) {
	m := ir.NewModule("t", ir.NVPTX64)
	fn := m.Func("f")
	p := fn.ParamPtr("p")
	fn.Entry().I32.Load(p, ir.DeviceScope)
	wantSticky(t, m, ir.ErrPlacement)
}

// Two scopes on one atomic is a contradiction, not a list.
func TestTwoScopesOnAtomic(t *testing.T) {
	m := ir.NewModule("t", ir.NVPTX64)
	fn := m.Func("f")
	p := fn.ParamPtr("p")
	fn.Entry().I32.AtomicLoad(p, ir.Acquire, ir.DeviceScope, ir.SystemScope)
	wantSticky(t, m, ir.ErrOrdering)
}

// singlethread is the scope of one thread; it does not combine with another.
func TestFenceSingleThreadAndScope(t *testing.T) {
	m := ir.NewModule("t", ir.NVPTX64)
	fn := m.Func("f")
	fn.Entry().Fence(ir.SeqCst, ir.SingleThread, ir.FenceDevice)
	wantSticky(t, m, ir.ErrOrdering)
}

// A shared global is a workgroup's storage, not a linker's symbol.
func TestComdatOnShared(t *testing.T) {
	m := ir.NewModule("t", ir.NVPTX64)
	m.Global("tile", ir.Shared, ir.StoreI32.FType()).Comdat()
	wantSticky(t, m, ir.ErrPlacement)
}

// Three axes, and the fault for a fourth.
func TestAxisOutOfRange(t *testing.T) {
	m := ir.NewModule("t", ir.NVPTX64)
	fn := m.Func("f")
	fn.Entry().I32.WorkitemID(ir.Axis(3))
	wantSticky(t, m, ir.ErrPlacement)
}

// The device targets admit no ext-float and no vector, which the builder
// says at the first use rather than the lowering.
func TestDeviceLayoutRejectsF80AndV128(t *testing.T) {
	m := ir.NewModule("t", ir.AMDGCN)
	fn := m.Func("f")
	fn.Entry().F80()
	wantSticky(t, m, ir.ErrLayout)

	m = ir.NewModule("t", ir.NVPTX64)
	fn = m.Func("f")
	fn.Entry().V128()
	wantSticky(t, m, ir.ErrLayout)
}
