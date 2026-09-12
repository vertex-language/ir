package verify_test

// §19.20 (a kernel's shape and reachability) and §19.21 (a shared global
// is zeroed). One case per clause, each expected to fail with the sentinel
// at module scope, plus the module that passes.

import (
	"testing"

	"github.com/vertex-language/ir"
	"github.com/vertex-language/ir/verify"
)

func kernelModule() (*ir.Module, *ir.Func) {
	m := ir.NewModule("t", ir.NVPTX64)
	k := m.Func("k").Export().CallConv(ir.Kernel)
	return m, k
}

func TestKernelAccepts(t *testing.T) {
	m, k := kernelModule()
	p := k.ParamPtr("p")
	n := k.ParamI32("n")
	k.Entry().I32.Store(n, p)
	k.Entry().Return()

	m.Global("tile", ir.Shared, ir.Array(64, i32()))
	m.Global("count", ir.RW, i32()).Init(ir.Lit(ir.Int(1)))

	if err := verify.Module(m); err != nil {
		t.Fatalf("verify: %v, want nil", err)
	}
}

func TestKernelWithResult(t *testing.T) {
	m, k := kernelModule()
	k.ReturnsI32()
	k.Entry().Return(k.Entry().I32.Const(0))
	wantItemFault(t, verify.Module(m), verify.ErrKernel)
}

func TestKernelVariadic(t *testing.T) {
	m, k := kernelModule()
	k.Variadic()
	k.Entry().Return()
	wantItemFault(t, verify.Module(m), verify.ErrKernel)
}

func TestKernelByVal(t *testing.T) {
	m, k := kernelModule()
	s := m.Struct("s").Field("a", i32())
	k.ParamPtr("p", ir.ByVal(s))
	k.Entry().Return()
	wantItemFault(t, verify.Module(m), verify.ErrKernel)
}

func TestKernelNaked(t *testing.T) {
	m, k := kernelModule()
	k.Naked().AsmBody("ret;")
	wantItemFault(t, verify.Module(m), verify.ErrKernel)
}

func TestKernelImport(t *testing.T) {
	m := ir.NewModule("t", ir.NVPTX64)
	m.ImportFunc("k", ir.NewSig().Conv(ir.Kernel))
	wantItemFault(t, verify.Module(m), verify.ErrKernel)
}

func TestKernelCalled(t *testing.T) {
	m, k := kernelModule()
	k.Entry().Return()
	f := m.Func("f").CallConv(ir.CCC)
	f.Entry().Call(k)
	f.Entry().Return()
	e := wantItemFault(t, verify.Module(m), verify.ErrKernel)
	if e.Detail == "" {
		t.Errorf("no detail naming the caller")
	}
}

func TestKernelAddressTaken(t *testing.T) {
	m, k := kernelModule()
	k.Entry().Return()
	f := m.Func("f").ReturnsPtr()
	f.Entry().Return(f.Entry().Ptr.GetAddr(k))
	wantItemFault(t, verify.Module(m), verify.ErrKernel)
}

func TestSharedInitialized(t *testing.T) {
	m := ir.NewModule("t", ir.NVPTX64)
	m.Global("tile", ir.Shared, i32()).Init(ir.Lit(ir.Int(1)))
	wantItemFault(t, verify.Module(m), verify.ErrShared)
}
