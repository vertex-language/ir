package ptx_test

// The execution oracle: the CUDA driver, which JITs PTX text and runs it.
//
// No toolkit and no cgo. nvcuda.dll ships with every NVIDIA driver, and
// syscall.NewLazyDLL reaches it directly; a machine without one, or with
// one and no GPU, skips rather than fails. This is the strongest check
// the README describes — arm64's "linked and run" — and it proves what
// the golden text cannot: that the bytes compute, that the .param
// marshalling is right across a real call, that a barrier really
// synchronises a workgroup, that an atomic really counts.
//
// The reference is always computed a different way in Go, not by
// restating the lowering.

import (
	"errors"
	"fmt"
	"runtime"
	"strings"
	"syscall"
	"testing"
	"unsafe"

	"github.com/vertex-language/ir"
	lower "github.com/vertex-language/ir/lower/ptx"
	"github.com/vertex-language/ir/verify"
	"github.com/vertex-language/ptx"
	"github.com/vertex-language/ptx/text"
)

type cuda struct {
	dll *syscall.LazyDLL

	init, driverVersion, deviceGet, deviceGetAttribute, ctxCreate, ctxDestroy,
	ctxSynchronize, moduleLoadDataEx, moduleUnload, moduleGetFunction,
	memAlloc, memFree, memcpyHtoD, memcpyDtoH, launchKernel, getErrorString *syscall.LazyProc

	ctx uintptr
	sm  ptx.Target
}

var (
	cudaOnce    bool
	cudaShared  *cuda
	cudaSkipMsg string
)

// device opens the driver once per test binary and skips the test when
// there is nothing to run on.
func device(t *testing.T) *cuda {
	t.Helper()
	if !cudaOnce {
		cudaOnce = true
		cudaShared, cudaSkipMsg = openCUDA()
	}
	if cudaShared == nil {
		t.Skip(cudaSkipMsg)
	}
	return cudaShared
}

func openCUDA() (*cuda, string) {
	c := &cuda{dll: syscall.NewLazyDLL("nvcuda.dll")}
	if err := c.dll.Load(); err != nil {
		return nil, "no nvcuda.dll: " + err.Error()
	}
	procs := map[string]**syscall.LazyProc{
		"cuInit": &c.init, "cuDriverGetVersion": &c.driverVersion,
		"cuDeviceGet": &c.deviceGet, "cuDeviceGetAttribute": &c.deviceGetAttribute,
		"cuCtxCreate_v2": &c.ctxCreate, "cuCtxDestroy_v2": &c.ctxDestroy,
		"cuCtxSynchronize": &c.ctxSynchronize,
		"cuModuleLoadDataEx": &c.moduleLoadDataEx, "cuModuleUnload": &c.moduleUnload,
		"cuModuleGetFunction": &c.moduleGetFunction,
		"cuMemAlloc_v2": &c.memAlloc, "cuMemFree_v2": &c.memFree,
		"cuMemcpyHtoD_v2": &c.memcpyHtoD, "cuMemcpyDtoH_v2": &c.memcpyDtoH,
		"cuLaunchKernel": &c.launchKernel, "cuGetErrorString": &c.getErrorString,
	}
	for name, p := range procs {
		*p = c.dll.NewProc(name)
		if err := (*p).Find(); err != nil {
			return nil, "nvcuda.dll lacks " + name
		}
	}
	if r, _, _ := c.init.Call(0); r != 0 {
		return nil, "cuInit: " + c.errString(r)
	}
	var dev int32
	if r, _, _ := c.deviceGet.Call(uintptr(unsafe.Pointer(&dev)), 0); r != 0 {
		return nil, "cuDeviceGet: " + c.errString(r)
	}
	var major, minor int32
	const attrMajor, attrMinor = 75, 76
	c.deviceGetAttribute.Call(uintptr(unsafe.Pointer(&major)), attrMajor, uintptr(dev))
	c.deviceGetAttribute.Call(uintptr(unsafe.Pointer(&minor)), attrMinor, uintptr(dev))
	c.sm = ptx.Target{SM: int(major*10 + minor)}
	if r, _, _ := c.ctxCreate.Call(uintptr(unsafe.Pointer(&c.ctx)), 0, uintptr(dev)); r != 0 {
		return nil, "cuCtxCreate: " + c.errString(r)
	}
	return c, ""
}

func (c *cuda) errString(r uintptr) string {
	var p *byte
	c.getErrorString.Call(r, uintptr(unsafe.Pointer(&p)))
	if p == nil {
		return fmt.Sprintf("CUDA error %d", r)
	}
	var b []byte
	for q := unsafe.Pointer(p); *(*byte)(q) != 0; q = unsafe.Add(q, 1) {
		b = append(b, *(*byte)(q))
	}
	return string(b)
}

func (c *cuda) check(what string, r uintptr) error {
	if r != 0 {
		return fmt.Errorf("%s: %s", what, c.errString(r))
	}
	return nil
}

// options is what Lower is given for this device: its own SM, and an ISA
// its driver accepts. 8.0 is CUDA 12.0's and every 12.x driver JITs it.
func (c *cuda) options() lower.Options {
	sm := c.sm
	if sm.SM > 90 {
		sm = ptx.SM90
	}
	return lower.Options{SM: sm, ISA: ptx.ISA80}
}

// module is a loaded PTX module.
type module struct {
	c *cuda
	h uintptr
}

// load JITs the text. The error log buffer is what ptxas would have said.
func (c *cuda) load(src string) (*module, error) {
	image := append([]byte(src), 0)
	logBuf := make([]byte, 8192)
	const jitErrorLogBuffer, jitErrorLogBufferSizeBytes = 5, 6
	opts := [2]uint32{jitErrorLogBuffer, jitErrorLogBufferSizeBytes}
	vals := [2]uintptr{uintptr(unsafe.Pointer(&logBuf[0])), uintptr(len(logBuf))}
	var h uintptr
	r, _, _ := c.moduleLoadDataEx.Call(
		uintptr(unsafe.Pointer(&h)), uintptr(unsafe.Pointer(&image[0])),
		2, uintptr(unsafe.Pointer(&opts[0])), uintptr(unsafe.Pointer(&vals[0])))
	runtime.KeepAlive(image)
	runtime.KeepAlive(logBuf)
	if r != 0 {
		log := strings.TrimRight(string(logBuf[:strings.IndexByte(string(logBuf), 0)]), "\n")
		return nil, fmt.Errorf("cuModuleLoadDataEx: %s\n%s\n--- ptx ---\n%s", c.errString(r), log, src)
	}
	return &module{c: c, h: h}, nil
}

func (m *module) unload() { m.c.moduleUnload.Call(m.h) }

func (m *module) function(name string) (uintptr, error) {
	var f uintptr
	cname := append([]byte(name), 0)
	r, _, _ := m.c.moduleGetFunction.Call(uintptr(unsafe.Pointer(&f)), m.h, uintptr(unsafe.Pointer(&cname[0])))
	runtime.KeepAlive(cname)
	return f, m.c.check("cuModuleGetFunction "+name, r)
}

// A buffer is device memory mirrored from a host slice.
type buffer struct {
	c    *cuda
	ptr  uint64
	size uintptr
}

func (c *cuda) alloc(size uintptr) (*buffer, error) {
	b := &buffer{c: c, size: size}
	r, _, _ := c.memAlloc.Call(uintptr(unsafe.Pointer(&b.ptr)), size)
	return b, c.check("cuMemAlloc", r)
}

func (b *buffer) free() { b.c.memFree.Call(uintptr(b.ptr)) }

func (b *buffer) upload(host unsafe.Pointer) error {
	r, _, _ := b.c.memcpyHtoD.Call(uintptr(b.ptr), uintptr(host), b.size)
	return b.c.check("cuMemcpyHtoD", r)
}

func (b *buffer) download(host unsafe.Pointer) error {
	r, _, _ := b.c.memcpyDtoH.Call(uintptr(host), uintptr(b.ptr), b.size)
	return b.c.check("cuMemcpyDtoH", r)
}

// launch runs f over grid×block work-items with the given arguments,
// each a pointer to a host value of the parameter's type, and waits.
func (c *cuda) launch(f uintptr, grid, block [3]uint32, shared uint32, args ...unsafe.Pointer) error {
	params := make([]uintptr, len(args))
	for i, a := range args {
		params[i] = uintptr(a)
	}
	var pp uintptr
	if len(params) > 0 {
		pp = uintptr(unsafe.Pointer(&params[0]))
	}
	r, _, _ := c.launchKernel.Call(f,
		uintptr(grid[0]), uintptr(grid[1]), uintptr(grid[2]),
		uintptr(block[0]), uintptr(block[1]), uintptr(block[2]),
		uintptr(shared), 0, pp, 0)
	runtime.KeepAlive(params)
	runtime.KeepAlive(args)
	if err := c.check("cuLaunchKernel", r); err != nil {
		return err
	}
	r, _, _ = c.ctxSynchronize.Call()
	return c.check("cuCtxSynchronize", r)
}

// run lowers m, loads it, and hands the test the loaded module.
func run(t *testing.T, m *ir.Module) (*cuda, *module) {
	t.Helper()
	c := device(t)
	if err := verify.Module(m); err != nil {
		t.Fatalf("verify: %v", err)
	}
	pm, err := lower.Lower(m, c.options())
	if err != nil {
		t.Fatalf("lower: %v", err)
	}
	src, err := text.Print(pm)
	if err != nil {
		t.Fatalf("print: %v", err)
	}
	mod, err := c.load(src)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(mod.unload)
	return c, mod
}

// deviceSlice allocates a device buffer holding a host slice and returns
// the device pointer as a launch argument.
func deviceSlice[T any](t *testing.T, c *cuda, host []T) (*buffer, *uint64) {
	t.Helper()
	var zero T
	b, err := c.alloc(uintptr(len(host)) * unsafe.Sizeof(zero))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(b.free)
	if len(host) > 0 {
		if err := b.upload(unsafe.Pointer(&host[0])); err != nil {
			t.Fatal(err)
		}
	}
	return b, &b.ptr
}

func readBack[T any](t *testing.T, b *buffer, host []T) {
	t.Helper()
	if len(host) == 0 {
		return
	}
	if err := b.download(unsafe.Pointer(&host[0])); err != nil {
		t.Fatal(err)
	}
}

var errSkip = errors.New("skip")

// —— the tests ——

// vector_add: the first kernel, on the hardware.
func TestRunVecAdd(t *testing.T) {
	c, mod := run(t, vecadd())
	f, err := mod.function("vector_add")
	if err != nil {
		t.Fatal(err)
	}

	const n = 1000
	a, b, out := make([]float32, n), make([]float32, n), make([]float32, n)
	for i := range a {
		a[i] = float32(i) * 0.5
		b[i] = float32(n-i) * 0.25
	}
	_, pa := deviceSlice(t, c, a)
	_, pb := deviceSlice(t, c, b)
	bc, pc := deviceSlice(t, c, out)
	nn := int32(n)

	if err := c.launch(f, [3]uint32{(n + 255) / 256, 1, 1}, [3]uint32{256, 1, 1}, 0,
		unsafe.Pointer(pa), unsafe.Pointer(pb), unsafe.Pointer(pc), unsafe.Pointer(&nn)); err != nil {
		t.Fatal(err)
	}
	readBack(t, bc, out)
	for i := range out {
		if want := a[i] + b[i]; out[i] != want {
			t.Fatalf("out[%d] = %v, want %v", i, out[i], want)
		}
	}
}
