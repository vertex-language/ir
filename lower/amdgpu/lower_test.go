package amdgpu_test

// The strongest check short of a GPU: the code object's bytes decode under
// llvm-objdump into the instructions the lowering meant, and the
// descriptor says what clang's would for the same kernel. Both halves
// need the WSL LLVM; without it the tests check what the object says
// about itself and stop there.

import (
	"bytes"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/vertex-language/amdgpu/feature"
	"github.com/vertex-language/amdgpu/obj"
	objelf "github.com/vertex-language/amdgpu/obj/elf"

	"github.com/vertex-language/ir"
	lower "github.com/vertex-language/ir/lower/amdgpu"
	"github.com/vertex-language/ir/verify"
)

// vecadd is c[i] = a[i] + b[i] with no bounds check: the straight line of
// milestone 26.
func vecadd() *ir.Module {
	m := ir.NewModule("vecadd", ir.AMDGCN)
	fn := m.Func("vector_add").Export().CallConv(ir.Kernel).NoUnwind()
	a := fn.ParamPtr("a", ir.NoAlias)
	b := fn.ParamPtr("b", ir.NoAlias)
	c := fn.ParamPtr("c", ir.NoAlias)

	entry := fn.Entry()
	gid := entry.I32.WorkgroupID(ir.X)
	gsz := entry.I32.WorkgroupSize(ir.X)
	base := entry.I32.Mul(gid, gsz)
	tid := entry.I32.WorkitemID(ir.X)
	i := entry.I32.Add(base, tid).Named("i")
	off := entry.I64.Shl(entry.I64.ZExtI32(i), entry.I64.Const(2))
	pa := entry.Ptr.Add(a, off)
	pb := entry.Ptr.Add(b, off)
	pc := entry.Ptr.Add(c, off)
	entry.F32.Store(entry.F32.Add(entry.F32.Load(pa), entry.F32.Load(pb)), pc)
	entry.Return()
	return m
}

func lowerObj(t *testing.T, m *ir.Module, opts lower.Options) *obj.Object {
	t.Helper()
	if err := verify.Module(m); err != nil {
		t.Fatalf("verify: %v", err)
	}
	o, err := lower.Lower(m, opts)
	if err != nil {
		t.Fatalf("lower: %v", err)
	}
	return o
}

// disassemble is llvm-objdump's view of the code object's .text, one
// instruction per line without the address and encoding columns, or nil
// where there is no llvm-objdump to ask.
func disassemble(t *testing.T, o *obj.Object, asic feature.ASIC) []string {
	t.Helper()
	objdump := findWSLTool(t, "llvm-objdump")
	if objdump == nil {
		t.Log("no llvm-objdump; the bytes go unchecked")
		return nil
	}
	var buf bytes.Buffer
	if err := objelf.WriteHSACO(&buf, o); err != nil {
		t.Fatalf("WriteHSACO: %v", err)
	}
	path := filepath.Join(t.TempDir(), "k.hsaco")
	if err := os.WriteFile(path, buf.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}
	args := append(objdump[1:], "-d", "--mcpu="+strings.ToLower(asic.String()), wslPath(path))
	out, err := exec.Command(objdump[0], args...).CombinedOutput()
	if err != nil {
		t.Fatalf("llvm-objdump: %v\n%s", err, out)
	}
	var lines []string
	for _, ln := range strings.Split(string(out), "\n") {
		ln = strings.TrimSpace(ln)
		if ln == "" || strings.HasSuffix(ln, ":") || strings.Contains(ln, "file format") || strings.HasPrefix(ln, "Disassembly") {
			continue
		}
		if i := strings.Index(ln, "//"); i >= 0 {
			ln = strings.TrimSpace(ln[:i])
		}
		lines = append(lines, strings.Join(strings.Fields(ln), " "))
	}
	return lines
}

func TestVecAdd(t *testing.T) {
	o := lowerObj(t, vecadd(), lower.Options{ASIC: feature.GFX942})
	kds := o.KernelDescriptors()
	if len(kds) != 1 || kds[0].Name != "vector_add" {
		t.Fatalf("descriptors = %+v", kds)
	}
	kd := kds[0]
	// Three pointers, then the hidden group size x at the next 8-byte
	// boundary — the metadata clang publishes for the same signature.
	if kd.KernargSize != 26 || len(kd.Args) != 4 || kd.Args[3].Offset != 24 || kd.Args[3].Kind != obj.HiddenGroupSizeX {
		t.Errorf("kernarg layout: size %d, args %+v", kd.KernargSize, kd.Args)
	}
	// 76 VGPRs: the pairs start at v64 and the last is v[74:75]; 34
	// SGPRs: the argument pairs land in s[32:33]. rsrc2 is the two user
	// SGPRs and workgroup id x; rsrc1's VGPR granule on gfx942 is 8, so
	// 76 is 9 (76/8 rounded up, minus one), and 34 SGPRs plus the 6
	// reserved is 40, a granule of 4.
	if kd.VGPRCount != 76 || kd.SGPRCount != 34 {
		t.Errorf("registers: %d VGPRs, %d SGPRs", kd.VGPRCount, kd.SGPRCount)
	}
	if kd.ComputePgmRsrc2 != 0x84 {
		t.Errorf("rsrc2 = %#x", kd.ComputePgmRsrc2)
	}
	if got, want := kd.ComputePgmRsrc1&0x3ff, uint32(9|4<<6); got != want {
		t.Errorf("rsrc1 register fields = %#x, want %#x", got, want)
	}
	same(t, disassemble(t, o, feature.GFX942), `
s_load_dwordx2 s[32:33], s[0:1], 0x0
s_waitcnt vmcnt(0) lgkmcnt(0)
v_mov_b32_e32 v64, s32
v_mov_b32_e32 v65, s33
s_load_dwordx2 s[32:33], s[0:1], 0x8
s_waitcnt vmcnt(0) lgkmcnt(0)
v_mov_b32_e32 v66, s32
v_mov_b32_e32 v67, s33
s_load_dwordx2 s[32:33], s[0:1], 0x10
s_waitcnt vmcnt(0) lgkmcnt(0)
v_mov_b32_e32 v68, s32
v_mov_b32_e32 v69, s33
v_mov_b32_e32 v1, s2
s_load_dword s8, s[0:1], 0x18
s_waitcnt vmcnt(0) lgkmcnt(0)
v_mov_b32_e32 v2, s8
v_and_b32_e32 v2, 0xffff, v2
v_mul_lo_u32 v3, v1, v2
v_and_b32_e32 v1, 0x3ff, v0
v_add_u32_e32 v2, v3, v1
v_mov_b32_e32 v70, v2
v_mov_b32_e32 v71, 0
v_mov_b32_e32 v72, 2
v_mov_b32_e32 v73, 0
v_lshlrev_b64 v[74:75], v72, v[70:71]
v_add_co_u32_e32 v70, vcc, v64, v74
v_addc_co_u32_e32 v71, vcc, v65, v75, vcc
v_add_co_u32_e32 v64, vcc, v66, v74
v_addc_co_u32_e32 v65, vcc, v67, v75, vcc
v_add_co_u32_e32 v66, vcc, v68, v74
v_addc_co_u32_e32 v67, vcc, v69, v75, vcc
flat_load_dword v1, v[70:71]
s_waitcnt vmcnt(0) lgkmcnt(0)
flat_load_dword v2, v[64:65]
s_waitcnt vmcnt(0) lgkmcnt(0)
v_add_f32_e32 v3, v1, v2
flat_store_dword v[66:67], v3
s_waitcnt vmcnt(0) lgkmcnt(0)
s_endpgm`)
}

// same compares a disassembly to the expected listing; a nil disassembly
// is no llvm-objdump, which is not a failure.
func same(t *testing.T, got []string, want string) {
	t.Helper()
	if got == nil {
		return
	}
	g := strings.Join(got, "\n")
	w := strings.TrimSpace(want)
	if g != w {
		t.Errorf("disassembly:\n%s\nwant:\n%s", g, w)
	}
}

// findWSLTool is an LLVM tool by name: on PATH, or in the WSL Ubuntu's
// ~/tools/bin. Nil when neither is there.
func findWSLTool(t *testing.T, name string) []string {
	t.Helper()
	if p, err := exec.LookPath(name); err == nil {
		return []string{p}
	}
	if wsl, err := exec.LookPath("wsl"); err == nil {
		cmd := exec.Command(wsl, "-e", "/home/cloud/tools/bin/"+name, "--version")
		if cmd.Run() == nil {
			return []string{wsl, "-e", "/home/cloud/tools/bin/" + name}
		}
	}
	return nil
}

// wslPath is a Windows path as WSL spells it.
func wslPath(p string) string {
	p = strings.ReplaceAll(p, "\\", "/")
	if len(p) > 2 && p[1] == ':' {
		return "/mnt/" + strings.ToLower(p[:1]) + p[2:]
	}
	return p
}
