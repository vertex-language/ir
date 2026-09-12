package ptx_test

// Golden text. The printer is the only thing that can show what the
// lowering built, and PTX text is the artifact, so the strongest check
// short of running it is the exact string.

import (
	"strings"
	"testing"

	"github.com/vertex-language/ir"
	lower "github.com/vertex-language/ir/lower/ptx"
	"github.com/vertex-language/ir/verify"
	"github.com/vertex-language/ptx"
	"github.com/vertex-language/ptx/text"
)

// vecadd is the kernel from ir/text's golden: c[i] = a[i] + b[i].
func vecadd() *ir.Module {
	m := ir.NewModule("vecadd", ir.NVPTX64)
	fn := m.Func("vector_add").Export().CallConv(ir.Kernel).NoUnwind()
	a := fn.ParamPtr("a", ir.NoAlias)
	b := fn.ParamPtr("b", ir.NoAlias)
	c := fn.ParamPtr("c", ir.NoAlias)
	n := fn.ParamI32("n")

	entry := fn.Entry()
	body := fn.Block("body")
	done := fn.Block("done")

	gid := entry.I32.WorkgroupID(ir.X)
	gsz := entry.I32.WorkgroupSize(ir.X)
	base := entry.I32.Mul(gid, gsz)
	tid := entry.I32.WorkitemID(ir.X)
	i := entry.I32.Add(base, tid).Named("i")
	entry.BrIf(entry.I32.ULt(i, n), body.To(), done.To())

	off := body.I64.Shl(body.I64.ZExtI32(i), body.I64.Const(2))
	pa := body.Ptr.Add(a, off)
	pb := body.Ptr.Add(b, off)
	pc := body.Ptr.Add(c, off)
	body.F32.Store(body.F32.Add(body.F32.Load(pa), body.F32.Load(pb)), pc)
	body.Return()
	done.Return()
	return m
}

func lowerText(t *testing.T, m *ir.Module, opts lower.Options) string {
	t.Helper()
	if err := verify.Module(m); err != nil {
		t.Fatalf("verify: %v", err)
	}
	pm, err := lower.Lower(m, opts)
	if err != nil {
		t.Fatalf("lower: %v", err)
	}
	src, err := text.New(pm).WithHeader("").Print()
	if err != nil {
		t.Fatalf("print: %v", err)
	}
	return src
}

func same(t *testing.T, got, want string) {
	t.Helper()
	want = strings.TrimPrefix(want, "\n")
	if got != want {
		t.Errorf("printed:\n%s\nwant:\n%s", got, want)
	}
}

func TestVecAdd(t *testing.T) {
	src := lowerText(t, vecadd(), lower.Options{SM: ptx.SM75})
	same(t, src, `
.version 8.0
.target sm_75
.address_size 64

.visible .entry vector_add(
	.param .b64 _p_a,
	.param .b64 _p_b,
	.param .b64 _p_c,
	.param .b32 _p_n
)
{
	.reg .b64   %rd<11>;
	.reg .b32   %r<8>;
	.reg .pred  %p<2>;
	.reg .f32   %f<4>;

	ld.param.u64            %rd1, [_p_a];
	ld.param.u64            %rd2, [_p_b];
	ld.param.u64            %rd3, [_p_c];
	ld.param.u32            %r1, [_p_n];
$L_entry:
	mov.u32                 %r2, %ctaid.x;
	mov.u32                 %r3, %ntid.x;
	mul.lo.s32              %r4, %r2, %r3;
	mov.u32                 %r5, %tid.x;
	add.s32                 %r6, %r4, %r5;
	setp.lo.u32             %p1, %r6, %r1;
	@%p1 bra                $L_body;
	bra                     $L_done;
$L_body:
	cvt.u64.u32             %rd4, %r6;
	mov.b64                 %rd5, 2;
	and.b64                 %rd7, %rd5, 63;
	cvt.u32.u64             %r7, %rd7;
	shl.b64                 %rd6, %rd4, %r7;
	add.s64                 %rd8, %rd1, %rd6;
	add.s64                 %rd9, %rd2, %rd6;
	add.s64                 %rd10, %rd3, %rd6;
	ld.f32                  %f1, [%rd8];
	ld.f32                  %f2, [%rd9];
	add.rn.f32              %f3, %f1, %f2;
	st.f32                  [%rd10], %f3;
	ret;
$L_done:
	ret;
}
`)
}
