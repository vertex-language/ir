package globals

import (
	"testing"

	"github.com/vertex-language/ir"
)

// FuncBinding is what carries a function's linkage below vir. Without it a
// static function is a global symbol, and two objects that both define one
// collide on a name neither program can see.
func TestFuncBinding(t *testing.T) {
	m := ir.NewModule("m", ir.X86_64Linux)

	plain := m.Func("plain")
	internal := m.Func("internal").Internal()
	exported := m.Func("exported").Export()
	weak := m.Func("weak").Export().Weak()

	tests := []struct {
		fn   *ir.Func
		want Binding
	}{
		{plain, Global},   // no linkage stated: a function is callable
		{internal, Local}, // static
		{exported, Global},
		{weak, Weak}, // an emitted inline definition
	}
	for _, tt := range tests {
		if got := FuncBinding(tt.fn); got != tt.want {
			t.Errorf("FuncBinding(%s) = %v, want %v", tt.fn.Name(), got, tt.want)
		}
	}
}

// TestSectionForRelocatedConst: a read-only global whose bytes name
// another symbol cannot go where nothing may write.
//
// .rodata is mapped read-only from the file and (__TEXT,__const) is
// inside the text segment, so a table of addresses placed there is
// either faulted on when the loader rebases it or silently left
// holding what the file had. It has to be relro, which is writable
// while the loader works and read-only afterwards -- and it has to be
// relro rather than .data, or a table of function pointers stays
// writable for the life of the process.
//
// The distinction is a property of the initializer, not of the domain,
// which is the same reading sectionFor already gave rw when it split
// .bss from .data.
func TestSectionForRelocatedConst(t *testing.T) {
	m := ir.NewModule("m", ir.X86_64Linux)
	target := m.Func("target")
	ptr := ir.StorePtr.FType()

	tests := []struct {
		name string
		g    *ir.Global
		want Kind
	}{
		{"a constant of plain bytes",
			m.Global("bytes", ir.RO, ir.StoreI64.FType()).Init(ir.Lit(ir.Int(7))),
			ROData},
		{"a constant naming a symbol",
			m.Global("one", ir.RO, ptr).Init(ir.RelocInit(target)),
			RelROData},
		{"a constant array of them",
			m.Global("table", ir.RO, ir.Array(2, ptr)).
				Init(ir.List(ir.RelocInit(target), ir.RelocInit(target))),
			RelROData},
		{"a relocation nested in a struct",
			m.Global("boxed", ir.RO, ptr).
				Init(ir.Fields(ir.Val("f", ir.RelocInit(target)))),
			RelROData},
		{"a relocation with an addend is still one",
			m.Global("adjusted", ir.RO, ptr).
				Init(ir.RelocInit(target).Plus(ir.Int(8))),
			RelROData},
		{"writable data is unaffected",
			m.Global("rw", ir.RW, ptr).Init(ir.RelocInit(target)),
			Data},
		{"and so is bss",
			m.Global("zeroed", ir.RW, ptr),
			BSS},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := sectionFor(tt.g); got != tt.want {
				t.Errorf("sectionFor = %v, want %v", got, tt.want)
			}
		})
	}
}
