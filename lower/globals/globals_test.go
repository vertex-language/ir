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
