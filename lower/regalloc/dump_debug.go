package regalloc

import (
	"fmt"
	"os"

	"github.com/vertex-language/ir/lower/mir"
)

// debugDump prints f with its assignment to stderr under IR_REGALLOC_DUMP=1,
// for finding where an allocation goes wrong.
func debugDump(f *mir.Func, pool *Pool, assigned map[mir.VReg]PhysReg) {
	if os.Getenv("IR_REGALLOC_DUMP") != "1" {
		return
	}
	r := func(v mir.VReg) string {
		p, ok := assigned[v]
		pin := ""
		if _, isPin := pool.pinned.get(v); isPin {
			pin = "!"
		}
		if !ok {
			return fmt.Sprintf("v%d=?", v)
		}
		return fmt.Sprintf("v%d=%s%d", v, pin, p)
	}
	for _, b := range f.Blocks {
		fmt.Fprintf(os.Stderr, "%s:\n", b.Label)
		for _, in := range b.Instrs {
			fmt.Fprintf(os.Stderr, "   %T%+v copy=%v defs[", in.Op, in.Op, in.Copy)
			for _, d := range in.Defs {
				fmt.Fprint(os.Stderr, r(d), " ")
			}
			fmt.Fprint(os.Stderr, "] uses[")
			for _, u := range in.Uses {
				fmt.Fprint(os.Stderr, r(u), " ")
			}
			fmt.Fprintln(os.Stderr, "]")
		}
	}
	fmt.Fprintln(os.Stderr, "----")
}
