package ptx

import (
	"fmt"

	"github.com/vertex-language/ir"
)

// asm is §G4. Not yet: the template is GCC's, the operands substitute
// as register names, and the text goes out through Body.Raw — but the
// constraint letters a CUDA frontend writes ("r", "l", "f", "d") want a
// table before a wrong one is silently accepted.
func (x *fn) asm(in *ir.Inst) error {
	return fmt.Errorf("inline assembly is not lowered yet")
}

var _ = ir.VAsm
