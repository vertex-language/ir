// Command vir is a debugging tool for the Vertex IR: it builds a named
// sample module and shows you what this repo makes of it.
//
//	vir list                every sample, with what shape it is
//	vir cat <sample>        print it as .vir
//	vir verify <sample>     run §19 over it and report every fault
//
// It is not a compiler driver, and the missing verb says why. There is no
// `vir fmt`, because formatting means reading .vir back in and this repo
// has no parser — deliberately: a reader would be a second front door for
// constructing a module, with its own invariants to defend, and nothing
// in the pipeline consumes one. So `vir` takes the name of a module it
// builds by calling the same public API a frontend would, and .vir stays
// what the README says it is, an output format.
//
// It imports nothing you cannot: ir, ir/text, and ir/verify.
package main

import (
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"text/tabwriter"

	"github.com/vertex-language/ir"
	"github.com/vertex-language/ir/text"
	"github.com/vertex-language/ir/verify"
)

func main() {
	flag.Usage = usage
	flag.Parse()

	if err := run(os.Stdout, flag.Args()); err != nil {
		fmt.Fprintln(os.Stderr, "vir: "+err.Error())
		os.Exit(1)
	}
}

func usage() {
	fmt.Fprint(os.Stderr, `usage: vir <command> [sample]

	list              every sample this tool can build
	cat <sample>      print the sample as .vir
	verify <sample>   run §19 over the sample and report every fault

There is no fmt: this repo has no .vir parser, so there is nothing to
read back in and reformat. Run "vir list" for the sample names.
`)
}

// run is main's body with its output as a parameter, which is the whole
// of what makes this tool testable: every command writes to w, and the
// error it returns is the tool's own failure rather than anything it
// found in a module.
func run(w io.Writer, args []string) error {
	if len(args) == 0 {
		usage()
		return errors.New("no command")
	}

	switch cmd := args[0]; cmd {
	case "list":
		if len(args) > 1 {
			return fmt.Errorf("list takes no arguments")
		}
		return list(w)

	case "cat", "verify":
		if len(args) != 2 {
			return fmt.Errorf("%s takes exactly one sample name; run \"vir list\"", cmd)
		}
		s, ok := lookup(args[1])
		if !ok {
			return fmt.Errorf("no sample named %q; run \"vir list\"", args[1])
		}
		if cmd == "cat" {
			return cat(w, s.build())
		}
		return check(w, s.build())

	default:
		usage()
		return fmt.Errorf("unknown command %q", cmd)
	}
}

// list prints the sample table.
func list(w io.Writer) error {
	tw := tabwriter.NewWriter(w, 0, 0, 3, ' ', 0)
	for _, s := range samples {
		fmt.Fprintf(tw, "%s\t%s\n", s.name, s.about)
	}
	return tw.Flush()
}

// cat prints m as .vir.
//
// text.Print refuses a module carrying a sticky builder error, since
// there is nothing faithful to print, and returns that error rather than
// a partial module. Nothing here has to check m.Err() first.
func cat(w io.Writer, m *ir.Module) error {
	return text.Print(w, m)
}

// check runs the verifier and prints every fault it found, one per line,
// on stdout — these are the tool's output, not its own failure, which is
// why they do not go to stderr and why finding them is not an exit code.
//
// verify.Errors is unwrapped rather than printed whole: its Error method
// is written for a caller who wants one line, and prints "and 3 more".
// The whole point of a verifier over a finished module is that every
// fault is there at once, so a tool whose job is to show them shows them
// all.
func check(w io.Writer, m *ir.Module) error {
	err := verify.Module(m)
	if err == nil {
		fmt.Fprintln(w, "ok")
		return nil
	}

	var faults verify.Errors
	if !errors.As(err, &faults) {
		// The module's own sticky builder error, which verify.Module
		// returns ahead of every §19 rule: one fault, first-wins, and
		// not a list.
		fmt.Fprintln(w, err)
		return nil
	}
	for _, f := range faults {
		fmt.Fprintln(w, f)
	}
	return nil
}
