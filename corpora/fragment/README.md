# Fragment corpus

The second corpus for `flatppl-testsuite`: small, hand-written, self-contained
FlatPPL models (not converted from a foreign format) that exercise one
measure-algebra construct each — `superpose`, `truncate` (+ `normalize`),
`pushfwd` (+ `bijection`), and `kchain` (+ `kernelof`) over a discrete latent.
Unlike `corpora/hs3/` (a parameterized likelihood scanned over a theta
vector), each fragment already ends in a fixed-point
`lp = logdensityof(m, <point>)` binding, so the gate needs no `__score__`
append and no theta: it determinizes and scores `lp` directly.

## Contents

| Path | What |
|------|------|
| `manifest.json` | Index of the 8 fragments. |
| `<name>/<name>.flatppl` | One self-contained model, ending in `lp = logdensityof(...)`. |
| `<name>/expected.json` | Frozen expected value for `lp` (schema mirrors `corpora/hs3/`; check kind `logdensity_value`). |
| `gen_expected.py` | Reproduces every `expected.json` value from an INDEPENDENT scipy oracle (not on the default test path — see its docstring). |
| `gate.py` | Prints a `test_id -> PASS/SKIP/MISMATCH` table (`pixi run fragment`). |
| `tests/test_fragment_gate.py` | The corpus's own pytest definitions; `tests/test_fragment.py` at the repo root is a one-line shim that re-exports them. |

## Oracle

Every `expected.json` value was originally computed with Julia's
`Distributions.jl 0.25` (an INDEPENDENT oracle — never the sibling FlatPPL
engine); `gen_expected.py` independently re-derives the same closed-form
values with `scipy.stats` and reproduces them to `<= 1e-12`. `±inf`
(`frag_trunc_out`, scored out of its truncation support) has no JSON
Infinity literal, so it is frozen as the string `"-inf"` and compared with
exact `==` rather than a tolerance band.

## Run

```sh
pixi run fragment                              # formatted table (gate.py)
pixi run test                                  # pytest, incl. this corpus via its shim
pixi run python corpora/fragment/gen_expected.py   # regenerate + verify expected.json
```

## Numeric gate status

This corpus's numeric gate (`corpora/fragment/tests/test_fragment_gate.py`)
requires a determinizer/engine fix pair that is not yet merged to `main`:
flatppl-rust's `logsumexp` determinizer lowering emitting a vector argument
(blocks `superpose`, `kchain_bern`, `kchain_cat`), and flatppl-js's
value-level `x in interval(lo, hi)` evaluation plus a fixed-phase `±inf`
materialiser fix (blocks `trunc_in`, `trunc_out`, `norm_trunc`). All 8
fragments verify GREEN against those fix binaries; the gate goes green on
`main` once they land and the testsuite's pins are bumped.
