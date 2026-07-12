# Bayesian-inference corpus

The third corpus for `flatppl-testsuite`: small, hand-written, self-contained
FlatPPL models (not converted from a foreign format) that exercise the
`bayesupdate` measure-algebra pipeline over a Bayesian posterior. `bi1`
through `bi4` express the SAME model — a `Normal`/`Exponential` prior over
two scalar parameters feeding a 10-point `iid(Normal(...))` likelihood — four
different ways (an explicit `joint` prior, a `lawof(record(...))` prior, a
`disintegrate`d joint, and a `restrict`ed joint), so all four MUST score the
identical `lp`: that 4-way construction equivalence is itself part of what
this corpus checks. `eight_schools` is Rubin's hierarchical eight-schools
model (`mu`, a half-Cauchy `tau`, and 8 `theta`s). Like `corpora/fragment/`,
every model already ends in a fixed-point `lp = logdensityof(posterior,
<point>)` binding, so the gate needs no `__score__` append: it determinizes
and scores `lp` directly.

## Contents

| Path | What |
|------|------|
| `manifest.json` | Index of the 5 posteriors. |
| `<name>/<name>.flatppl` | One self-contained model, ending in `lp = logdensityof(...)`. |
| `<name>/expected.json` | Frozen expected value for `lp` (schema mirrors `corpora/fragment/`; check kind `logdensity_value`). |
| `gen_expected.py` | Reproduces every `expected.json` value from an INDEPENDENT scipy oracle (not on the default test path — see its docstring). |
| `gate.py` | Prints a `test_id -> PASS/SKIP/MISMATCH` table (`pixi run bayesian_inference`). |
| `tests/test_bayesian_inference_gate.py` | The corpus's own pytest definitions; `tests/test_bayesian_inference.py` at the repo root is a one-line shim that re-exports them. |

## Oracle

Every `expected.json` value is computed with `scipy.stats` closed forms (an
INDEPENDENT oracle — never the sibling FlatPPL engine): `bi1`-`bi4` share one
`Normal`/`Exponential`/derived-`Normal`-likelihood log-density, and
`eight_schools` sums a `Normal` prior on `mu`, a half-Cauchy log-density on
`tau`, the `theta | mu, tau` `Normal`s, and the `y | theta` `Normal`
likelihood. `gen_expected.py` asserts its own scipy computation reproduces
the frozen value to `<= 1e-12` before writing each `expected.json`.

## Run

```sh
pixi run bayesian_inference                                # formatted table (gate.py)
pixi run test                                              # pytest, incl. this corpus via its shim
pixi run python corpora/bayesian_inference/gen_expected.py  # regenerate + verify expected.json
```

## Numeric gate status

This corpus's numeric gate (`corpora/bayesian_inference/tests/test_bayesian_inference_gate.py`)
requires determinizer work that is not yet merged to `main`: `bi1` needs the
theta-derived-parameter inline (`a = f_a(theta2)`, `b = f_b(theta1, theta2)`
feeding the likelihood's `mu`/`sigma`), and `bi3`/`bi4` need the
`disintegrate`/`restrict` lowerings. All 5 posteriors verify GREEN against
those fix binaries; the gate goes green on `main` once they land and the
testsuite's pins are bumped.
