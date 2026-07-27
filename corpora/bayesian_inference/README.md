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

One directory per posterior (5 in total), each a unified test dir:

| Path | What |
|------|------|
| `<test_id>/<test_id>.flatppl` | One self-contained model, ending in `lp = logdensityof(...)`. |
| `<test_id>/test.json` | `test_type: "logdensity"`, `engines: ["det-js"]`, the frozen `expected` scalar for `lp`, tolerances. |
| `<test_id>/test.py` | INDEPENDENT scipy oracle: `oracle()` reproduces `expected` in closed form. |

`tests/test_unified.py` discovers every directory here automatically; there is no
per-corpus gate script or manifest anymore.

## Oracle

Every `test.py::oracle()` is computed with `scipy.stats` closed forms (an
INDEPENDENT oracle — never the sibling FlatPPL engine): `bi1`-`bi4` share one
`Normal`/`Exponential`/derived-`Normal`-likelihood log-density, and
`eight_schools` sums a `Normal` prior on `mu`, a half-Cauchy log-density on
`tau`, the `theta | mu, tau` `Normal`s, and the `y | theta` `Normal`
likelihood.

## Run

```sh
pixi run test                                                          # pytest, incl. every dir here
pixi run unified                                                        # the unified harness alone
PYTHONPATH=$PWD/src pixi run -e stablehlo regen corpora/bayesian_inference/<test_id>  # refreeze
```
