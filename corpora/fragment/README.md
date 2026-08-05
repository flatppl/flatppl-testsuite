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

One directory per fragment (18 in total), each a unified test dir:

| Path | What |
|------|------|
| `<test_id>/<test_id>.flatppl` | One self-contained model, ending in `lp = logdensityof(...)`. |
| `<test_id>/test.json` | `test_type: "logdensity"`, `engines: ["det-js"]`, the frozen `expected` scalar for `lp`, tolerances. |
| `<test_id>/test.py` | INDEPENDENT oracle: `oracle()` reproduces `expected` in closed form (scipy / Julia Distributions.jl). |

`tests/test_unified.py` discovers every directory here automatically; there is no
per-corpus gate script or manifest anymore.

## Oracle

Every `test.py::oracle()` is an INDEPENDENT closed-form computation (scipy or
Julia `Distributions.jl` — never the sibling FlatPPL engine) of the same
measure-algebra construct the fragment exercises: `superpose`, `truncate` (+
`normalize`), `pushfwd` (+ `bijection`), `kchain` (+ `kernelof`), `densityof`.
`frag_trunc_out` (scored out of its truncation support) freezes `-inf` and is
compared with exact equality rather than a tolerance band.

`shared_latent_record` is the one case whose oracle is a MULTIVARIATE law: two
record fields drawn over one shared latent are correlated, so the joint is
`MvNormal` with a rank-one-plus-diagonal covariance, not the product of the two
per-field marginals. The product is 0.199 nats away at the frozen point, which is
what the case discriminates against.

## Run

```sh
pixi run test                                                    # pytest, incl. every dir here
pixi run unified                                                  # the unified harness alone
PYTHONPATH=$PWD/src pixi run -e stablehlo regen corpora/fragment/<test_id>   # refreeze from test.py::oracle()
```
