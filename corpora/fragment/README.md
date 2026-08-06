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

One directory per fragment (21 in total), each a unified test dir:

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

`shared_latent_joint` and `shared_latent_joint_positional` are the `joint`
spellings of the same shared-latent shape (keyword `joint(y1 = lawof(y1), y2 =
lawof(y2))` and positional `joint(lawof(y1), lawof(y2))`): §06 "Equivalent record law" makes a `joint` over two reified components
equivalent to the plain record law, so both freeze the SAME expected value as
`shared_latent_record`.

`joint_singular_refusal` gates the degenerate case: `joint(a = lawof(y), b =
lawof(y))` reifies the SAME draw into both components, which has no density
w.r.t. the product reference measure. There is no closed-form value to
freeze, so this dir has no `oracle()`; it sets `"allow_skip": true` and
relies on the determiniser's exit-3 refusal producing a tolerated
`skipped`/`DETERMINIZE_SKIP` result instead.

## Run

```sh
pixi run test                                                    # pytest, incl. every dir here
pixi run unified                                                  # the unified harness alone
PYTHONPATH=$PWD/src pixi run -e stablehlo regen corpora/fragment/<test_id>   # refreeze from test.py::oracle()
```
