# Fragment corpus

The second corpus for `flatppl-testsuite`: small, hand-written, self-contained
FlatPPL models (not converted from a foreign format) that exercise one
measure-algebra construct each — `superpose`, `truncate` (+ `normalize`),
`pushfwd` (+ `bijection`), and `kchain` (+ `kernelof`) over a discrete latent.
Unlike `corpora/hs3/` (a parametric likelihood scanned over a theta
vector), each fragment already ends in a fixed-point
`lp = logdensityof(m, <point>)` binding, so the gate needs no `__score__`
append and no theta: it determinizes and scores `lp` directly.

## Contents

One directory per fragment (31 in total), each a unified test dir:

| Path | What |
|------|------|
| `<test_id>/<test_id>.flatppl` | One self-contained model, ending in `lp = logdensityof(...)`. |
| `<test_id>/query.flatppl` | The SAME query as an `inputs`/`outputs` ABI, for the StableHLO path. |
| `<test_id>/test.json` | `test_type: "logdensity"`, `engines: ["det-js", "stablehlo"]`, the frozen `expected` scalar for `lp`, tolerances. |
| `<test_id>/test.py` | INDEPENDENT oracle: `oracle()` reproduces `expected` in closed form (scipy / Julia Distributions.jl). |

`tests/test_unified.py` discovers every directory here automatically; there is no
per-corpus gate script or manifest anymore.

## Both paths, one query, one frozen value

Every dir runs its query on the det-js path AND the StableHLO path. The model
file is unchanged: det-js still scores its own `lp` binding (Mode A), while
`query.flatppl` restates the same query with the point as an `inputs`/`outputs`
ABI argument, which is what `flatppl stablehlo` needs. The `"stablehlo"` block
in `test.json` carries only that ABI shape (`inputs`, `points`, an f32
`tolerance`) — `expected` deliberately stays at the top level, so ONE frozen
oracle value gates both paths and `regen` keeps owning it.

`joint_singular_refusal` pins a StableHLO refusal instead (`"stablehlo": {"status": "refuses",
"allow_skip": true, "refusal": "<verbatim exit-3 message>"}`):
the repeated draw lies on a diagonal and has no product-reference density.
Its det-js case also allows the refusal, with a NaN expected value to prevent
an arbitrary emitted scalar from passing. The three shared-latent fixtures
score their frozen densities on both paths.

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

`shared_latent_joint` and `shared_latent_joint_positional` retain their
correlated oracles using one `lawof(record(y1 = y1, y2 = y2))`.
Their queries expose separate scalar inputs and one vector input, respectively.
Separate law components in `joint` are independent, so they would not preserve
these fixtures' shared latent.

`joint_singular_refusal` gates the degenerate case:
`lawof(record(a = y, b = y))` retains the same draw in both fields, which has
no density w.r.t. the product reference measure. There is no closed-form value to
freeze, so this dir has no `oracle()`; it sets `"allow_skip": true` and
relies on the determiniser's exit-3 refusal producing a tolerated
`skipped`/`DETERMINIZE_SKIP` result instead.

## Run

```sh
pixi run test                                                    # pytest, incl. every dir here
pixi run unified                                                  # the unified harness alone
PYTHONPATH=$PWD/src pixi run -e stablehlo regen corpora/fragment/<test_id>   # refreeze from test.py::oracle()
```
