# Sample corpus

The third corpus for `flatppl-testsuite`: a small, hand-written,
self-contained FlatPPL model that exercises the FlatPDL **sample path** —
`rand(rng, lawof(record(...)))`, lowered by flatppl-rust's sample
determinizer to a rng-threaded `builtin_sample` chain and evaluated by
flatppl-js. Unlike `corpora/fragment/` (a fixed-point
`lp = logdensityof(m, <point>)`, one scalar per model), a sample-path model
gives ONE random realization per seed — there is no single scalar to freeze.
Instead this corpus **seed-sweeps** the determinized model over many seeds
to build an empirical sample set, and checks its mean/var/cov against the
model's closed-form structural moments.

## Contents

One directory per sample-path model (one so far: `hier_normal`), a unified test dir:

| Path | What |
|------|------|
| `<test_id>/<test_id>.flatppl` | The sample-path model, ending in `rand(rng, lawof(record(...)))`. |
| `<test_id>/<test_id>_density.flatppl` | A companion model with the IDENTICAL joint law, no `rand(...)` wrapper — used only by the sampling<->density secondary check (see below), named by `test.json`'s `density_model`. |
| `<test_id>/query.flatppl` | The density query as an `inputs`/`outputs` ABI, for the StableHLO row (see below). |
| `<test_id>/test.json` | `test_type: "sample"`, `engines: ["det-js", "stablehlo"]`, per-check frozen closed-form moments + Monte-Carlo tolerances (`checks[i].expected`/`atol`), or a top-level `stat` recipe (the stablehlo-sample shape) — see `unified/regen.py`'s docstring for which shape applies. |
| `<test_id>/test.py` | INDEPENDENT closed-form oracle: `stat()` returns structural moments (mean/var/cov) + the Monte-Carlo tolerance formulas; may also expose a per-point joint log-density for the density-consistency check. |

`tests/test_unified.py` discovers every directory here automatically; there is no
per-corpus gate script or manifest anymore.

## The model: `hier_normal`

```
s = rnginit([42, 0, 0, 0])
mu = draw(Normal(mu = 0.0, sigma = 10.0))
y1 = draw(Normal(mu = mu, sigma = 1.0))
y2 = draw(Normal(mu = mu, sigma = 1.0))
draws = rand(s, lawof(record(mu = mu, y1 = y1, y2 = y2)))
```

`y1` and `y2` share the SAME draw of the latent `mu` — they are not two
independent draws. That shared ancestry has a statistical signature:
`Cov(y1, y2) = Var(mu) = 100`. If the determinizer instead sampled `mu`
independently per consumer (the bug this corpus exists to catch), `y1` and
`y2` would come out independent and this covariance would land near 0 —
about 44 standard errors away from the tolerance band at N=4000. **This is
the point of this test dir**: passing `cov_y1_y2` is the statistical proof
that the sample path preserves shared-ancestor identity end-to-end.

`Var(mu) = 100` also swamps the observation noise in every marginal, which
leaves all seven mean/var/cov checks blind to `sigma_y` — raising it from 1
to 2 in the model keeps them all inside their `±11` bands. `var_y1_minus_y2`
closes that: `y1 - y2 = eps1 - eps2` cancels the shared `mu` exactly, so it
pins `2 * sigma_y**2 = 2` to `±0.224`, and the same mutation moves it to
7.93. It is also the only check here that catches `y2` aliased to `y1`
(reads exactly 0; `cov_y1_y2` then becomes `Var(y1)` and still passes).

## The seed sweep

A fixed seed gives one deterministic realization, so `scoring/sample_sweep.cjs`
substitutes the `rnginit([...])` byte-vector for N distinct seeds and
re-materialises `mu`, `y1`, `y2` for each — in ONE Node process (not a
subprocess per seed; see that script's header comment for why). The
`(sample, det-js)` runner (`unified/runners/sample_detjs.py`) reduces the N
realizations to empirical mean/var/cov and compares them to `test.py`'s
closed-form moments within a `k * Monte-Carlo-SE` tolerance (`k=5`, N=4000 —
frozen per-check as `checks[i].atol`).

## Sampling<->density secondary check

For a handful of the swept realizations, the runner also checks that the
closed-form joint log-density at that exact `(mu, y1, y2)` point matches
what the determinized DENSITY path returns for the same law
(`logdensityof(lawof(record(...)), <point>)`, scored via the det-js engine)
— against `hier_normal_density.flatppl`, a companion model with the
identical joint law but no `rand(...)` wrapper. (Appending a second
`lawof(...)` query onto the sample-path model after `rand()` has already
consumed the stochastic-phase graph is refused by the determinizer —
confirmed empirically — hence the separate companion model.)

## StableHLO path

The StableHLO row scores that SAME density query, at one frozen point, via a
`"stablehlo"` engine block that overrides `test_type` to `"logdensity"` and
`model` to the companion density file, plus `query.flatppl`'s
`inputs`/`outputs` ABI. It is the one block in the corpus that carries its own
`expected`, because a `sample` dir has no top-level scalar to share; the value
comes from this dir's `test.py::logdensity` and
`tests/core/test_engine_override_rows.py` re-derives it, since `regen` does not
refreeze a block.

The sample path itself is NOT scored on StableHLO: the emitter needs at least
one `inputs` argument (`inputs = ()` is a parse error) and this model has no
free parameter, and the `(sample, stablehlo)` runner's check kinds
(`distribution`, `key_reproducibility`, `key_advance`, `fanout_distribution`)
do not include the `sample_stats`/`cov` checks this dir exists for.

## Oracle

Every stat frozen in `test.json` is an INDEPENDENT closed-form fact about the
model (never derived from flatppl-js output): `mu ~ N(0, 10)` =>
`E[mu]=0, Var[mu]=100`; `y_i = mu + eps_i`, `eps_i ~ N(0, 1)` =>
`E[y_i]=0, Var[y_i]=101, Cov(y1,y2)=Var(mu)=100, Var(y1-y2)=2*Y_SIGMA**2=2`.
See `test.py` for the derivation and the Monte-Carlo tolerance formulas.

## Run

```sh
pixi run test                                                       # pytest, incl. every dir here
pixi run unified                                                     # the unified harness alone
PYTHONPATH=$PWD/src pixi run -e stablehlo regen corpora/sample/hier_normal   # refreeze from test.py::stat()
```
