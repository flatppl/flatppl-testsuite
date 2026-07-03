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

| Path | What |
|------|------|
| `manifest.json` | Index of the sample models (one so far: `hier_normal`). |
| `oracle.py` | INDEPENDENT closed-form oracle: structural moments + per-point joint log-density. |
| `gen_expected.py` | Derives `expected.json` from `oracle.py` + the Monte-Carlo tolerance formulas (not on the default test path — see its docstring). |
| `<name>/<name>.flatppl` | The sample-path model, ending in `rand(rng, lawof(record(...)))`. |
| `<name>/<name>_density.flatppl` | A companion model with the IDENTICAL joint law, no `rand(...)` wrapper — used only by the sampling<->density secondary check (see below). |
| `<name>/expected.json` | Frozen closed-form moments + Monte-Carlo tolerances for `<name>`. |
| `gate.py` | Prints a `test_id::check -> PASS/SKIP/MISMATCH` table (`pixi run sample`). |
| `tests/test_sample_gate.py` | The corpus's own pytest definitions; `tests/test_sample.py` at the repo root is a one-line shim that re-exports them. |

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
the point of the gate**: passing `cov_y1_y2` is the statistical proof that
the sample path preserves shared-ancestor identity end-to-end.

## The seed sweep

A fixed seed gives one deterministic realization, so `scoring/sample_sweep.cjs`
substitutes the `rnginit([...])` byte-vector for N distinct seeds and
re-materialises `mu`, `y1`, `y2` for each — in ONE Node process (not a
subprocess per seed; see that script's header comment for why). The Python
suite (`suites/sample_gate.py`) reduces the N realizations to empirical
mean/var/cov and compares them to `oracle.py`'s closed-form moments within a
`k * Monte-Carlo-SE` tolerance (`k=5`, N=4000 — see `gen_expected.py`).

## Sampling<->density secondary check

For a handful of the swept realizations, the suite also checks that the
closed-form joint log-density at that exact `(mu, y1, y2)` point matches
what the determinized DENSITY path returns for the same law
(`logdensityof(lawof(record(...)), <point>)`, scored via the det-js engine)
— against `hier_normal_density.flatppl`, a companion model with the
identical joint law but no `rand(...)` wrapper. (Appending a second
`lawof(...)` query onto the sample-path model after `rand()` has already
consumed the stochastic-phase graph is refused by the determinizer —
confirmed empirically — hence the separate companion model.)

## Oracle

Every stat in `expected.json` is an INDEPENDENT closed-form fact about the
model (never derived from flatppl-js output): `mu ~ N(0, 10)` =>
`E[mu]=0, Var[mu]=100`; `y_i = mu + eps_i`, `eps_i ~ N(0, 1)` =>
`E[y_i]=0, Var[y_i]=101, Cov(y1,y2)=Var(mu)=100`. See `oracle.py` and
`gen_expected.py` for the derivation and the Monte-Carlo tolerance formulas.

## Run

```sh
pixi run sample                                  # formatted table (gate.py)
pixi run test                                    # pytest, incl. this corpus via its shim
pixi run python corpora/sample/gen_expected.py   # regenerate expected.json
```

## Numeric gate status

This corpus's numeric gate (`corpora/sample/tests/test_sample_gate.py`)
requires a determinizer/engine fix pair that is not yet merged to `main`:
flatppl-rust's sample-path determinizer (`rand(rng, lawof(...))` -> a
rng-threaded `builtin_sample` chain with shared-ancestor preservation) and
flatppl-js's get0-on-tuple fix (needed to destructure the
`(variate, RngState)` tuple `builtin_sample` returns). The gate verifies
GREEN against those fix binaries; it goes green on `main` once they land
and the testsuite's pins are bumped.
