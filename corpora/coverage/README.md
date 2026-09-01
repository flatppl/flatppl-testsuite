# Coverage corpus

Full worked models, each written to score a language axis no other suite
row reaches. Grounded in `flatppl-dev/testsuite-coverage-proposals.md`
(the 2026-09-01 coverage survey); every model is a unified test dir
scored on the convert-free det-js path, with an independent scipy or
closed-form oracle in `test.py`.

Two kinds of dir live here:

* **Scored rows** (`status: "lowers"`) — the engine lowers and the
  frozen values are checked at 1e-9.
* **Refusal pins** (`status: "refuses"`, `allow_skip: true`) — the rust
  determiniser currently refuses the construct, so the unified harness
  records DETERMINIZE_SKIP. The frozen `expected` values are REAL
  oracle values, not sentinels: the moment the lowering lands, the skip
  disappears and the numeric compare takes over. The refusal message
  observed at authoring time is quoted in each dir's `test.py`.

| Dir | Axis | Status at authoring (2026-09-01) |
|---|---|---|
| `spectral_lines` | `ksuperpose` mixture + Dirichlet-simplex latent posterior | lowers, matches scipy |
| `ar1_drift` | `markovchain` trajectory density | determiniser refuses ("deferred to a later task") |
| `kscan_walk` | `kscan` with exogenous step sizes | determiniser refuses (same deferral) |
| `sensor_calibration` | `aggregate` (einsum) linear predictor in a likelihood | lowers, matches scipy |
| `dose_surface` | §06 `weighted` arity rule, both spellings over one measure | lowers, matches closed form |
| `allele_freq` | Dirichlet prior + iid `Categorical` likelihood | lowers, matches scipy |
| `two_instruments` | `joint_likelihood`, derived kernel parameter, free-`elementof` prior | lowers, matches closed form |
| `beam_bunch` | unbinned `PoissonProcess`, latent-weighted `superpose` intensity | determiniser refuses (no PoissonProcess type rule) |
| `b_mass_peak` | `particle-physics` standard-module members in a fit | determiniser refuses (cross-module ref) |
| `out_of_window` | `filter` -> derived `iid` size 0 (empty product) | determiniser refuses (no dynamic iid size) |
| `paired_assay` | table variate: `iid` over a record law, table observation | determiniser refuses (no record-measure unroll) |
| `censored_lifetimes` | `truncate` with a LATENT bound, exact `-inf` gate point | lowers, matches closed form incl. -inf |

Engine gaps found while authoring that could NOT be expressed as rows
here (no green shape and no refusal to pin) are recorded in
`flatppl-dev/testsuite-coverage-proposals.md` per proposal.

## Run

```sh
pixi run test                                              # includes every dir here
PYTHONPATH=$PWD/src pixi run python -m flatppl_testsuite.unified.regen corpora/coverage/<id>   # refreeze
```
