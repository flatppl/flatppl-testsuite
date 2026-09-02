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
| `ar1_drift` | `markovchain` trajectory density | authored as a refusal pin; now lowers, matches closed form |
| `kscan_walk` | `kscan` with exogenous step sizes | authored as a refusal pin; now lowers, matches scipy |
| `sensor_calibration` | `aggregate` (einsum) linear predictor in a likelihood | lowers, matches scipy |
| `dose_surface` | §06 `weighted` arity rule, both spellings over one measure | lowers, matches closed form |
| `allele_freq` | Dirichlet prior + iid `Categorical` likelihood | lowers, matches scipy |
| `two_instruments` | `joint_likelihood`, derived kernel parameter, free-`elementof` prior | lowers, matches closed form |
| `beam_bunch` | unbinned `PoissonProcess`, latent-weighted `superpose` intensity | determiniser refuses (no PoissonProcess type rule) |
| `b_mass_peak` | `particle-physics` standard-module members in a fit | determiniser refuses (cross-module ref) |
| `out_of_window` | `filter` -> derived `iid` size 0 (empty product) | determiniser refuses (no dynamic iid size) |
| `paired_assay` | table variate: `iid` over a record law, table observation | lowers, matches scipy (record-measure unroll landed 2026-09-01) |
| `censored_lifetimes` | `truncate` with a LATENT bound, exact `-inf` gate point | lowers, matches closed form incl. -inf |
| `mv_mixture` | §06 `ksuperpose` family axes by parameter rank: a MULTIVARIATE mixture (N x d `mu`, N x d x d `cov`) | determiniser refuses (no per-component slice extraction) |
| `mv_mixture_sample` | the same mixture's sample path, per-draw component selection | determiniser refuses (ksuperpose sampling unimplemented) |
| `stdmod_interp_poly6` | §09 standard-module FUNCTION members, one per lowered module: `particle-physics` (both degree-6 HistFactory interpolators, both extrapolation branches), `polynomials` (`legendre`), `distances` (`euclidean`) | lowers, matches a 6x6 solve of §09's C² conditions cross-checked against ROOT, plus `scipy.special` and the norm written out |

`stdmod_interp_poly6` is the FUNCTION-member counterpart to `b_mass_peak`'s
DISTRIBUTION members, and it is the only row anywhere that scores the
determiniser's §09 function lowering. Note that a LOWERED member needs no JS
catalogue entry: the lowering leaves base ops only, so the `distances` member
scores on det-js even though `flatppl-js/packages/engine/standard-modules.ts`
registers no `distances` module. `corpora/hs3/conversions/histfactory`
uses the same two members through the `convert` runner, which determinizes
too since 2026-09-02 (before that it scored via the environment-selected
engine, defaulting to pure `js`, so it never determinized at all).
`tests/core/test_hs3_absolute_density.py` scores the same members absolutely.

Engine gaps found while authoring that could NOT be expressed as rows
here (no green shape and no refusal to pin) are recorded in
`flatppl-dev/testsuite-coverage-proposals.md` per proposal.

## Run

```sh
pixi run test                                              # includes every dir here
PYTHONPATH=$PWD/src pixi run python -m flatppl_testsuite.unified.regen corpora/coverage/<id>   # refreeze
```
