# pyhf corpus

172 vendored pyhf workspaces, converted with `flatppl convert --from pyhf` and
scored against **pyhf's own absolute `Model.logpdf`** at six parameter points
each. 1032 frozen numbers. The whole matrix of the pyhf import audit
(`flatppl-dev/audit-fix-pyhf.md`), which found five wrong-number defect classes
in the converter — every one of which converted at exit 0 and passed every gate
the suite had, because the suite had no way to hold a pyhf fixture at all.

The refusal side lives beside this, in `corpora/pyhf-rejects/`.

## What makes it different from `corpora/hs3/`

**The comparison is absolute, not a Delta.** Both HS3 flavours compare an
offset-invariant `twice_delta_nll` against a reference point in the same model,
so a normalization difference legitimately drops out. pyhf and the FlatPPL
lowering both carry the full Poisson normalization, so the absolute values must
agree — and they have to be compared that way: the staterror constraint-form
defect changes the normalization, which a Delta inside one model would partly
cancel. Do not port these into a `twice_delta_nll_*` check kind.

**The oracle is the foreign tool itself.** ROOT is the HS3 corpus's provenance;
pyhf is this corpus's. The rust converter is the artifact under test and the
FlatPPL engine scores its output, so neither can stand in as the reference.

| Path | What |
|------|------|
| `<fixture>/pyhf.json` | The vendored pyhf workspace. |
| `<fixture>/test.json` | `fixture_kind: "pyhf"`; a `static_integrity` check and a `logpdf_points` check carrying the frozen pyhf vector, the parameter points, the binding, and the tolerance. |
| `gen_expected.py` | Regenerates every fixture's frozen vector from live pyhf. |

`tests/test_unified.py` discovers these directories like every other corpus, so
`pixi run test` gates them and CI needs no pyhf environment. The `pyhf` pixi
environment exists only to regenerate:

```sh
FLATPPL_BIN=/path/to/flatppl pixi run -e pyhf gen-pyhf
```

The 172 rows add roughly 100 s to `pixi run test`: 415 s without them against
504 s, 510 s and 561 s with them over three runs on the same machine. The
spread between those three is machine load, not the corpus.

## The `logpdf_points` check

```
convert --from pyhf  ->  logdensityof(<binding>, <record>) at each point
                     ->  compare_vectors against the frozen pyhf logpdf
```

Scoring is the det-js path (`unified/detjs_exec.log_density_points`), batched:
each point needs its own `determinize`, because theta is spliced into the source
before lowering, but the whole batch's Node evaluation runs in one process. That
is what keeps 1032 points inside a couple of minutes: one Node start per
fixture instead of one per point.

**Tolerance is `atol = 1e-9`, `rtol = 0`.** Measured over all 1032 points, the
worst absolute difference is **1.819e-12**, on `sw3_norm_norm_shap_shap_stat`
whose log-density is about -1.9e+3 — a relative difference of 1e-15, one or two
ulp of a double. The band is ~550x that, so float reassociation between pyhf's
numpy reduction order and the engine's cannot reach it, while the smallest
defect the audit found (1.276e+0) is nine orders of magnitude outside it.

`rtol` stays 0 deliberately. The corpus's deepest log-density is -4.66e+3, so an
`rtol` of 1e-12 would admit 4.7e-9 there, five times looser than `atol`, on
exactly the rows where a constant offset from a wrong normalization hides best.

## Parameter points

Six per fixture: `suggested_init()` plus five drawn uniformly inside
`suggested_bounds()` clipped to init ± 2.5, with every `suggested_fixed()`
component held at its init. `SEED = 137`, and the draw is one vectorised
`rng.uniform` per point, so the whole corpus's point sets reproduce from
scratch. These are the audit's own sets: regenerating from an empty tree
reproduces all 1032 points and all 1032 values bit for bit.

A fixture whose `test.json` already carries points reuses them, so an ordinary
regen is a pure re-measurement whose diff shows only moved values.

The record shape per parameter comes from the converter's own emitted
`elementof` declaration, not a guess: pyhf's per-bin kinds (shapesys,
staterror, shapefactor) become a vector even at one bin, while normfactor,
normsys, histosys and lumi stay scalar. A workspace the converter refuses makes
the generator fail loudly instead of skipping.

## Every fixture scores the binding the converter names

All 172 rows score the module's own `likelihood`. That was briefly not true. A
workspace whose only parameters are unconstrained — a normfactor or a
shapefactor, which pyhf leaves free — has exactly one likelihood term, so the
converter emits `likelihood = <channel>_likelihood`, a bare alias, and
`flatppl determinize` refused to score through one:

```
determinize: refuse `c_likelihood` in `likelihood` (…): expected likelihoodof
```

Nine fixtures worked around it by freezing the aliased name, each recording why
in a `binding_note`. That was never a converter defect — a hand-written
`top = lik; logdensityof(top, …)` was refused identically — and rust `637a3e9`
now follows the alias. The workaround is gone: regenerating rebound exactly
those nine (`m_normfactor`, `m_shapefactor`, `poi_bounds_inits`, `sw1_none`,
`sw1_shap`, `sw2_none`, `sw2_shap`, `sw3_none`, `sw3_shap`) to `likelihood` and
moved **no** frozen value and **no** point, which is what a pure lowering fix
should do.

## Coverage

| Group | Count | What |
|---|---|---|
| `m_<kind>` | 7 | one modifier kind alone: normfactor, normsys, histosys, shapesys, shapefactor, staterror, lumi |
| `sw1_*`, `sw2_*`, `sw3_*` | 108 | every subset of the six per-sample kinds on one background sample, at 1, 2 and 3 bins, beside a signal sample carrying the POI |
| `lum1_*`, `lum2_*` | 8 | lumi paired with each other kind, at 1 and 2 bins |
| `two_hist`, `two_norm`, `two_shap`, `two_stat` | 4 | each kind in two channels, per-channel names for the per-bin kinds |
| `pyhfval_*` | 10 | pyhf's own `tests/test_validation.py` workspaces |
| named | 35 | the surface items and defect classes the audit called out individually (below) |

The named fixtures: `one_bin`, `many_bins`, `two_channels`,
`three_channels_all_kinds`, `all_kinds_one_sample`, `two_histosys`,
`three_histosys`, `histosys_shapesys`, `staterror_plus_shapesys`,
`staterror_two_samples`, `shared_normfactor_across_channels`,
`normfactor_shared_across_channels_diff_bins`,
`shared_normsys_across_channels`, `shared_histosys_across_samples`,
`shapefactor_shared_across_channels`, `normsys_histosys_share_a_name`,
`lumi_multi_channel`, `lumi_with_normsys`, `normsys_auxdata_override`,
`staterror_sigmas_override`, `shapesys_factors_override`, `fixed_normsys`,
`poi_bounds_inits`, `shapesys_zero_unc_bin`, `shapesys_zero_nominal_bin`,
`staterror_zero_err_bin`, `staterror_zero_nominal_bin`, `multichan_old`,
`two_measurements_diff_poi`, `two_measurements_conflicting_auxdata`,
`asimov_fractional_counts`, and the four spanning-staterror fixtures below.

### The fractional observed count

`asimov_fractional_counts` is the corpus's only workspace whose observations are
not all whole numbers, and the only row that scores
`hepphys.ContinuedPoisson` as the count likelihood. Its two channels are Asimov
data at `mu = 1` with every systematic at nominal, so each bin's observation is
the per-bin total of the sample nominals:

| Channel | Observed | Count likelihood |
|---|---|---|
| `ca` | `[70.0, 70.0]` | `Poisson` |
| `cb` | `[34.0, 22.75]` | `hepphys.ContinuedPoisson` |

The rule the converter applies is **per channel, not per bin** (rust
`78e0b03`, `validate_observed_counts`): a channel whose counts are all
nonnegative integers keeps `Poisson`, and a channel holding any finite
nonnegative non-integer count scores `hepphys.ContinuedPoisson` for every bin.
`cb` mixes `34.0` with `22.75` inside one channel precisely to pin that, and
`ca` beside it pins that a fractional channel does not drag its neighbour off
`Poisson`.

Both distributions carry the same log-density on a whole number
(§09: `ContinuedPoisson`'s density is `rate^x e^-rate / Gamma(x+1)`, the Poisson
density with the factorial continued by the gamma function), which is why pyhf
scores the whole workspace through one `lgamma` form and the two channels still
agree with it. §08 gives `Poisson` the support `nonnegintegers`, so the
pre-`78e0b03` lowering is not merely different but unscoreable: the emitted
source with `Poisson.(cb_expected)` in place of the continued form evaluates to
`-inf` on this fixture, against a frozen `-17.317288936632934`.

The independent oracle is the closed form summed by hand in Julia
(`SpecialFunctions.loggamma`), which reaches pyhf and the engine from neither
side:

| Point | Julia closed form | pyhf 0.7.6 | det-js engine |
|---|---|---|---|
| 0 (`mu = 1`) | `-17.31728893663294` | `-17.317288936632934` | `-17.31728893663292` |
| 1 (`mu = 3.0036…`) | `-52.78522704517505` | `-52.78522704517504` | `-52.78522704517503` |

### The spanning staterror

pyhf lets one staterror name span channels: it gets ONE paramset over the union
of every carrying channel's bins, and each channel masks its own slice. The
converter used to emit a single `cartpow` multiplied into both channels, which
correlated components pyhf keeps independent and gave the constraint only the
first channel's bins; it then refused the shape outright rather than emit a
wrong number. Rust `main` `106a7d2` emits the spanning form, and `3e64224`
sizes an `auxdata` override by the spanning component count, so four fixtures
became scoreable at once:

| Fixture | Components | Shape |
|---|--:|---|
| `staterror_shared_across_channels` | 5 | two 2-bin channels |
| `staterror_span_three` | 6 | three channels |
| `staterror_span_uneq` | 5 | a 2-bin and a 3-bin channel |
| `staterror_span_aux` | 4 | an `auxdata` override across every component |

All four need a converter at or after `3e64224`. How an older one fails depends
on which era it is from, and neither era produces a usable number:

- from `ba8b7c8` to `49c657b` the converter REFUSES the shape, naming pyhf's
  per-channel staterror convention;
- before `ba8b7c8` it converts at exit 0 and emits the wrong arity. On
  `staterror_shared_across_channels` it declares
  `st = elementof(cartpow(posreals, 2))`, one channel's bins, where pyhf's
  paramset has 4 components. The frozen 4-element record then does not bind and
  the row fails as `no derivation for '__score__'`, which is the arity defect
  made visible rather than a masked wrong number.

`staterror_shared_across_channels` arrived here by promotion: it was a pinned
refusal in `corpora/pyhf-rejects/` whose row was written to fail on its exit
code the moment the lowering landed, which is what happened.

Names are the audit's, verbatim, so each row maps to a line of the coverage
table in `flatppl-dev/audit-fix-pyhf.md`.

`interpcode` has no row because it is not part of the pyhf JSON surface: every
modifier schema sets `additionalProperties: false` and none declares one, so an
interpolation code is a `pyhf.Model` construction argument, not a workspace
field.

## The gate bites

Scored with a `flatppl` built from rust `main` at `0b5fc1c`, the immediate
parent of the first pyhf import fix, **88 of the 171 rows fail**:

| Outcome | Rows |
|---|---|
| numeric mismatch | 80 |
| unscoreable (`no derivation for '__score__'`) | 8 |
| pass | 83 |

The worst pre-fix difference is **908.5**, on three of the two-bin sweep rows
carrying both a shapesys and a staterror. The eight unscoreable rows split in
two: the four degenerate-bin fixtures, whose pre-fix lowering makes the density
NaN or `+inf` so the engine derives nothing, and the four spanning-staterror
fixtures, whose pre-fix declaration has the wrong component count. Both are
reported as failures, not skips.

At `3e64224` all 171 rows pass, with a worst difference of **1.819e-12** on
`sw3_norm_norm_shap_shap_stat`. The 83 rows that pass on both binaries pin what
the fixes had to leave alone.

Both binaries predate `asimov_fractional_counts`, so that row is outside the
counts above; its own pre-fix evidence is in "The fractional observed count".
Its worst difference at rust `78e0b03` is **1.421e-14**, inside the corpus's
1.819e-12 worst case, so the corpus band is unchanged.

## Attribution

`pyhfval_*` are built from the `spec_*` fixtures of pyhf's
`tests/test_validation.py` at pyhf `v0.7.6` together with the
`validation/data/*.json` bindata they read, with observations set to the rounded
nominal total per bin. pyhf is Apache-2.0.

Every other workspace was written against `pyhf/schemas/1.0.0/defs.json`, the
workspace schema pyhf 0.7.6 installs, and measured with pyhf 0.7.6 on the numpy
backend at 64-bit precision. All but `asimov_fractional_counts` come from the
import audit.
