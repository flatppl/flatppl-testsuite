# pyhf corpus

167 vendored pyhf workspaces, converted with `flatppl convert --from pyhf` and
scored against **pyhf's own absolute `Model.logpdf`** at six parameter points
each. 1002 frozen numbers. The whole matrix of the pyhf import audit
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

The 167 rows add about 60 s to `pixi run test`.

## The `logpdf_points` check

```
convert --from pyhf  ->  logdensityof(<binding>, <record>) at each point
                     ->  compare_vectors against the frozen pyhf logpdf
```

Scoring is the det-js path (`unified/detjs_exec.log_density_points`), batched:
each point needs its own `determinize`, because theta is spliced into the source
before lowering, but the whole batch's Node evaluation runs in one process. That
is what keeps 1002 points at ~60 s rather than ~8 minutes.

**Tolerance is `atol = 1e-9`, `rtol = 0`.** Measured over all 1002 points, the
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
reproduces all 1002 points and all 1002 values bit for bit.

A fixture whose `test.json` already carries points reuses them, so an ordinary
regen is a pure re-measurement whose diff shows only moved values.

The record shape per parameter comes from the converter's own emitted
`elementof` declaration, not a guess: pyhf's per-bin kinds (shapesys,
staterror, shapefactor) become a vector even at one bin, while normfactor,
normsys, histosys and lumi stay scalar. A workspace the converter refuses makes
the generator fail loudly instead of skipping.

## The nine `binding_note` fixtures

A workspace with no constrained parameter has exactly one likelihood term, so
the converter emits `likelihood = <channel>_likelihood` — a bare alias.
`flatppl determinize` refuses to score through a bare alias:

```
determinize: refuse `c_likelihood` in `likelihood` (…): expected likelihoodof
```

That is a determiniser gap with no converter involved. A hand-written
`top = lik; logdensityof(top, …)` is refused identically. The generator resolves
the alias and freezes the aliased binding, recording why in the check's
`binding_note`, so those nine fixtures still gate the converter numerically
rather than being deleted from the corpus by an unrelated gap. When the
determiniser learns to follow an alias, a regen moves the binding back on its
own. Logged in `flatppl-dev/TODO-hs3-converter.md`.

## Coverage

| Group | Count | What |
|---|---|---|
| `m_<kind>` | 7 | one modifier kind alone: normfactor, normsys, histosys, shapesys, shapefactor, staterror, lumi |
| `sw1_*`, `sw2_*`, `sw3_*` | 108 | every subset of the six per-sample kinds on one background sample, at 1, 2 and 3 bins, beside a signal sample carrying the POI |
| `lum1_*`, `lum2_*` | 8 | lumi paired with each other kind, at 1 and 2 bins |
| `two_hist`, `two_norm`, `two_shap`, `two_stat` | 4 | each kind in two channels, per-channel names for the per-bin kinds |
| `pyhfval_*` | 10 | pyhf's own `tests/test_validation.py` workspaces |
| named | 30 | the surface items and defect classes the audit called out individually (below) |

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
`two_measurements_diff_poi`, `two_measurements_conflicting_auxdata`.

Names are the audit's, verbatim, so each row maps to a line of the coverage
table in `flatppl-dev/audit-fix-pyhf.md`.

`interpcode` has no row because it is not part of the pyhf JSON surface: every
modifier schema sets `additionalProperties: false` and none declares one, so an
interpolation code is a `pyhf.Model` construction argument, not a workspace
field.

## The gate bites

Scored with a `flatppl` built from rust `main` at `0b5fc1c`, the immediate
parent of the first pyhf import fix, **84 of the 167 rows fail**:

| Outcome | Rows |
|---|---|
| numeric mismatch | 80 |
| unscoreable (`no derivation for '__score__'`) | 4 |
| pass | 83 |

The worst pre-fix difference is **908.5**, on three of the two-bin sweep rows
that carry both a shapesys and a staterror. The four unscoreable rows are the
degenerate-bin fixtures: before the fix the lowering makes the density NaN or
`+inf` and the engine derives nothing, which the runner reports as a failure,
not a skip.

At `49c657b` all 167 rows pass, with a worst difference of 1.819e-12. The 83
rows that pass on both binaries pin what the fixes had to leave alone.

## Attribution

`pyhfval_*` are built from the `spec_*` fixtures of pyhf's
`tests/test_validation.py` at pyhf `v0.7.6` together with the
`validation/data/*.json` bindata they read, with observations set to the rounded
nominal total per bin. pyhf is Apache-2.0.

Every other workspace was written for the import audit against
`pyhf/schemas/1.0.0/defs.json`, the workspace schema pyhf 0.7.6 installs, and
measured with pyhf 0.7.6 on the numpy backend at 64-bit precision.
