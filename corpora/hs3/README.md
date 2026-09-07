# HS3 corpus

The first corpus for `flatppl-testsuite`: frozen HS3 models, converted to FlatPPL
and scored against frozen expected values. Self-contained — no upstream checkout
needed to run it.

## Contents

The one corpus whose test dirs have `test_type: "convert"` — the model under
test is a foreign HS3 JSON fixture, not FlatPPL, and the `(convert, det-js)`
runner (`unified/runners/convert_detjs.py`) drives it via the shared helper
library `src/flatppl_testsuite/suites/hs3_import.py` (`score_scan`/`score_points`/
`_names_in_source`).

| Path | What |
|------|------|
| `fixtures/<id>/` | One vendored HS3 model: `hs3.json` (input), `metadata.json` (provenance), `model.flatppl` (golden conversion, asserted against live converter output by `tests/core/test_hs3_assemble.py` — the runner's `fixture` arm reads only `hs3.json`, so that test is the ONLY thing that catches a stale golden), `test.json` (`fixture_kind: "fixture"`; `static_integrity`/`structure_import`/`twice_delta_nll_scan` checks, the last carrying the frozen ROOT `expected` vector). |
| `conversions/<model>/` | Three HS3-paper Appendix A models (`gaussian`, `product`, `histfactory`): `<model>.hs3.json`, `<model>.flatppl`, `<model>_root.py`, `test.json` (`fixture_kind: "conversion"`; one `twice_delta_nll_points` check with the frozen ROOT `expected` vector). See `conversions/README.md`. |
| `conversions/gen_expected.py` | Regenerates every `conversions/<model>/test.json`'s frozen ROOT vector from the live ROOT/RooFit oracle (needs the `root` pixi env — `unified/regen.py` deliberately does not reproduce this offline). |
| `ATTRIBUTION.md` | Source, commit, license, and the deliberate `rf103` deviation. **Read this before editing any fixture.** |

`tests/test_unified.py` discovers every directory here automatically; there is no
per-corpus gate script, manifest, or comparison-table script anymore.

## Refusal pins

Until 2026-09-02 the runner scored through the environment-selected engine,
which defaults to pure JS, so these rows never ran `determinize` at all and the
`det-js` label was false for the whole corpus. It now names the det-js path the
way its sibling runners do (`tests/core/test_detjs_runners_are_det_js.py` guards
that). Three fixtures turned out not to lower, and each is pinned with
`status: "refuses"`, `allow_skip: true`, and the verbatim exit-3 message:
`rf103_interprfuncs`, `rf203_ranges`, `rf207_comptools`. The frozen `expected`
vectors stay REAL ROOT values, so each row starts comparing numbers the moment
the lowering lands. `allow_skip` is per-dir, so it also covers the dir's
`static_integrity` and `structure_import` checks; both pass today, and a
regression in either would show up as a skip rather than a failure.

The pinned message is the SHALLOWER of two stacked determiniser gaps. It fires
on `__M__ = <pdf>`, the measure ALIAS binding that `formats/hs3/importer.assemble`
emits on its prenormalized branch; the refusal is not construct-specific, since
`M = g1` for a bare `Normal` refuses identically. Inline the name into `iid(...)`
and the alias gap gives way to the real blocker, `normalize of an unnormalized
measure needs a closed-form mass rule; totalmass is not FlatPDL`, on the
converter's generic-pdf shape
`normalize(truncate(weighted(x -> polynomial(...), Lebesgue(reals)), interval(...)))`.
A polynomial's mass over an interval is closed form, so that is a capability
gap, not a conformant refusal. Fixing the alias gap alone moves the message
without turning any row green.

### The four rf30x conditional/composition fixtures

`rf301_composition`, `rf302_utilfuncs`, `rf303_conditional` and
`rf305_condcorrprod` are pinned the same way, and they arrived already
refusing. They needed the converter to APPLY a function named by a
distribution field: before that landed, all four converted at exit 0 and
emitted `model = Normal(mu = fy, sigma = sigma)`, passing the lambda itself as
a parameter, which the engine rejected outright with `record field 'mu': a
function may not appear inside a record (spec §04)`. That was a hard failure,
which `allow_skip` does NOT tolerate, so the four could not be vendored until
the converter applied the function.

They now refuse, and on TWO DIFFERENT gaps. Closing one will not light all
four:

| Fixtures | Refused on | The gap |
|---|---|---|
| rf301, rf302, rf303 | the CONVERTER's own emitted `normalize` | `normalize of an unnormalized measure needs a closed-form mass rule; totalmass is not FlatPDL`, on the conditional lowering `normalize(logweighted(<lambda over the observable record>, Lebesgue(...)))` |
| rf305 | the `normalize(truncate(...))` the HARNESS adds in `assemble` | the closed-form Z goes through `builtin_touniform`, the CDF only for a univariate continuous base (§07); this pdf's base is multivariate |

The rf301/302/303 gap is the same `totalmass` blocker the paragraph above
names, reached by a different route. rf305's is the multivariate-normalize
gap and is its own item.

Each dir's frozen `expected` stays a REAL ROOT vector, so all four start
comparing numbers as their gap closes. Meanwhile `static_integrity` and
`structure_import` both pass on all four and are what they gate.

**Do not read a green scan row here as agreement without checking the column.**
These are 2-D conditional models. The converter emits a joint-normalized density
over the observable RECORD, but `importer.assemble`'s 1-D path observes
`data_columns(...)[0]`, one column. Upstream reordered these datasets' axes after
the ROOT vector was frozen, so that column is now `y` in rf301, rf302 and rf305
while the Gaussian's variate is `x` and `y` is the conditioning observable. Each
dir records its `upstream_axis_order` and repeats the caveat in its
`refusal_note`. Whoever closes the gap needs `assemble` to handle the
multivariate observable record; the frozen vector cannot vouch for which axis was
scored. See `ATTRIBUTION.md` for the commit-level detail.

## Run

```sh
pixi run test                                                  # pytest, incl. every dir here
pixi run unified                                                # the unified harness alone
pixi run -e root python corpora/hs3/conversions/gen_expected.py  # refreeze the conversions' ROOT vectors
```

## Vendored vs upstream

The vendored set is only the fixtures the harness converts **and** scores end to
end (`rf101_basics`, `rf103_interprfuncs`). To run against the full upstream suite
instead, set `HS3SUITE` to an HS3TestSuite checkout. The `rf103` `test.json`'s
frozen `expected` is an intentional split/loosened variant that lives only here —
see `ATTRIBUTION.md` for why.
