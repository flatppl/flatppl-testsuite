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

All four now refuse on the SAME gap, and refuse **by design**:

```
determinize: refuse normalize: normalize of an unnormalized measure needs a
closed-form mass rule; `totalmass` is not FlatPDL
```

The conditional lowering is
`normalize(logweighted(<lambda over the observable record>, Lebesgue(...)))`.
FlatPDL has no numeric `totalmass`, so there is no closed-form mass rule to
reach for and the determiniser is right to refuse. These rows are therefore
**permanent** refusals, not pending ones: do not read them as work queued
behind a capability. Their frozen `expected` vectors stay real ROOT values so
the rows remain honest and would start comparing numbers if the fragment ever
grew a mass rule, but nothing is scheduled to make that happen.

`static_integrity` and `structure_import` pass on all four and are what they
gate.

rf305 used to refuse one level further out, with a different message, and that
one WAS the harness's own defect rather than a property of the model:
`assemble` consulted the prenormalized notion once for the whole product, so a
`joint(...)` head made it re-wrap every factor. rf305 is
`joint(y = gaussy, x = gaussx)` with `gaussy` a raw `Normal` and `gaussx` the
conditional lowering, already `normalize`-headed; wrapping `gaussx` put a
`normalize` node under a `truncate` over a multivariate base, and the resulting
`builtin_touniform` message masked the real blocker. `assemble` now consults
the notion per factor, so rf305 reports the same refusal as its three siblings.

**Do not read a green scan row here as agreement without checking what was
observed.** These are 2-D conditional models and the converter emits a density
over the observable RECORD, while `assemble` observes one column per factor.
The observation is now bound by the converter's own `% observable:` annotation
rather than by `data_columns(...)[0]`, which matters because upstream reordered
these datasets' axes after the ROOT vectors were frozen -- a positional pick
follows a reorder silently, a name does not. Where the converter annotates no
observable, the caller's column still stands, which is the case for rf301,
rf302 and rf303. Each dir records its `upstream_axis_order` and repeats the
caveat in its `refusal_note`. See `ATTRIBUTION.md` for the commit-level detail.

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
