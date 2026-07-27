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
| `fixtures/<id>/` | One vendored HS3 model: `hs3.json` (input), `metadata.json` (provenance), `model.flatppl` (golden conversion), `test.json` (`fixture_kind: "fixture"`; `static_integrity`/`structure_import`/`twice_delta_nll_scan` checks, the last carrying the frozen ROOT `expected` vector). |
| `conversions/<model>/` | Three HS3-paper Appendix A models (`gaussian`, `product`, `histfactory`): `<model>.hs3.json`, `<model>.flatppl`, `<model>_root.py`, `test.json` (`fixture_kind: "conversion"`; one `twice_delta_nll_points` check with the frozen ROOT `expected` vector). See `conversions/README.md`. |
| `conversions/gen_expected.py` | Regenerates every `conversions/<model>/test.json`'s frozen ROOT vector from the live ROOT/RooFit oracle (needs the `root` pixi env — `unified/regen.py` deliberately does not reproduce this offline). |
| `ATTRIBUTION.md` | Source, commit, license, and the deliberate `rf103` deviation. **Read this before editing any fixture.** |

`tests/test_unified.py` discovers every directory here automatically; there is no
per-corpus gate script, manifest, or comparison-table script anymore.

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
