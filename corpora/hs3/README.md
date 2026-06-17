# HS3 corpus

The first corpus for `flatppl-testsuite`: frozen HS3 models, converted to FlatPPL
and scored against frozen expected values. Self-contained — no upstream checkout
needed to run it.

## Contents

| Path | What |
|------|------|
| `manifest.json` | Index of the vendored fixtures (subset of the upstream HS3TestSuite index). |
| `fixtures/<id>/` | One vendored HS3 model: `hs3.json` (input), `expected.json` (frozen results), `metadata.json` (provenance), `model.flatppl` (golden conversion). |
| `conversions/` | Three HS3-paper Appendix A models (`gaussian`, `product`, `histfactory`) as HS3 + golden FlatPPL + ROOT oracle. See `conversions/README.md`. |
| `tests/` | The corpus's own pytest definitions. `tests/test_hs3.py` at the repo root is a one-line shim that re-exports them. |
| `run_comparisons.py` | Prints formatted comparison tables (fixture 2ΔNLL vs frozen; conversions vs ROOT). |
| `ATTRIBUTION.md` | Source, commit, license, and the deliberate `rf103` deviation. **Read this before editing any fixture.** |

## Run

```sh
pixi run hs3                      # formatted tables (run_comparisons.py)
pixi run test                     # pytest, incl. this corpus via its shim
```

## Vendored vs upstream

The vendored set is only the fixtures the harness converts **and** scores end to
end (`rf101_basics`, `rf103_interprfuncs`). To run against the full upstream suite
instead, set `HS3SUITE` to an HS3TestSuite checkout. The `rf103` `expected.json`
is an intentional split/loosened variant that lives only here — see
`ATTRIBUTION.md` for why.
