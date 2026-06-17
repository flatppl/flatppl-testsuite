# Vendored HS3 corpus

These fixtures are vendored from the **HS3TestSuite** — backend-neutral HS3
conformance fixtures.

- Source: https://github.com/flatppl/HS3TestSuite
- Commit: `9d04e321ae6fddd283a35507f14ecf852eb7df61` (branch `master`)

Each fixture is a frozen HS3 model (`hs3.json`) with machine-readable expected
results (`expected.json`) and provenance (`metadata.json`); `manifest.json` is a
subset of the upstream index covering only the vendored fixtures. The per-fixture
`metadata.json` records the original RooFit tutorial each model derives from.

## Vendored subset

Only the fixtures the harness currently converts **and** scores end to end:

- `rf101_basics` — Gaussian, unbinned. Verbatim from upstream `master`.
- `rf103_interprfuncs` — `generic_dist` / `generic_function`. `hs3.json` and
  `metadata.json` are verbatim from upstream `master`; **`expected.json` is the
  harness's split/loosened variant** (not upstream): its `twice_delta_nll_scan`
  is split into `_match` (tight) and `_diverge` (loosened) checks. The diverging
  scan points differ from the frozen RooFit values by ~1e-3 relative — RooFit's
  `RooGenericPdf` numeric normalizer carries ~1e-4 error in `log Z`, which over
  the 2000-event dataset yields the gap. An independent scipy quadrature (to
  ~1e-11) confirms the FlatPPL value is the more accurate one; see the `note`
  fields in `expected.json`. This deviation is intentional and lives only here,
  not upstream.

To run against the full upstream suite instead of this vendored subset, set
`HS3SUITE` to an HS3TestSuite checkout.

## `conversions/` — HS3 paper Appendix A

`conversions/` holds three models from **Appendix A of the HS3 paper**, each as an
HS3 input (`<model>.hs3`), its golden FlatPPL conversion (`<model>.flatppl`), and a
RooFit/ROOT oracle (`<model>_root.py`); `repro_hs3.sh` / `repro_hs3_js.cjs` reproduce
the FlatPPL-vs-ROOT comparison, and `score_js.cjs` is the single-point JS scorer.

- HS3 paper: https://arxiv.org/abs/2606.01760
- HS3 spec: https://hep-statistics-serialization-standard.github.io/

Models: `gaussian` (A.1), `product` (A.2, normalized Gaussian product), `histfactory`
(A.3, 2-bin with normsys + Barlow–Beeston staterror). The `.flatppl` goldens are
verified against the ROOT oracles (gaussian agrees on the absolute log-density;
histfactory agrees on Δ(log L) — ROOT drops the per-bin `log n!` term).
