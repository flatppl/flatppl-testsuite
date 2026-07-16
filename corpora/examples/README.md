# Examples corpus

The fourth corpus for `flatppl-testsuite`: determinises the worked
`.flatppl` models in the sibling `../flatppl-examples` checkout and scores
them via the convert-free det-js path, exactly like `corpora/bayesian_inference/`
and `corpora/fragment/`. Unlike those two, the flatppl-examples models are
pure model DEFINITIONS — each ends in `posterior = bayesupdate(L, prior)`
with no query — so this corpus constructs the query itself: for every
manifest entry, `logdensityof(posterior, theta_i)` is appended and scored at
each point `theta_i` in that entry's theta grid, via
`DetJsScoreEngine.log_density` (append `__score__ = ...`, `flatppl
determinize`, `score_flatpdl.cjs`).

An entry's `status` says what should happen: `"lowers"` means the query is
expected to determinize and score, compared to a frozen INDEPENDENT oracle;
`"refuses"` means the determinizer is expected to reject the query (exit 3,
`DeterminizeRefused`); `"unscoreable"` means the determinizer DOES lower the
query (same structural claim as `"lowers"`) but the score stage then crashes
on a named, documented engine/determiniser gap (see "Unscoreable entries"
below). Both `"refuses"` and `"unscoreable"` name a `reason` — a required
substring of the refusal message, resp. the score-stage crash message (the
whole `reason` string must be a substring, so any human-facing
categorization lives in the sibling `category`/`note` fields the Suite
ignores, not in `reason`). Models that carry no scoreable `posterior` query
are excluded from the manifest entirely — see the top-level `excluded` list.

## Excluded models

The top-level `excluded` list names the flatppl-examples models this corpus
does not score, with the reason:

| Model | Why excluded |
|-------|--------------|
| `minimal` | No `posterior` binding — ends in a kernel application, no query to construct. |
| `aggregates` | No `posterior` binding — a purely deterministic showcase of the `aggregate` array primitive (no random variables). |
| `bayesian_inference_common` | Module include (loaded by variants 3/4), not a standalone model — no `posterior`. |
| `bayesian_inference_priors` | Module include (loaded by `bayesian_inference_common`), not a standalone model — no `posterior`. |
| `bayesian_inference_3` | Uses `load_module(...)`. `DetJsScoreEngine` determinizes a temp-file COPY of the appended source in a system temp dir, and `load_module` resolves relative to that input file's directory — so the sibling module cannot be found and the model is unscoreable by this harness as written. (With the module co-located the determiniser still REFUSES: a `~` draw from a module-namespaced distribution — `common.theta1_dist` — is not resolved to a built-in constructor, unlike a local alias. Determiniser gap; see the sweep notes.) |
| `bayesian_inference_4` | Same as `bayesian_inference_3` (`load_module` unresolvable from the temp dir; module-namespaced draw distribution refuse). |

## Unscoreable / refuses entries

None currently — every manifest entry is `"lowers"`. The determiniser and
flatppl-js gaps the former `"unscoreable"` (`ex_best_estimation`,
`ex_capture_recapture`, `ex_partial_pooling`, `ex_gamma_reparam`) and
`"refuses"` (`ex_dissimilar_mixture`, `ex_linear_regression`,
`ex_zero_inflated_binomial`) entries were pinned against have all landed
(auto-splat, Uniform-interval fixed-eval, the bare-`sqrt`/dependent-prior
bijection path, positional `Dirac`, `normalize(superpose(...))` mixture mass,
and residual-user-call inlining), so each now lowers and is checked against
its frozen independent oracle.

The status mechanism remains for future gaps. A `"refuses"` entry asserts the
determiniser exits 3 (`DeterminizeRefused`, checked against `reason`); an
`"unscoreable"` entry asserts only the lowering half (`flatppl determinize`
exits 0) plus a score-stage crash matching its documented `reason` — its
frozen `expected.json` is not checked while pending, and `ExamplesGateSuite`
flags (rather than swallows) both a regression (stops determinizing) and an
improvement (starts scoring cleanly). Re-tag an entry `"unscoreable"` /
`"refuses"` if a new gap appears, or back to `"lowers"` when it is fixed — a
`"lowers"` entry's frozen `expected.json` is then checked against the oracle,
with no need to regenerate it unless the model itself changed.

## Contents

| Path | What |
|------|------|
| `manifest.json` | `examples: [...]` entries (`test_id`, `model`, `binding`, `theta` grid, `status`, plus `reason`/`category`/`note`/`pending_oracle` for `"refuses"`/`"unscoreable"` entries); `excluded` lists non-posterior models. |
| `<test_id>/expected.json` | Frozen oracle values for a `"lowers"` (or `"unscoreable"`, pending) entry, one `checks[]` item per theta-grid point (schema mirrors `corpora/bayesian_inference/`'s `checks` list, plus an `index` selecting the grid point). No directory for `"refuses"` entries — there is no oracle value, only the expected refusal. |
| `gen_expected.py` | Reproduces every `<test_id>/expected.json` from an INDEPENDENT oracle. |
| `gate.py` | Prints a `test_id::check -> LOWERS/LOWERS(unscoreable: ...)/REFUSE/MISMATCH` table (`pixi run examples`). |
| `tests/test_examples_gate.py` | The corpus's own pytest definitions; `tests/test_examples.py` at the repo root is a one-line shim that re-exports them. |

## Oracle

Every `"lowers"` entry's `expected.json` is computed with an INDEPENDENT
oracle (scipy / Julia Distributions.jl — never the sibling FlatPPL engine),
reproducing the same prior-plus-likelihood log-density the manifest's query
constructs, at each theta-grid point. See `gen_expected.py`'s docstring. An
`"unscoreable"` entry's `expected.json` was computed the same way before its
gap was discovered — it stays frozen (`pending_oracle: true`) but unchecked
until the entry flips back to `"lowers"`.

## Run

```sh
pixi run examples                                    # formatted table (gate.py)
pixi run test                                        # pytest, incl. this corpus via its shim
pixi run python corpora/examples/gen_expected.py     # regenerate + verify expected.json
```

## Status

14 scored entries — 7 `"lowers"`, 4 `"unscoreable"`, 3 `"refuses"` — plus 6
`excluded` models. `pixi run examples` reports 0 MISMATCH / exit 0: the 7
`"lowers"` entries score-match their frozen oracle, the 3 `"refuses"`
entries raise `DeterminizeRefused` with the documented `reason` substring,
and the 4 `"unscoreable"` entries determinize cleanly and crash at the score
stage with the documented `reason` substring (see "Unscoreable entries").
