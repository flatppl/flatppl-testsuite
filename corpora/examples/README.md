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

## Unscoreable entries

Four `"lowers"`-shaped entries determinize successfully (the determinizer's
structural job is done) but the score stage then crashes on a named,
currently-open gap, so their status is `"unscoreable"` instead of
`"lowers"`:

| `test_id` | `category` | Gap |
|-----------|------------|-----|
| `ex_best_estimation` | `engine-gap` | `Uniform(interval(0.1, 20.0))` (`sigma1`/`sigma2`) — flatppl-js's fixed-value evaluator marks the Uniform-interval prior "not evaluable in sampler context" and falls back to a generic "no derivation" instead of propagating that reason (Buffy #248). |
| `ex_capture_recapture` | `engine-gap` | `Uniform(interval(0, rcp_max))` (`rcp`) — same flatppl-js gap as above (Buffy #248). |
| `ex_partial_pooling` | `engine-gap` | `Uniform(interval(0.0, 1.0))` (`phi`) — same flatppl-js gap as above (Buffy #248). |
| `ex_gamma_reparam` | `determiniser-gap` | `Gamma(gamma_shape_rate(mu, sigma))` — `gamma_shape_rate` is a multi-output function (`record(shape = ..., rate = ...)`); the determinizer's auto-splat doesn't distribute its record output across `Gamma`'s `shape`/`rate` keyword arguments, so the emitted flatpdl leaves `shape` holding the whole two-field record and drops `rate` entirely (Buffy #247). |

Each entry keeps its `<test_id>/expected.json`, frozen from before the gap
was discovered, and marks it `"pending_oracle": true` — `ExamplesGateSuite`
never reads that file for an `"unscoreable"` entry, so nothing currently
verifies it matches. `ExamplesGateSuite.run` asserts the structural half only
(`flatppl determinize` exits 0) and checks the score-stage crash message
against `reason`; it flags — rather than silently swallows — a regression
(the entry stops determinizing) or an improvement (the entry starts scoring
cleanly). When the gap is fixed, flip the entry's `status` back to
`"lowers"` (and drop `reason`/`category`/`note`/`pending_oracle`) — its
frozen `expected.json` immediately starts being checked against the oracle,
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
