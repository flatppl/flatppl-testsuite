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
`DeterminizeRefused`), with `reason` naming a required substring of the
refusal message (the whole `reason` string must be a substring, so any
human-facing categorization lives in the sibling `category`/`note` fields the
Suite ignores, not in `reason`). Models that carry no scoreable `posterior`
query are excluded from the manifest entirely — see the top-level `excluded`
list.

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

## Contents

| Path | What |
|------|------|
| `manifest.json` | `examples: [...]` entries (`test_id`, `model`, `binding`, `theta` grid, `status`, `reason?`); `excluded` lists non-posterior models. Starts empty (Task 1); Task 2 populates it. |
| `<test_id>/expected.json` | Frozen oracle values for a `"lowers"` entry, one `checks[]` item per theta-grid point (schema mirrors `corpora/bayesian_inference/`'s `checks` list, plus an `index` selecting the grid point). No directory for `"refuses"` entries — there is no oracle value, only the expected refusal. |
| `gen_expected.py` | Reproduces every `<test_id>/expected.json` from an INDEPENDENT oracle. Stub only until Task 3 (see its docstring). |
| `gate.py` | Prints a `test_id::check -> LOWERS/REFUSE/MISMATCH` table (`pixi run examples`). |
| `tests/test_examples_gate.py` | The corpus's own pytest definitions; `tests/test_examples.py` at the repo root is a one-line shim that re-exports them. |

## Oracle

Every `"lowers"` entry's `expected.json` is meant to be computed with an
INDEPENDENT oracle (scipy / Julia Distributions.jl — never the sibling
FlatPPL engine), reproducing the same prior-plus-likelihood log-density the
manifest's query constructs, at each theta-grid point. See
`gen_expected.py`'s docstring; oracle functions land in Task 3.

## Run

```sh
pixi run examples                                    # formatted table (gate.py)
pixi run test                                        # pytest, incl. this corpus via its shim
pixi run python corpora/examples/gen_expected.py     # regenerate + verify expected.json
```

## Status

Manifest populated (Task 2): 14 scored entries — 11 `"lowers"`, 3
`"refuses"` — plus 6 `excluded` models. The `"refuses"` entries verify
cleanly through the gate today (each raises `DeterminizeRefused` with its
`reason` substring). The `"lowers"` entries have **no `expected.json` yet**
(Task 3), and the Task-1 Suite cannot represent a "lowers, oracle-pending"
entry: `ExamplesGateSuite.run` reads each `"lowers"` entry's
`<test_id>/expected.json` unconditionally, so `pixi run examples` currently
raises `FileNotFoundError` on the first `"lowers"` entry until Task 3 freezes
the oracles (or the Suite is taught to skip an entry whose `expected.json` is
absent). Task 3 fills in `gen_expected.py`'s per-model oracle functions and
writes the `expected.json` files.
