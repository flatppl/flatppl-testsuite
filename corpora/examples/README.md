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
refusal message. `minimal` (no `posterior` binding) is excluded from the
manifest entirely — see its top-level `excluded` list.

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

Scaffold only (Task 1): `manifest.json` has zero entries, so `pixi run
examples` prints an empty table and exits 0, and the pytest parametrization
over manifest entries is empty. Task 2 populates the manifest (and adds
`<test_id>/expected.json` for each `"lowers"` entry); Task 3 fills in
`gen_expected.py`'s oracle functions.
