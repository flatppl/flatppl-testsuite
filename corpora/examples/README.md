# Examples corpus

The fourth corpus for `flatppl-testsuite`: vendored copies of the worked
`.flatppl` models from `../flatppl-examples`, scored via the convert-free
det-js path, exactly like `corpora/bayesian_inference/` and
`corpora/fragment/`. Unlike those two, the flatppl-examples models are pure
model DEFINITIONS — each ends in `posterior = bayesupdate(L, prior)` with no
query — so each test dir's `test.json` carries a `binding` (`"posterior"`)
and a `points` theta grid instead of a fixed point; the `(logdensity,
det-js)` runner appends `__score__ = logdensityof(<binding>, <theta
record>)` per point and scores it (Mode B of `unified/runners/logdensity_detjs.py`).

A `test.json` may also carry a `status` field (`"lowers"`, currently the only
value present — every entry here determinizes and scores cleanly). Models
that carry no scoreable `posterior` query were never vendored here at all —
see "Excluded models" below. A determiniser refusal (`DeterminizeRefused`,
exit 3) is a SKIP under the unified harness, not a failure, regardless of
`status` — it means the model uses a construct outside the determiniser's
current density fragment.

## Excluded models

The flatppl-examples models this corpus does not vendor/score, with the reason:

| Model | Why excluded |
|-------|--------------|
| `minimal` | No `posterior` binding — ends in a kernel application, no query to construct. |
| `aggregates` | No `posterior` binding — a purely deterministic showcase of the `aggregate` array primitive (no random variables). |
| `bayesian_inference_common` | Module include (loaded by variants 3/4), not a standalone model — no `posterior`. |
| `bayesian_inference_priors` | Module include (loaded by `bayesian_inference_common`), not a standalone model — no `posterior`. |
| `bayesian_inference_3` | Uses `load_module(...)`. `DetJsScoreEngine` determinizes a temp-file COPY of the appended source in a system temp dir, and `load_module` resolves relative to that input file's directory — so the sibling module cannot be found and the model is unscoreable by this harness as written. (With the module co-located the determiniser still REFUSES: a `~` draw from a module-namespaced distribution — `common.theta1_dist` — is not resolved to a built-in constructor, unlike a local alias. Determiniser gap.) |
| `bayesian_inference_4` | Same as `bayesian_inference_3` (`load_module` unresolvable from the temp dir; module-namespaced draw distribution refuse). |

## Contents

One directory per scored example (14 in total), each a unified test dir:

| Path | What |
|------|------|
| `<test_id>/model.flatppl` | The vendored example model (pure definition, ending in `posterior = bayesupdate(L, prior)`). |
| `<test_id>/test.json` | `test_type: "logdensity"`, `engines: ["det-js"]`, `binding: "posterior"`, the `points` theta grid, the frozen `expected` list (one value per point, same order), tolerances, `status`. |
| `<test_id>/test.py` | INDEPENDENT oracle: `oracle(point)` reproduces the matching `expected[i]` in closed form. |

`tests/test_unified.py` discovers every directory here automatically; there is no
per-corpus gate script or manifest anymore.

## Oracle

Every `test.py::oracle(point)` is computed with an INDEPENDENT oracle (scipy /
Julia Distributions.jl — never the sibling FlatPPL engine), reproducing the
same prior-plus-likelihood log-density the model's `posterior` query
constructs, at each theta-grid point.

## Run

```sh
pixi run test                                                     # pytest, incl. every dir here
pixi run unified                                                   # the unified harness alone
PYTHONPATH=$PWD/src pixi run -e stablehlo regen corpora/examples/<test_id>   # refreeze
```
