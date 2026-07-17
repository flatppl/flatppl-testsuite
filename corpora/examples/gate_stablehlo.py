#!/usr/bin/env python3
"""Examples corpus gate under the StableHLO backend.

The sibling `gate.py` scores each flatppl-examples posterior under **det-js**
(append `__score__ = logdensityof(binding, theta)` to the model, determinize,
score). This gate instead scores the SAME posteriors under the **StableHLO**
backend, and it does so through a FlatPPL **query module** rather than by
splicing a query line into the example: per (example, theta-point) it writes a
tiny query module

    m = load_module("<relpath to the pristine example>")
    score = logdensityof(m.<binding>, record(<theta point as a literal>))

emits `@logdensity` from the local `flatppl` binary (`FLATPPL_BIN`, built with
the `stablehlo` feature), executes it under Enzyme-JAX, and compares the number
to the frozen INDEPENDENT scipy oracle in `corpora/examples/<test_id>/
expected.json` (the very oracle `gen_expected.py` froze for the det-js gate).
The example files themselves are never modified — they are composed as modules.

Theta is inlined as a literal record (no `elementof` free parameters), so
`@logdensity` takes no arguments: this is a value gate (the frozen oracle
carries `logdensity_value` only, no gradient), and a concrete point sidesteps
the free-parameter/argument-order and name-collision concerns that a
parameterised query module would carry.

`load_module` is referenced by a path computed at RUNTIME with
`os.path.relpath` (from the generated query module's temp directory to the
resolved flatppl-examples checkout) — never an absolute path, which would not
survive a move to CI. A relative path also lets an example's own transitive
`load_module` dependencies (e.g. `bayesian_inference_1` -> `_common`/`_priors`)
resolve against the example's real location.

The StableHLO emitter covers a narrower op set than det-js, so this gate is a
COVERAGE REPORT: an example whose posterior the emitter/determiniser cannot yet
lower is reported `REFUSE` (informational — a known gap, not a failure). Only a
real MISMATCH (emitted a number outside the f32 tolerance) trips a nonzero exit.

    FLATPPL_BIN=/path/to/target/release/flatppl pixi run -e stablehlo examples-shlo
"""
from __future__ import annotations

import json
import math
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent            # corpora/examples
REPO = HERE.parents[1]                             # repo root
sys.path.insert(0, str(REPO / "corpora" / "stablehlo"))

import executor  # noqa: E402

# f32 execution tolerance (the emitted modules are single-precision; the frozen
# oracle's own 1e-9 atol/rtol is a det-js/f64 figure and does not apply here).
# Combined absolute + relative so it is meaningful for both near-zero and
# large-magnitude (e.g. -406) log-densities.
VALUE_ATOL = 1e-3
VALUE_RTOL = 1e-4

# Expected StableHLO backend outcome per example (a separate axis from the
# manifest's det-js `status`, which every example here shares as "lowers").
# `("scores", None)`   — the posterior emits, executes, and matches the frozen
#                        scipy oracle at every theta point.
# `("refuses", <sub>)` — the emitter/determiniser cannot yet lower it; <sub> is
#                        a required substring of the refusal message. The named
#                        StableHLO op/dist gap:
#   builtin_touniform             — eight_schools, dissimilar_mixture (§07 touniform / CDF)
#   broadcast of a user function  — poisson_glm_link
#   Dirac distribution            — zero_inflated_binomial
# Both directions of drift fail the gate: a "scores" entry that starts refusing
# is a regression; a "refuses" entry that starts scoring is a fixed gap to
# promote (flip to "scores" so its numbers get checked against the oracle).
EXPECTED: dict[str, tuple[str, str | None]] = {
    "ex_bayesian_inference_1": ("scores", None),
    "ex_bayesian_inference_2": ("scores", None),
    "ex_best_estimation": ("scores", None),
    "ex_capture_recapture": ("scores", None),
    "ex_eight_schools": ("refuses", "builtin_touniform"),
    "ex_gamma_reparam": ("scores", None),
    "ex_hierarchical_logistic": ("scores", None),
    "ex_partial_pooling": ("scores", None),
    "ex_poisson_glm_link": ("refuses", "callable must be a bare builtin name"),
    "ex_poisson_model": ("scores", None),
    "ex_rasch_1pl": ("scores", None),
    "ex_dissimilar_mixture": ("refuses", "builtin_touniform"),
    "ex_linear_regression": ("scores", None),
    "ex_zero_inflated_binomial": ("refuses", "no lowering for distribution 'Dirac'"),
}


def examples_dir() -> Path:
    """The flatppl-examples checkout, resolved the same way
    `flatppl_testsuite.config.CONFIG` does — `FLATPPL_EXAMPLES_DIR`, else the
    sibling `../flatppl-examples` of the repo root. (Resolved here directly so
    this gate stays runnable in the minimal, no-default-feature `stablehlo`
    env, which does not install the `flatppl_testsuite` package.)"""
    env = os.environ.get("FLATPPL_EXAMPLES_DIR")
    return Path(env) if env else REPO.parent / "flatppl-examples"


def render_value(v) -> str:
    """A JSON theta value -> FlatPPL literal syntax: numbers verbatim, a list
    as a `[...]` vector literal (recursively)."""
    if isinstance(v, bool):  # guard: bool is a subclass of int
        raise ValueError(f"unexpected boolean theta value {v!r}")
    if isinstance(v, (int, float)):
        return repr(v)
    if isinstance(v, list):
        return "[" + ", ".join(render_value(x) for x in v) + "]"
    raise ValueError(f"unsupported theta value type {type(v).__name__}: {v!r}")


def render_record(theta: dict) -> str:
    """A theta dict -> `record(k1 = v1, k2 = v2, ...)`."""
    fields = ", ".join(f"{k} = {render_value(v)}" for k, v in theta.items())
    return f"record({fields})"


def query_module_source(model_rel: str, binding: str, theta: dict) -> str:
    return (
        'flatppl_compat = "0.1"\n'
        f'm = load_module("{model_rel}")\n'
        f"score = logdensityof(m.{binding}, {render_record(theta)})\n"
    )


@dataclass
class CheckResult:
    test_id: str
    check_id: str
    status: str    # "passed" | "failed"
    outcome: str   # display label: SCORES | REFUSE | REGRESSED | IMPROVED | MISMATCH | ERROR
    message: str = ""


def score_point(model_path: Path, binding: str, theta: dict) -> float:
    """Emit + execute the query module for one theta point; return the scalar.

    Raises `executor.EmitRefused` if the emitter/determiniser refuses to lower
    the posterior (a StableHLO coverage gap)."""
    with tempfile.TemporaryDirectory() as td:
        model_rel = os.path.relpath(model_path, td)
        query = Path(td) / "query.flatppl"
        query.write_text(query_module_source(model_rel, binding, theta))
        src = executor.emit(query, "logdensity")
    return executor.value(src, [])


def run() -> list[CheckResult]:
    manifest = json.loads((HERE / "manifest.json").read_text())
    ex_root = examples_dir() / "examples"
    results: list[CheckResult] = []

    for ex in manifest.get("examples", []):
        # This gate scores; entries the det-js corpus documents as non-scoring
        # (`refuses`/`unscoreable`) are out of scope here — it reports only what
        # the StableHLO backend does with the models det-js already lowers.
        if ex.get("status") != "lowers":
            continue
        test_id = ex["test_id"]
        model_path = ex_root / ex["model"]
        binding = ex["binding"]
        theta_grid = ex["theta"]
        expect, reason = EXPECTED.get(test_id, ("scores", None))

        expected_doc = json.loads((HERE / test_id / "expected.json").read_text())
        for check in expected_doc["checks"]:
            check_id = check["id"]
            theta_i = theta_grid[check["index"]]
            exp = float(check["expected"])
            results.append(
                _check_one(test_id, check_id, model_path, binding, theta_i, exp, expect, reason)
            )
    return results


def _check_one(test_id, check_id, model_path, binding, theta_i, exp, expect, reason) -> CheckResult:
    try:
        got = score_point(model_path, binding, theta_i)
    except executor.EmitRefused as e:
        detail = str(e).splitlines()[0][:100]
        if expect == "refuses":
            if reason and reason not in str(e):
                return CheckResult(test_id, check_id, "failed", "REFUSE",
                                   f"refused, but reason changed: want {reason!r} in {detail!r}")
            return CheckResult(test_id, check_id, "passed", "REFUSE", detail)
        return CheckResult(test_id, check_id, "failed", "REGRESSED",
                           f"expected to score but emit refused: {detail}")
    except Exception as e:  # noqa: BLE001 — surface any executor error as detail
        return CheckResult(test_id, check_id, "failed", "ERROR",
                           f"executor error: {str(e).splitlines()[0][:100]}")

    # Emitted + executed to a number.
    if expect == "refuses":
        return CheckResult(test_id, check_id, "failed", "IMPROVED",
                           f"expected to refuse ({reason!r}) but scored {got:.6f} — "
                           f"promote to 'scores' and check against the oracle")
    d = abs(got - exp)
    tol = VALUE_ATOL + VALUE_RTOL * abs(exp)
    if d <= tol or (math.isinf(exp) and got == exp):
        return CheckResult(test_id, check_id, "passed", "SCORES",
                           f"Δ={d:.2e} (got {got:.6f} vs {exp:.6f})")
    return CheckResult(test_id, check_id, "failed", "MISMATCH",
                       f"Δ={d:.2e} > {tol:.2e} (got {got:.6f} vs scipy {exp:.6f})")


def render(results: list[CheckResult]) -> str:
    labels = [f"{r.test_id}::{r.check_id}" for r in results]
    width = max((len(x) for x in labels), default=8)
    ow = max((len(r.outcome) for r in results), default=8)
    lines = [
        "=" * 78,
        "EXAMPLES CORPUS UNDER StableHLO — score a load_module posterior query, "
        "execute via Enzyme-JAX",
        "=" * 78,
        "",
        f"  {'test_id :: check':<{width}}  {'outcome':<{ow}}  detail",
        f"  {'-' * width}  {'-' * ow}  ------",
    ]
    for r, label in zip(results, labels):
        lines.append(f"  {label:<{width}}  {r.outcome:<{ow}}  {r.message}")
    n_scores = sum(r.outcome == "SCORES" for r in results)
    n_refuse = sum(r.outcome == "REFUSE" and r.status == "passed" for r in results)
    n_fail = sum(r.status == "failed" for r in results)
    lines += [
        "",
        f"  {n_scores} SCORES, {n_refuse} REFUSE (expected), {n_fail} FAIL "
        f"(of {len(results)} checks)",
    ]
    return "\n".join(lines)


def main() -> int:
    if not executor.binary_supports_stablehlo():
        print("SKIP: FLATPPL_BIN does not expose the `stablehlo` subcommand "
              "(build with --features stablehlo and set FLATPPL_BIN).")
        return 0
    if not executor.executor_available():
        print("SKIP: jax + enzyme_ad not importable in this environment.")
        return 0
    results = run()
    print(render(results))
    # Coverage report: only a real MISMATCH (a number outside tolerance, or an
    # executor error on a model that DID emit) is a hard failure. A REFUSE is an
    # unlowered posterior — a known StableHLO gap, reported, not a trip wire.
    return 1 if any(r.status == "failed" for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())
