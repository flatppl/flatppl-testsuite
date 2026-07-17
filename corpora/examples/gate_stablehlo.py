#!/usr/bin/env python3
"""Examples corpus gate under the StableHLO backend.

The sibling `gate.py` scores each flatppl-examples posterior under **det-js**
(append `__score__ = logdensityof(binding, theta)` to the model, determinize,
score). This gate instead scores the SAME posteriors under the **StableHLO**
backend, and it does so through a FlatPPL **query module** rather than by
splicing a query line into the example: per example it writes a tiny
PARAMETERIZED query module using the `inputs`/`outputs` ABI (design doc
`flatppl-rust/docs/superpowers/specs/2026-07-17-inputs-outputs-abi-design.md`)

    m = load_module("<relpath to the pristine example>")
    t_f1 = elementof(<domain(f1)>)
    ...
    t_fk = elementof(<domain(fk)>)
    inputs = (t_f1, ..., t_fk)
    outputs = logdensityof(m.<binding>, record(f1 = t_f1, ..., fk = t_fk))

emits `@logdensity` ONCE per example from the local `flatppl` binary
(`FLATPPL_BIN`, built with the `stablehlo` feature), then executes that one
compiled module at every theta grid point by feeding theta as runtime
arguments (in `inputs` order), comparing each number to the frozen
INDEPENDENT scipy oracle in `corpora/examples/<test_id>/expected.json` (the
very oracle `gen_expected.py` froze for the det-js gate). The example files
themselves are never modified — they are composed as modules.

Free-param **binding** names are prefixed `t_<field>` so they do not shadow
the example's own internal bindings of the same name (e.g. a bare `alpha`
would collide with `linear-regression.flatppl`'s own `alpha ~ Normal(...)`
and refuse in the determiniser as a name collision). Record **field** names
stay the posterior variate names the example itself uses.

Each field's domain (`DOMAINS` below) is derived from that field's PRIOR
SUPPORT by reading the example source — not guessed, not inferred from the
theta grid's sampled values. A vector field (theta value is a list of length
N) uses the vector form of the scalar domain, `cartpow(<scalar domain>, N)`;
its runtime arg is a (possibly nested) list, which `executor._to_arg` turns
into the right-shaped f32 array.

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

Fallback coverage: the ABI is additive/opt-in (design doc "Fallback +
migration") — a model with neither `inputs` nor `outputs` still emits via the
legacy last-public-binding convention, with a one-line deprecation warning on
stderr. `check_legacy_fallback_warns` below exercises that path directly
(against the CLI, not through `executor.emit` which does not surface stderr
on success) rather than duplicating a whole second concrete-point scoring
path through every example — the warning is the cheaper, still-real signal
that the fallback (not just the ABI path) stays live.
"""
from __future__ import annotations

import json
import math
import os
import subprocess
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
    "ex_eight_schools": ("scores", None),
    "ex_gamma_reparam": ("scores", None),
    "ex_hierarchical_logistic": ("scores", None),
    "ex_partial_pooling": ("scores", None),
    "ex_poisson_glm_link": ("scores", None),
    "ex_poisson_model": ("scores", None),
    "ex_rasch_1pl": ("scores", None),
    # Emits under StableHLO (touniform now lowers) but executes to nan: a
    # `superpose` mixand is evaluated off its own support and a positive-support
    # builder returns nan there (Buffy #365). A documented downstream gap, not a
    # refusal and not yet a match — reported XFAIL (fails the gate only if it
    # regresses to an emit refusal, or if #365 is fixed and it starts matching,
    # either of which is a re-triage signal).
    "ex_dissimilar_mixture": ("known_gap", "#365 superpose mixand nan off-support"),
    "ex_linear_regression": ("scores", None),
    "ex_zero_inflated_binomial": ("scores", None),
}

# Per-field FlatPPL domain expression, keyed by test_id then theta field name.
# Derived from each field's PRIOR SUPPORT in the example source (read, not
# guessed): unconstrained real -> `reals`; positive (a scale, an
# InverseGamma/Gamma draw, or a `sqrt(...)` reparam) -> `posreals`; a
# probability/Beta/Uniform-on-a-bounded-interval -> that literal
# `interval(lo, hi)`; a vector field (theta value is a length-N list) -> the
# vector form `cartpow(<scalar domain>, N)`. Field order matches the manifest
# theta dict's key order (fixed per example; `inputs`/args follow the same
# order).
DOMAINS: dict[str, dict[str, str]] = {
    # theta1/theta2 are declared in-source as `elementof(reals)` (the model's
    # own point: domain declarations + `joint`) — mirror that declared type
    # rather than theta2_dist = Exponential(1)'s narrower positive support.
    "ex_bayesian_inference_1": {"theta1": "reals", "theta2": "reals"},
    # Same posterior via stochastic `~` bindings: theta2 ~ Exponential(1),
    # genuinely positive-support here.
    "ex_bayesian_inference_2": {"theta1": "reals", "theta2": "posreals"},
    "ex_best_estimation": {
        "mu1": "reals",
        "mu2": "reals",
        "sigma1": "interval(0.1, 20.0)",  # sigma1 ~ Uniform(interval(0.1, 20.0))
        "sigma2": "interval(0.1, 20.0)",
        "nu": "posreals",                 # nu ~ Exponential(1/29)
    },
    # rcp ~ Uniform(interval(0, rcp_max)); rcp_max = 10 / (15 - 5 + 10) = 0.5.
    "ex_capture_recapture": {"rcp": "interval(0.0, 0.5)"},
    "ex_eight_schools": {
        "mu": "reals",                    # mu ~ Normal(0, 5)
        "tau": "posreals",                # tau ~ normalize(truncate(Cauchy(...), interval(0, inf)))
        "theta": "cartpow(reals, 8)",     # theta ~ iid(Normal(mu, tau), J=8)
    },
    "ex_gamma_reparam": {"mu": "reals", "sigma": "posreals"},
    "ex_hierarchical_logistic": {
        "mu_a": "reals",                  # mu_a ~ Normal(0, 1)
        "sigma_a": "posreals",            # sigma_a ~ Gamma(shape=4, rate=2)
        "a": "cartpow(reals, 3)",         # a ~ iid(Normal(mu_a, sigma_a), G=3)
        "b": "reals",                     # b ~ locscale(StudentT(3), 0, 2.5)
    },
    "ex_partial_pooling": {
        "phi": "interval(0.0, 1.0)",      # phi ~ Uniform(interval(0, 1))
        "kappa": "posreals",              # kappa ~ Gamma(2.0, 0.05)
        "theta": "cartpow(interval(0.0, 1.0), 8)",  # theta ~ iid(Beta(...), N=8)
    },
    "ex_poisson_glm_link": {"intercept": "reals", "slope": "reals"},
    "ex_poisson_model": {"lambda": "posreals"},       # lambda ~ Gamma(2, 1)
    "ex_rasch_1pl": {
        "theta": "cartpow(reals, 4)",     # theta ~ iid(Normal(0, 1.5), P=4)
        "b": "cartpow(reals, 5)",         # b ~ iid(Normal(0, 1.5), I=5)
    },
    "ex_dissimilar_mixture": {
        "p": "interval(0.0, 1.0)",        # p ~ Beta(2, 2)
        "mu": "reals",                    # mu ~ Normal(0, 1)
        "sigma": "posreals",              # sigma = sqrt(sigma2), sigma2 ~ InverseGamma(2, 2)
        "shape": "posreals",              # shape ~ normalize(truncate(Normal(0,5), interval(0,inf)))
        "rate": "posreals",               # rate ~ normalize(truncate(Normal(0,5), interval(0,inf)))
    },
    "ex_linear_regression": {
        "alpha": "reals",                 # alpha ~ Normal(0, sigma * 3)
        "beta": "reals",                  # beta ~ Normal(0, sigma * 3)
        "sigma": "posreals",              # sigma = sqrt(sigma2), sigma2 ~ InverseGamma(5, 5)
    },
    "ex_zero_inflated_binomial": {
        "p": "interval(0.0, 1.0)",        # p ~ Beta(1.5, 1.5)
        "psi": "interval(0.0, 1.0)",      # psi ~ Beta(1.5, 1.5)
    },
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
    as a `[...]` vector literal (recursively). Used only by the legacy
    concrete-point fallback probe below — the ABI path feeds theta as runtime
    args, not literals."""
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


def query_module_source(
    model_rel: str, binding: str, fields: list[str], domains: dict[str, str]
) -> str:
    """The ABI query module for one example: a `t_<field>` free-param binding
    per theta field (elementof its prior-support domain), an `inputs` tuple in
    field order, and a single `outputs` query — `logdensityof(m.<binding>,
    record(<field> = t_<field>, ...))`. `inputs` is a bare value (not a
    1-tuple) when there is exactly one field, matching `outputs`'s own
    single-query bare form."""
    lines = ['flatppl_compat = "0.1"', f'm = load_module("{model_rel}")']
    for f in fields:
        lines.append(f"t_{f} = elementof({domains[f]})")
    if len(fields) == 1:
        lines.append(f"inputs = t_{fields[0]}")
    else:
        lines.append("inputs = (" + ", ".join(f"t_{f}" for f in fields) + ")")
    record_fields = ", ".join(f"{f} = t_{f}" for f in fields)
    lines.append(f"outputs = logdensityof(m.{binding}, record({record_fields}))")
    return "\n".join(lines) + "\n"


def legacy_query_module_source(model_rel: str, binding: str, theta: dict) -> str:
    """The pre-ABI concrete-point query module (theta inlined, no free
    params): used only to probe that the fallback path still fires its
    deprecation warning, not for scoring."""
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


def emit_query(model_path: Path, binding: str, fields: list[str], domains: dict[str, str]) -> str:
    """Emit the ABI query module for one example ONCE; returns the StableHLO
    source text (raises `executor.EmitRefused` on a determiniser/emitter
    refusal, or another exception on an executor error)."""
    with tempfile.TemporaryDirectory() as td:
        model_rel = os.path.relpath(model_path, td)
        query = Path(td) / "query.flatppl"
        query.write_text(query_module_source(model_rel, binding, fields, domains))
        return executor.emit(query, "logdensity")


def score_point(src: str, fields: list[str], theta: dict) -> float:
    """Execute an already-emitted ABI module at one theta point: a scalar
    field becomes a float arg, a vector field a list arg, in `inputs` order
    (== field order)."""
    args = [theta[f] for f in fields]
    return executor.value(src, args)


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
        fields = list(theta_grid[0].keys())
        domains = DOMAINS[test_id]

        # Emit ONCE per example: one compiled module scores every theta point.
        try:
            src: str | Exception = emit_query(model_path, binding, fields, domains)
        except Exception as e:  # noqa: BLE001 — captured, dispatched per-check below
            src = e

        expected_doc = json.loads((HERE / test_id / "expected.json").read_text())
        for check in expected_doc["checks"]:
            check_id = check["id"]
            theta_i = theta_grid[check["index"]]
            exp = float(check["expected"])
            results.append(
                _check_one(test_id, check_id, src, fields, theta_i, exp, expect, reason)
            )
    return results


def _check_one(test_id, check_id, src, fields, theta_i, exp, expect, reason) -> CheckResult:
    if isinstance(src, Exception):
        e = src
        if isinstance(e, executor.EmitRefused):
            detail = str(e).splitlines()[0][:100]
            if expect == "refuses":
                if reason and reason not in str(e):
                    return CheckResult(test_id, check_id, "failed", "REFUSE",
                                       f"refused, but reason changed: want {reason!r} in {detail!r}")
                return CheckResult(test_id, check_id, "passed", "REFUSE", detail)
            if expect == "known_gap":
                return CheckResult(test_id, check_id, "failed", "REGRESSED",
                                   f"known gap ({reason}) expected to emit but emit refused: {detail}")
            return CheckResult(test_id, check_id, "failed", "REGRESSED",
                               f"expected to score but emit refused: {detail}")
        return CheckResult(test_id, check_id, "failed", "ERROR",
                           f"executor error: {str(e).splitlines()[0][:100]}")

    try:
        got = score_point(src, fields, theta_i)
    except Exception as e:  # noqa: BLE001 — surface any executor error as detail
        return CheckResult(test_id, check_id, "failed", "ERROR",
                           f"executor error: {str(e).splitlines()[0][:100]}")

    # Emitted + executed to a number.
    if expect == "refuses":
        return CheckResult(test_id, check_id, "failed", "IMPROVED",
                           f"expected to refuse ({reason!r}) but scored {got:.6f} — "
                           f"promote to 'scores' and check against the oracle")
    if expect == "known_gap":
        # A documented downstream defect: emits but does not match (typically
        # nan). XFAIL — expected. If it now MATCHES, the gap is fixed → fail so
        # it gets promoted to "scores" and its numbers start being checked.
        tol = VALUE_ATOL + VALUE_RTOL * abs(exp)
        if math.isnan(got) or abs(got - exp) > tol:
            return CheckResult(test_id, check_id, "passed", "XFAIL",
                               f"known gap ({reason}): got {got} vs {exp:.6f}")
        return CheckResult(test_id, check_id, "failed", "FIXED",
                           f"known gap ({reason}) now matches (got {got:.6f}) — "
                           f"promote to 'scores'")
    d = abs(got - exp)
    tol = VALUE_ATOL + VALUE_RTOL * abs(exp)
    if d <= tol or (math.isinf(exp) and got == exp):
        return CheckResult(test_id, check_id, "passed", "SCORES",
                           f"Δ={d:.2e} (got {got:.6f} vs {exp:.6f})")
    return CheckResult(test_id, check_id, "failed", "MISMATCH",
                       f"Δ={d:.2e} > {tol:.2e} (got {got:.6f} vs scipy {exp:.6f})")


def check_legacy_fallback_warns() -> CheckResult:
    """Fallback coverage (design doc "Fallback + migration"): a model with
    neither `inputs` nor `outputs` must still emit via the legacy
    last-public-binding convention, printing a one-line deprecation warning to
    stderr. Probes the CLI directly (`executor.emit` only returns stdout on
    success) with a legacy concrete-point query module built from the first
    scoring example's first theta point — the exact form every check used
    before this migration."""
    manifest = json.loads((HERE / "manifest.json").read_text())
    ex = next(e for e in manifest["examples"] if e["test_id"] == "ex_linear_regression")
    model_path = examples_dir() / "examples" / ex["model"]
    theta = ex["theta"][0]
    with tempfile.TemporaryDirectory() as td:
        model_rel = os.path.relpath(model_path, td)
        query = Path(td) / "query.flatppl"
        query.write_text(legacy_query_module_source(model_rel, ex["binding"], theta))
        proc = subprocess.run(
            [str(executor.flatppl_bin()), "stablehlo", str(query), "--mode", "logdensity"],
            capture_output=True, text=True,
        )
    ok = (
        proc.returncode == 0
        and "no inputs/outputs bindings" in proc.stderr
        and "declare inputs/outputs" in proc.stderr
    )
    if ok:
        return CheckResult("fallback", "legacy_deprecation_warning", "passed", "WARNS",
                           "legacy (no inputs/outputs) model still emits + warns on stderr")
    return CheckResult("fallback", "legacy_deprecation_warning", "failed", "ERROR",
                       f"exit={proc.returncode} stderr={proc.stderr.strip()[:200]!r}")


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
    n_xfail = sum(r.outcome == "XFAIL" for r in results)
    n_fail = sum(r.status == "failed" for r in results)
    lines += [
        "",
        f"  {n_scores} SCORES, {n_refuse} REFUSE (expected), {n_xfail} XFAIL "
        f"(known gap), {n_fail} FAIL (of {len(results)} checks)",
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
    ex_root = examples_dir() / "examples"
    if not ex_root.is_dir():
        print(f"SKIP: no flatppl-examples checkout at {ex_root} "
              "(set FLATPPL_EXAMPLES_DIR, or clone the sibling ../flatppl-examples).")
        return 0
    results = run()
    results.append(check_legacy_fallback_warns())
    print(render(results))
    # Coverage report: only a real MISMATCH (a number outside tolerance, or an
    # executor error on a model that DID emit) is a hard failure. A REFUSE is an
    # unlowered posterior — a known StableHLO gap, reported, not a trip wire.
    return 1 if any(r.status == "failed" for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())
