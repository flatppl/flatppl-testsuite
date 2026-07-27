"""Runner: test_type=sample, engine=det-js.

Ported from ``suites/sample_gate.py`` -- see that module's docstring for the
full rationale. This is NOT the ``(sample, stablehlo)`` runner's 7-check
distribution/independence/key parity set (``sample_stablehlo.py`` +
``sample_checks.py``); it is the legacy sample-path numeric gate, carrying
exactly the two check kinds that gate used:

* ``sample_stats`` -- seed-sweep the determinized model
  (``detjs_exec.sample_sweep``, N distinct seeds from a fixed ``seed_base``)
  into an empirical sample set, reduce a named field (or field pair) to
  mean/var/cov, and compare against a frozen closed-form-oracle value within
  a frozen Monte-Carlo tolerance. ``test.json``'s ``checks`` entries, their
  ``expected``/``atol`` values, ``n_samples`` and ``seed_base`` are copied
  VERBATIM from the legacy ``corpora/sample/hier_normal/expected.json`` --
  changing the seed base or draw count would make this a different
  statistical test, not the same one ported.

* ``density_consistency`` -- for the first ``n_points`` swept realizations,
  the closed-form joint log-density at that exact point must match what the
  det-js engine's density path returns for the SAME law evaluated at that
  point, scored against the directory's companion ``density_model`` (a model
  with the identical joint law but no ``rand(...)`` wrapper -- appending a
  second query after ``rand()`` has already consumed the stochastic-phase
  graph is refused by the determinizer). Sampled points are only known at
  runtime, so there is no frozen scalar to compare against; this check is
  the one place in this harness that loads a directory's ``test.py`` and
  calls its oracle LIVE, at test time, rather than only offline under
  ``regen.py`` (this runner dispatches on ``(test_type, engine)`` alone, so
  the formula must live in the directory's own ``test.py``, not here --
  otherwise a second ``(sample, det-js)`` directory would silently get
  scored against this one's model).

A determiniser refusal (exit 3) is a SKIP, not a failure, for every check in
the directory (mirroring ``sample_gate.py``'s per-check skip loop).
"""
from __future__ import annotations

import statistics
from pathlib import Path

from flatppl_testsuite.scoring.compare import compare_scalar
from flatppl_testsuite.scoring.result import (
    CheckResult, DETERMINIZE_SKIP, NUMERIC_MISMATCH, UNSCOREABLE,
)
from flatppl_testsuite.unified import detjs_exec as ex
from flatppl_testsuite.unified.loader import TestSpec, load_test_module


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def _var(xs: list[float]) -> float:
    return statistics.variance(xs)  # sample variance, ddof=1 (matches gen_expected.py's SE)


def _cov(xs: list[float], ys: list[float]) -> float:
    mx, my = _mean(xs), _mean(ys)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (len(xs) - 1)


def run(spec: TestSpec, dir: Path) -> list[CheckResult]:
    tid = dir.name
    body = spec.body
    model_path = dir / body["model"]
    density_model_path = dir / body["density_model"]
    n_samples = body["n_samples"]
    seed_base = body.get("seed_base", 0)
    bindings = body["bindings"]
    checks = body["checks"]

    try:
        realizations = ex.sample_sweep(model_path, n_samples, bindings, base=seed_base)
    except ex.DeterminizeRefused as e:
        return [
            CheckResult(tid, check["id"], "skipped", DETERMINIZE_SKIP, str(e))
            for check in checks
        ]

    fields = {b: [r[b] for r in realizations] for b in bindings}
    results: list[CheckResult] = []

    for check in checks:
        check_id = check["id"]
        kind = check["kind"]

        if kind == "sample_stats":
            try:
                stat = check["stat"]
                if stat == "mean":
                    got = _mean(fields[check["field"]])
                elif stat == "var":
                    got = _var(fields[check["field"]])
                elif stat == "cov":
                    f1, f2 = check["fields"]
                    got = _cov(fields[f1], fields[f2])
                else:
                    raise ValueError(f"unknown sample_stats stat {stat!r}")
                compare_scalar(got, check["expected"], {"atol": check["atol"], "rtol": 0.0})
            except AssertionError as e:
                results.append(CheckResult(tid, check_id, "failed", NUMERIC_MISMATCH, str(e)))
                continue
            except Exception as e:  # noqa: BLE001
                results.append(CheckResult(tid, check_id, "failed", UNSCOREABLE, str(e)))
                continue
            results.append(CheckResult(tid, check_id, "passed"))

        elif kind == "density_consistency":
            n_points = min(check["n_points"], len(realizations))
            tol = {"atol": check["atol"], "rtol": check["rtol"]}
            oracle = load_test_module(dir)
            status, tag, detail = "passed", "", ""
            # Both the variate field names and the scored binding come from the
            # test dir, NOT from this runner: `_RUNNERS` dispatches on
            # (test_type, engine) alone, so hardcoding one directory's schema
            # here would silently score a future second (sample, det-js) dir
            # against hier_normal's fields and binding.
            density_binding = body.get("density_binding")
            if not density_binding:
                results.append(CheckResult(
                    tid, check_id, "failed", UNSCOREABLE,
                    "test.json declares `density_model` but no `density_binding` "
                    "(the binding in that model to score)",
                ))
                continue
            for point in realizations[:n_points]:
                theta = {b: point[b] for b in bindings}
                try:
                    got = ex.log_density_at(density_model_path, density_binding, theta)
                    want = oracle.logdensity(**theta)
                    compare_scalar(got, want, tol)
                except ex.DeterminizeRefused as e:
                    status, tag, detail = "skipped", DETERMINIZE_SKIP, str(e)
                    break
                except AssertionError as e:
                    status, tag, detail = "failed", NUMERIC_MISMATCH, str(e)
                    break
                except Exception as e:  # noqa: BLE001
                    status, tag, detail = "failed", UNSCOREABLE, str(e)
                    break
            results.append(CheckResult(tid, check_id, status, tag, detail))
        else:
            raise ValueError(f"unknown check kind {kind!r}")

    return results
