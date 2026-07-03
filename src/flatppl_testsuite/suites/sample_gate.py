"""The sample suite: the numeric gate for the FlatPDL sample path —
``rand(rng, lawof(record(...)))`` lowered by flatppl-rust's sample
determinizer to a rng-threaded ``builtin_sample`` chain, evaluated by
flatppl-js (with its get0-on-tuple fix).

A fixed seed gives ONE deterministic realization of a sample-path model, so
this gate cannot compare a single scalar to one frozen value the way
``fragment_gate.py`` does. Instead it SEED-SWEEPS the determinized model
over N distinct seeds (``scoring/sample_sweep.cjs``, one Node process for
the whole sweep — see that script's header) to build an empirical sample
set, reduces it to mean/var/cov, and compares those empirical stats to a
closed-form INDEPENDENT oracle (``corpora/sample/oracle.py``) within a
Monte-Carlo tolerance (``corpora/sample/gen_expected.py``).

THE POINT OF THIS GATE: the ``cov(y1, y2) ~= 100`` check
(``corpora/sample/hier_normal/expected.json``'s ``cov_y1_y2``) is the
statistical proof that the shared latent ``mu`` is sampled ONCE and reused
by both ``y1``'s and ``y2``'s kernels, rather than resampled independently
per consumer. If the determinizer got shared-ancestor identity wrong, y1
and y2 would come out independent and this covariance would land near 0 —
tens of standard errors outside the tolerance band (see
``gen_expected.py``'s note for the exact figure). This is deliberately NOT
a check that can be forced green by loosening a tolerance: an independent
draw of mu per consumer is a qualitatively different (and wrong) number,
not a slightly-off one.

A secondary check verifies sampling<->density agreement: for a handful of
the swept realizations, the closed-form joint log-density at that exact
point (``corpora/sample/oracle.py::logdensity``, duplicated inline below —
see the comment at ``_closed_form_logdensity``) must match what the
determinized DENSITY path (``logdensityof(lawof(record(...)), <point>)``,
scored via the existing det-js engine) returns for the SAME law evaluated
at that point. This runs against ``hier_normal_density.flatppl``, a
companion model with the identical joint law but no ``rand(...)`` wrapper:
appending a second ``lawof(...)`` query onto the sample-path model AFTER
``rand()`` has already consumed the stochastic-phase graph is refused by
the determinizer (confirmed empirically — exit 3, "lawof's argument is not
stochastic-phase"), hence the separate density-only companion model.
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

from .base import Suite, register

_TESTSUITE_ROOT = Path(__file__).resolve().parents[3]  # repo root
SAMPLE_CORPUS: Path = _TESTSUITE_ROOT / "corpora" / "sample"
SAMPLE_MANIFEST: Path = SAMPLE_CORPUS / "manifest.json"


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def _var(xs: list[float]) -> float:
    return statistics.variance(xs)  # sample variance, ddof=1 (matches gen_expected.py's SE)


def _cov(xs: list[float], ys: list[float]) -> float:
    mx, my = _mean(xs), _mean(ys)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (len(xs) - 1)


def _closed_form_logdensity(mu: float, y1: float, y2: float) -> float:
    """Independent closed-form joint log-density, duplicated from
    ``corpora/sample/oracle.py::logdensity`` (a suite under ``src/`` does
    not reach into ``corpora/`` at runtime elsewhere in this toolkit — see
    e.g. ``fragment_gate.py`` — so this mirrors that formula rather than
    importing across the boundary). ``tests/test_sample_gate.py`` asserts
    the two implementations agree, so they cannot silently drift apart."""
    from scipy.stats import norm

    return (
        norm.logpdf(mu, 0.0, 10.0)
        + norm.logpdf(y1, mu, 1.0)
        + norm.logpdf(y2, mu, 1.0)
    )


class SampleGateSuite(Suite):
    name = "sample"

    def run(self, selected: set[str] | None = None,
            oracles: tuple[str, ...] = ()) -> list["CheckResult"]:
        from ..scoring.engine import DeterminizeRefused, get_engine, sample_sweep
        from ..scoring.compare import compare_scalar
        from ..scoring.result import (
            CheckResult, NUMERIC_MISMATCH, DETERMINIZE_SKIP, UNSCOREABLE)

        manifest = json.loads(SAMPLE_MANIFEST.read_text())
        results: list[CheckResult] = []

        for entry in manifest["models"]:
            test_id = entry["test_id"]
            if selected is not None and test_id not in selected:
                continue

            model_dir = SAMPLE_CORPUS / entry["path"]
            expected_doc = json.loads((model_dir / "expected.json").read_text())
            model_path = model_dir / expected_doc["model"]
            density_model_path = model_dir / expected_doc["density_model"]
            n_samples = expected_doc["n_samples"]
            seed_base = expected_doc.get("seed_base", 0)
            bindings = expected_doc["bindings"]

            try:
                realizations = sample_sweep(model_path, n_samples, bindings, base=seed_base)
            except DeterminizeRefused as e:
                for check in expected_doc["checks"]:
                    results.append(CheckResult(
                        test_id=test_id, check_id=check["id"],
                        status="skipped", tag=DETERMINIZE_SKIP, message=str(e),
                    ))
                continue
            except Exception as e:  # noqa: BLE001
                for check in expected_doc["checks"]:
                    results.append(CheckResult(
                        test_id=test_id, check_id=check["id"],
                        status="failed", tag=UNSCOREABLE, message=str(e),
                    ))
                continue

            fields = {b: [r[b] for r in realizations] for b in bindings}

            for check in expected_doc["checks"]:
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
                        compare_scalar(got, check["expected"],
                                        {"atol": check["atol"], "rtol": 0.0})
                    except AssertionError as e:
                        results.append(CheckResult(
                            test_id=test_id, check_id=check_id,
                            status="failed", tag=NUMERIC_MISMATCH, message=str(e),
                        ))
                        continue
                    except Exception as e:  # noqa: BLE001
                        results.append(CheckResult(
                            test_id=test_id, check_id=check_id,
                            status="failed", tag=UNSCOREABLE, message=str(e),
                        ))
                        continue
                    results.append(CheckResult(
                        test_id=test_id, check_id=check_id, status="passed",
                    ))

                elif kind == "density_consistency":
                    n_points = min(check["n_points"], len(realizations))
                    tol = {"atol": check["atol"], "rtol": check["rtol"]}
                    engine = get_engine("det-js")
                    status, tag, detail = "passed", "", ""
                    for point in realizations[:n_points]:
                        theta = {"mu": point["mu"], "y1": point["y1"], "y2": point["y2"]}
                        try:
                            got = engine.log_density(density_model_path, "m", theta)
                            want = _closed_form_logdensity(**theta)
                            compare_scalar(got, want, tol)
                        except DeterminizeRefused as e:
                            status, tag, detail = "skipped", DETERMINIZE_SKIP, str(e)
                            break
                        except AssertionError as e:
                            status, tag, detail = "failed", NUMERIC_MISMATCH, str(e)
                            break
                        except Exception as e:  # noqa: BLE001
                            status, tag, detail = "failed", UNSCOREABLE, str(e)
                            break
                    results.append(CheckResult(
                        test_id=test_id, check_id=check_id,
                        status=status, tag=tag, message=detail,
                    ))
                else:
                    raise ValueError(f"unknown check kind {kind!r}")

        return results


register(SampleGateSuite())
