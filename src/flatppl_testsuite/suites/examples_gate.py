"""The examples suite: determinises each flatppl-examples model's
constructed posterior query and either compares it to a frozen INDEPENDENT
oracle (status ``lowers``), asserts the determiniser refuses it (status
``refuses``), or — for a query the determiniser DOES lower but a named
engine/determiniser gap prevents from scoring — asserts just the lowering
half and documents the gap (status ``unscoreable``).

Unlike ``bayesian_inference``/``fragment`` (each model already ends in a
fixed-point ``lp = logdensityof(...)`` binding), the flatppl-examples models
are pure model DEFINITIONS ending in ``posterior = bayesupdate(L, prior)``
with no query — so this suite builds the query itself, per manifest entry,
via ``DetJsScoreEngine.log_density(model, binding, theta_i)`` (appends
``__score__ = logdensityof(binding, theta_i)``, runs ``flatppl determinize``,
scores the result with ``score_flatpdl.cjs``), once per point in the entry's
theta grid. Examples with no ``posterior`` binding (e.g. ``minimal``) are not
in the manifest at all (see its top-level ``excluded`` list).

Manifest schema (``corpora/examples/manifest.json``, key ``examples``): each
entry is::

    {
      "test_id": str,          # e.g. "ex_eight_schools"
      "model": str,            # filename under flatppl-examples/examples/
      "binding": str,          # the measure binding to query, e.g. "posterior"
      "theta": [dict, ...],    # the theta grid; each dict renders to a record()
      "status": "lowers" | "refuses" | "unscoreable",
      "reason": str | None,    # `refuses`/`unscoreable`: required substring of
                               # the refusal message, resp. the score-stage
                               # crash message
      "category": str | None, # human-facing gap categorization (e.g.
                               # "determiniser-gap", "engine-gap"); ignored by
                               # this Suite, informational only
      "note": str | None,     # human-facing explanation of the gap; ignored
                               # by this Suite, informational only
      "pending_oracle": bool, # `unscoreable` only: True means expected.json
                               # (if present) is frozen but NOT YET checked
                               # against this entry — scoring is skipped
                               # entirely, so nothing verifies it matches
    }

A ``lowers`` entry additionally needs a frozen oracle at
``corpora/examples/<test_id>/expected.json`` (schema mirrors
``bayesian_inference``/``fragment``'s ``checks`` list — ``kind
"logdensity_value"``, ``expected``, ``tolerance`` — plus an ``index`` field
selecting the theta grid point each check scores: ``checks[i]`` is scored at
``theta[checks[i]["index"]]``). A ``refuses`` entry needs no expected.json:
every point in its theta grid is expected to raise ``DeterminizeRefused``. An
``unscoreable`` entry MAY keep the same ``<test_id>/expected.json`` frozen
from before the gap was discovered (``pending_oracle: true`` marks it
unchecked) — this Suite never reads it for an ``unscoreable`` entry, so a
future fix need only flip the status back to ``lowers`` to start comparing
against it.

Outcome mapping (deliberately reusing the existing tag vocabulary in
``scoring/result.py`` rather than inventing new ones):

- ``lowers`` + determinizes + matches oracle -> ``passed``.
- ``lowers`` + ``DeterminizeRefused`` -> ``failed``/``UNSCOREABLE`` (a
  regression: this corpus's premise is that these models lower).
- ``lowers`` + determinizes but diverges from the oracle -> ``failed``/``NUMERIC_MISMATCH``.
- ``refuses`` + ``DeterminizeRefused`` with ``reason`` as a substring (or no
  ``reason`` given) -> ``passed``.
- ``refuses`` + ``DeterminizeRefused`` but the message doesn't contain
  ``reason`` -> ``failed``/``UNSCOREABLE`` (refused for the wrong reason).
- ``refuses`` + determinizes+scores anyway -> ``failed``/``NUMERIC_MISMATCH``
  (an improvement to surface: the determinizer now handles this construct).
- ``unscoreable`` + determinizes, score stage crashes with ``reason`` as a
  substring -> ``passed``/``UNSCOREABLE`` (the documented gap, unchanged;
  ``engine.log_density`` runs determinize then score in one call, so
  reaching a score-stage crash — rather than ``DeterminizeRefused`` or a
  "determinize failed" ``RuntimeError`` — already proves determinize alone
  exited 0, with no second subprocess call needed to check that separately).
- ``unscoreable`` + ``DeterminizeRefused`` -> ``failed``/``UNSCOREABLE`` (a
  regression: the determiniser used to lower this and now refuses it
  outright, a strictly worse state than the documented gap).
- ``unscoreable`` + score stage crashes but the message doesn't contain
  ``reason`` -> ``failed``/``UNSCOREABLE`` (the crash signature changed —
  re-triage before assuming it's still the documented gap).
- ``unscoreable`` + determinizes+scores cleanly -> ``failed``/``NUMERIC_MISMATCH``
  (an improvement to surface: the gap looks fixed — flip status to
  ``lowers`` and it starts being checked against the frozen oracle).
"""
from __future__ import annotations

import json
from pathlib import Path

from .base import Suite, register
from ..config import CONFIG

_TESTSUITE_ROOT = Path(__file__).resolve().parents[3]  # repo root
EXAMPLES_CORPUS: Path = _TESTSUITE_ROOT / "corpora" / "examples"
EXAMPLES_MANIFEST: Path = EXAMPLES_CORPUS / "manifest.json"


def _parse_expected(value: float | str) -> float:
    """``expected.json`` freezes +-inf as the JSON string "-inf"/"inf"
    (standard JSON has no Infinity literal); ``float()`` parses those
    case-insensitively to the actual IEEE-754 infinities. Everything else is
    already a JSON number."""
    return float(value)


class ExamplesGateSuite(Suite):
    name = "examples"

    def run(self, selected: set[str] | None = None,
            oracles: tuple[str, ...] = ()) -> list["CheckResult"]:
        from ..scoring.compare import compare_scalar
        from ..scoring.engine import DetJsScoreEngine, DeterminizeRefused
        from ..scoring.result import CheckResult, NUMERIC_MISMATCH, UNSCOREABLE

        manifest = json.loads(EXAMPLES_MANIFEST.read_text())
        results: list[CheckResult] = []
        engine = DetJsScoreEngine()

        for ex in manifest.get("examples", []):
            test_id = ex["test_id"]
            if selected is not None and test_id not in selected:
                continue

            model_path = CONFIG.examples_dir / "examples" / ex["model"]
            binding = ex["binding"]
            theta_grid = ex["theta"]
            status = ex["status"]

            if status == "lowers":
                expected_doc = json.loads(
                    (EXAMPLES_CORPUS / test_id / "expected.json").read_text()
                )
                for check in expected_doc["checks"]:
                    check_id = check["id"]
                    theta_i = theta_grid[check["index"]]
                    expected = _parse_expected(check["expected"])

                    try:
                        got = engine.log_density(model_path, binding, theta_i)
                    except DeterminizeRefused as e:
                        results.append(CheckResult(
                            test_id=test_id, check_id=check_id,
                            status="failed", tag=UNSCOREABLE,
                            message=f"expected to lower but determinize refused: {e}",
                        ))
                        continue
                    except Exception as e:  # noqa: BLE001
                        results.append(CheckResult(
                            test_id=test_id, check_id=check_id,
                            status="failed", tag=UNSCOREABLE, message=str(e),
                        ))
                        continue

                    try:
                        compare_scalar(got, expected, check["tolerance"])
                    except AssertionError as e:
                        results.append(CheckResult(
                            test_id=test_id, check_id=check_id,
                            status="failed", tag=NUMERIC_MISMATCH, message=str(e),
                        ))
                        continue

                    results.append(CheckResult(
                        test_id=test_id, check_id=check_id, status="passed",
                    ))

            elif status == "refuses":
                reason = ex.get("reason")
                for i, theta_i in enumerate(theta_grid):
                    check_id = f"theta_{i}"
                    try:
                        engine.log_density(model_path, binding, theta_i)
                    except DeterminizeRefused as e:
                        if reason and reason not in str(e):
                            results.append(CheckResult(
                                test_id=test_id, check_id=check_id,
                                status="failed", tag=UNSCOREABLE,
                                message=(
                                    f"refused but reason mismatch: expected "
                                    f"substring {reason!r} in {e!r}"
                                ),
                            ))
                            continue
                        results.append(CheckResult(
                            test_id=test_id, check_id=check_id, status="passed",
                        ))
                        continue
                    except Exception as e:  # noqa: BLE001
                        results.append(CheckResult(
                            test_id=test_id, check_id=check_id,
                            status="failed", tag=UNSCOREABLE, message=str(e),
                        ))
                        continue

                    # Determinize + score succeeded where a refusal was
                    # expected: an improvement to surface, not a hard crash.
                    results.append(CheckResult(
                        test_id=test_id, check_id=check_id,
                        status="failed", tag=NUMERIC_MISMATCH,
                        message="expected DeterminizeRefused but the model lowered and scored",
                    ))

            elif status == "unscoreable":
                # This entry's premise: `flatppl determinize` DOES lower the
                # query (the structural claim this corpus still gates on) but
                # the score stage then crashes on a named, documented
                # engine/determiniser gap (see the manifest's `category`/
                # `note`). `engine.log_density` runs determinize-then-score
                # in one call, so a score-stage crash can *only* be reached
                # after determinize exits 0 (a determinize-stage failure
                # instead raises `DeterminizeRefused`, or a distinct
                # "determinize failed: ..." `RuntimeError`) — that already
                # verifies the structural claim without a second, redundant
                # `flatppl determinize` subprocess call.
                reason = ex.get("reason")
                for i, theta_i in enumerate(theta_grid):
                    check_id = f"theta_{i}"
                    try:
                        got = engine.log_density(model_path, binding, theta_i)
                    except DeterminizeRefused as e:
                        # Regression: the determiniser used to lower this
                        # query; now it refuses outright, a strictly worse
                        # state than the documented score-stage gap.
                        results.append(CheckResult(
                            test_id=test_id, check_id=check_id,
                            status="failed", tag=UNSCOREABLE,
                            message=(
                                f"regression: unscoreable entry no longer "
                                f"determinizes (expected determinize to still "
                                f"exit 0, only scoring to crash): {e}"
                            ),
                        ))
                        continue
                    except RuntimeError as e:
                        if reason and reason not in str(e):
                            # Still crashes, but not with the documented
                            # signature — re-triage rather than assume it's
                            # the same known gap.
                            results.append(CheckResult(
                                test_id=test_id, check_id=check_id,
                                status="failed", tag=UNSCOREABLE,
                                message=(
                                    f"unscoreable but crash reason changed: "
                                    f"expected substring {reason!r} in {e!r}"
                                ),
                            ))
                            continue
                        results.append(CheckResult(
                            test_id=test_id, check_id=check_id,
                            status="passed", tag=UNSCOREABLE,
                            message=(
                                f"documented gap ({ex.get('category', 'gap')}): "
                                f"determinize OK, score stage crashed as "
                                f"expected: {e}"
                            ),
                        ))
                        continue
                    except Exception as e:  # noqa: BLE001
                        results.append(CheckResult(
                            test_id=test_id, check_id=check_id,
                            status="failed", tag=UNSCOREABLE, message=str(e),
                        ))
                        continue

                    # Scored cleanly where the documented crash was
                    # expected: the gap looks fixed — an improvement to
                    # surface, not a hard failure.
                    results.append(CheckResult(
                        test_id=test_id, check_id=check_id,
                        status="failed", tag=NUMERIC_MISMATCH,
                        message=(
                            f"expected the documented score-stage crash but "
                            f"it scored cleanly (got={got!r}) -- the gap may "
                            f"be fixed; flip status to \"lowers\" to check "
                            f"against the frozen oracle"
                        ),
                    ))

            else:
                raise ValueError(f"{test_id}: unknown status {status!r}")

        return results


register(ExamplesGateSuite())
