"""The fragment suite: small, self-contained FlatPPL models that already end
in a fixed-point ``lp = logdensityof(m, <point>)`` binding.

Unlike the HS3 corpus (a parameterized likelihood scanned over a theta
vector), each fragment needs no ``__score__`` append and no theta: it is
scored through the convert-free det-js path (``flatppl determinize`` ->
``score_flatpdl.cjs``) exactly as written, and the resulting scalar is
compared to a frozen INDEPENDENT oracle (Julia Distributions.jl / scipy — see
``corpora/fragment/gen_expected.py``) recorded in each fragment's
``expected.json``.
"""
from __future__ import annotations

import json
from pathlib import Path

from .base import Suite, register

_TESTSUITE_ROOT = Path(__file__).resolve().parents[3]  # repo root
FRAGMENT_CORPUS: Path = _TESTSUITE_ROOT / "corpora" / "fragment"
FRAGMENT_MANIFEST: Path = FRAGMENT_CORPUS / "manifest.json"


def _parse_expected(value: float | str) -> float:
    """``expected.json`` freezes +-inf as the JSON string "-inf"/"inf"
    (standard JSON has no Infinity literal); ``float()`` parses those
    case-insensitively to the actual IEEE-754 infinities. Everything else is
    already a JSON number."""
    return float(value)


class FragmentGateSuite(Suite):
    name = "fragment"

    def run(self, selected: set[str] | None = None,
            oracles: tuple[str, ...] = ()) -> list["CheckResult"]:
        from ..scoring.compare import compare_scalar
        from ..scoring.engine import DeterminizeRefused, score_binding
        from ..scoring.result import (
            CheckResult, NUMERIC_MISMATCH, DETERMINIZE_SKIP, UNSCOREABLE)

        manifest = json.loads(FRAGMENT_MANIFEST.read_text())
        results: list[CheckResult] = []

        for frag in manifest["fragments"]:
            test_id = frag["test_id"]
            if selected is not None and test_id not in selected:
                continue

            frag_dir = FRAGMENT_CORPUS / frag["path"]
            expected_doc = json.loads((frag_dir / "expected.json").read_text())
            model_path = frag_dir / expected_doc["model"]

            for check in expected_doc["checks"]:
                check_id = check["id"]
                if check["kind"] != "logdensity_value":
                    continue

                binding = check["binding"]
                expected = _parse_expected(check["expected"])

                try:
                    got = score_binding(model_path, binding)
                except DeterminizeRefused as e:
                    results.append(CheckResult(
                        test_id=test_id, check_id=check_id,
                        status="skipped", tag=DETERMINIZE_SKIP, message=str(e),
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

        return results


register(FragmentGateSuite())
