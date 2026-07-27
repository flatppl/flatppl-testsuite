"""Runner: test_type=convert, engine=det-js.

The `hs3` corpus is structurally unlike every other unified corpus: the model
under test is NOT FlatPPL but a foreign HS3 JSON fixture, and the frozen
reference is ROOT/RooFit's `twice_delta_nll_scan` vector rather than a scalar
or a per-point scalar list. `test.json` carries a `fixture_kind` of either
`"fixture"` (the HS3TestSuite corpus, vendored HS3 -> convert -> assemble ->
score) or `"conversion"` (the HS3-paper worked examples, already committed as
`.flatppl` and scored directly, no convert step).

Each `checks` entry is copied VERBATIM from the legacy
`corpora/hs3/{fixtures,conversions}/*/expected.json` (frozen ROOT vectors,
tolerances, targets) -- this runner ports `suites/hs3_import.py`'s three check
kinds (`static_integrity`, `structure_import`, `twice_delta_nll_scan`) and the
conversions corpus's `twice_delta_nll_points`, preserving their comparisons
and tolerances exactly. It reuses `score_scan`/`score_points` and the
structure-checking helpers from `suites/hs3_import.py` rather than
reimplementing them, so there is exactly one scoring path for the frozen ROOT
vectors, shared with `corpora/hs3/run_comparisons.py` and the legacy gate.

Skip semantics (legacy tags, preserved distinctly):
* a conversion failure (an HS3 construct the converter doesn't legalize yet)
  is `CONVERT_SKIP`;
* a determiniser refusal (the model legalizes to FlatPPL but the determiniser
  can't yet lower it to FlatPDL) is `DETERMINIZE_SKIP`.
"""
from __future__ import annotations

import json
from pathlib import Path

from flatppl_testsuite.scoring.compare import compare_vectors
from flatppl_testsuite.scoring.engine import DeterminizeRefused
from flatppl_testsuite.scoring.result import (
    CheckResult, CONVERT_SKIP, DETERMINIZE_SKIP, NUMERIC_MISMATCH, UNSCOREABLE,
)
from flatppl_testsuite.formats.hs3.importer import convert, SkipUnimplemented
from flatppl_testsuite.suites.hs3_import import _names_in_source, score_points, score_scan
from flatppl_testsuite.unified.loader import TestSpec


def _run_fixture(tid: str, dir: Path, body: dict) -> list[CheckResult]:
    hs3_path = dir / body.get("hs3", "hs3.json")
    hs3_doc = json.loads(hs3_path.read_text())
    results: list[CheckResult] = []

    for check in body["checks"]:
        check_id = check["id"]
        kind = check["kind"]

        if kind == "static_integrity":
            # Pass if the HS3 JSON is valid (already parsed above).
            results.append(CheckResult(tid, check_id, "passed"))
            continue

        if kind == "structure_import":
            target = check.get("target", {})
            pdf_names = target.get("pdfs", [])
            func_names = target.get("functions", [])
            data_names = target.get("data", [])

            try:
                src = convert(hs3_path)
            except SkipUnimplemented as e:
                results.append(CheckResult(
                    tid, check_id, "skipped", CONVERT_SKIP, e.hs3_type,
                ))
                continue
            except Exception as e:  # noqa: BLE001
                results.append(CheckResult(tid, check_id, "failed", UNSCOREABLE, str(e)))
                continue

            bound = _names_in_source(src)
            has_likelihoods = bool(hs3_doc.get("likelihoods"))

            missing = []
            for name in pdf_names:
                if name not in bound:
                    missing.append(f"pdf:{name}")
            for name in func_names:
                if name not in bound:
                    missing.append(f"function:{name}")
            for name in data_names:
                if name not in bound and not has_likelihoods:
                    declared = [d.get("name") for d in hs3_doc.get("data", [])]
                    if name not in declared:
                        missing.append(f"data:{name}")
                elif name not in bound:
                    missing.append(f"data:{name}")

            if missing:
                results.append(CheckResult(
                    tid, check_id, "failed", UNSCOREABLE,
                    "missing bindings: " + ", ".join(missing),
                ))
            else:
                results.append(CheckResult(tid, check_id, "passed"))
            continue

        if kind == "twice_delta_nll_scan":
            try:
                actual_vec = score_scan(hs3_doc, hs3_path, check)
            except SkipUnimplemented as e:
                results.append(CheckResult(
                    tid, check_id, "skipped", CONVERT_SKIP, e.hs3_type,
                ))
                continue
            except DeterminizeRefused as e:
                results.append(CheckResult(
                    tid, check_id, "skipped", DETERMINIZE_SKIP, str(e),
                ))
                continue
            except Exception as e:  # noqa: BLE001
                results.append(CheckResult(tid, check_id, "failed", UNSCOREABLE, str(e)))
                continue

            try:
                compare_vectors(actual_vec, check["expected"], check["tolerance"])
            except AssertionError as e:
                results.append(CheckResult(tid, check_id, "failed", NUMERIC_MISMATCH, str(e)))
                continue

            results.append(CheckResult(tid, check_id, "passed"))
            continue

        raise ValueError(f"unknown check kind {kind!r}")

    return results


def _run_conversion(tid: str, dir: Path, body: dict) -> list[CheckResult]:
    model_file = dir / body["model"]
    results: list[CheckResult] = []

    for check in body["checks"]:
        check_id = check["id"]
        if check["kind"] != "twice_delta_nll_points":
            raise ValueError(f"unknown check kind {check['kind']!r}")

        try:
            actual_vec = score_points(model_file, check)
        except DeterminizeRefused as e:
            results.append(CheckResult(tid, check_id, "skipped", DETERMINIZE_SKIP, str(e)))
            continue
        except Exception as e:  # noqa: BLE001
            results.append(CheckResult(tid, check_id, "failed", UNSCOREABLE, str(e)))
            continue

        try:
            compare_vectors(actual_vec, check["expected"], check["tolerance"])
        except AssertionError as e:
            results.append(CheckResult(tid, check_id, "failed", NUMERIC_MISMATCH, str(e)))
            continue

        results.append(CheckResult(tid, check_id, "passed"))

    return results


def run(spec: TestSpec, dir: Path) -> list[CheckResult]:
    tid = dir.name
    body = spec.body
    fixture_kind = body.get("fixture_kind")
    if fixture_kind == "fixture":
        return _run_fixture(tid, dir, body)
    if fixture_kind == "conversion":
        return _run_conversion(tid, dir, body)
    raise ValueError(f"{dir}: unknown fixture_kind {fixture_kind!r}")
