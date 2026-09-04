"""Runner: test_type=convert, engine=det-js.

The `hs3` corpus is structurally unlike every other unified corpus: the model
under test is NOT FlatPPL but a foreign HS3 JSON fixture, and the frozen
reference is ROOT/RooFit's `twice_delta_nll_scan` vector rather than a scalar
or a per-point scalar list. `test.json` carries a `fixture_kind` of
`"fixture"` (the HS3TestSuite corpus, vendored HS3 -> convert -> assemble ->
score), `"conversion"` (the HS3-paper worked examples, already committed as
`.flatppl` and scored directly, no convert step), `"pyhf"` (a vendored pyhf
workspace -> convert --from pyhf -> score, against pyhf's own absolute
`Model.logpdf`), or `"pyhf_reject"` (a document with no log-density to score,
whose converter EXIT CODE and refusal message are what the row holds).

Each `checks` entry is copied VERBATIM from the legacy
`corpora/hs3/{fixtures,conversions}/*/expected.json` (frozen ROOT vectors,
tolerances, targets) -- this runner ports `suites/hs3_import.py`'s three check
kinds (`static_integrity`, `structure_import`, `twice_delta_nll_scan`) and the
conversions corpus's `twice_delta_nll_points`, preserving their comparisons
and tolerances exactly. It reuses `score_scan`/`score_points` and the
structure-checking helpers from `suites/hs3_import.py` rather than
reimplementing them, so there is exactly one scoring path for the frozen ROOT
vectors, shared with `corpora/hs3/run_comparisons.py` and the legacy gate.

Scoring goes through `detjs_exec`, named here, the way every other det-js
runner names it. It used to go through `scoring.engine.get_engine()`, which
reads `FLATPPL_ENGINE` and defaults to `"js"` -- so under a plain `pixi run
test` this whole det-js-labelled corpus scored in pure JS and never ran
`determinize` at all (provable: point `FLATPPL_BIN` at a wrapper that fails on
`determinize` and all 8 rows still passed). `tests/core/test_detjs_runners_are_det_js.py`
guards the property.

Skip semantics (legacy tags, preserved distinctly):
* a conversion failure (an HS3 construct the converter doesn't legalize yet)
  is `CONVERT_SKIP`;
* a determiniser refusal (the model legalizes to FlatPPL but the determiniser
  can't yet lower it to FlatPDL) is `DETERMINIZE_SKIP`.
"""
from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

from flatppl_testsuite.scoring.compare import compare_vectors
from flatppl_testsuite.scoring.engine import DeterminizeRefused
from flatppl_testsuite.scoring.result import (
    CheckResult, CONVERT_SKIP, DETERMINIZE_SKIP, NUMERIC_MISMATCH, UNSCOREABLE,
)
from flatppl_testsuite.formats.hs3.importer import convert, convert_raw, SkipUnimplemented
from flatppl_testsuite.suites.hs3_import import _names_in_source, score_points, score_scan
from flatppl_testsuite.unified import detjs_exec as ex
from flatppl_testsuite.unified.loader import TestSpec


def _static_integrity(tid: str, check: dict, doc: dict) -> CheckResult:
    """The vendored fixture must still be the content its frozen oracle vector
    was computed against. Both the legacy gate and the first unified port
    passed unconditionally ("it parsed"), while the recorded per-fixture hashes
    went unread -- so a vendored copy could drift from its expected vector
    silently.

    Hash the CANONICAL form, not the raw bytes: the canonical form matches all
    5 recorded HS3 hashes, whereas raw bytes match only 3 (two fixtures carry
    formatting churn with identical semantic content), so raw-byte hashing
    would fail them for a difference that cannot affect any result.
    """
    check_id = check["id"]
    want = check.get("canonical_sha256")
    if not want:
        return CheckResult(
            tid, check_id, "failed", UNSCOREABLE,
            "static_integrity declares no `canonical_sha256`, so it verifies nothing",
        )
    got = hashlib.sha256(
        json.dumps(doc, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if got == want:
        return CheckResult(tid, check_id, "passed")
    return CheckResult(
        tid, check_id, "failed", NUMERIC_MISMATCH,
        f"vendored fixture has drifted: canonical sha256 {got} != {want}",
    )


def _run_fixture(tid: str, dir: Path, body: dict) -> list[CheckResult]:
    hs3_path = dir / body.get("hs3", "hs3.json")
    hs3_doc = json.loads(hs3_path.read_text())
    results: list[CheckResult] = []

    for check in body["checks"]:
        check_id = check["id"]
        kind = check["kind"]

        if kind == "static_integrity":
            results.append(_static_integrity(tid, check, hs3_doc))
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
                actual_vec = score_scan(hs3_doc, hs3_path, check,
                                        log_density=ex.log_density_at)
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
            actual_vec = score_points(model_file, check,
                                      log_density=ex.log_density_at)
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


def _run_pyhf(tid: str, dir: Path, body: dict) -> list[CheckResult]:
    """`fixture_kind: "pyhf"` -- convert a pyhf workspace, score ABSOLUTE
    log-densities, compare against pyhf's own `Model.logpdf`.

    Unlike both HS3 flavours this subtracts no reference point. pyhf and the
    FlatPPL lowering both carry the full Poisson normalization, so the absolute
    values must agree -- and they have to be compared that way, because the
    defect class this corpus exists to catch (a staterror read with the wrong
    constraint form) changes the normalization, which a Delta against a
    reference point inside the same model would partly cancel.
    """
    source_path = dir / body.get("source", "pyhf.json")
    source_doc = json.loads(source_path.read_text())
    results: list[CheckResult] = []

    for check in body["checks"]:
        check_id = check["id"]
        kind = check["kind"]

        if kind == "static_integrity":
            results.append(_static_integrity(tid, check, source_doc))
            continue

        if kind != "logpdf_points":
            raise ValueError(f"unknown check kind {kind!r}")

        try:
            src = convert(source_path, source_format="pyhf")
        except SkipUnimplemented as e:
            results.append(CheckResult(tid, check_id, "skipped", CONVERT_SKIP, e.hs3_type))
            continue
        except Exception as e:  # noqa: BLE001
            results.append(CheckResult(tid, check_id, "failed", UNSCOREABLE, f"convert: {e}"))
            continue

        with tempfile.TemporaryDirectory() as tmp:
            model_path = Path(tmp) / "model.flatppl"
            model_path.write_text(src)
            try:
                scores = ex.log_density_points(model_path, check["binding"], check["points"])
            except DeterminizeRefused as e:
                results.append(CheckResult(tid, check_id, "skipped", DETERMINIZE_SKIP, str(e)))
                continue
            except Exception as e:  # noqa: BLE001
                results.append(CheckResult(tid, check_id, "failed", UNSCOREABLE, f"score: {e}"))
                continue

        errored = [(i, s.error) for i, s in enumerate(scores) if s.value is None]
        if errored:
            results.append(CheckResult(
                tid, check_id, "failed", UNSCOREABLE,
                "; ".join(f"point {i}: {msg}" for i, msg in errored),
            ))
            continue

        try:
            compare_vectors([s.value for s in scores], check["expected"], check["tolerance"])
        except AssertionError as e:
            results.append(CheckResult(tid, check_id, "failed", NUMERIC_MISMATCH, str(e)))
            continue

        results.append(CheckResult(tid, check_id, "passed"))

    return results


def _run_pyhf_reject(tid: str, dir: Path, body: dict) -> list[CheckResult]:
    """`fixture_kind: "pyhf_reject"` -- assert the converter's OUTCOME on a
    document, not a number.

    One document per validation-failure class, plus the handful pyhf and the
    converter disagree about. There is nothing to score: an exit-1 document has
    no log-density. What the row holds is the exit code and, for a refusal, a
    substring of the message, so the converter cannot start accepting a
    document pyhf rejects (or drift to a reason that names a different defect)
    with a green run.

    The corpus also records what pyhf itself does with each document, measured
    rather than copied (`corpora/pyhf-rejects/gen_expected.py`). A row whose
    `pyhf_agrees` is false is a KNOWN, reasoned mismatch listed in that
    corpus's README, not a defect.
    """
    source_path = dir / body.get("source", "pyhf.json")
    source_doc = json.loads(source_path.read_text())
    results: list[CheckResult] = []

    for check in body["checks"]:
        check_id = check["id"]
        kind = check["kind"]

        if kind == "static_integrity":
            results.append(_static_integrity(tid, check, source_doc))
            continue

        if kind != "convert_outcome":
            raise ValueError(f"unknown check kind {kind!r}")

        code, stderr = convert_raw(source_path, source_format="pyhf")
        want_code = check["expect_exit"]
        if code != want_code:
            results.append(CheckResult(
                tid, check_id, "failed", NUMERIC_MISMATCH,
                f"converter exited {code}, expected {want_code}"
                + (f"; stderr: {stderr.strip()}" if stderr.strip() else ""),
            ))
            continue

        want_msg = check.get("stderr_contains")
        if want_msg and want_msg not in stderr:
            results.append(CheckResult(
                tid, check_id, "failed", NUMERIC_MISMATCH,
                f"refusal no longer names the same defect: expected "
                f"{want_msg!r} in stderr, got {stderr.strip()!r}",
            ))
            continue

        results.append(CheckResult(tid, check_id, "passed"))

    return results


def run(spec: TestSpec, dir: Path) -> list[CheckResult]:
    tid = dir.name
    body = spec.body
    fixture_kind = body.get("fixture_kind")
    if fixture_kind == "pyhf_reject":
        return _run_pyhf_reject(tid, dir, body)
    if fixture_kind == "fixture":
        return _run_fixture(tid, dir, body)
    if fixture_kind == "conversion":
        return _run_conversion(tid, dir, body)
    if fixture_kind == "pyhf":
        return _run_pyhf(tid, dir, body)
    raise ValueError(f"{dir}: unknown fixture_kind {fixture_kind!r}")
