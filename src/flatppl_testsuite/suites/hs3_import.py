"""The HS3 import suite: the module that knows the HS3 corpus manifest,
its static_integrity/structure_import/twice_delta_nll_scan check kinds, and its
expected.json schema."""
from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path

from .base import Suite, register

_HS3_MODULE_ROOT = Path(__file__).resolve().parents[3]  # repo root
HS3_CORPUS: Path = Path(os.environ.get("HS3SUITE", _HS3_MODULE_ROOT / "corpora" / "hs3"))
HS3_MANIFEST: Path = HS3_CORPUS / "manifest.json"


def _binding_is_prenormalized(src: str, pdf_name: str) -> bool:
    """True if the converted `<pdf_name> = ...` RHS starts with `normalize(` —
    an already range-normalized pdf (mixture / chebychev / polynomial / generic).
    Such a pdf is iid'd directly; `assemble` must not re-wrap it in another
    normalize (which makes a `normalize` node the truncate base — unscoreable)."""
    m = re.search(rf"(?m)^{re.escape(pdf_name)}\s*=\s*(.*)$", src)
    return bool(m) and m.group(1).lstrip().startswith("normalize(")


def _names_in_source(src: str) -> set[str]:
    """Return the set of binding names defined in FlatPPL source."""
    names: set[str] = set()
    for m in re.finditer(r"^\s*(\w+)\s*=", src, re.MULTILINE):
        names.add(m.group(1))
    return names


def score_scan(hs3_doc: dict, hs3_path: Path, check: dict) -> list[float]:
    """Score the 2DeltaNLL vector for a `twice_delta_nll_scan` check.

    Convert the fixture, then either score the converter's emitted likelihood
    binding (fixtures WITH a `likelihoods` block) or build the check-time iid
    likelihood (generic-family dists are already range-normalized, so they are
    iid'd directly; raw dists are range-normalized). Propagates
    `SkipUnimplemented`; raises `RuntimeError` (stage-prefixed) on
    convert/assemble/score failure. Single scoring path shared by the suite
    runner and the comparison-table script.
    """
    from ..formats.hs3.importer import (
        convert, SkipUnimplemented, data_columns, assemble)
    from ..scoring.flatppl_engine import twice_delta_nll

    target = check.get("target", {})
    pdf_name = target.get("pdf")
    data_name = target.get("data")

    try:
        src = convert(hs3_path)
    except SkipUnimplemented:
        raise
    except Exception as e:
        raise RuntimeError(f"convert: {e}") from e

    data_observable_names = {
        a["name"]
        for d in hs3_doc.get("data", [])
        for a in d.get("axes", [])
    }
    try:
        if hs3_doc.get("likelihoods"):
            scoreable_src, binding = src, pdf_name
        else:
            prenormalized = _binding_is_prenormalized(src, pdf_name)
            # Single observable for the 1-D scoring path; the converter names the
            # embedded table column after the dataset's observable axis.
            column = data_columns(hs3_path, data_name)[0]
            scoreable_src, binding = assemble(
                src, pdf_name, data_name, column, data_observable_names,
                prenormalized=prenormalized)
    except Exception as e:
        raise RuntimeError(f"assemble: {e}") from e

    reference = {k: v for k, v in check["reference_point"].items()
                 if k not in data_observable_names}
    with tempfile.NamedTemporaryFile(suffix=".flatppl", mode="w", delete=False) as tf:
        tf.write(scoreable_src)
        model_path = Path(tf.name)
    try:
        return twice_delta_nll(
            model_path, binding, check["scan_parameter"], check["scan_points"], reference)
    except Exception as e:
        raise RuntimeError(f"score: {e}") from e
    finally:
        model_path.unlink(missing_ok=True)


def score_points(model_file: Path, check: dict) -> list[float]:
    """Score a `twice_delta_nll_points` check against the committed FlatPPL model.

    The conversions corpus (HS3 paper examples) ships a worked `<model>.flatppl`;
    this scores its likelihood binding directly over the check's multi-parameter
    theta points and returns the 2DeltaNLL vector (offset-invariant, so it lines
    up with the frozen ROOT vector regardless of normalization conventions).
    Single scoring path shared by the suite runner and the comparison-table
    script. Raises `RuntimeError` (stage-prefixed) on failure.
    """
    from ..scoring.flatppl_engine import twice_delta_nll_points
    try:
        return twice_delta_nll_points(
            model_file, check["binding"], check["reference_point"], check["points"])
    except Exception as e:
        raise RuntimeError(f"score: {e}") from e


class HS3ImportSuite(Suite):
    name = "hs3_import"

    def run(self, selected: set[str] | None = None,
            oracles: tuple[str, ...] = ()) -> list["CheckResult"]:
        """Run the harness over the selected fixtures; optionally cross-check oracles."""
        from ..formats.hs3.importer import convert, SkipUnimplemented
        from ..formats.hs3 import engines as _oracle_mod
        from ..scoring.compare import compare_vectors
        from ..scoring.result import CheckResult, CONVERT_SKIP, UNSCOREABLE, NUMERIC_MISMATCH

        manifest = json.loads(HS3_MANIFEST.read_text())
        results: list[CheckResult] = []

        for fixture_meta in manifest["fixtures"]:
            test_id = fixture_meta["test_id"]
            if selected is not None and test_id not in selected:
                continue

            fixture_dir = HS3_CORPUS / fixture_meta["path"]
            hs3_path = fixture_dir / "hs3.json"
            expected_path = fixture_dir / "expected.json"

            expected_doc = json.loads(expected_path.read_text())
            hs3_doc = json.loads(hs3_path.read_text())

            for check in expected_doc["checks"]:
                check_id = check["id"]
                kind = check["kind"]

                if kind == "static_integrity":
                    # Pass if JSON is valid (we already loaded it above).
                    results.append(CheckResult(
                        test_id=test_id, check_id=check_id, status="passed",
                    ))

                elif kind == "structure_import":
                    target = check.get("target", {})
                    pdf_names = target.get("pdfs", [])
                    func_names = target.get("functions", [])
                    data_names = target.get("data", [])

                    try:
                        src = convert(hs3_path)
                    except SkipUnimplemented as e:
                        results.append(CheckResult(
                            test_id=test_id, check_id=check_id,
                            status="skipped", tag=CONVERT_SKIP,
                            message=e.hs3_type,
                        ))
                        continue
                    except Exception as e:
                        results.append(CheckResult(
                            test_id=test_id, check_id=check_id,
                            status="failed", tag=UNSCOREABLE,
                            message=str(e),
                        ))
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
                        # When no likelihoods block, the converter may not emit a
                        # data binding; check the HS3 doc declares the dataset.
                        if name not in bound and not has_likelihoods:
                            declared = [d.get("name") for d in hs3_doc.get("data", [])]
                            if name not in declared:
                                missing.append(f"data:{name}")
                        elif name not in bound:
                            missing.append(f"data:{name}")

                    if missing:
                        results.append(CheckResult(
                            test_id=test_id, check_id=check_id,
                            status="failed", tag=UNSCOREABLE,
                            message="missing bindings: " + ", ".join(missing),
                        ))
                    else:
                        results.append(CheckResult(
                            test_id=test_id, check_id=check_id, status="passed",
                        ))

                elif kind == "twice_delta_nll_scan":
                    try:
                        actual_vec = score_scan(hs3_doc, hs3_path, check)
                    except SkipUnimplemented as e:
                        results.append(CheckResult(
                            test_id=test_id, check_id=check_id,
                            status="skipped", tag=CONVERT_SKIP, message=e.hs3_type,
                        ))
                        continue
                    except Exception as e:
                        results.append(CheckResult(
                            test_id=test_id, check_id=check_id,
                            status="failed", tag=UNSCOREABLE, message=str(e),
                        ))
                        continue

                    try:
                        compare_vectors(actual_vec, check["expected"], check["tolerance"])
                    except AssertionError as e:
                        results.append(CheckResult(
                            test_id=test_id, check_id=check_id,
                            status="failed", tag=NUMERIC_MISMATCH,
                            message=str(e),
                        ))
                        continue

                    oracle_notes: list[str] = []
                    for ob in oracles:
                        try:
                            ov = _oracle_mod.run_oracle(ob, test_id)
                            if len(actual_vec) != len(ov):
                                oracle_notes.append(
                                    f"oracle[{ob}]: length mismatch "
                                    f"(flatppl={len(actual_vec)}, oracle={len(ov)})"
                                )
                            else:
                                diffs = [abs(a - o) for a, o in zip(actual_vec, ov)]
                                max_diff = max(diffs) if diffs else 0.0
                                oracle_notes.append(
                                    f"oracle[{ob}]={ov}; max|Δ vs flatppl|={max_diff:.6g}"
                                )
                        except RuntimeError as exc:
                            oracle_notes.append(f"oracle[{ob}]=unavailable: {exc}")

                    results.append(CheckResult(
                        test_id=test_id, check_id=check_id, status="passed",
                        message="; ".join(oracle_notes),
                    ))

        # Conversions corpus (HS3 paper examples): score the committed
        # <model>.flatppl directly against its frozen ROOT 2DeltaNLL vector. No
        # convert/assemble — the worked example is the unit under test, re-pinned
        # structurally elsewhere (tests/test_conversions.py).
        for conv in manifest.get("conversions", []):
            test_id = conv["test_id"]
            if selected is not None and test_id not in selected:
                continue
            conv_dir = HS3_CORPUS / conv["path"]
            expected_doc = json.loads((conv_dir / "expected.json").read_text())
            model_file = conv_dir / expected_doc["model"]

            for check in expected_doc["checks"]:
                check_id = check["id"]
                if check["kind"] != "twice_delta_nll_points":
                    continue
                try:
                    actual_vec = score_points(model_file, check)
                except Exception as e:  # noqa: BLE001
                    results.append(CheckResult(
                        test_id=test_id, check_id=check_id,
                        status="failed", tag=UNSCOREABLE, message=str(e),
                    ))
                    continue
                try:
                    compare_vectors(actual_vec, check["expected"], check["tolerance"])
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


register(HS3ImportSuite())
