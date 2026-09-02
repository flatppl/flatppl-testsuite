"""Shared HS3 helper library.

Formerly the legacy HS3 gate (a `Suite` registered into `suites/base.py`'s
registry and driven by `runner.py`/`pixi run harness`); that gate and the
whole `Suite` registry it lived in are retired (superseded by the unified
per-test-directory harness, `tests/test_unified.py`). This module survives as
a shared helper library: `score_scan`/`score_points`/`_names_in_source` are
now imported by `unified/runners/convert_detjs.py` (the `(convert, det-js)`
runner that drives the migrated `corpora/hs3/` test directories), and
`_binding_is_prenormalized` by `tests/test_prenormalized_structural.py`.
`HS3_CORPUS`/`HS3_MANIFEST` are kept as the corpus-root/manifest-path
constants they always were, for whatever future code wants a canonical
HS3-corpus-root path -- but note `HS3_MANIFEST` no longer points at a real
file (the legacy `manifest.json` was deleted alongside the gate; nothing in
this module reads it at import time)."""
from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path

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


def score_scan(hs3_doc: dict, hs3_path: Path, check: dict,
               *, log_density) -> list[float]:
    """Score the 2DeltaNLL vector for a `twice_delta_nll_scan` check.

    Convert the fixture, then either score the converter's emitted likelihood
    binding (fixtures WITH a `likelihoods` block) or build the check-time iid
    likelihood (generic-family dists are already range-normalized, so they are
    iid'd directly; raw dists are range-normalized). Propagates
    `SkipUnimplemented` and `DeterminizeRefused` (the latter when the model
    can't be legalized to FlatPDL); raises `RuntimeError` (stage-prefixed) on
    convert/assemble/score failure. Single scoring path shared by the suite
    runner and the comparison-table script.

    `log_density` is the caller's scorer, required so the environment cannot
    pick one (see `scoring/flatppl_engine`).
    """
    from ..formats.hs3.importer import (
        convert, SkipUnimplemented, data_columns, assemble)
    from ..scoring.engine import DeterminizeRefused
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
            model_path, binding, check["scan_parameter"], check["scan_points"],
            reference, log_density=log_density)
    except DeterminizeRefused:
        raise
    except Exception as e:
        raise RuntimeError(f"score: {e}") from e
    finally:
        model_path.unlink(missing_ok=True)


def score_points(model_file: Path, check: dict, *, log_density) -> list[float]:
    """Score a `twice_delta_nll_points` check against the committed FlatPPL model.

    The conversions corpus (HS3 paper examples) ships a worked `<model>.flatppl`;
    this scores its likelihood binding directly over the check's multi-parameter
    theta points and returns the 2DeltaNLL vector (offset-invariant, so it lines
    up with the frozen ROOT vector regardless of normalization conventions).
    Single scoring path shared by the suite runner and the comparison-table
    script. Propagates `DeterminizeRefused`; raises `RuntimeError`
    (stage-prefixed) on other failure.

    `log_density` is the caller's scorer, required so the environment cannot
    pick one (see `scoring/flatppl_engine`).
    """
    from ..scoring.engine import DeterminizeRefused
    from ..scoring.flatppl_engine import twice_delta_nll_points
    try:
        return twice_delta_nll_points(
            model_file, check["binding"], check["reference_point"],
            check["points"], log_density=log_density)
    except DeterminizeRefused:
        raise
    except Exception as e:
        raise RuntimeError(f"score: {e}") from e
