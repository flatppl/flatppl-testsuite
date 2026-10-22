"""A det-js refusal must name its cause, not just "no derivation".

A fixed binding the engine cannot evaluate gets no derivation, and the reason
lives in `orchestrator.buildDerivations`'s diagnostics, not the analyzer's. The
scorers used to report only the materialiser's "no derivation for '<binding>'".

The refused binding passes a vector of vectors as `MvNormal`'s `cov`. §03
"Arrays": "Vectors of vectors are not interpreted as matrices implicitly", and
§08 gives `cov` as a matrix. The determiniser now refuses that source
statically, so the refusal tests feed the scorers the FlatPDL it used to emit
directly. They test the scorers, whatever the determiniser does upstream. The
`rowstack` spelling is the control on the full det-js path: it must score
scipy's `multivariate_normal.logpdf([0.7, -1.2], [0.2, -0.1], [[1, .3], [.3, 2]])`.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from flatppl_testsuite.config import CONFIG
from flatppl_testsuite.unified import detjs_exec
from flatppl_testsuite.unified.loader import load_test
from flatppl_testsuite.unified.runners import logdensity_detjs

SCIPY_LOGPDF = -2.6954601010511796

# The FlatPDL `flatppl determinize` emitted for the vector-of-vectors model.
REFUSED_FLATPDL = (
    "lp = builtin_logdensityof(MvNormal, record(mu = [0.2, -0.1], "
    "cov = [[1.0, 0.3], [0.3, 2.0]]), [0.7, -1.2])\n")


def _js_available() -> bool:
    return (CONFIG.flatppl_js_dir / "packages" / "engine" / "index.ts").exists()


def _bin_available() -> bool:
    return CONFIG.flatppl_bin.exists() or shutil.which(str(CONFIG.flatppl_bin)) is not None


needs_js = pytest.mark.skipif(not _js_available(), reason="requires a flatppl-js checkout")
needs_detjs = pytest.mark.skipif(
    not (_js_available() and _bin_available()),
    reason="requires a determinize-capable flatppl binary and a flatppl-js checkout")


def _case(tmp_path: Path, cov: str) -> Path:
    (tmp_path / "model.flatppl").write_text(
        f"m = MvNormal(mu = [0.2, -0.1], cov = {cov})\n"
        "lp = logdensityof(m, [0.7, -1.2])\n")
    (tmp_path / "query.flatppl").write_text(
        "t_a = elementof(reals)\nt_b = elementof(reals)\n"
        "inputs = (t_a, t_b)\noutputs = logdensityof(m, [t_a, t_b])\n")
    (tmp_path / "test.json").write_text(json.dumps({
        "test_type": "logdensity", "engines": ["det-js"], "model": "model.flatppl",
        "binding": "lp", "inputs": ["t_a", "t_b"], "points": [{"t_a": 0.7, "t_b": -1.2}],
        "expected": [SCIPY_LOGPDF], "tolerance": {"atol": 1e-9, "rtol": 1e-9},
    }))
    return tmp_path


@needs_js
def test_a_refused_fixed_binding_reaches_the_check_result(tmp_path, monkeypatch):
    # The real batch scorer on the refused FlatPDL, behind the runner's ABI path.
    monkeypatch.setattr(detjs_exec, "score_abi_points",
                        lambda *a, **k: detjs_exec._score_flatpdl_batch([REFUSED_FLATPDL], "lp"))
    case = _case(tmp_path, "rowstack([[1.0, 0.3], [0.3, 2.0]])")
    [result] = logdensity_detjs.run(load_test(case), case)
    assert result.status == "failed", result
    assert "vector-of-vectors" in result.message and "rowstack" in result.message, result.message


@needs_js
def test_a_refused_named_binding_names_its_cause(tmp_path):
    flatpdl = tmp_path / "m.flatpdl.flatppl"
    flatpdl.write_text(REFUSED_FLATPDL)
    with pytest.raises(RuntimeError, match="vector-of-vectors.*rowstack"):
        detjs_exec._score_flatpdl_binding(flatpdl, "lp")


@needs_detjs
def test_the_rowstack_spelling_scores_the_scipy_value(tmp_path):
    case = _case(tmp_path, "rowstack([[1.0, 0.3], [0.3, 2.0]])")
    [result] = logdensity_detjs.run(load_test(case), case)
    assert result.status == "passed", result
    assert detjs_exec.score_binding(case / "model.flatppl", "lp") == pytest.approx(
        SCIPY_LOGPDF, rel=1e-12, abs=1e-12)
