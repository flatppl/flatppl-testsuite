"""Formula preparation failures must not become skipped numeric checks."""
import subprocess
from pathlib import Path

from flatppl_testsuite.scoring.result import UNSCOREABLE
from flatppl_testsuite.unified import harness


def test_formula_conversion_refusal_is_a_failed_check(monkeypatch):
    directory = (Path(__file__).resolve().parents[2] / "corpora" / "formula"
                 / "prediction_interaction")

    def refuse(command, **kwargs):
        raise subprocess.CalledProcessError(3, command, stderr="formula conversion refused")

    monkeypatch.setattr(harness.subprocess, "run", refuse)
    results = harness.run_test_dir(directory, engines=["det-js"])

    assert len(results) == 1
    result = results[0]
    assert result.test_id == directory.name
    assert result.check_id == "convert"
    assert result.status == "failed"
    assert result.tag == UNSCOREABLE
    assert result.message
