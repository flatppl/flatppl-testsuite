"""Formula conversion retains file-backed data and reports preparation failures."""
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from flatppl_testsuite.scoring.result import UNSCOREABLE
from flatppl_testsuite.unified import detjs_exec, harness


@pytest.mark.skipif(not detjs_exec.engine_available(), reason="det-js path unavailable")
def test_formula_conversion_consumes_sidecars_and_reports_refusals(tmp_path, monkeypatch):
    source = (Path(__file__).resolve().parents[2] / "corpora" / "formula"
              / "family_gaussian")
    directory = tmp_path / source.name
    shutil.copytree(source, directory)
    manifest = directory / "test.json"
    body = json.loads(manifest.read_text())
    body["model"] = "density.flatppl"
    # At theta=(0.4, 0.3, 1.2), the first centered mean is -0.05.
    # Replacing y[1]=2 with 3 changes only its Normal likelihood factor.
    body["expected"][0] -= ((3 + 0.05) ** 2 - (2 + 0.05) ** 2) / (2 * 1.2**2)
    manifest.write_text(json.dumps(body))
    run = subprocess.run

    def convert_then_change_data(command, **kwargs):
        result = run(command, **kwargs)
        if command[1] == "convert":
            sidecar = Path(command[5]).with_suffix(".data.json")
            data = json.loads(sidecar.read_text())
            data["y_data"][0] = 3
            sidecar.write_text(json.dumps(data))
        return result

    monkeypatch.setattr(harness.subprocess, "run", convert_then_change_data)
    results = harness.run_test_dir(directory, engines=["det-js"])
    assert results and all(result.status == "passed" for result in results), results
    assert all(result.test_id == directory.name for result in results)

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
