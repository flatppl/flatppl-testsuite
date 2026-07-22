import json
from pathlib import Path

import pytest

from flatppl_testsuite.unified import loader


def _write(dir: Path, obj: dict) -> Path:
    dir.mkdir(parents=True, exist_ok=True)
    (dir / "test.json").write_text(json.dumps(obj))
    return dir


def test_load_valid_logdensity(tmp_path):
    d = _write(tmp_path / "t", {
        "test_type": "logdensity",
        "engines": ["stablehlo"],
        "measure": "posterior",
        "inputs": ["a"],
        "points": [{"a": 0.5}],
        "expected": [-1.0],
        "tolerance": {"value_atol_f32": 1e-4},
    })
    spec = loader.load_test(d)
    assert spec.test_type == "logdensity"
    assert spec.engines == ["stablehlo"]
    assert spec.body["measure"] == "posterior"


def test_missing_test_type_raises(tmp_path):
    d = _write(tmp_path / "t", {"engines": ["stablehlo"]})
    with pytest.raises(ValueError, match="test_type"):
        loader.load_test(d)


def test_empty_engines_raises(tmp_path):
    d = _write(tmp_path / "t", {"test_type": "logdensity", "engines": []})
    with pytest.raises(ValueError, match="engines"):
        loader.load_test(d)


def test_discover_finds_dirs_with_test_json(tmp_path):
    _write(tmp_path / "a", {"test_type": "logdensity", "engines": ["stablehlo"]})
    (tmp_path / "b").mkdir()  # no test.json
    _write(tmp_path / "c", {"test_type": "logdensity", "engines": ["stablehlo"]})
    found = loader.discover_test_dirs(tmp_path)
    assert [p.name for p in found] == ["a", "c"]
