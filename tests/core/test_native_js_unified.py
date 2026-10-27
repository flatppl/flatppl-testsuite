"""The native JS row scores source directly, with source-relative dependencies."""
import json
from pathlib import Path

import pytest

from flatppl_testsuite.unified import detjs_exec
from flatppl_testsuite.unified.harness import run_test_dir
from tests.test_unified import _gate_engine, assert_results_acceptable


def test_native_abi_does_not_determinize_and_resolves_relative_import(tmp_path, monkeypatch):
    _gate_engine("js")
    (tmp_path / "dependency.flatppl").write_text("v = 3.0\n")
    (tmp_path / "model.flatppl").write_text(
        '%%%\nExample:\nx = 99.0\n%%%\nx = 0.0\nf(t) = t\nm = load_module("dependency.flatppl")\n')
    (tmp_path / "query.flatppl").write_text("inputs = x\noutputs = f(x) + m.v\n")
    (tmp_path / "test.json").write_text(json.dumps({
        "test_type": "logdensity", "engines": ["js"], "inputs": ["x"],
        "points": [{"x": 2.0}, {"x": 4.0}], "expected": [5.0, 7.0],
    }))
    def refuse(*args, **kwargs):
        raise AssertionError("native JS must not determinize")
    monkeypatch.setattr(detjs_exec, "determinize_abi", refuse)
    results = run_test_dir(tmp_path, engines=["js"])
    assert len(results) == 2
    assert_results_acceptable(results, allow_skip=False)
    assert sorted(p.name for p in tmp_path.glob(".abi-*.flatppl")) == []


@pytest.mark.parametrize("body, reason", [
    ("bad = reverse(2.0)\noutputs = bad\n", "statically"),
    ("z ~ Normal(0, 1)\noutputs = logdensityof(Normal(z, 1), 0.5)\n", "reads draw"),
])
def test_native_abi_refuses_invalid_sources(tmp_path: Path, body: str, reason: str):
    _gate_engine("js")
    (tmp_path / "model.flatppl").write_text("x = 0.0\n" + body)
    (tmp_path / "query.flatppl").write_text("inputs = x\n")
    (tmp_path / "test.json").write_text(json.dumps({
        "test_type": "logdensity", "engines": ["js"], "inputs": ["x"],
        "points": [{"x": 2.0}], "expected": [0.0],
    }))
    [result] = run_test_dir(tmp_path, engines=["js"])
    assert result.status == "failed"
    assert reason in result.message


def test_native_abi_resolves_relative_data(tmp_path: Path):
    _gate_engine("js")
    (tmp_path / "data.json").write_text("[1.0, 2.0, 4.0]")
    (tmp_path / "model.flatppl").write_text(
        'x = 0.0\na = load_data("data.json", cartpow(reals, 3))\n')
    (tmp_path / "query.flatppl").write_text("inputs = x\noutputs = x + sum(a)\n")
    (tmp_path / "test.json").write_text(json.dumps({
        "test_type": "logdensity", "engines": ["js"], "inputs": ["x"],
        "points": [{"x": 2.0}], "expected": [9.0],
    }))
    assert_results_acceptable(run_test_dir(tmp_path, engines=["js"]), allow_skip=False)


def test_required_native_engine_does_not_skip_missing_prerequisites(monkeypatch):
    from flatppl_testsuite.unified.runners import logdensity_js
    from tests import test_unified

    monkeypatch.setattr(logdensity_js, "engine_available", lambda: False)
    monkeypatch.setattr(test_unified, "_REQUIRED_ENGINES", {"js"})
    with pytest.raises(pytest.fail.Exception, match="engine 'js' is required"):
        test_unified._gate_engine("js")
