"""Gradient checks retain sensitivity across interior and boundary scales."""

from pathlib import Path

import numpy as np
import pytest

from flatppl_testsuite.unified.loader import load_test
from flatppl_testsuite.unified.runners import gradient_stablehlo as runner


@pytest.mark.parametrize("name", ["pushfwd_tanh", "pushfwd_invlogit"])
def test_gradient_checks_detect_zero_interior_derivatives(monkeypatch, name):
    directory = Path(__file__).resolve().parents[2] / "corpora/stablehlo-gradient" / name
    spec = load_test(directory)
    reference = {
        point["y"]: expected["y"]
        for point, expected in zip(spec.body["points"], spec.body["expected_grad"])
    }
    monkeypatch.setattr(runner.ex, "emit_concat", lambda *args: "emitted")
    monkeypatch.setattr(runner.ex, "gradient", lambda src, args, argnums: [
        float(np.float32(reference[args[0]])),
    ])
    assert all(r.status == "passed" for r in runner.run(spec, directory))

    monkeypatch.setattr(runner.ex, "gradient", lambda src, args, argnums: [
        0.0 if abs(reference[args[0]]) < 10 else float(np.float32(reference[args[0]])),
    ])
    results = runner.run(spec, directory)
    assert sum(r.status == "failed" for r in results) == sum(abs(v) < 10 for v in reference.values())
