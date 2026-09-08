"""The fanout runner validates one call's shape before collecting its values."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from flatppl_testsuite.unified.loader import load_test
from flatppl_testsuite.unified.runners import sample_stablehlo as runner


@pytest.mark.parametrize("shape,status", [((), "failed"), ((200,), "passed")])
def test_normal_fanout_requires_the_declared_batch(monkeypatch, shape, status):
    directory = Path(__file__).resolve().parents[2] / "corpora/stablehlo-sample/normal"
    spec = load_test(directory)
    spec = replace(spec, body={
        **spec.body, "checks": ["fanout_distribution"], "draws": 10000,
    })
    monkeypatch.setattr(runner.ex, "emit_concat", lambda *args: "executor double")

    def sample_call(src, key, args):
        rng = np.random.default_rng(int(key[0]))
        value = np.asarray(rng.normal(args[0], args[1], size=shape or None))
        return value, np.array([int(key[0]) + 1, int(key[1])], dtype=np.uint64)

    monkeypatch.setattr(runner.ex, "sample_call", sample_call)
    results = runner.run(spec, directory)

    assert len(results) == 1
    assert results[0].status == status
