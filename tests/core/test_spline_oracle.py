"""The independent spline target preserves the executed Stan coordinate map."""
import hashlib
import json
from pathlib import Path
import runpy

import numpy as np
import pytest


def test_spline_regeneration_matches_actual_stan_and_raw_gradient():
    root = Path(__file__).resolve().parents[2] / "corpora" / "formula"
    evidence = root / "oracles" / "splines"
    stan = json.loads((evidence / "score.json").read_text())
    mapping = json.loads((evidence / "mapping.json").read_text())
    rotation = np.array(mapping["stan_to_native"])
    np.testing.assert_array_equal(rotation.T @ rotation, np.eye(4))
    assert hashlib.sha256((evidence / "model.stan").read_bytes()).hexdigest() == stan["stan_sha256"]
    derivative = np.array(stan["gradient"])
    derivative[2:6] = rotation @ derivative[2:6]
    derivative[6:] /= [stan["point"]["sds_1"][0], stan["point"]["sigma"]]
    np.testing.assert_array_equal(derivative, stan["native_gradient"])
    for name in ["thinplate_gaussian", "thinplate_gaussian_gradient"]:
        directory = root / name
        frozen = json.loads((directory / "test.json").read_text())
        oracle = runpy.run_path(str(directory / "test.py"))
        point = frozen["points"][0]
        np.testing.assert_array_equal(point["theta"], stan["native_point"])
        assert oracle["oracle"](point) == pytest.approx(stan["lp"], abs=1e-10, rel=0)
        np.testing.assert_allclose(oracle["grad_oracle"](point)["theta"], derivative,
                                   atol=1e-10, rtol=0)
        if frozen["test_type"] == "logdensity":
            assert frozen["expected"] == [stan["lp"]]
        else:
            np.testing.assert_array_equal(frozen["expected_grad"][0]["theta"], derivative)
