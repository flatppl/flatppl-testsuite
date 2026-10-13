"""Original posteriorDB targets: native JS and differentiated StableHLO."""

import json
from pathlib import Path

import numpy as np
import pytest

from flatppl_testsuite.scoring.engine import JsScoreEngine
from flatppl_testsuite.unified import stablehlo_exec as ex
from tests.test_unified import _gate_engine


CASES = sorted((Path(__file__).resolve().parents[1] / "corpora/posteriordb").glob("*/test.json"))


@pytest.mark.parametrize("manifest", CASES, ids=lambda path: path.parent.name)
def test_native_js_posterior(manifest):
    """Exercise JS's own measure layer and file loader, without Rust lowering."""
    _gate_engine("det-js")
    spec = json.loads(manifest.read_text())
    engine = JsScoreEngine()
    for point, expected in zip(spec["points"], spec["expected"], strict=True):
        parameters = {name.removeprefix("p_"): value for name, value in point.items()}
        actual = engine.log_density(manifest.parent / "model.flatppl", "Pi_post", parameters)
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-8)


@pytest.mark.stablehlo_only
@pytest.mark.parametrize("precision", ["f32", "f64"])
@pytest.mark.parametrize("manifest", CASES, ids=lambda path: path.parent.name)
def test_stablehlo_posterior_gradients(manifest, precision):
    """Check full constrained gradients against frozen, independent Stan AD."""
    _gate_engine("stablehlo")
    spec = json.loads(manifest.read_text())
    source = ex.emit_concat(manifest.parent, "logdensity", dtype=precision)
    atol, rtol = (3e-4, 3e-5) if precision == "f32" else (1e-9, 1e-10)
    inputs = spec["inputs"]
    for point, expected_value, expected_gradient in zip(
        spec["points"], spec["expected"], spec["expected_grad"], strict=True,
    ):
        arguments = [point[name] for name in inputs]
        np.testing.assert_allclose(ex.value(source, arguments), expected_value, rtol=rtol, atol=atol)
        gradient = ex.gradient(source, arguments, list(range(len(inputs))))
        for name, actual in zip(inputs, gradient, strict=True):
            np.testing.assert_allclose(actual, expected_gradient[name], rtol=rtol, atol=atol)
