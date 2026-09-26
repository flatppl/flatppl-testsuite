"""Shared cell arrays retain their scope, batch domain, and derivatives."""

import subprocess

import numpy as np
import pytest

from flatppl_testsuite.unified import stablehlo_exec as ex
from tests.test_unified import _gate_engine


@pytest.mark.stablehlo_only
@pytest.mark.parametrize("count", [1, 4])
def test_shared_cell_arrays_use_runtime_points(tmp_path, count):
    _gate_engine("stablehlo")
    (tmp_path / "model.flatppl").write_text(
        "x = elementof(reals)\n"
        "rates = exp.([x, x + 1.0, x + 2.0])\n"
        "L = likelihoodof(Normal(sum(rates) + sum(rates .* rates), 1.0), 0.2)\n"
    )
    query = tmp_path / "query.flatppl"
    query.write_text(
        'm = load_module("model.flatppl")\n'
        "score(t) = logdensityof(m.L, record(x = t))\n"
        f"points = elementof(cartpow(reals, {count}))\n"
        "inputs = points\noutputs = sum(score.(points))\n"
    )
    serialized = tmp_path / "lowered.flatppl"
    subprocess.run(
        [str(ex.flatppl_bin()), "determinize", str(query), "--keep", "inputs",
         "--keep", "outputs", "-o", str(serialized)],
        check=True, capture_output=True, text=True,
    )
    source = ex.emit(serialized, "logdensity", dtype="f64")
    points = np.linspace(-1.0, 0.5, count)
    rates = np.exp(points[:, None] + np.arange(3))
    mean = rates.sum(axis=1) + (rates * rates).sum(axis=1)
    expected = (-0.5 * (np.log(2 * np.pi) + (mean - 0.2)**2)).sum()
    gradient = -(mean - 0.2) * (rates.sum(axis=1) + 2 * (rates * rates).sum(axis=1))
    np.testing.assert_allclose(ex.value(source, [points]), expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(ex.gradient(source, [points], [0]), [gradient], rtol=1e-12, atol=1e-12)
