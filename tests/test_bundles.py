"""Execute one independently built ZIP through both engine hosts."""

import math
from pathlib import Path
import subprocess
from zipfile import ZIP_DEFLATED, ZipFile

import pytest

from flatppl_testsuite.config import CONFIG
from flatppl_testsuite.unified import stablehlo_exec as ex
from tests.test_unified import _gate_engine


PORTABLE = Path(__file__).resolve().parents[1] / "corpora/bundles/portable.zip"
QUERY = """left = elementof(reals)
right = elementof(reals)
a = load_module("portable.zip", theta = left)
b = load_module("portable.zip", theta = right)
inputs = (left, right)
outputs = a.outputs + 2*b.outputs
"""
POSTERIOR = """source = load_module("portable.zip")
observed_sum = sum(source.observations)
theta ~ Normal(0, 1)
y ~ iid(Normal(theta, 1), 2)
prior = lawof(record(theta = theta))
K = kernelof(y, theta = theta)
L = likelihoodof(K, source.observations)
posterior = bayesupdate(L, prior)
"""


def density(theta):
    return -1.5 * math.log(2 * math.pi) - 0.5 * (
        theta**2 + (1 - theta)**2 + (2 - theta)**2
    )


@pytest.fixture
def imported_bundle(tmp_path):
    path = tmp_path / "query.zip"
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("main.flatppl", QUERY)
        archive.writestr("portable.zip", PORTABLE.read_bytes())
    return path


def test_native_js_archive_density(tmp_path):
    _gate_engine("det-js")
    engine = str(CONFIG.flatppl_js_dir / "packages/engine")
    prior = subprocess.run(
        [CONFIG.node_bin, str(CONFIG.scorer), str(PORTABLE), "prior.Pi", "0.0", "--engine", engine],
        check=True, capture_output=True, text=True,
    )
    assert float(prior.stdout) == pytest.approx(-0.5 * math.log(2 * math.pi), abs=1e-12)
    posterior = tmp_path / "posterior.zip"
    with ZipFile(posterior, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("main.flatppl", POSTERIOR)
        archive.writestr("portable.zip", PORTABLE.read_bytes())
    observed_sum = subprocess.run(
        [CONFIG.node_bin, str(CONFIG.flatpdl_scorer), str(posterior), "observed_sum", "--engine", engine],
        check=True, capture_output=True, text=True,
    )
    assert float(observed_sum.stdout) == 3.0
    for theta in (0.0, 1.0, -2.0):
        scored = subprocess.run(
            [CONFIG.node_bin, str(CONFIG.scorer), str(posterior), "posterior", f"record(theta = {theta})", "--engine", engine],
            check=True, capture_output=True, text=True,
        )
        assert float(scored.stdout) == pytest.approx(density(theta), abs=1e-11)


@pytest.mark.stablehlo_only
@pytest.mark.parametrize("precision,tolerance", [("f32", 2e-5), ("f64", 1e-11)])
def test_stablehlo_archive_values_and_gradients(precision, tolerance):
    _gate_engine("stablehlo")
    source = ex.emit(PORTABLE, "logdensity", dtype=precision)
    for theta in (0.0, 1.0, -2.0):
        assert ex.value(source, [theta]) == pytest.approx(density(theta), abs=tolerance)
        assert ex.gradient(source, [theta], [0]) == pytest.approx([3 - 3*theta], abs=tolerance)


@pytest.mark.stablehlo_only
def test_stablehlo_nested_archive_substitutions(imported_bundle):
    _gate_engine("stablehlo")
    source = ex.emit(imported_bundle, "logdensity", dtype="f32")
    assert ex.value(source, [0.0, 2.0]) == pytest.approx(density(0) + 2*density(2), abs=2e-5)
    assert ex.gradient(source, [0.0, 2.0], [0, 1]) == pytest.approx([3.0, -6.0], abs=2e-5)
