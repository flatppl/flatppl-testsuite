"""The JS scorers refuse a score that reads a free draw.

§13 "Output reduction": draw nodes take their values from the explicit point,
unless marginalized out. A density whose measure reads a draw outside every
lawof has no single value; the JS engine evaluates it per atom (decisions-log
2026-10-21: no implicit marginal). The scorers materialise one atom, so without
the check they reported one random conditional density as the score. They now
refuse with the wording of the flatppl-rust determiniser. Legal spellings
(lawof, a kernelof likelihood, a posterior) still score their closed forms.
"""
from __future__ import annotations

import math
import shutil
from pathlib import Path

import pytest

from flatppl_testsuite.config import CONFIG
from flatppl_testsuite.scoring import worker

pytestmark = pytest.mark.skipif(
    not (CONFIG.flatppl_js_dir / "packages" / "engine" / "index.ts").exists()
    or shutil.which(CONFIG.node_bin) is None and not Path(CONFIG.node_bin).exists(),
    reason="requires Node and a flatppl-js checkout",
)

_SHARED = ("z = draw(Normal(mu = 0.4, sigma = 1.0))\n"
           "M = joint(f1 = Normal(mu = z, sigma = 1.0), f2 = Normal(mu = z, sigma = 1.5))\n")


def _model(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "m.flatppl"
    path.write_text(text)
    return path


def _logn(x: float, mu: float, var: float) -> float:
    return -0.5 * math.log(2 * math.pi * var) - (x - mu) ** 2 / (2 * var)


def test_score_js_refuses_a_measure_that_reads_a_free_draw(tmp_path):
    with pytest.raises(RuntimeError) as raised:
        worker.score_js_file(_model(tmp_path, _SHARED), "M", "record(f1 = 0.5, f2 = 0.7)")
    msg = str(raised.value)
    assert "the density query in `M` reads draw `z`" in msg
    assert "neither in its point nor integrated out by a `lawof`" in msg
    assert "put `z` in the point, or score `lawof(...)` of the value instead" in msg


def test_score_js_names_an_inline_draw(tmp_path):
    model = _model(tmp_path, "M = Normal(mu = draw(Normal(mu = 0.0, sigma = 1.0)), sigma = 1.0)\n")
    with pytest.raises(RuntimeError) as raised:
        worker.score_js_file(model, "M", "0.5")
    assert "the density query in `M` reads an inline `draw`" in str(raised.value)


def test_score_binding_refuses_a_free_draw_density_value(tmp_path):
    model = _model(tmp_path, "z = draw(Normal(mu = 0.0, sigma = 1.0))\n"
                             "lp = logdensityof(Normal(mu = z, sigma = 1.0), 0.5)\n")
    with pytest.raises(RuntimeError) as raised:
        worker.score_binding_file(model, "lp")
    assert "the density query in `lp` reads draw `z`" in str(raised.value)


@pytest.mark.parametrize("text, binding, theta, expected", [
    # lawof(y) with y ~ Normal(z, 1), z ~ Normal(0, 1): y ~ Normal(0, 2).
    ("z = draw(Normal(mu = 0.0, sigma = 1.0))\ny = draw(Normal(mu = z, sigma = 1.0))\nM = lawof(y)\n",
     "M", "0.5", _logn(0.5, 0.0, 2.0)),
    # A kernelof likelihood integrates its ancestor draw: obs | theta ~ Normal(theta, 2).
    ("theta = elementof(reals)\nz = draw(Normal(mu = 0.0, sigma = 1.0))\n"
     "y = draw(Normal(mu = theta + z, sigma = 1.0))\nK = kernelof(y, theta = theta)\nL = likelihoodof(K, 0.5)\n",
     "L", "record(theta = 0.2)", _logn(0.5, 0.2, 2.0)),
    # A joint law: log N(0.1; 0, 1) + log N(0.3; 0.1, 1).
    ("a ~ Normal(mu = 0.0, sigma = 1.0)\nb ~ Normal(mu = a, sigma = 1.0)\nM = lawof(record(a = a, b = b))\n",
     "M", "record(a = 0.1, b = 0.3)", _logn(0.1, 0.0, 1.0) + _logn(0.3, 0.1, 1.0)),
])
def test_score_js_still_scores_marginals_and_likelihoods(tmp_path, text, binding, theta, expected):
    assert worker.score_js_file(_model(tmp_path, text), binding, theta) == pytest.approx(expected, abs=1e-12)


def test_score_binding_still_scores_a_density_through_lawof(tmp_path):
    model = _model(tmp_path, "z = draw(Normal(mu = 0.0, sigma = 1.0))\n"
                             "lp = logdensityof(lawof(z), 0.1) + 1.0\n")
    assert worker.score_binding_file(model, "lp") == pytest.approx(_logn(0.1, 0.0, 1.0) + 1.0, abs=1e-12)
