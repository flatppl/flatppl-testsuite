"""End-to-end det-js engine checks: convert-free determinize -> score(js).

These shell out to a real `flatppl determinize` binary (CONFIG.flatppl_bin)
and a real Node 24 + flatppl-js checkout (CONFIG.node_bin / CONFIG.flatppl_js_dir),
so they're skipped if either isn't available — unlike tests/core/test_engine.py's
pure-Python engine-seam tests, these exercise the actual subprocess pipeline.

The gaussian case is the first end-to-end numeric proof of the determiniser's
density lowering: the oracle is the closed-form log-normal-density, computed
here from `math` (independent of both flatppl-js and the sibling engine).
"""
from __future__ import annotations

import math
import shutil

import pytest

from flatppl_testsuite.config import CONFIG
from flatppl_testsuite.scoring.engine import DeterminizeRefused, get_engine

def _flatppl_bin_available() -> bool:
    return CONFIG.flatppl_bin.exists() or shutil.which(str(CONFIG.flatppl_bin)) is not None


pytestmark = pytest.mark.skipif(
    not _flatppl_bin_available()
    or not (CONFIG.flatppl_js_dir / "packages" / "engine" / "index.ts").exists(),
    reason="requires a determinize-capable flatppl binary and a flatppl-js checkout",
)


def test_det_js_scores_gaussian(tmp_path):
    model = tmp_path / "g.flatppl"
    model.write_text(
        "mu = elementof(reals)\n"
        "sigma = elementof(posreals)\n"
        "g = Normal(mu = mu, sigma = sigma)\n"
        "obs = likelihoodof(iid(g, 1), [1.27])\n"
    )
    value = get_engine("det-js").log_density(model, "obs", {"mu": 0.0, "sigma": 1.0})
    oracle = -0.5 * math.log(2 * math.pi) - 0.5 * 1.27**2
    assert math.isclose(value, oracle, rel_tol=0, abs_tol=1e-9), (
        f"det-js={value!r} oracle={oracle!r} delta={value - oracle!r}"
    )


def test_det_js_refuses_continuous_kchain(tmp_path):
    model = tmp_path / "k.flatppl"
    model.write_text(
        "mu = draw(Normal(mu = 0.0, sigma = 1.0))\n"
        "pp = kchain(lawof(record(mu = mu)), x -> Normal(mu = get(x, \"mu\"), sigma = 1.0))\n"
    )
    with pytest.raises(DeterminizeRefused):
        get_engine("det-js").log_density(model, "pp", {})


# det-js ≈ js cross-check (Phase 4b.3): a secondary internal check that the
# determiniser preserves the density it lowers. NOT the primary numeric proof
# — that is det-js vs the frozen ROOT vector (see corpora/hs3/tests/test_runner.py
# under FLATPPL_ENGINE=det-js). Here we just confirm the two independent
# scoring paths (determinize+materialise vs the measure-algebra interpreter)
# agree pointwise on models that are known to determinize.
_CROSS_CHECK_MODELS = [
    pytest.param(
        "mu = elementof(reals)\n"
        "sigma = elementof(posreals)\n"
        "g = Normal(mu = mu, sigma = sigma)\n"
        "obs = likelihoodof(iid(g, 1), [1.27])\n",
        "obs",
        [{"mu": 0.0, "sigma": 1.0}, {"mu": 1.5, "sigma": 0.5}, {"mu": -2.0, "sigma": 2.0}],
        id="gaussian_single_obs",
    ),
    pytest.param(
        "mu = elementof(reals)\n"
        "sigma = elementof(posreals)\n"
        "g = Normal(mu = mu, sigma = sigma)\n"
        "obs = likelihoodof(iid(g, 3), [1.27, -0.4, 2.1])\n",
        "obs",
        [{"mu": 0.0, "sigma": 1.0}, {"mu": 0.5, "sigma": 1.5}],
        id="gaussian_iid_literal_count",
    ),
]


@pytest.mark.parametrize("source, binding, thetas", _CROSS_CHECK_MODELS)
def test_det_js_matches_js_engine(tmp_path, source, binding, thetas):
    model = tmp_path / "model.flatppl"
    model.write_text(source)
    js = get_engine("js")
    det_js = get_engine("det-js")
    for theta in thetas:
        js_value = js.log_density(model, binding, theta)
        det_value = det_js.log_density(model, binding, theta)
        assert math.isclose(det_value, js_value, rel_tol=0, abs_tol=1e-9), (
            f"theta={theta}: det-js={det_value!r} js={js_value!r} "
            f"delta={det_value - js_value!r}"
        )
