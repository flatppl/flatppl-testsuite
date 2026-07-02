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
