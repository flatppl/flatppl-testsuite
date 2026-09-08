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
from pathlib import Path

import pytest

from flatppl_testsuite.config import CONFIG
from flatppl_testsuite.scoring.engine import DeterminizeRefused, get_engine, sample_sweep
from flatppl_testsuite.unified.detjs_exec import log_density_points, score_abi_points

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


def _record_scores(model, binding, points, scorer):
    if scorer == "single":
        return [get_engine("det-js").log_density(model, binding, point) for point in points]
    scores = log_density_points(model, binding, points)
    assert all(score.error is None for score in scores)
    return [score.value for score in scores]


@pytest.mark.parametrize("scorer", ["single", "batch"])
def test_det_js_scores_boolean_record(tmp_path, scorer):
    model = tmp_path / "boolean.flatppl"
    model.write_text("a ~ Bernoulli(0.3)\nM = lawof(record(a = a))\n")
    values = _record_scores(model, "M", [{"a": True}, {"a": False}], scorer)
    assert values == pytest.approx([math.log(0.3), math.log(0.7)], rel=0, abs=1e-9)


@pytest.mark.parametrize("scorer", ["single", "batch"])
def test_det_js_scores_nested_relative_module(tmp_path, scorer):
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "leaf.flatppl").write_text("M = joint(x = Normal(2.0, 3.0))\n")
    (sub / "helper.flatppl").write_text('leaf = load_module("leaf.flatppl")\nM = leaf.M\n')
    model = tmp_path / "model.flatppl"
    model.write_text('helpers = load_module("sub/helper.flatppl")\n')
    values = _record_scores(model, "helpers.M", [{"x": 3.0}], scorer)
    oracle = -math.log(3) - 0.5 * math.log(2 * math.pi) - 1 / 18
    assert values == pytest.approx([oracle], rel=0, abs=1e-9)


def test_det_js_refuses_continuous_kchain(tmp_path):
    # theta must be non-empty: an empty record renders as `record()`, which spec
    # §04 forbids ("Nullary calls (`f()`) are not allowed"), so inference refused
    # at exit 1 before the kchain refusal under test could fire at exit 3.
    model = tmp_path / "k.flatppl"
    model.write_text(
        "mu = draw(Normal(mu = 0.0, sigma = 1.0))\n"
        "pp = kchain(lawof(record(mu = mu)), x -> Normal(mu = get(x, \"mu\"), sigma = 1.0))\n"
    )
    with pytest.raises(DeterminizeRefused):
        get_engine("det-js").log_density(model, "pp", {"mu": 0.0})


@pytest.fixture
def sample_model():
    return Path(__file__).resolve().parents[2] / "corpora/sample/hier_normal/hier_normal.flatppl"


def test_sample_sweep_uses_all_seed_bytes(sample_model):
    bindings = ["mu", "y1", "y2"]
    base = sample_sweep(sample_model, 2, bindings, base=0)
    assert sample_sweep(sample_model, 2, bindings, base=0) == base
    third_byte = sample_sweep(sample_model, 2, bindings, base=1 << 16)
    fourth_byte = sample_sweep(sample_model, 2, bindings, base=1 << 24)
    last_index = sample_sweep(sample_model, 1, bindings, base=(1 << 32)-1)
    assert base != third_byte
    assert base != fourth_byte
    assert third_byte != fourth_byte
    assert last_index != base[:1]


@pytest.mark.parametrize("n,base", [(1.5, 0), (1, 0.5), (1, -1), (1, 1 << 32), (2, (1 << 32)-1)])
def test_sample_sweep_rejects_unrepresentable_seed_ranges(sample_model, n, base):
    with pytest.raises(RuntimeError):
        sample_sweep(sample_model, n, ["mu"], base=base)


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


@pytest.mark.parametrize("scorer", ["single", "batch", "abi"])
def test_det_js_scores_a_model_in_a_read_only_directory(tmp_path, scorer):
    directory = tmp_path / "readonly"
    directory.mkdir()
    model, query = directory / "model.flatppl", directory / "query.flatppl"
    model.write_text("m = joint(a = Normal(0.0, 1.0))\n")
    query.write_text("x = elementof(reals)\nlp = logdensityof(m, record(a=x))\n"
                     "inputs = x\noutputs = lp\n")
    directory.chmod(0o555)
    try:
        if scorer == "abi":
            scores = score_abi_points(model, query, ["x"], [{"x": 1.0}])
            assert all(score.error is None for score in scores)
            values = [score.value for score in scores]
        else:
            values = _record_scores(model, "m", [{"a": 1.0}], scorer)
    finally:
        directory.chmod(0o755)
    assert values == pytest.approx([-0.5 * math.log(2 * math.pi) - 0.5], rel=0, abs=1e-9)
