"""Native numeric checks of spec §07's explicit (value, new_state) contract."""
from __future__ import annotations

import math
import shutil

import pytest

from flatppl_testsuite.config import CONFIG
from flatppl_testsuite.scoring.engine import sample_sweep


pytestmark = pytest.mark.skipif(
    not (CONFIG.flatppl_bin.exists() or shutil.which(str(CONFIG.flatppl_bin)))
    or not (CONFIG.flatppl_js_dir / "packages" / "engine" / "index.ts").exists(),
    reason="requires a determinize-capable flatppl binary and a flatppl-js checkout",
)


@pytest.fixture
def rand_model(tmp_path):
    model = tmp_path / "rand_states.flatppl"
    model.write_text(
        "s = rnginit([42, 0, 0, 0])\n"
        "first, s2 = rand(s, Normal(0.0, 1.0))\n"
        "second, s3 = rand(s2, Normal(0.0, 1.0))\n"
        "replay, unused = rand(s, Normal(0.0, 1.0))\n"
    )
    return model


def test_rand_preserves_explicit_state_sources(rand_model):
    bindings = ["first", "second", "replay"]
    samples = sample_sweep(rand_model, 8, bindings)
    assert sample_sweep(rand_model, 8, bindings) == samples
    assert sample_sweep(rand_model, 8, list(reversed(bindings))) == samples
    assert all(row["first"] == row["replay"] for row in samples)
    assert any(row["first"] != row["second"] for row in samples)


def test_sequential_rand_has_independent_normal_noise(rand_model):
    n = 4096
    samples = sample_sweep(rand_model, n, ["first", "second"])
    assert len(samples) == n
    assert all(math.isfinite(row[key]) for row in samples for key in ("first", "second"))

    # Under independent standard Normals, Var(X)=1, Var(X²)=2 and Var(XY)=1.
    # These six-SE bands use the analytic oracle, not a fitted sample variance.
    for key in ("first", "second"):
        mean = math.fsum(row[key] for row in samples) / n
        second_moment = math.fsum(row[key] ** 2 for row in samples) / n
        assert abs(mean) < 6 / math.sqrt(n), (key, "mean", mean)
        assert abs(second_moment - 1) < 6 * math.sqrt(2 / n), (
            key, "second moment", second_moment,
        )
    cross_moment = math.fsum(row["first"] * row["second"] for row in samples) / n
    assert abs(cross_moment) < 6 / math.sqrt(n), ("cross moment", cross_moment)
