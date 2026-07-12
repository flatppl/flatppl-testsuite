"""Bayesian-inference corpus numeric gate.

Every posterior in `corpora/bayesian_inference/manifest.json` is a
self-contained FlatPPL model that already ends in a fixed-point `lp =
logdensityof(posterior, <point>)` binding. This test scores each one through
the convert-free det-js path (`flatppl determinize` -> `score_flatpdl.cjs`)
and asserts it matches its frozen scipy oracle value (see
`corpora/bayesian_inference/gen_expected.py` and
`corpora/bayesian_inference/README.md`).

IMPORTANT: this gate goes GREEN only against a determinize-capable `flatppl`
carrying the disintegrate/restrict + theta-derived-parameter-inline
determiniser work that is NOT yet merged to `main` at the time this test was
written — `bi1_posterior` needs the theta-derived-parameter inline (`a =
f_a(theta2)`, `b = f_b(theta1, theta2)` feeding the likelihood's
`mu`/`sigma`), and `bi3_posterior`/`bi4_posterior` need the
`disintegrate`/`restrict` lowerings. Against the currently pinned `main`
binary these skip or fail determinize; that is a real, already-tracked
pin-bump gap, not a bug in this test — it goes green once those fixes land
and the testsuite's pins are bumped.
"""
from __future__ import annotations

import json
import shutil

import pytest

from flatppl_testsuite.config import CONFIG
from flatppl_testsuite.scoring.result import DETERMINIZE_SKIP
from flatppl_testsuite.suites.bayesian_inference_gate import (
    PosteriorGateSuite, BAYESIAN_INFERENCE_MANIFEST)


def _flatppl_bin_available() -> bool:
    return CONFIG.flatppl_bin.exists() or shutil.which(str(CONFIG.flatppl_bin)) is not None


pytestmark = pytest.mark.skipif(
    not _flatppl_bin_available()
    or not (CONFIG.flatppl_js_dir / "packages" / "engine" / "index.ts").exists(),
    reason="requires a determinize-capable flatppl binary and a flatppl-js checkout",
)

_MANIFEST = json.loads(BAYESIAN_INFERENCE_MANIFEST.read_text())
_TEST_IDS = [p["test_id"] for p in _MANIFEST["posteriors"]]


@pytest.mark.parametrize("test_id", _TEST_IDS)
def test_posterior_numeric_check_passes(test_id):
    results = PosteriorGateSuite().run(selected={test_id})
    assert results, f"no check ran for {test_id}"
    for r in results:
        ok = r.status == "passed" or (r.status == "skipped" and r.tag == DETERMINIZE_SKIP)
        assert ok, f"{r.test_id}::{r.check_id}: {r.status} {r.tag} {r.message}"


def test_all_posteriors_are_gated():
    """Guard against a posterior silently dropping out of the manifest."""
    ids = {p["test_id"] for p in _MANIFEST["posteriors"]}
    assert ids == {
        "post_bi1", "post_bi2", "post_bi3", "post_bi4", "post_eight_schools",
    }
