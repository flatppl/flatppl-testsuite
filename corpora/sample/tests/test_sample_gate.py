"""Sample corpus numeric gate.

The sample corpus's model (``corpora/sample/hier_normal/hier_normal.flatppl``)
ends in ``rand(rng, lawof(record(mu = mu, y1 = y1, y2 = y2)))`` — a
hierarchical Normal where y1 and y2 share the SAME draw of the latent mu.
This test seed-sweeps the determinized FlatPDL over N seeds
(``scoring/sample_sweep.cjs``) and asserts the empirical mean/var/cov match
the model's closed-form structural moments (see
``corpora/sample/oracle.py`` and ``corpora/sample/gen_expected.py``), most
importantly ``cov(y1, y2) ~= 100`` — THE SHARED-ANCESTOR CATCH: if the
determinizer had sampled mu independently per consumer, y1 and y2 would be
independent and this covariance would land near 0, tens of standard errors
outside the tolerance band.

IMPORTANT: this gate goes GREEN only against fix binaries that are not yet
merged to `main` at the time this test was written — flatppl-rust's
sample-path determinizer (`rand(rng, lawof(...))` -> a rng-threaded
`builtin_sample` chain with shared-ancestor preservation) and flatppl-js's
get0-on-tuple fix (needed to destructure the `(variate, RngState)` tuple
`builtin_sample` returns). Against the currently pinned `main`, this corpus
fails determinize or scoring; that is a real, already-tracked gap, not a
bug in this test — it goes green once those fixes land and the testsuite's
pins are bumped (mirrors `corpora/fragment/tests/test_fragment_gate.py`).
"""
from __future__ import annotations

import json
import math
import shutil

import pytest

from flatppl_testsuite.config import CONFIG
from flatppl_testsuite.scoring.result import DETERMINIZE_SKIP
from flatppl_testsuite.suites.sample_gate import (
    SampleGateSuite, SAMPLE_MANIFEST, _closed_form_logdensity,
)


def _flatppl_bin_available() -> bool:
    return CONFIG.flatppl_bin.exists() or shutil.which(str(CONFIG.flatppl_bin)) is not None


pytestmark = pytest.mark.skipif(
    not _flatppl_bin_available()
    or not (CONFIG.flatppl_js_dir / "packages" / "engine" / "index.ts").exists(),
    reason="requires a sample-path-determinize-capable flatppl binary and a "
           "get0-tuple-fixed flatppl-js checkout",
)

_MANIFEST = json.loads(SAMPLE_MANIFEST.read_text())
_TEST_IDS = [m["test_id"] for m in _MANIFEST["models"]]


@pytest.mark.parametrize("test_id", _TEST_IDS)
def test_sample_numeric_check_passes(test_id):
    results = SampleGateSuite().run(selected={test_id})
    assert results, f"no check ran for {test_id}"
    for r in results:
        ok = r.status == "passed" or (r.status == "skipped" and r.tag == DETERMINIZE_SKIP)
        assert ok, f"{r.test_id}::{r.check_id}: {r.status} {r.tag} {r.message}"


def test_cov_y1_y2_check_is_gated():
    """Guard against the shared-ancestor covariance check silently dropping
    out of the frozen expected.json (the whole point of this corpus)."""
    expected = json.loads(
        (SAMPLE_MANIFEST.parent / "hier_normal" / "expected.json").read_text()
    )
    ids = {c["id"] for c in expected["checks"]}
    assert "cov_y1_y2" in ids
    cov_check = next(c for c in expected["checks"] if c["id"] == "cov_y1_y2")
    assert cov_check["fields"] == ["y1", "y2"]
    assert cov_check["expected"] == 100.0


def test_all_sample_models_are_gated():
    """Guard against a sample model silently dropping out of the manifest."""
    ids = {m["test_id"] for m in _MANIFEST["models"]}
    assert ids == {"sample_hier_normal"}


def test_closed_form_logdensity_matches_oracle_module():
    """`sample_gate.py::_closed_form_logdensity` is a deliberate duplicate of
    `corpora/sample/oracle.py::logdensity` (suites/ doesn't reach into
    corpora/ at runtime elsewhere in this toolkit). Pin them together so a
    future edit to one can't silently drift from the other."""
    from corpora.sample.oracle import logdensity as oracle_logdensity

    for mu, y1, y2 in [(0.0, 0.0, 0.0), (-18.386210602342178, -18.85413393410393,
                                          -19.263306789040588), (5.2, 4.5, 6.1)]:
        a = _closed_form_logdensity(mu, y1, y2)
        b = oracle_logdensity(mu, y1, y2)
        assert math.isclose(a, b, rel_tol=0, abs_tol=1e-12), (mu, y1, y2, a, b)
