"""Fragment corpus numeric gate.

Every fragment in `corpora/fragment/manifest.json` is a self-contained
FlatPPL model that already ends in a fixed-point `lp = logdensityof(m,
<point>)` binding. This test scores each one through the convert-free det-js
path (`flatppl determinize` -> `score_flatpdl.cjs`) and asserts it matches
its frozen Julia/scipy oracle value (see `corpora/fragment/gen_expected.py`
and `corpora/fragment/README.md`).

IMPORTANT: this gate goes GREEN only against fix binaries that are not yet
merged to `main` at the time this test was written — flatppl-rust's
determinizer lowering `logsumexp` to a vector argument (superpose, kchain_*),
and flatppl-js's value-level `x in interval(lo, hi)` evaluation plus a
fixed-phase +-inf materialiser fix (trunc_in, trunc_out, norm_trunc). Against
the currently pinned `main`, several of these fragments fail determinize or
scoring; that is a real, already-tracked gap, not a bug in this test — it
goes green once those fixes land and the testsuite's pins are bumped.
"""
from __future__ import annotations

import json
import shutil

import pytest

from flatppl_testsuite.config import CONFIG
from flatppl_testsuite.scoring.result import DETERMINIZE_SKIP
from flatppl_testsuite.suites.fragment_gate import FragmentGateSuite, FRAGMENT_MANIFEST


def _flatppl_bin_available() -> bool:
    return CONFIG.flatppl_bin.exists() or shutil.which(str(CONFIG.flatppl_bin)) is not None


pytestmark = pytest.mark.skipif(
    not _flatppl_bin_available()
    or not (CONFIG.flatppl_js_dir / "packages" / "engine" / "index.ts").exists(),
    reason="requires a determinize-capable flatppl binary and a flatppl-js checkout",
)

_MANIFEST = json.loads(FRAGMENT_MANIFEST.read_text())
_TEST_IDS = [f["test_id"] for f in _MANIFEST["fragments"]]


@pytest.mark.parametrize("test_id", _TEST_IDS)
def test_fragment_numeric_check_passes(test_id):
    results = FragmentGateSuite().run(selected={test_id})
    assert results, f"no check ran for {test_id}"
    for r in results:
        ok = r.status == "passed" or (r.status == "skipped" and r.tag == DETERMINIZE_SKIP)
        assert ok, f"{r.test_id}::{r.check_id}: {r.status} {r.tag} {r.message}"


def test_all_fragments_are_gated():
    """Guard against a fragment silently dropping out of the manifest."""
    ids = {f["test_id"] for f in _MANIFEST["fragments"]}
    assert ids == {
        "frag_superpose", "frag_trunc_in", "frag_trunc_out", "frag_norm_trunc",
        "frag_pushfwd_affine", "frag_pushfwd_exp", "frag_kchain_bern", "frag_kchain_cat",
    }
