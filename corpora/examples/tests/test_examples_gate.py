"""Examples corpus numeric gate.

Every entry in `corpora/examples/manifest.json` names a flatppl-examples
model, a binding to query (typically `posterior`), and a theta grid; this
test builds `logdensityof(binding, theta_i)` for each grid point via the
convert-free det-js path (`flatppl determinize` -> `score_flatpdl.cjs`) and
either compares it to a frozen oracle (`status: "lowers"`) or asserts the
determinizer refuses it (`status: "refuses"`) — see
`flatppl_testsuite.suites.examples_gate` for the full schema and outcome
mapping.

Scaffold only (Task 1): `manifest.json` has zero entries, so the
parametrization below is empty (no-op — an empty parametrize collects zero
tests, not a failure) and `test_all_examples_are_gated` is a placeholder.
Task 2 populates the manifest.
"""
from __future__ import annotations

import json
import shutil

import pytest

from flatppl_testsuite.config import CONFIG
from flatppl_testsuite.suites.examples_gate import (
    ExamplesGateSuite, EXAMPLES_MANIFEST)


def _flatppl_bin_available() -> bool:
    return CONFIG.flatppl_bin.exists() or shutil.which(str(CONFIG.flatppl_bin)) is not None


pytestmark = pytest.mark.skipif(
    not _flatppl_bin_available()
    or not (CONFIG.flatppl_js_dir / "packages" / "engine" / "index.ts").exists()
    or not (CONFIG.examples_dir / "examples").is_dir(),
    reason="requires a determinize-capable flatppl binary, a flatppl-js checkout, "
           "and a flatppl-examples checkout",
)

_MANIFEST = json.loads(EXAMPLES_MANIFEST.read_text())
_TEST_IDS = [ex["test_id"] for ex in _MANIFEST.get("examples", [])]


@pytest.mark.parametrize("test_id", _TEST_IDS)
def test_example_numeric_check_passes(test_id):
    results = ExamplesGateSuite().run(selected={test_id})
    assert results, f"no check ran for {test_id}"
    for r in results:
        assert r.status == "passed", f"{r.test_id}::{r.check_id}: {r.status} {r.tag} {r.message}"


def test_all_examples_are_gated():
    """Guard against a flatppl-examples posterior silently dropping out of
    the manifest.

    The manifest is empty until Task 2 populates it with every
    non-excluded flatppl-examples model that defines a `posterior` binding
    — there is no roster to check yet, so this just documents the
    intent and stays green on the empty manifest.
    """
    if not _TEST_IDS:
        pytest.skip("manifest.json has no examples yet (Task 2 populates it)")
    ids = {ex["test_id"] for ex in _MANIFEST["examples"]}
    assert ids == set(_TEST_IDS)
