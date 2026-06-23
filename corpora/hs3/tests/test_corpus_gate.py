"""Full-corpus numeric gate.

Every manifest entry's 2DeltaNLL check — the `fixtures/` scans and the
`conversions/` point clouds — must match its frozen ROOT vector. This is what
`pixi run test` uses to catch a converter/engine regression; before this, only
`rf101_basics` was gated in pytest (the rest ran only in the manual
`pixi run hs3`), which is how the conversions silently broke once.
"""

import json

import pytest

from flatppl_testsuite.runner import run
from flatppl_testsuite.suites.hs3_import import HS3_MANIFEST

_MANIFEST = json.loads(HS3_MANIFEST.read_text())
_NUMERIC_IDS = (
    [f["test_id"] for f in _MANIFEST.get("fixtures", [])]
    + [c["test_id"] for c in _MANIFEST.get("conversions", [])]
)


@pytest.mark.parametrize("test_id", _NUMERIC_IDS)
def test_corpus_numeric_checks_pass(test_id):
    results = run(selected={test_id})
    nll = [r for r in results if "twice_delta_nll" in r.check_id]
    assert nll, f"no 2DeltaNLL check ran for {test_id}"
    failed = [r for r in nll if r.status != "passed"]
    assert not failed, "; ".join(
        f"{r.check_id}: {r.tag} {r.message}" for r in failed
    )


def test_conversions_are_gated():
    """Guard against the conversions silently dropping out of the manifest."""
    conv_ids = {c["test_id"] for c in _MANIFEST.get("conversions", [])}
    assert conv_ids == {"conv_gaussian", "conv_product", "conv_histfactory"}
