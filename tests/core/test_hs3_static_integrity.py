"""`static_integrity` must actually verify the vendored fixture, not just parse it.

`corpora/hs3/` vendors a subset of the upstream HS3TestSuite fixtures. The point
of a `static_integrity` check is that a vendored copy has not drifted from the
content its frozen ROOT vector was computed against -- otherwise the scan vector
and the fixture can disagree silently and the corpus asserts nothing meaningful.

Both the legacy gate and the first unified port implemented it as an
unconditional pass ("the JSON parsed, so it's fine"), while the legacy
`manifest.json` carried per-fixture `canonical_sha256`/`sha256` hashes that
nothing ever read. So the data for a real check existed and was never wired up.

The hash is over the CANONICAL form -- `json.dumps(doc, sort_keys=True,
separators=(",", ":"))` -- not the raw bytes. Verified against the recorded
hashes: the canonical form matches all 5 vendored fixtures, whereas raw bytes
match only 3 (`rf203_ranges` and `rf207_comptools` carry byte-level formatting
churn with identical semantic content). Hashing raw bytes would therefore fail
two fixtures for a difference that cannot affect any result.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from flatppl_testsuite.unified.loader import load_test
from flatppl_testsuite.unified.runners import convert_detjs

_CORPORA = Path(__file__).resolve().parents[2] / "corpora"
_FIXTURES = _CORPORA / "hs3" / "fixtures"


def _fixture_dirs() -> list[Path]:
    return sorted(d for d in _FIXTURES.iterdir() if (d / "test.json").exists())


_IDS = [d.name for d in _fixture_dirs()]


@pytest.mark.parametrize("dir", _fixture_dirs(), ids=_IDS)
def test_every_fixture_declares_its_canonical_hash(dir: Path):
    body = json.loads((dir / "test.json").read_text())
    kinds = [c["kind"] for c in body["checks"]]
    if "static_integrity" not in kinds:
        pytest.skip("no static_integrity check declared")
    check = next(c for c in body["checks"] if c["kind"] == "static_integrity")
    assert check.get("canonical_sha256"), (
        f"{dir.name}: static_integrity declares no canonical_sha256, so it cannot "
        "verify anything"
    )


@pytest.mark.parametrize("dir", _fixture_dirs(), ids=_IDS)
def test_static_integrity_passes_on_the_pristine_fixture(dir: Path):
    results = convert_detjs.run(load_test(dir), dir)
    integrity = [r for r in results if "static_integrity" in r.check_id]
    assert integrity, f"{dir.name}: no static_integrity result produced"
    assert all(r.status == "passed" for r in integrity), [
        (r.check_id, r.status, r.message) for r in integrity
    ]


def test_static_integrity_fails_on_a_tampered_fixture(tmp_path: Path):
    """The whole point: a changed fixture must be caught.

    Copies a fixture, perturbs a NUMBER inside the HS3 document (semantic, so the
    canonical form genuinely changes), and asserts the check fails.
    """
    import shutil

    src = _FIXTURES / "rf101_basics"
    d = tmp_path / "rf101_basics"
    shutil.copytree(src, d)

    doc = json.loads((d / "hs3.json").read_text())
    # Perturb the first numeric leaf we can find, so the canonical form changes.
    def perturb(node) -> bool:
        if isinstance(node, dict):
            for k, v in node.items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    node[k] = v + 1.5
                    return True
                if perturb(v):
                    return True
        elif isinstance(node, list):
            for i, v in enumerate(node):
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    node[i] = v + 1.5
                    return True
                if perturb(v):
                    return True
        return False

    assert perturb(doc), "found no numeric leaf to perturb"
    (d / "hs3.json").write_text(json.dumps(doc, indent=2) + "\n")

    results = convert_detjs.run(load_test(d), d)
    integrity = [r for r in results if "static_integrity" in r.check_id]
    assert integrity, "no static_integrity result produced"
    assert any(r.status == "failed" for r in integrity), (
        "a tampered fixture passed static_integrity: "
        f"{[(r.check_id, r.status, r.message) for r in integrity]}"
    )
