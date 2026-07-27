"""The hs3 conversions' ROOT refresh must update `test.json`, in place.

`corpora/hs3/conversions/gen_expected.py` is the ONLY way to regenerate those
three frozen `twice_delta_nll_points` vectors: `unified/regen.py` deliberately
refuses `test_type: "convert"` and points the user here, because the values are
ROOT/RooFit oracle numbers produced in the separate `root` pixi env, not by an
offline `test.py`.

It used to write a legacy `expected.json` in the pre-migration schema. After the
unified migration the live file is `test.json` and `expected.json` is gone, so
the refresh became a silent no-op: it ran, printed success, wrote a file nothing
reads, and left the frozen vectors untouched.

The ROOT-dependent half (`root_logL`) cannot run here -- it needs the `root` env
and the HS3SUITE checkout. So the merge is a pure function taking already-computed
numbers, and that is what these tests pin: the part that was broken was the
serialisation, not the physics.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

_CONV = Path(__file__).resolve().parents[2] / "corpora" / "hs3" / "conversions"

# Import without executing the ROOT-dependent module top-level.
from flatppl_testsuite.unified.loader import load_test_module  # noqa: E402


def _merge():
    """The merge helper under test, imported lazily so a missing ROOT does not
    break collection (the module guards its ROOT import for exactly this)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_gen_expected", _CONV / "gen_expected.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.merged_doc


MODELS = ["gaussian", "histfactory", "product"]


@pytest.mark.parametrize("model", MODELS)
def test_merge_preserves_the_unified_envelope(model: str):
    """Refreshing must not strip test_type/engines/fixture_kind/model."""
    merged_doc = _merge()
    current = json.loads((_CONV / model / "test.json").read_text())
    check = current["checks"][0]

    # Arity must match the check's own point count (merged_doc enforces this).
    fresh = [float(i) for i in range(len(check["points"]))]
    out = merged_doc(current, expected=fresh, backend="root 6.99.00")

    for key in ("test_type", "engines", "fixture_kind", "model"):
        assert out[key] == current[key], f"{model}: {key} not preserved"
    assert out["test_type"] == "convert"
    # The check keeps its identity and its points; only `expected` moves.
    assert out["checks"][0]["id"] == check["id"]
    assert out["checks"][0]["kind"] == check["kind"]
    assert out["checks"][0]["points"] == check["points"]
    assert out["checks"][0]["tolerance"] == check["tolerance"]
    assert out["checks"][0]["expected"] == fresh
    assert out["backend"] == "root 6.99.00"


def test_merge_does_not_emit_the_legacy_schema():
    merged_doc = _merge()
    current = json.loads((_CONV / "gaussian" / "test.json").read_text())
    n = len(current["checks"][0]["points"])
    out = merged_doc(current, expected=[0.0] * n, backend="root 6.99.00")
    for dead in ("schema_version", "test_id", "reference_backend"):
        assert dead not in out, f"legacy key {dead!r} reintroduced"


def test_merge_is_identity_when_the_numbers_are_unchanged():
    """Re-running the refresh with the values already frozen must be a no-op,
    so a genuine ROOT change is the only thing that ever shows up in a diff."""
    merged_doc = _merge()
    current = json.loads((_CONV / "gaussian" / "test.json").read_text())
    out = merged_doc(
        current,
        expected=current["checks"][0]["expected"],
        backend=current["backend"],
    )
    assert out == current


@pytest.mark.parametrize("model", MODELS)
def test_no_legacy_expected_json_remains(model: str):
    assert not (_CONV / model / "expected.json").exists(), (
        f"{model}: a legacy expected.json is present again -- the live file is test.json"
    )


def test_merge_rejects_a_wrong_length_vector():
    """A refresh producing the wrong number of values must raise, not silently
    freeze a truncated vector against the declared points."""
    merged_doc = _merge()
    current = json.loads((_CONV / "gaussian" / "test.json").read_text())
    with pytest.raises(ValueError, match="points"):
        merged_doc(current, expected=[0.0], backend="root 6.99.00")
