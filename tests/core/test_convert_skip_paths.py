"""A fixture the converter cannot handle must SKIP with its HS3 type, not fail.

The hs3 corpus's operating rule is that a fixture which won't convert is a
signal to extend the converter, not something to work around -- so the harness
distinguishes:

* `CONVERT_SKIP` -- `convert()` raised `SkipUnimplemented`; the message carries
  the offending HS3 type so the coverage gap is named, not anonymous.
* `DETERMINIZE_SKIP` -- conversion succeeded but the determiniser can't lower
  the result to FlatPDL.

Neither path is exercised by the corpus: all 8 vendored fixtures convert and
lower cleanly today. The legacy suite covered it with a monkeypatched
`test_run_unimplemented_skips`, deleted along with the gates. Without a test,
the branch that classifies a coverage gap could rot unnoticed and a future
unconvertible fixture would surface as an opaque failure instead of a named gap.

These tests inject the exception rather than shipping a deliberately-broken
fixture, so no corpus file has to be sacrificed to cover the path.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from flatppl_testsuite.formats.hs3.importer import SkipUnimplemented
from flatppl_testsuite.scoring.result import CONVERT_SKIP, DETERMINIZE_SKIP
from flatppl_testsuite.unified.loader import load_test
from flatppl_testsuite.unified.runners import convert_detjs

_CORPORA = Path(__file__).resolve().parents[2] / "corpora"
_FIXTURE = _CORPORA / "hs3" / "fixtures" / "rf101_basics"


def test_unconvertible_fixture_skips_naming_its_hs3_type(monkeypatch):
    """`SkipUnimplemented` -> CONVERT_SKIP, message == the HS3 type."""
    def boom(*a, **k):
        raise SkipUnimplemented("fft_conv_pdf")

    monkeypatch.setattr(convert_detjs, "convert", boom)

    results = convert_detjs.run(load_test(_FIXTURE), _FIXTURE)
    assert results, "no results produced"

    skipped = [r for r in results if r.status == "skipped"]
    assert skipped, [(r.check_id, r.status, r.tag) for r in results]
    assert all(r.tag == CONVERT_SKIP for r in skipped), (
        f"expected every skip tagged {CONVERT_SKIP}: "
        f"{[(r.check_id, r.tag) for r in skipped]}"
    )
    assert all(r.message == "fft_conv_pdf" for r in skipped), (
        "the skip must NAME the unsupported HS3 type so the gap is not anonymous: "
        f"{[(r.check_id, r.message) for r in skipped]}"
    )
    assert not [r for r in results if r.status == "failed"], (
        "an unconvertible fixture must skip, never fail"
    )


def test_determiniser_refusal_is_tagged_separately(monkeypatch):
    """A refusal AFTER conversion is DETERMINIZE_SKIP, a distinct diagnosis."""
    from flatppl_testsuite.unified import detjs_exec

    def refuse(*a, **k):
        raise detjs_exec.DeterminizeRefused("refuse draw (node NodeId(7))")

    monkeypatch.setattr(convert_detjs, "score_scan", refuse)

    results = convert_detjs.run(load_test(_FIXTURE), _FIXTURE)
    tags = {r.tag for r in results if r.status == "skipped"}
    assert DETERMINIZE_SKIP in tags, (
        f"expected a {DETERMINIZE_SKIP} skip; got {[(r.check_id, r.status, r.tag) for r in results]}"
    )
    assert CONVERT_SKIP not in tags, (
        "a post-conversion refusal must NOT be reported as a conversion gap"
    )


def test_the_two_skip_tags_are_distinct_strings():
    """They are separate diagnoses; collapsing them would lose the distinction
    between 'the converter cannot read this' and 'the determiniser cannot lower
    what the converter produced'."""
    assert CONVERT_SKIP != DETERMINIZE_SKIP


def test_the_fixture_used_here_normally_passes():
    """Guards the tests above: if the fixture stopped converting on its own, the
    monkeypatched assertions would pass for the wrong reason."""
    results = convert_detjs.run(load_test(_FIXTURE), _FIXTURE)
    assert results
    assert not [r for r in results if r.status != "passed"], [
        (r.check_id, r.status, r.tag, r.message) for r in results
    ]
    body = json.loads((_FIXTURE / "test.json").read_text())
    assert body["test_type"] == "convert"
