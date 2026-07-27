"""`regen` must round-trip a frozen value without corrupting the file.

Two properties, both of which regen has to hold for every test_type:

* **JSON-spec validity.** Python's `json.dumps` emits bare `Infinity` / `-Infinity`
  / `NaN` by default, which are NOT valid JSON (RFC 8259 has no such literals).
  Python and `jq` accept them; a strict parser -- notably JavaScript's
  `JSON.parse`, and this repo scores through a Node engine -- rejects them. The
  corpus therefore stores non-finite frozen values as the STRINGS "inf" / "-inf" /
  "nan" (see `detjs_exec.parse_expected`), and regen must preserve that encoding
  rather than rewriting it to a bare literal.
* **Value preservation.** Regenerating must not change what the test asserts:
  the value the oracle produces has to compare equal to the value already frozen.

`corpora/fragment/trunc_out` is the case that exercises both: a truncation gate
scored outside its support, whose density is exactly 0 and whose log-density is
therefore `-inf`.

Each test regenerates into a COPY so the corpus is never mutated.
"""
from __future__ import annotations

import json
import math
import re
import shutil
from pathlib import Path

import pytest

from flatppl_testsuite.unified.regen import regen_dir

_CORPORA = Path(__file__).resolve().parents[2] / "corpora"
_TRUNC_OUT = _CORPORA / "fragment" / "trunc_out"

# Bare non-finite literals, as a JSON *token* (not inside a string).
_BARE_NONFINITE = re.compile(r'(?<!")\b(-?Infinity|NaN)\b(?!")')


def _copy_dir(src: Path, dst: Path) -> Path:
    shutil.copytree(src, dst)
    return dst


def test_regen_keeps_non_finite_expected_json_valid(tmp_path: Path):
    d = _copy_dir(_TRUNC_OUT, tmp_path / "trunc_out")
    before = json.loads((d / "test.json").read_text())
    assert before["expected"] == "-inf", "fixture drifted: expected the string form"

    regen_dir(d)

    text = (d / "test.json").read_text()
    bare = _BARE_NONFINITE.search(text)
    assert bare is None, (
        f"regen wrote the bare JSON-invalid literal {bare.group(0)!r}; non-finite "
        'frozen values must stay in their string form ("inf"/"-inf"/"nan")'
    )

    after = json.loads(text)
    assert math.isinf(float(after["expected"])) and float(after["expected"]) < 0, (
        f"regen changed the frozen value to {after['expected']!r}"
    )


def test_regen_preserves_the_frozen_value(tmp_path: Path):
    """Regen is value-preserving: the oracle reproduces what is already frozen."""
    d = _copy_dir(_TRUNC_OUT, tmp_path / "trunc_out")
    before = float(json.loads((d / "test.json").read_text())["expected"])
    regen_dir(d)
    after = float(json.loads((d / "test.json").read_text())["expected"])
    assert (before == after) or (math.isnan(before) and math.isnan(after)), (
        f"regen changed the frozen value: {before!r} -> {after!r}"
    )


@pytest.mark.parametrize(
    "dirname", ["superpose", "pushfwd_exp", "kchain_bern"], ids=lambda s: s
)
def test_regen_is_value_preserving_for_finite_dirs(tmp_path: Path, dirname: str):
    """The finite Mode-A dirs must round-trip unchanged too."""
    d = _copy_dir(_CORPORA / "fragment" / dirname, tmp_path / dirname)
    before = json.loads((d / "test.json").read_text())["expected"]
    regen_dir(d)
    after = json.loads((d / "test.json").read_text())["expected"]
    assert repr(before) == repr(after), f"{dirname}: {before!r} -> {after!r}"
