"""Per-engine override blocks in `test.json`.

A dir describes ONE model, but the det-js and StableHLO paths can need
different scoring shapes for the same query: det-js scores a Mode A model's
own `lp = logdensityof(m, <point>)` binding, while the StableHLO emitter
refuses a module with no `inputs`/`outputs` ABI and needs that point as a
compiled-function argument. A `"stablehlo": {...}` block carries the ABI
shape, `loader.merged_body` applies it, and the det-js case keeps the body it
always had -- so adding a StableHLO row cannot move a det-js verdict.

The frozen-value check guards the one block that had to carry its own
`expected`: `unified/regen.py` refreezes only the top-level key, so a block
value that regen never writes is exactly where a corpus drifts silently.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from flatppl_testsuite.unified.loader import load_test_module, merged_body

_CORPORA = Path(__file__).resolve().parents[2] / "corpora"


def test_merge_replaces_only_the_declared_keys():
    body = {"test_type": "logdensity", "binding": "lp", "expected": -1.0,
            "tolerance": {"atol": 1e-9}, "stablehlo": {"tolerance": {"value_atol_f32": 1e-5}}}
    merged = merged_body(body, "stablehlo")
    assert merged["tolerance"] == {"value_atol_f32": 1e-5}, "tolerance must be replaced whole"
    assert merged["expected"] == -1.0, "the ONE frozen oracle value must survive the merge"
    assert merged["binding"] == "lp"


def test_merge_is_a_noop_for_an_engine_with_no_block():
    body = {"expected": -1.0, "stablehlo": {"points": []}}
    assert merged_body(body, "det-js") == body


def test_allow_skip_in_a_block_does_not_leak_to_the_other_engine():
    """A StableHLO refusal pin must not also let the det-js case skip."""
    body = {"stablehlo": {"allow_skip": True}}
    assert merged_body(body, "stablehlo").get("allow_skip") is True
    assert merged_body(body, "det-js").get("allow_skip") is None


def _dirs_with_own_expected():
    for tj in sorted(_CORPORA.rglob("test.json")):
        body = json.loads(tj.read_text())
        for engine in body["engines"]:
            block = body.get(engine)
            if isinstance(block, dict) and "expected" in block:
                yield pytest.param(tj.parent, engine, block,
                                   id=f"{tj.parent.relative_to(_CORPORA)}::{engine}")


@pytest.mark.parametrize("dir,engine,block", list(_dirs_with_own_expected()))
def test_a_block_expected_still_matches_the_dirs_own_oracle(dir, engine, block):
    mod = load_test_module(dir)
    got = [float(mod.logdensity(*[pt[name] for name in block["inputs"]]))
           for pt in block["points"]]
    # 1e-12, not float identity: an oracle that goes through a linear solve or
    # lgamma differs by ~1e-14 relative between libm/BLAS builds (CI vs a
    # laptop), while a real drift of the frozen values is >= 1e-6.
    assert got == pytest.approx(block["expected"], abs=0, rel=1e-12), (
        f"{dir.name}: the {engine!r} block's frozen `expected` no longer matches "
        "test.py's oracle -- regen does not refreeze a block, so update it by hand"
    )
