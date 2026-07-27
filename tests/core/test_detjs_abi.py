"""det-js scores an inputs/outputs ABI query module, not just a named binding.

The ABI (flatppl-design §12, "Compilation ABI: `inputs` and `outputs`") is how a
query designates its arguments and results. The StableHLO path consumes it
natively: one compiled module, the point passed as runtime args. det-js
historically did not — `DetJsScoreEngine.log_density` appended its own
`__score__ = logdensityof(binding, <record literal>)` and re-determinized per
point, so the two engines scored *different source* for the same test.

`detjs_exec.score_abi_points` closes that: determinize the concatenated
model+query ONCE, then per point bind each binding named in the module's own
`inputs` to that point's value and evaluate `outputs`.

The oracle here is the frozen `expected` in each test dir's `test.json`, which
was produced offline by that dir's independent scipy `test.py` — never by an
engine. Matching it means det-js-via-ABI agrees with the independent oracle,
and (since the StableHLO path checks the same frozen values) with the other
engine.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from flatppl_testsuite.unified import detjs_exec as ex

_CORPORA = Path(__file__).resolve().parents[2] / "corpora"

# det-js is f64 and the oracle is closed-form, so agreement is near-exact --
# far tighter than the f32 band the StableHLO path needs.
_ATOL = 1e-9
_RTOL = 1e-9


def _abi_example_dirs() -> list[Path]:
    """Example dirs carrying an ABI query (the corpus this path is for)."""
    return sorted(
        d for d in (_CORPORA / "examples").iterdir()
        if (d / "query.flatppl").exists() and (d / "test.json").exists()
    )


_DIRS = _abi_example_dirs()
_IDS = [d.name for d in _DIRS]


@pytest.mark.skipif(not ex.engine_available(), reason="det-js path unavailable")
@pytest.mark.parametrize("dir", _DIRS, ids=_IDS)
def test_abi_scoring_matches_the_frozen_oracle(dir: Path):
    body = json.loads((dir / "test.json").read_text())
    got = ex.score_abi_points(
        model=dir / body["model"],
        query=dir / "query.flatppl",
        fields=body["inputs"],
        points=body["points"],
    )
    want = [float(v) for v in body["expected"]]

    assert len(got) == len(want), (
        f"{len(body['points'])} points and {len(want)} expected values "
        f"-> {len(got)} scores"
    )
    for i, (g, w) in enumerate(zip(got, want)):
        assert abs(g - w) <= _ATOL + _RTOL * abs(w), (
            f"{dir.name} point {i} {body['points'][i]}: det-js-via-ABI gave {g!r}, "
            f"frozen independent oracle is {w!r}"
        )


def test_the_guard_sees_the_corpus():
    assert _DIRS, "no ABI example dirs found -- this test is vacuous"
