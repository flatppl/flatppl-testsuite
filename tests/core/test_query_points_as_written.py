"""A logdensity case freezes the density at its query point as written.

Every engine reads the point from `test.json` as written, at its own
precision. A point that an f32 executor would round is therefore written as
its exact f32 value (`0.9999998807907104`, not `0.99999988`), so an f32 and an
f64 engine score the same point and one frozen value serves both. An oracle
that rounded the point itself would freeze a different point than the one an
f64 engine reads.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest

_CORPORA = Path(__file__).resolve().parents[2] / "corpora"

# The dirs whose points needed it; each value must already be f32-exact.
_F32_EXACT = [
    "stablehlo/pushfwd_asinh",
    "stablehlo/pushfwd_invlogit",
    "stablehlo/pushfwd_log_exponential",
    "stablehlo/pushfwd_log_gamma",
    "stablehlo/pushfwd_tanh",
]

_ROUNDS_A_POINT = re.compile(r"(?:_f32|float32)\(\s*point\b")


def _logdensity_dirs():
    for tj in sorted(_CORPORA.rglob("test.json")):
        if json.loads(tj.read_text()).get("test_type") == "logdensity":
            yield tj.parent


def _reals(value):
    if isinstance(value, list):
        for v in value:
            yield from _reals(v)
    elif isinstance(value, float):
        yield value


@pytest.mark.parametrize("name", _F32_EXACT)
def test_each_real_query_value_is_f32_exact(name):
    body = json.loads((_CORPORA / name / "test.json").read_text())
    for point in body["points"]:
        for v in _reals(list(point.values())):
            assert float(np.float32(v)) == v, f"{name}: {v!r} is not an f32 value"


def test_no_logdensity_oracle_rounds_its_query_point():
    offenders = [str(d.relative_to(_CORPORA)) for d in _logdensity_dirs()
                 if (d / "test.py").exists()
                 and _ROUNDS_A_POINT.search((d / "test.py").read_text())]
    assert not offenders, f"oracles that round the query point: {offenders}"
