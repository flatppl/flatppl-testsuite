"""Score FlatPPL through the convert-free det-js path.

The unified counterpart of `stablehlo_exec`: the ONLY place the unified
runners touch the det-js engine. Everything here delegates to
`scoring/engine.py`, which owns the `flatppl determinize` -> `score_flatpdl.cjs`
subprocess pair; this module exists so runners import one stable surface
rather than reaching into the legacy scoring package directly.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from flatppl_testsuite.config import CONFIG
from flatppl_testsuite.scoring.engine import (  # noqa: F401  (re-exported)
    DeterminizeRefused,
    DetJsScoreEngine,
    score_binding,
)

_ENGINE = DetJsScoreEngine()


def log_density_at(model: Path, binding: str, theta: dict) -> float:
    """`logdensityof(binding, theta)` via the theta-splice path (Mode B)."""
    return _ENGINE.log_density(model, binding, theta)


def parse_expected(v):
    """Frozen expected value -> float. `±inf`/`nan` cannot round-trip through
    JSON, so they are stored as the STRINGS "inf"/"-inf"/"nan" (e.g.
    fragment's trunc_out, whose density outside the support is exactly 0).
    `float()` already parses those strings natively, so this is just a
    documented single entry point for runners to call."""
    return float(v)


@lru_cache(maxsize=1)
def engine_available() -> bool:
    """True if both subprocess halves of the det-js path are resolvable."""
    return Path(CONFIG.flatppl_bin).exists() and Path(CONFIG.flatpdl_scorer).exists()
