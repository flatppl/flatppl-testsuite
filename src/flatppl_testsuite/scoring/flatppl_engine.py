"""Score a FlatPPL model via the selected engine and form the 2DeltaNLL vector.

The actual scoring is delegated to a pluggable engine (see ``engine.py``); this
module is engine-agnostic. ``twice_delta_nll`` returns
``-2 * (logL(point) - logL(reference))`` — the quantity the suite's
``twice_delta_nll_scan`` check compares against the frozen expected vector. The
parameter-independent additive constant in FlatPPL's log-density cancels in the
difference, so the comparison is offset-invariant.
"""

from __future__ import annotations

from pathlib import Path

from .engine import get_engine


def log_density(model: Path, binding: str, theta: dict[str, object]) -> float:
    """Return logdensityof(binding, theta) via the active FlatPPL engine."""
    return get_engine().log_density(model, binding, theta)


def twice_delta_nll(model: Path, binding: str, scan_param: str,
                    scan_points: list[float], reference: dict[str, object]) -> list[float]:
    """Return the 2DeltaNLL vector over scan_points relative to the reference point."""
    ref = log_density(model, binding, reference)
    out = []
    for p in scan_points:
        theta = dict(reference)
        theta[scan_param] = p
        out.append(-2.0 * (log_density(model, binding, theta) - ref))
    return out


def twice_delta_nll_points(model: Path, binding: str,
                           reference: dict[str, object],
                           points: list[dict[str, object]]) -> list[float]:
    """Return the 2DeltaNLL vector over arbitrary multi-parameter theta points.

    Like ``twice_delta_nll`` but each point is a full theta record rather than a
    single scanned parameter, so multi-parameter clouds (e.g. HistFactory, where
    ``mu`` and several systematics move together) are scored directly. The
    parameter-independent additive constant cancels in the difference, so the
    comparison is offset-invariant — this is what makes HistFactory's ROOT
    ``Sum log(n_k!)`` convention offset drop out.
    """
    ref = log_density(model, binding, reference)
    return [-2.0 * (log_density(model, binding, p) - ref) for p in points]
