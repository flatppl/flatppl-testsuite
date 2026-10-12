"""Regenerate from captured mgcv design and independent Gaussian arithmetic."""
from pathlib import Path
import runpy

_EVALUATE = runpy.run_path(str(
    Path(__file__).resolve().parent.parent / "oracles" / "splines" / "analytic.py"
))["evaluate"]


def oracle(point):
    return _EVALUATE(point)[0]


def grad_oracle(point):
    return _EVALUATE(point)[1]
