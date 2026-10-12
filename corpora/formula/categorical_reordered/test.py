"""Regenerate the categorical target independently from its brms matrices."""
import json
from pathlib import Path
import runpy

_DIR = Path(__file__).resolve().parent
_REFERENCE = json.loads((_DIR / "oracle.json").read_text())
_EVALUATE = runpy.run_path(str(_DIR.parent / "oracles" / "categorical" / "analytic.py"))["evaluate"]


def oracle(point):
    return _EVALUATE(_REFERENCE, point)[0]


def grad_oracle(point):
    return _EVALUATE(_REFERENCE, point)[1]
