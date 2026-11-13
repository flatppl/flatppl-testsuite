"""Lineshape `ls_LNR30_NR` of the Λb0 → p K− γ (LHCb, arXiv:2403.03710) model, real and imaginary part.

At sigma = 3.2 the oracle is the JSON checksum of `LNR30_NR` (amplitude-serialization
7f01d66).
"""
import runpy
from pathlib import Path

_ORACLE = runpy.run_path(str(Path(__file__).resolve().parent.parent / "_lhcb" / "oracle.py"))
MODEL = "lb2pkg-lhcb-2765817"


def oracle(point):
    return _ORACLE["lineshape"](MODEL, "LNR30_NR", {"s": point["q_s"], "imaginary": point["q_imaginary"]})
