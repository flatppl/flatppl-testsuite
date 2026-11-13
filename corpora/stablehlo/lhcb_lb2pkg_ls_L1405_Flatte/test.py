"""Lineshape `ls_L1405_Flatte` of the Λb0 → p K− γ (LHCb, arXiv:2403.03710) model, real and imaginary part.

At sigma = 3.2 the oracle is the JSON checksum of `L1405_Flatte` (amplitude-serialization
7f01d66).
"""
import runpy
from pathlib import Path

_ORACLE = runpy.run_path(str(Path(__file__).resolve().parent.parent / "_lhcb" / "oracle.py"))
MODEL = "lb2pkg-lhcb-2765817"


def oracle(point):
    return _ORACLE["lineshape"](MODEL, "L1405_Flatte", {"s": point["q_s"], "imaginary": point["q_imaginary"]})
