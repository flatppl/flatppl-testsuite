"""Lineshape `ls_L1405_Flatte` of the Λc+ → p π+ K− (LHCb, arXiv:2208.03262) model, real and imaginary part.

At sigma = 3.2 the oracle is the JSON checksum of `L1405_Flatte` (amplitude-serialization
7f01d66).
"""
import runpy
from pathlib import Path

_ORACLE = runpy.run_path(str(Path(__file__).resolve().parent.parent / "_lhcb" / "oracle.py"))
MODEL = "lc2ppik-lhcb-2683025"


def oracle(point):
    return _ORACLE["lineshape"](MODEL, "L1405_Flatte", {"s": point["q_s"], "imaginary": point["q_imaginary"]})
