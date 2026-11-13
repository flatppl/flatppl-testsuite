"""Lineshape `ls_L1670_BW` of the Λc+ → p π+ K− (LHCb, arXiv:2208.03262) model, real and imaginary part.

At sigma = 3.2 the oracle is the JSON checksum of `L1670_BW` (amplitude-serialization
7f01d66). At the other sigma it is spec §09
`resonance_breitwigner` with the JSON parameters (mass 1.67, width 0.03, ma 0.493677, mb 0.938272046, l 0, d 1.5),
evaluated from the §09 text in _lhcb/oracle.py.
"""
import runpy
from pathlib import Path

_ORACLE = runpy.run_path(str(Path(__file__).resolve().parent.parent / "_lhcb" / "oracle.py"))
MODEL = "lc2ppik-lhcb-2683025"


def oracle(point):
    return _ORACLE["lineshape"](MODEL, "L1670_BW", {"s": point["q_s"], "imaginary": point["q_imaginary"]})
