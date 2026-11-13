"""Lineshape `ls_L2000_BW` of the Λc+ → p π+ K− (LHCb, arXiv:2208.03262) model, real and imaginary part.

At sigma = 3.2 the oracle is the JSON checksum of `L2000_BW` (amplitude-serialization
7f01d66). At the other sigma it is spec §09
`resonance_breitwigner` with the JSON parameters (mass 1.98819, width 0.17926, ma 0.493677, mb 0.938272046, l 0, d 1.5),
evaluated from the §09 text in _lhcb/oracle.py.
"""
import runpy
from pathlib import Path

_ORACLE = runpy.run_path(str(Path(__file__).resolve().parent.parent / "_lhcb" / "oracle.py"))
MODEL = "lc2ppik-lhcb-2683025"


def oracle(point):
    return _ORACLE["lineshape"](MODEL, "L2000_BW", {"s": point["q_s"], "imaginary": point["q_imaginary"]})
