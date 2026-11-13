"""Lineshape `ls_L1800_BW` of the Λb0 → p K− γ (LHCb, arXiv:2403.03710) model, real and imaginary part.

At sigma = 3.2 the oracle is the JSON checksum of `L1800_BW` (amplitude-serialization
7f01d66). At the other sigma it is spec §09
`resonance_breitwigner` with the JSON parameters (mass 1.8, width 0.2, ma 0.938, mb 0.493, l 0, d 1.5),
evaluated from the §09 text in _lhcb/oracle.py.
"""
import runpy
from pathlib import Path

_ORACLE = runpy.run_path(str(Path(__file__).resolve().parent.parent / "_lhcb" / "oracle.py"))
MODEL = "lb2pkg-lhcb-2765817"


def oracle(point):
    return _ORACLE["lineshape"](MODEL, "L1800_BW", {"s": point["q_s"], "imaginary": point["q_imaginary"]})
