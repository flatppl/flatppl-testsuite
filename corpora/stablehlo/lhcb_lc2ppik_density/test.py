"""Log-density of `decay_rate` of the Λc+ → p π+ K− (LHCb, arXiv:2208.03262) model (spec §06 `weighted`).

decay_rate = weighted(rate_fn, Lebesgue(support = decay_space)) has density
rate = phase_space_density * intensity_event inside decay_space and zero outside.
The model writes phase_space_density = q_31_2 * q_31, the breakup momenta of its two
vertices. At the amplitude-serialization validation points the oracle returns
log(phase_space_density) + log(JSON checksum). The last two points lie outside
decay_space (a mass below threshold, and |cos theta| > 1), where the density is zero.
"""
import runpy
from pathlib import Path

_ORACLE = runpy.run_path(str(Path(__file__).resolve().parent.parent / "_lhcb" / "oracle.py"))
MODEL = "lc2ppik-lhcb-2683025"
import math


def oracle(point):
    if not _ORACLE["three_body_inside"](MODEL, point, prefix="q_"):
        return -math.inf
    for ref in _ORACLE["REFERENCES"][MODEL]["intensity"]:
        if all(ref["coordinates"][name[2:]] == value for name, value in point.items()):
            return math.log(_ORACLE["three_body_phase_space"](MODEL, point, prefix="q_")) + math.log(ref["value"])
    raise KeyError("not a validation point")
