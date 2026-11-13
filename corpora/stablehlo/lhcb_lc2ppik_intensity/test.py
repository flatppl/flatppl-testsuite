"""Intensity of the Λc+ → p π+ K− (LHCb, arXiv:2208.03262) model at the validation points of amplitude-serialization
7f01d66. The oracle is the JSON checksum of the point whose coordinates match
(references.json).
"""
import runpy
from pathlib import Path

_ORACLE = runpy.run_path(str(Path(__file__).resolve().parent.parent / "_lhcb" / "oracle.py"))
MODEL = "lc2ppik-lhcb-2683025"


def oracle(point):
    for ref in _ORACLE["REFERENCES"][MODEL]["intensity"]:
        if all(ref["coordinates"][name] == value for name, value in point.items()):
            return ref["value"]
    raise KeyError("not a validation point")
