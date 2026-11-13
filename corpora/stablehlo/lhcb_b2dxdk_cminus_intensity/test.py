"""Intensity of the B+ → D0 π+ D− K+, charge-conjugate model C− model at the validation points of B2DxDK.jl.

The points of commit 85af4cc, which the FlatPPL model was translated from, come first,
then the two points that commit 9d78e67 replaced them with. The oracle is the JSON
checksum of the point whose four-momenta match (references.json).
"""
import runpy
from pathlib import Path

_ORACLE = runpy.run_path(str(Path(__file__).resolve().parent.parent / "_lhcb" / "oracle.py"))
MODEL = "b2dxdk-lhcb-2794793-Cminus"


def oracle(point):
    refs = _ORACLE["REFERENCES"][MODEL]
    for ref in refs["translated_from"]["intensity"] + refs["later_points"]["intensity"]:
        if [list(p) for p in ref["momenta"]] == [list(point[f"p{i}"]) for i in range(1, 5)]:
            return ref["value"]
    raise KeyError("not a validation point")
