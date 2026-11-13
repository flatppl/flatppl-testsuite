"""Log-density of `decay_rate` of the B+ → D0 π+ D− K+, charge-conjugate model C− model (spec §06 `weighted`).

decay_rate = weighted(rate_fn, Lebesgue(support = decay_space)) has density
rate = phase_space_density * intensity_event inside decay_space and zero outside.
The model writes phase_space_density = q_12_3_4 * q_12_3 * q_12, the breakup momenta
of its three vertices. The points are the decay coordinates of the B2DxDK.jl
validation four-momenta, found with the model's own frame functions; the oracle
maps a point back to four-momenta, takes the JSON checksum of the matching
validation point, and returns log(phase_space_density) + log(checksum).
The last two points lie outside decay_space (m_12_3 below m_12 + m3, and
|cos theta_12| > 1), where the density is zero.
"""
import runpy
from pathlib import Path

_ORACLE = runpy.run_path(str(Path(__file__).resolve().parent.parent / "_lhcb" / "oracle.py"))
MODEL = "b2dxdk-lhcb-2794793-Cminus"
import math

import numpy as np


def oracle(point):
    x = dict(point)
    m0, m1, m2, m3, m4 = _ORACLE["MASSES"][MODEL]
    inside = (_ORACLE["B_M12"] + m3 <= x["q_m_12_3"] <= m0 - m4
              and all(-1 <= x[k] <= 1 for k in x if "cos_theta" in k)
              and all(-math.pi <= x[k] <= math.pi for k in x if "phi" in k))
    if not inside:
        return -math.inf
    momenta = _ORACLE["momenta"](MODEL, x)
    refs = _ORACLE["REFERENCES"][MODEL]
    for ref in refs["translated_from"]["intensity"] + refs["later_points"]["intensity"]:
        if max(np.max(np.abs(np.asarray(a) - b)) for a, b in zip(ref["momenta"], momenta)) < 1e-9:
            return math.log(_ORACLE["four_body_phase_space"](MODEL, x)) + math.log(ref["value"])
    raise KeyError("not a validation point")
