"""Independent oracle for ex_capture_recapture.

``rcp ~ Uniform(interval(0, rcp_max))`` with ``rcp_max = M_obs / (C_obs -
R_obs + M_obs) = 0.5``; ``R ~ Binomial(C_obs, rcp)`` against the fixed
``R_obs = 5`` out of ``C_obs = 15`` trials.
"""
from scipy.stats import binom, uniform

_M_OBS, _C_OBS, _R_OBS = 10, 15, 5
_RCP_MAX = _M_OBS / (_C_OBS - _R_OBS + _M_OBS)


def oracle(point: dict) -> float:
    rcp = point["rcp"]
    lp = uniform.logpdf(rcp, loc=0.0, scale=_RCP_MAX) + binom.logpmf(_R_OBS, _C_OBS, rcp)
    return float(lp)
