"""Independent closed-form oracle for the HistFactory conversion.

The other seven `corpora/hs3/` dirs carry no `test.py`: their oracle is the
external frozen ROOT/RooFit vector, and `unified/regen.py` refuses
`test_type: "convert"` for exactly that reason. This one is the exception,
because the model is small enough to write out in closed form: a two-bin
Poisson product, three unit-Gaussian nuisance constraints, and §09's
`ContinuedPoisson` staterror term.

Having it here buys two things the frozen ROOT vector cannot give:

* an ABSOLUTE per-point log-density, where the ROOT vector is a 2DeltaNLL
  difference and therefore blind to a constant shift in every hs3 density;
* the `"stablehlo"` block's own frozen `expected`, which `regen` never
  refreezes, so `tests/core/test_engine_override_rows.py` re-derives it from
  this module on every run.

Independence is not taken on trust. `tests/core/test_hs3_absolute_density.py`
feeds this oracle's absolute values through the same 2DeltaNLL difference the
dir's frozen ROOT vector holds and requires them to agree, which pins
everything here except the offset -- and the offset is the closed-form Poisson
and ContinuedPoisson normalisation, which the absolute test then asserts.

§09 specifies `interp_poly6_exp` only as C^2 conditions and does not write the
six coefficients out, so the spec text alone does not pin the boundary FIRST
derivative: its own extrapolation form f(a) = f(+-1) exp((a-+1) f'(+-1)/f(+-1))
reduces C^2 to f''(+-1) = f'(+-1)^2/f(+-1), which leaves f'(+-1) free. §09's
table names pyhf code4 as the reference and that closes it -- the polynomial
matches the exponential interpolation center*(right/center)^a on the right and
its mirror on the left. So `_poly6_exp` SOLVES the 6x6 C^2 system numerically
rather than restating a coefficient the implementation carries.
"""
from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
from scipy.stats import norm, poisson

_DIR = Path(__file__).resolve().parent
_MODEL = _DIR / "histfactory.flatppl"

# The interpolation anchors, i.e. the normsys hi/lo the modifier declares. Not
# read back from the model: only the DATA is, per `_vector` below.
_ANCHOR_LO, _ANCHOR_CTR, _ANCHOR_HI = 0.95, 1.0, 1.05


def _vector(binding: str) -> np.ndarray:
    """One of the model's data vectors, READ from the model rather than restated,
    so a future dataset edit fails the tests instead of inviting an oracle 'fix'."""
    src = _MODEL.read_text()
    m = re.search(rf"^{binding} = \[([^\]]*)\]", src, re.M)
    assert m, f"could not read {binding} from the model"
    return np.array([float(v) for v in m.group(1).split(",") if v.strip()])


OBSERVED = _vector("model_channel1_observed")
SIGNAL = _vector("model_channel1_signal_nominal")
BKG1 = _vector("model_channel1_background1_nominal")
BKG2 = _vector("model_channel1_background2_nominal")
TAU = _vector("mcstat_tau")


def _poly6_exp(left, center, right, alpha):
    """§09's interp_poly6_exp: a 6th-order polynomial on [-1, 1] whose C^2
    conditions match the exponential continuation, solved here as a 6x6 system."""
    lo, hi = math.log(left / center), math.log(right / center)
    if alpha > 1.0:
        return center * (right / center) ** alpha
    if alpha < -1.0:
        return center * (left / center) ** (-alpha)
    A = np.zeros((6, 6))
    for i in range(1, 7):
        A[0, i - 1] = 1.0                          # f(+1) - center
        A[1, i - 1] = (-1.0) ** i                  # f(-1) - center
        A[2, i - 1] = i                            # f'(+1)
        A[3, i - 1] = i * (-1.0) ** (i - 1)        # f'(-1)
        A[4, i - 1] = i * (i - 1)                  # f''(+1)
        A[5, i - 1] = i * (i - 1) * (-1.0) ** i    # f''(-1)
    b = np.array([right - center, left - center,
                  right * hi, -left * lo,
                  right * hi * hi, left * lo * lo])
    a = np.linalg.solve(A, b)
    return center + sum(a[i - 1] * alpha ** i for i in range(1, 7))


def oracle(point: dict) -> float:
    """The model's root `likelihood` log-density at one parameter point, in
    closed form and independent of any FlatPPL engine."""
    mcstat = np.asarray(point["mcstat"], dtype=float)
    f = _poly6_exp
    nu = (SIGNAL * f(_ANCHOR_LO, _ANCHOR_CTR, _ANCHOR_HI, point["syst1"]) * point["mu"]
          + BKG1 * f(_ANCHOR_LO, _ANCHOR_CTR, _ANCHOR_HI, point["syst2"]) * mcstat
          + BKG2 * f(_ANCHOR_LO, _ANCHOR_CTR, _ANCHOR_HI, point["syst3"]) * mcstat)
    lp = float(np.sum(poisson.logpmf(OBSERVED, nu)))
    for name in ("syst1", "syst2", "syst3"):
        lp += float(norm.logpdf(0.0, point[name], 1.0))
    # §09 ContinuedPoisson at x = tau, rate = mcstat * tau: the staterror
    # constraint's aux data is the effective count itself.
    rate = mcstat * TAU
    lp += float(np.sum(TAU * np.log(rate) - rate
                       - np.array([math.lgamma(t + 1.0) for t in TAU])))
    return lp


def logdensity(mu, syst1, syst2, syst3, mcstat) -> float:
    """The `"stablehlo"` block's ABI signature, in its `inputs` order -- the
    entry point `tests/core/test_engine_override_rows.py` calls to re-derive
    that block's frozen `expected`."""
    return oracle({"mu": mu, "syst1": syst1, "syst2": syst2, "syst3": syst3,
                   "mcstat": mcstat})
