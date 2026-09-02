"""Independent scipy oracle: §09 ContinuedPoisson log-density at rate = 4.5.

§09's density is `lambda**x * exp(-lambda) / Gamma(x + 1)` for `x >= 0`, so the
log form is `x*log(lambda) - lambda - gammaln(x + 1)`. Below zero the variate is
off the stated `nonnegreals` support and §08's shared rule makes the density
zero, i.e. a `-inf` log-density.

The mask is not cosmetic. `Gamma` has poles at the non-positive integers, so the
unmasked formula returns a finite but WRONG value on `-1 < x < 0` (Gamma is
finite there, so `x = -0.5` would score about -3.1 instead of `-inf`) and a
`+inf` at `x = -1`. `x = -0.5` and `x = -1.5` are in the grid to catch exactly
that, since a builder that dropped the mask still passes every non-negative
point.

`x = 3.7` and `x = 12.5` are the points §09 adds this measure for: a non-integer
variate, which `Poisson`'s counting-measure mass cannot score at all.
"""
import math

from scipy.special import gammaln

_RATE = 4.5


def oracle(point: dict) -> float:
    x = point["x"]
    if x < 0.0:
        return -math.inf
    return x * math.log(_RATE) - _RATE - float(gammaln(x + 1.0))


def logdensity(x) -> float:
    return oracle({"x": x})
