"""Independent oracle for cov_stdmod_interp_poly6 (scipy + a 6x6 linear solve).

A function member has no base-op spelling in the surface language, so it reaches
a density only through the determiniser's §09 function lowering. This row is the
one place that lowering is scored numerically, on both the determinised and the
StableHLO path, and it carries one member from each of the three modules the
lowering covers:

* `particle-physics` -- `interp_poly6_exp` and `interp_poly6_lin`, the two
  members the HS3 converter emits by default (`crates/hs3/src/histfactory.rs`,
  normsys and histosys respectively). Oracle: a 6x6 solve of §09's own C^2
  conditions, cross-checked against ROOT (see below).
* `polynomials` -- `legendre` at a literal degree. Oracle:
  `scipy.special.eval_legendre`, which shares no code with the unrolled
  three-term recursion the pass emits.
* `distances` -- `euclidean`. Oracle: the norm written out directly. `alpha`
  rides inside the vector, so the member is not constant-folded away.

`corpora/hs3/conversions/histfactory` uses the same two interpolators, but every
point in its frozen ROOT vector is inside `[-1, 1]`, so that vector never touches
either extrapolation branch. The two points past `|alpha| = 1` here do.

§09 specifies both interpolators only as C^2 conditions and does not write the
six coefficients out, so the spec text alone does not pin the boundary FIRST
derivative: for `interp_poly6_exp` its own extrapolation form
`f(a) = f(+-1) exp((a -+ 1) f'(+-1) / f(+-1))` reduces C^2 to
`f''(+-1) = f'(+-1)^2 / f(+-1)`, which leaves `f'(+-1)` free. §09's table names
pyhf code4/code4p as the reference and that closes it: the polynomial matches the
exponential interpolation `center * (right/center)^a` on the right (and its
mirror on the left) for `interp_poly6_exp`, and the piecewise-LINEAR
interpolation continued outward for `interp_poly6_lin`.

So this oracle SOLVES §09's 6x6 C^2 system numerically under that reading rather
than restating any coefficient the implementation carries. Both interpolators
were then confirmed against ROOT 6.40.02 over exactly this alpha grid, outside
both FlatPPL engines -- `FlexibleInterpVar` with interpCode 4 for
`interp_poly6_exp` and `PiecewiseInterpolation` with interpCode 4 for
`interp_poly6_lin` -- agreeing to 4.4e-16 at every point, extrapolation
included.
"""
import math

import numpy as np
from scipy.special import eval_legendre
from scipy.stats import norm

# The model's anchors and data, restated here on purpose: this file is the
# INDEPENDENT oracle, so it must not read them back out of the model it checks.
EXP_LO, EXP_CTR, EXP_HI = 0.95, 1.0, 1.05
LIN_LO, LIN_CTR, LIN_HI = 0.8, 1.0, 1.3
NOMINAL = 20.0
OBSERVED = 21.5
LEGENDRE_DEGREE = 3
# `probe_point` carries alpha in slot 0; only the constant slots appear here.
PROBE_CONST = (2.0, 2.0)
REFERENCE_POINT = (0.0, 0.0, 1.0)


def _poly6_coeffs(left, center, right, d1_lo, d1_hi, d2_lo, d2_hi):
    """Solve §09's C^2 system for a_1..a_6, with f(0) = center fixing a_0.

    The six rows are §09's own conditions on
    `f(a) = center + sum_{i=1..6} a_i a^i`: value, first derivative and second
    derivative, each at a = +1 and a = -1.
    """
    A = np.zeros((6, 6))
    for i in range(1, 7):
        A[0, i - 1] = 1.0                          # f(+1) - center
        A[1, i - 1] = (-1.0) ** i                  # f(-1) - center
        A[2, i - 1] = i                            # f'(+1)
        A[3, i - 1] = i * (-1.0) ** (i - 1)        # f'(-1)
        A[4, i - 1] = i * (i - 1)                  # f''(+1)
        A[5, i - 1] = i * (i - 1) * (-1.0) ** i    # f''(-1)
    b = np.array([right - center, left - center, d1_hi, d1_lo, d2_hi, d2_lo])
    return np.linalg.solve(A, b)


def _poly6(left, center, right, alpha, d1_lo, d1_hi, d2_lo, d2_hi):
    a = _poly6_coeffs(left, center, right, d1_lo, d1_hi, d2_lo, d2_hi)
    return center + sum(a[i - 1] * alpha ** i for i in range(1, 7))


def interp_poly6_exp(left, center, right, alpha):
    """Degree-6 polynomial on [-1, 1], exponential continuation outside.

    pyhf CODE4 convention, which is what closes §09's underdetermined C^2 system:
    the boundary first derivative is the LOG-SLOPE one, i.e. the polynomial
    matches `center * (right/center)**alpha` at `alpha = +1` (and the mirrored
    `center * (left/center)**(-alpha)` at `alpha = -1`) in value, first and
    second derivative. Confirmed against ROOT's `FlexibleInterpVar` at
    interpCode 4.
    """
    lo, hi = math.log(left / center), math.log(right / center)
    if alpha > 1.0:
        return center * (right / center) ** alpha
    if alpha < -1.0:
        return center * (left / center) ** (-alpha)
    return _poly6(left, center, right, alpha,
                  d1_lo=-left * lo, d1_hi=right * hi,
                  d2_lo=left * lo * lo, d2_hi=right * hi * hi)


def interp_poly6_lin(left, center, right, alpha):
    """Degree-6 polynomial on [-1, 1], linear continuation outside.

    pyhf CODE4P convention: the boundary second derivative is zero and the
    boundary first derivative is the piecewise-LINEAR interpolation's own slope,
    `right - center` on the right and `center - left` on the left. Confirmed
    against ROOT's `PiecewiseInterpolation` at interpCode 4.
    """
    if alpha > 1.0:
        return right + (alpha - 1.0) * (right - center)
    if alpha < -1.0:
        return left + (alpha + 1.0) * (center - left)
    return _poly6(left, center, right, alpha,
                  d1_lo=center - left, d1_hi=right - center,
                  d2_lo=0.0, d2_hi=0.0)


def euclidean(u, v):
    """§09's `distances.euclidean`: the l2 norm of the difference."""
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(u, v)))


def oracle(point: dict) -> float:
    alpha = float(point["alpha"])
    expected = (NOMINAL * interp_poly6_exp(EXP_LO, EXP_CTR, EXP_HI, alpha)
                + interp_poly6_lin(LIN_LO, LIN_CTR, LIN_HI, alpha)
                + float(eval_legendre(LEGENDRE_DEGREE, alpha)))
    spread = euclidean((alpha,) + PROBE_CONST, REFERENCE_POINT)
    return float(
        norm.logpdf(OBSERVED, expected, spread)
        + norm.logpdf(0.0, alpha, 1.0)
    )
