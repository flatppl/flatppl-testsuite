"""Independent oracle: logdensity of `pushfwd(asinh, Normal(0, 1))` at `y`.

§06 "Engine contract for `pushfwd` density evaluation" gives the density as
`log densityof(M, f_inv(y)) - logvolume(f_inv(y))`, where `logvolume` is the
FORWARD map's log-volume element taken at the preimage. Here `f = asinh`, so
`f_inv = sinh` and `log|asinh'(x)| = -log cosh y` at `x = sinh y`, leaving

    log p(y) = log phi(sinh y) + log cosh y,    log phi(x) = -x**2/2 - log(2 pi)/2

which is what this file computes. Nothing here reads the executor's output.

## What the boundary points regress against

flatppl-rust #134 respelled the volume in the query point. The old spelling was
`-0.5 * log(1 + sinh(y) ** 2)`, and `sinh(y) ** 2` overflows f32 at `y` around
45: the term became `-inf` and the density `inf - inf`, i.e. NaN. So `y = 50`
and `y = 90` returned NaN before #134 and are the reason this directory exists.
`y = 44` sits just below the overflow, where the old spelling was already
finite, and pins that the respell did not move the finite region — measured
identical before and after, so it is a correctness pin rather than a witness.

## The f32 rounding is part of the oracle, not a fudge

The executor evaluates in f32, so the closed form is evaluated in f64 at the
f32-rounded query point and the RESULT is then rounded to f32 — one rule for
every point, no per-point casework. At `y = 50` and `y = 90` the f64 truth
(-3.4e42 and -1.9e77) is below f32's range, so the correctly rounded f32 value
IS `-inf`; that is the frozen value on its own terms, not a placeholder.
`compare_scalar` matches an infinite expected exactly and never matches NaN, so
those two points fail against a pre-#134 binary.

Freezing `-inf` is possible HERE and not in the `log` family, which is worth
knowing before copying either. `tests/sweep/test_oracle.py` cross-checks every
curated case the compositional oracle can express against an f64 value within
1e-9, and at an f32-overflow point the f64 truth is finite while the executor is
`-inf` — irreconcilable. `asinh` is not among that oracle's `_FORWARD_NAMES`, so
this directory is unexpressible to it and the conflict does not arise; `log` is,
which is why `pushfwd_log_exponential` records its overflow point instead of
pinning it.

`value_rtol_f32` carries `y = 44`, whose magnitude is 2e37: an absolute
tolerance cannot express agreement there, and the executor differs from the
closed form by 3e-6 relative.
"""
import math

import numpy as np


def _f32(v: float) -> float:
    """The query point as the f32 executor actually receives it."""
    return float(np.float32(v))


def _frozen(v: float) -> float:
    """The f64 closed form, or the infinity it overflows to in the executor's f32.

    Sibling directories freeze the f64 oracle and let the `value_atol_f32` /
    `grad_atol` band absorb the executor's f32 error, and this keeps that
    convention -- which `tests/sweep/test_oracle.py` depends on, since it
    cross-checks every curated case the compositional oracle can express against
    a 1e-9 band. Past f32's range there is no f64 value the executor could
    return and the correctly rounded f32 result is an infinity, so that is what
    gets frozen; `compare_scalar` matches an infinite expected exactly.
    """
    with np.errstate(over="ignore"):
        rounded = float(np.float32(v))
    return rounded if math.isinf(rounded) else v


def _log_phi(x: float) -> float:
    return -0.5 * x * x - 0.5 * math.log(2.0 * math.pi)


def oracle(point: dict) -> float:
    y = _f32(point["y"])
    return _frozen(_log_phi(math.sinh(y)) + math.log(math.cosh(y)))
