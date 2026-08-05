"""Independent oracle: logdensity of `pushfwd(invlogit, Normal(0, 1))` at `y`.

§06 "Engine contract for `pushfwd` density evaluation" gives
`log densityof(M, f_inv(y)) - logvolume(f_inv(y))`. With `f = invlogit` (the
logistic sigma), `f_inv = logit` and `sigma'(x) = sigma(x)(1 - sigma(x))`, which
at `x = logit y` is `y(1 - y)`. So

    log p(y) = log phi(logit y) - log y - log1p(-y)

which is what this file computes, in f64 at the f32-rounded query point, with
the result rounded to f32 because the executor is f32.

## What the boundary points regress against

flatppl-rust #134 respelled the volume in the query point. The old spelling was
`log sigma(logit y) + log(1 - sigma(logit y))`, a round trip through the forward
map: at `y = 1 - 6e-8` the inner `sigma(logit y)` returns exactly 1.0 in f32, so
`log(1 - 1) = -inf` and the density was `+inf`. That point is the regression
subject.

`y = 1 - 1.19e-7` is the next f32 down, which was FINITE before #134 as well as
after — D1 recorded it as the far side of a one-ulp-wide boundary. It is pinned
so a future change cannot fix the boundary by moving it.
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
    return _frozen(_log_phi(math.log(y / (1.0 - y))) - math.log(y) - math.log1p(-y))
