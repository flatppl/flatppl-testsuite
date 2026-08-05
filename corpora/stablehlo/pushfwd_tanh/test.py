"""Independent oracle: logdensity of `pushfwd(tanh, Normal(0, 1))` at `y`.

§06 "Engine contract for `pushfwd` density evaluation" gives
`log densityof(M, f_inv(y)) - logvolume(f_inv(y))`. With `f = tanh`,
`f_inv = atanh` and `log|tanh'(x)| = log(1 - tanh(x)**2) = log1p(-y**2)` at
`x = atanh y`, so

    log p(y) = log phi(atanh y) - log1p(-y**2)

which is what this file computes, in f64 at the f32-rounded query point, with
the result rounded to f32 because the executor is f32.

## What the boundary points regress against

flatppl-rust #134 respelled the volume in the query point. The old spelling ran
`log(1 - tanh(atanh y)**2)`, a round trip through the forward map: at
`y = 1 - 1.2e-7` (the nearest f32 below 1 where this bites) `tanh(atanh y)`
returns exactly 1.0, so the term was `log 0 = -inf` and the density `+inf`. Both
signs are pinned because the density is even in `y` while the intermediate
`atanh` is odd, so a sign error in the respell shows on one side only.

`y = 0.9` is the interior point D1 reported as 9 f32 ulp from the f64 closed
form — the accumulated f32 error of the whole arm, not of the respell (it was
21 ulp before). It sits inside `value_atol_f32` with room to spare and is kept
as the interior witness.
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
    return _frozen(_log_phi(math.atanh(y)) - math.log1p(-y * y))
