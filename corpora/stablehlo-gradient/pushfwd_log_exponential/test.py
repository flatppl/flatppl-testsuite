"""Independent analytic oracle: d/dy of `pushfwd(log, Exponential(1))`'s
logdensity.

The sibling `corpora/stablehlo/pushfwd_log_exponential` derives the density as
`-exp(y) + y`. Differentiating,

    d/dy = 1 - exp(y)

which is what this file computes — analytically, not by finite difference, and
never from the executor. Evaluated in f64 at the f32-rounded query point with the
result rounded to f32, matching the sibling's convention and the f32 executor.

## Interior points only, so this directory does not witness #134

The interior gradients are byte-identical across flatppl-rust #134, so this
directory pins correctness rather than the fix — as does its value-side sibling,
for the reason recorded there (over an `Exponential` base the saturating point
was already `-inf` before #134).

The saturating point `y = 100` is absent on purpose: its true derivative
`1 - e**100` is below f32's range, so the executor correctly returns `-inf`, and
the gradient runner's `max|got - want| < grad_atol` is `nan` for `-inf` against
`-inf` and can never pass. That boundary is gated on the value side by the
sibling logdensity directory, whose comparison handles an infinite expected.
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


def grad_oracle(point: dict) -> dict:
    y = _f32(point["y"])
    return {"y": _frozen(1.0 - math.exp(y))}
