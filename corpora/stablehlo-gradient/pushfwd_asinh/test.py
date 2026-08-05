"""Independent analytic oracle: d/dy of `pushfwd(asinh, Normal(0, 1))`'s
logdensity.

The sibling `corpora/stablehlo/pushfwd_asinh` derives the density as
`log phi(sinh y) + log cosh y`. Differentiating term by term,

    d/dy = -sinh(y) * cosh(y) + tanh(y)

which is what this file computes — analytically, not by finite difference, and
never from the executor. Evaluated in f64 at the f32-rounded query point with the
result rounded to f32, matching the sibling's convention and the f32 executor.

## Interior points only, so this directory does not witness #134

The interior gradients moved by about one f32 ulp across flatppl-rust #134 (at
`y = 0.5`, -0.1254834532737732 to -0.12548348307609558), which is far inside any
usable tolerance, so this directory pins correctness rather than the fix. The
gradient witnesses for #134 are the sibling `pushfwd_tanh` and
`pushfwd_invlogit` directories, whose boundary gradients were NaN before it.

The saturating points `y = 50` and `y = 90` are absent on purpose. Their true
derivatives (-6.7e42 and -3.7e77) are below f32's range, so the executor
correctly returns `-inf` — and the gradient runner compares with
`max|got - want| < grad_atol`, which is `nan` for `-inf` against `-inf` and so
can never pass. The value side of that boundary is gated by the sibling
logdensity directory, whose comparison does handle an infinite expected.
`y = 44` is likewise absent: finite, but at 4e37 it needs a relative tolerance
the gradient runner has no key for.
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
    return {"y": _frozen(-math.sinh(y) * math.cosh(y) + math.tanh(y))}
