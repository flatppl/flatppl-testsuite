"""Independent analytic oracle: d/dy of `pushfwd(invlogit, Normal(0, 1))`'s
logdensity.

The sibling `corpora/stablehlo/pushfwd_invlogit` derives the density as
`log phi(logit y) - log y - log1p(-y)`. Differentiating, with
`d logit/dy = 1/(y(1 - y))`,

    d/dy = -logit(y)/(y*(1 - y)) - 1/y + 1/(1 - y)

which is what this file computes — analytically, not by finite difference, and
never from the executor. Evaluated in f64 at the f32-rounded query point with the
result rounded to f32, matching the sibling's convention and the f32 executor.

## The boundary point is the regression subject

At `y = 1 - 6e-8` the gradient was NaN before flatppl-rust #134 (the old volume
spelling round-tripped through `sigma(logit y)`, which returns exactly 1.0 there,
so the term was `log 0` and its derivative undefined). It is now finite, about
`-2.62e8`, and NaN never satisfies the runner's comparison, so the point fails
against a pre-#134 binary. `y = 1 - 1.19e-7`, the next f32 down, was finite both
before and after and is pinned so the boundary cannot be "fixed" by moving it.

`grad_atol` is 64.0, which reads large only out of context: the boundary
derivative is 2.6e8, the analytic and executor values differ there by 16 in
absolute terms, i.e. 6e-8 relative — about f32 epsilon — and the gradient
runner's tolerance key is absolute-only with no `rtol` counterpart. The interior
points agree to 3e-7 absolute and are tightly gated in any case by the sibling
logdensity directory.
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
    logit = math.log(y / (1.0 - y))
    return {"y": _frozen(-logit / (y * (1.0 - y)) - 1.0 / y + 1.0 / (1.0 - y))}
