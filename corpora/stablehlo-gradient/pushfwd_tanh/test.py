"""Independent analytic oracle: d/dy of `pushfwd(tanh, Normal(0, 1))`'s
logdensity.

The sibling `corpora/stablehlo/pushfwd_tanh` derives the density as
`log phi(atanh y) - log1p(-y**2)`. Differentiating, with
`d atanh/dy = 1/(1 - y**2)`,

    d/dy = -atanh(y)/(1 - y**2) + 2*y/(1 - y**2)

which is what this file computes — analytically, not by finite difference, and
never from the executor. Evaluated in f64 at the f32-rounded query point with the
result rounded to f32, matching the sibling's convention and the f32 executor.

## The boundary points are the regression subject

At `y = ±(1 - 1.2e-7)` the gradient was NaN before flatppl-rust #134 (the old
volume spelling round-tripped through `tanh(atanh y)`, which returns exactly 1.0
there, so the term was `log 0` and its derivative undefined). It is now finite,
about `∓2.65e7`, and NaN never satisfies the runner's comparison, so these two
points fail against a pre-#134 binary. Both signs are pinned because the density
is even in `y` and its derivative odd, so the pair also pins the sign.

`grad_atol` is 8.0, which reads large only out of context: the boundary
derivative is 2.6e7, the analytic and executor values differ there by 2 in
absolute terms, i.e. 8e-8 relative, and the gradient runner's tolerance key is
absolute-only with no `rtol` counterpart. The interior points agree to 5e-7
absolute and are tightly gated in any case by the sibling logdensity directory.
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
    return {"y": _frozen((-math.atanh(y) + 2.0 * y) / (1.0 - y * y))}
