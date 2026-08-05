"""Independent oracle: logdensity of `pushfwd(log, Gamma(2, 1))` at `y`.

§06 "Engine contract for `pushfwd` density evaluation" gives
`log densityof(M, f_inv(y)) - logvolume(f_inv(y))`. With `f = log`,
`f_inv = exp` and `log|log'(x)| = -log x`, which at `x = exp y` is `-y`. §08's
`Gamma(shape, rate)` has log-density
`shape * log rate - lgamma(shape) + (shape - 1) * log x - rate * x`, so at
`shape = 2`, `rate = 1` (where `lgamma(2) = 0`)

    log p(y) = log(exp y) - exp(y) + y = 2 * y - exp(y)

which is what this file computes, in f64 at the f32-rounded query point, with
the result rounded to f32 because the executor is f32.

## This directory pins INTERIOR points only — the boundary is blocked

The saturating point for this family is `y = 100`, and it is deliberately absent.
flatppl-rust #134 fixed the volume term there, but the emitted density is STILL
NaN, because the NaN moved into the base term: `exp(100)` overflows f32 to `inf`,
and `builtin_logdensityof(Gamma, ...)` is NaN at an infinite argument since
`(shape - 1) * log x - rate * x` evaluates `inf - inf`. Measured with a #134
binary through this corpus's own executor: `nan`, gradient `-inf`.

That is a separate open defect in `crates/stablehlo`, recorded under the D1
entry in `flatppl-dev/TODO-flatppl-rust.md` ("`builtin_logdensityof(Gamma, ...)`
is NaN at an infinite argument"), and it wants the same `-inf`-where-the-base-
has-no-mass guard §06 asks for. Pinning `y = 100` here would freeze a NaN, which
`compare_scalar` refuses on principle ("NaN never matches"), so the point waits
for that fix rather than being written down as expected behaviour.

**When that defect is fixed, add `{"y": 100.0}` to `points` and re-run regen.**
The oracle above already returns the right answer for it: `2 * 100 - exp(100)`
is far below f32's range, so the correctly rounded f32 value is `-inf`. The
volume fix itself is gated meanwhile by the sibling
`corpora/stablehlo/pushfwd_log_exponential`, whose base has log-density `-inf`
there instead of NaN.
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


def oracle(point: dict) -> float:
    y = _f32(point["y"])
    return _frozen(2.0 * y - math.exp(y))
