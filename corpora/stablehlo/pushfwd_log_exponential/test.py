"""Independent oracle: logdensity of `pushfwd(log, Exponential(1))` at `y`.

§06 "Engine contract for `pushfwd` density evaluation" gives
`log densityof(M, f_inv(y)) - logvolume(f_inv(y))`. With `f = log`,
`f_inv = exp` and `log|log'(x)| = -log x`, which at `x = exp y` is `-y`. §08's
`Exponential(rate)` has log-density `log rate - rate * x`, so at `rate = 1`

    log p(y) = -exp(y) + y

which is what this file computes, in f64 at the f32-rounded query point, with
the result rounded to f32 because the executor is f32.

## Interior points only, and this directory does not witness #134

`y = 100`, the saturating point, is deliberately absent, for two independent
reasons.

**It is not a #134 witness.** Over an `Exponential` base that point is `-inf`
BEFORE flatppl-rust #134 as well as after — measured through this corpus's own
executor with binaries built at `482d26f`, `2e4f11e` and `5255d9a`, `-inf` at
all three. (The D1 report lists `log` at `y = 100` among five extra rows
"measured NaN at 482d26f, -inf after (over an Exponential base)". That
comparison crossed two bases — NaN over `Gamma`, `-inf` over `Exponential` — so
it establishes a before/after move for neither. Over `Exponential` there was
nothing to fix here.) The witnesses for #134 are the sibling `pushfwd_asinh`,
`pushfwd_tanh` and `pushfwd_invlogit` directories, which do fail against a
pre-#134 binary.

**And it is not expressible under the frozen-value convention.** The
compositional oracle in `flatppl_testsuite.sweep` CAN express this shape
(`pushfwd(log, ...)` over a recognised base), so `tests/sweep/test_oracle.py`
cross-checks this directory's frozen values against it within 1e-9. That oracle
works in f64, where the density at `y = 100` is a finite `-2.7e43`, while the f32
executor returns `-inf`. Freezing either one makes that gate disagree with the
other, and the schema has no per-point precision annotation to reconcile them. So
the point is recorded here rather than pinned. The `asinh` sibling keeps its
overflow points because `asinh` is not in the oracle's `_FORWARD_NAMES`, leaving
that directory unexpressible and the conflict absent.

The base is `Exponential` rather than `Gamma` deliberately. The same shape over
`Gamma` is STILL NaN at `y = 100` after #134, because
`builtin_logdensityof(Gamma, ...)` is NaN at an infinite argument, since
`(shape - 1) * log x - rate * x` is `inf - inf`. That is a separate open defect
in `crates/stablehlo`, recorded under the D1 entry in
`flatppl-dev/TODO-flatppl-rust.md`. So the `log` family's f32 overflow point is
currently pinned by NO directory: broken over `Gamma`, never broken over
`Exponential`, and unexpressible in f64 either way.
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
    return _frozen(-math.exp(y) + y)
