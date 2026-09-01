"""Independent oracle for weighted_arity (closed form).

The base `iid(Uniform(interval(0, 1)), 2)` has density 1 on the unit
square, so each weighted spelling's log-density at (0.5, 0.8) is
log(0.5 * 0.8^2) exactly; the scored binding is the SUM of the two
arity spellings (§06 `weighted`, "Weight arity") of one measure:

    lp = 2 * log(0.5 * 0.8^2)

The model has no free parameters, so the one authored point is empty and
`oracle` takes it only to match the runner's signature.
"""
import math


def oracle(point: dict) -> float:
    return 2.0 * math.log(0.5 * 0.8 ** 2)
