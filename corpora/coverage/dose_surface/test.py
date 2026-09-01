"""Independent oracle for cov_dose_surface (closed form).

The base `iid(Uniform(interval(0, 1)), 2)` has density 1 on the unit
square, so each weighted spelling's log-density at (0.5, 0.8) is
log(0.5 * 0.8^2) exactly; the scored binding is the SUM of the two
arity spellings (§06 `weighted`, "Weight arity") of one measure:

    lp = 2 * log(0.5 * 0.8^2)
"""
import math


def oracle() -> float:
    return 2.0 * math.log(0.5 * 0.8 ** 2)
