"""INDEPENDENT closed-form oracle for the Gaussian AR(1) chain sample-path model
(``markovchain_ar1.flatppl``):

    traj ~ markovchain(x -> Normal(rho * x, 1), init, 3),  rho = 0.8, init = 2

traj[i] = rho * traj[i-1] + z_i with independent standard Normal z_i and
traj[0] = init (spec §06 `markovchain`: the initial value is not part of the
trajectory). So traj[i] has mean rho^i * init and variance sum_{k<i} rho^(2k),
and Cov(traj[2], traj[3]) = rho * Var(traj[2]). That covariance is the check on
the carried state: steps drawn from `init` instead of the previous state make
the steps independent, so the covariance is 0. A key reused across steps gives
every step the same z, so traj[2] = rho^2 init + (1 + rho) z and traj[3] =
rho^3 init + (1 + rho + rho^2) z, and the covariance is (1 + rho)(1 + rho +
rho^2) = 4.392 instead of 1.312. The difference traj[3] - traj[2] =
(rho - 1) traj[2] + z_3 has variance (1 - rho)^2 Var(traj[2]) + 1.

A projection record of one chain draw has no joint density (its fields resolve
to one draw), so the `density_consistency` check scores the same law written as
three chained Normal draws (``markovchain_ar1_density.flatppl``).

``stat()`` runs offline under ``unified/regen.py`` to refreeze the
``sample_stats`` checks: ``expected`` is the closed-form moment, ``atol`` is K
times the Monte-Carlo standard error at N samples (normal theory).
``logdensity()`` is called live by the ``(sample, det-js)`` runner for the
``density_consistency`` check.
"""
from __future__ import annotations

import math

from scipy.stats import norm

RHO = 0.8
INIT = 2.0
N_SAMPLES = 4000
K = 5


def moments() -> dict[str, float]:
    mean = [RHO ** i * INIT for i in (1, 2, 3)]
    var = [sum(RHO ** (2 * k) for k in range(i)) for i in (1, 2, 3)]
    return {
        "mean_x1": mean[0], "mean_x3": mean[2],
        "var_x1": var[0], "var_x2": var[1], "var_x3": var[2],
        "cov_x2_x3": RHO * var[1],
        "var_x3_minus_x2": (1 - RHO) ** 2 * var[1] + 1.0,
    }


def stat() -> dict[str, dict[str, float]]:
    m = moments()
    n = N_SAMPLES
    se = {
        "mean_x1": math.sqrt(m["var_x1"] / n),
        "mean_x3": math.sqrt(m["var_x3"] / n),
        "var_x1": m["var_x1"] * math.sqrt(2 / n),
        "var_x3": m["var_x3"] * math.sqrt(2 / n),
        "cov_x2_x3": math.sqrt((m["var_x2"] * m["var_x3"] + m["cov_x2_x3"] ** 2) / n),
        "var_x3_minus_x2": m["var_x3_minus_x2"] * math.sqrt(2 / n),
    }
    return {key: {"expected": m[key], "atol": K * se[key]} for key in se}


def logdensity(x1: float, x2: float, x3: float) -> float:
    """log N(x1; rho init, 1) + log N(x2; rho x1, 1) + log N(x3; rho x2, 1)."""
    return (norm.logpdf(x1, RHO * INIT, 1.0) + norm.logpdf(x2, RHO * x1, 1.0)
            + norm.logpdf(x3, RHO * x2, 1.0))
