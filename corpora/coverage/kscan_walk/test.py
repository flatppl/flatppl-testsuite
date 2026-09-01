"""Independent oracle for cov_kscan_walk (scipy).

The kscan chain density is the product of the step conditionals:
traj_k ~ Normal(traj_{k-1}, sqrt(2 D dt_k)) with traj_0 = 0 excluded
from the trajectory (§06 `kscan`), plus the Gamma prior on D.

STATUS: LIVE. The rust determiniser unrolls the `kscan` trajectory into
its step conditionals, threading each `dts` element through its own step,
so all three points are numerically checked at 1e-9. This dir was
authored as a refusal pin alongside ar1_drift's `markovchain` sibling;
both lowerings landed together and the frozen values took over unchanged.
"""
import numpy as np
from scipy import stats

DTS = np.array([0.01, 0.02, 0.015, 0.018, 0.012])
TRAJ = np.array([0.1, -0.05, 0.2, 0.15, 0.3])


def oracle(point: dict) -> float:
    d = float(point["D"])
    prev = np.concatenate([[0.0], TRAJ[:-1]])
    lik = stats.norm.logpdf(TRAJ, prev, np.sqrt(2 * d * DTS)).sum()
    prior = stats.gamma.logpdf(d, a=2.0, scale=1 / 0.5)
    return float(lik + prior)
