"""Independent closed-form reference: the final state of a Gaussian AR(1) chain.

`traj ~ markovchain(x -> Normal(rho * x, 1), init, n)` gives
traj[i] = rho * traj[i-1] + z_i with independent standard Normal z_i and
traj[0] = init (spec §06 `markovchain`: the initial value is not part of the
trajectory). So traj[n] = rho^n * init + sum_{k<n} rho^k z_{n-k}, which is
Normal(rho^n * init, sum_{k<n} rho^(2k)). A step drawn from `init` instead of
the previous state gives Normal(rho * init, 1), and a key reused across steps
gives variance (sum_{k<n} rho^k)^2, so the KS check separates both.
"""
import math

PARAMS = {"rho": 0.8}
INIT = 2.0
N = 6


def stat() -> dict:
    rho = PARAMS["rho"]
    variance = sum(rho ** (2 * k) for k in range(N))
    return {
        "distribution": {
            "family": "norm",
            "kwargs": {"loc": rho ** N * INIT, "scale": math.sqrt(variance)},
        },
        "discrete": False,
    }
