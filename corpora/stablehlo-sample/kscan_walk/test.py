"""Independent closed-form reference: the final state of a diffusion random walk.

`traj ~ kscan((x, dt) -> Normal(x, sqrt(2 D dt)), init, dts)` gives
traj[i] = traj[i-1] + sqrt(2 D dt_i) z_i with independent standard Normal z_i
and traj[0] = init (spec §06 `kscan`). So the final state is
Normal(init, sqrt(2 D sum(dts))). Each step reads its own `dt`, so a scan that
held one step input fixed would change the variance.
"""
import math

PARAMS = {"D": 0.8}
INIT = 0.5
DTS = [0.01, 0.02, 0.015, 0.018, 0.012]


def stat() -> dict:
    return {
        "distribution": {
            "family": "norm",
            "kwargs": {"loc": INIT, "scale": math.sqrt(2.0 * PARAMS["D"] * sum(DTS))},
        },
        "discrete": False,
    }
