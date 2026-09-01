"""Independent oracle for cov_sensor_calibration (numpy + scipy).

The `aggregate(sum, [.i], X[.i, .j] * beta[.j])` predictor is the
matrix-vector product X @ beta (§04 multi-axis aggregation: the `.j`
axis is absent from the output axes and is summed). The posterior
log-density at beta is the iid-Normal prior plus the broadcast Normal
likelihood at the predicted means, noise sigma 0.5.

The point keys are `query.flatppl`'s ABI input names, not the model's
draw names: the dir scores under Mode ABI, so one query, one point set
and one frozen vector serve both det-js and stablehlo.
"""
import numpy as np
from scipy import stats

X = np.array([
    [0.57, -0.384, 0.306],
    [-0.449, 0.622, 0.821],
    [-1.383, 0.213, 1.351],
    [-0.352, -0.499, 0.455],
    [-1.104, -1.479, 1.251],
    [0.319, 0.277, 1.557],
    [1.528, -1.048, 0.097],
    [1.359, -1.219, 0.016],
])

# Same values as model.flatppl's y_obs (numpy seed 7331 at
# beta = (1.2, -0.7, 0.4), sigma 0.5).
Y_OBS = np.array([0.515, -1.0269, -1.0278, -0.796, 0.4253, 1.2745, 2.2125, 2.9271])


def oracle(point: dict) -> float:
    beta = np.asarray(point["beta_v"], dtype=float)
    lik = stats.norm.logpdf(Y_OBS, X @ beta, 0.5).sum()
    prior = stats.norm.logpdf(beta, 0.0, 1.0).sum()
    return float(lik + prior)
