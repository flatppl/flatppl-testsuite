"""Independent oracle for cov_ar1_drift (closed form + scipy).

The model is `x ~ markovchain(step_kernel, 0.0, 120, sigma_step, drift)` with
`step_kernel(prev, sigma, delta) = Normal(prev + delta, sigma)`, a Normal(0, 1)
prior on `drift` and a half-Cauchy prior on `sigma_step`. The chain density is the
product of the step conditionals (§06: jointchain-like, no marginalization), so with
increments d_k = x_k - x_{k-1}, S1 = sum d_k and S2 = sum d_k^2:

    log lik   = -n/2 log(2 pi) - n log s - (S2 - 2 delta S1 + n delta^2) / (2 s^2)
    log prior = Normal(0, 1).logpdf(delta) + log 2 + Cauchy(0, 1).logpdf(s)
                (half-Cauchy via normalize(truncate(Cauchy(0, 1), interval(0, inf))))

The posterior moments quoted in the example's header come from a two-dimensional
quadrature of this closed form (scipy.integrate.dblquad over drift in [-2, 2],
sigma_step in [0.3, 2.5], which holds all the mass): drift 0.3599 +- 0.0767,
sigma_step 0.8410 +- 0.0550, correlation -0.004.

STATUS: LIVE. The rust determiniser unrolls the `markovchain` trajectory
into its step conditionals, so all points are numerically checked at 1e-9.

The point keys are `query.flatppl`'s ABI input names, not the model's
draw names: the dir scores under Mode ABI, so one query, one point set
and one frozen vector serve both det-js and stablehlo.
"""
import numpy as np
from scipy import stats

N = 120

X_DATA = np.array([
    1.516427, 1.882717, 3.261518, 4.888037, 3.325587, 3.775154,
    4.261822, 2.324014, 2.208734, 2.256201, 2.185386, 1.056015,
    1.626074, 2.347466, 3.492224, 3.370195, 3.155436, 2.83785,
    2.201612, 4.624558, 5.183817, 6.681243, 6.568079, 6.306448,
    8.402202, 7.913185, 9.291051, 10.814278, 11.58481, 13.279839,
    14.254945, 14.798567, 16.384709, 16.97349, 16.006265, 16.154214,
    17.257363, 18.647199, 19.571369, 20.625459, 20.519087, 21.036068,
    21.067389, 21.481353, 22.247533, 22.052138, 22.950567, 21.871003,
    21.400535, 23.424888, 23.683817, 23.573373, 24.292698, 24.346151,
    25.499088, 24.674574, 24.755158, 24.915128, 24.921334, 25.082569,
    25.089259, 26.386966, 26.577786, 26.847697, 27.185248, 27.677056,
    27.565025, 27.758033, 28.950505, 29.314636, 30.40715, 31.223765,
    29.700574, 30.0341, 30.476632, 31.257038, 31.119398, 30.029854,
    30.790604, 31.291329, 31.194528, 30.32378, 32.0472, 32.423559,
    34.515547, 33.866483, 35.037138, 34.7215, 33.943814, 34.522598,
    35.258671, 35.364932, 35.200842, 36.132384, 37.695908, 37.130937,
    36.63698, 37.312778, 36.318208, 36.534821, 36.372995, 36.971762,
    38.028491, 38.180367, 36.881211, 37.83705, 38.661383, 39.505314,
    40.711684, 41.692369, 42.711966, 42.033105, 42.964984, 42.201666,
    42.501826, 42.38132, 42.918669, 43.7028, 43.263506, 43.443864,
])

_D = np.diff(np.concatenate([[0.0], X_DATA]))
_S1 = float(_D.sum())
_S2 = float((_D ** 2).sum())


def oracle(point: dict) -> float:
    s = float(point["sigma_step_v"])
    delta = float(point["drift_v"])
    ss = _S2 - 2 * delta * _S1 + N * delta ** 2
    lik = -N / 2 * np.log(2 * np.pi) - N * np.log(s) - ss / (2 * s * s)
    prior = stats.norm.logpdf(delta) + np.log(2.0) + stats.cauchy.logpdf(s)
    return float(lik + prior)
