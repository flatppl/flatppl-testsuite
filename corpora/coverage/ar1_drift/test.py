"""Independent oracle for cov_ar1_drift (closed form + scipy).

The model is `x ~ markovchain(prev -> Normal(prev + drift, sigma_step),
0.0, 120)` with drift 0.4 and a half-Cauchy prior on `sigma_step`. The
chain density is the product of the step conditionals (§06: jointchain-
like, no marginalization), so with increments d_k = x_k - x_{k-1} - drift
and S = sum d_k^2 = 83.15890438672801:

    log lik  = -n/2 log(2 pi) - n log s - S / (2 s^2)
    log prior = log 2 + Cauchy(0, 1).logpdf(s)      (half-Cauchy via
                normalize(truncate(Cauchy(0, 1), interval(0, inf))))

STATUS: the rust determiniser REFUSES `markovchain` density lowering
("deferred to a later task", verified live at setup-resolved main,
2026-09-01), so this dir carries `allow_skip: true` and the unified
harness records DETERMINIZE_SKIP. The frozen `expected` values below are
real oracle values: the moment the lowering lands, the skip disappears
and the numeric compare takes over.
"""
import numpy as np
from scipy import stats

DRIFT = 0.4
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

_INC = np.diff(np.concatenate([[0.0], X_DATA])) - DRIFT
_S = float((_INC ** 2).sum())


def oracle(point: dict) -> float:
    s = float(point["sigma_step"])
    lik = -N / 2 * np.log(2 * np.pi) - N * np.log(s) - _S / (2 * s * s)
    prior = np.log(2.0) + stats.cauchy.logpdf(s)
    return float(lik + prior)
