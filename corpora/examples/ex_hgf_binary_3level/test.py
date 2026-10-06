"""Independent scalar HGF recurrence with complex-step density derivatives."""

import numpy as np

OUTCOMES = [1,1,1,1,1,1,1,0, 0,0,0,0,0,0,0,1, 1,0,1,0,1,0,1,0, 1,1,0,0,1,1,0,0]
CHOICES = np.array([1,1,0,1,1,1,1,1, 1,0,0,1,0,0,0,0, 0,1,1,0,1,0,0,1, 1,0,1,0,1,1,0,0])


def _score(point):
    omega, omega_vol, noise = (point[k] for k in ("omega_v", "omega_vol_v", "log_noise_v"))
    if not (-4 <= np.real(omega) <= -1 and -6 <= np.real(omega_vol) <= -2):
        return -np.inf
    mean, precision, vol_mean, vol_precision = 0.0, 1.0, 0.0, 1.0
    means = []
    for outcome in OUTCOMES:
        predicted_precision = 1/(1/precision + np.exp(omega+vol_mean))
        predicted_vol_precision = 1/(1/vol_precision + np.exp(omega_vol))
        probability = 1/(1+np.exp(-mean))
        precision = predicted_precision + probability*(1-probability)
        change = (outcome-probability)/precision
        surprise = predicted_precision/precision + predicted_precision*change**2 - 1
        weight = np.exp(omega+vol_mean)*predicted_precision
        mean += change
        vol_mean += weight*surprise/(2*predicted_vol_precision)
        vol_precision = predicted_vol_precision + weight**2*(0.5+surprise) - 0.5*weight*surprise
        means.append(mean)
    logits = np.asarray(means)/np.exp(noise)
    likelihood = np.sum(CHOICES*logits - np.log1p(np.exp(logits)))
    prior = -np.log(12) - np.log(0.5*np.sqrt(2*np.pi)) - 0.5*(noise/0.5)**2
    return likelihood + prior


def oracle(point):
    return float(_score(point))


def grad_oracle(point):
    return {key: float(np.imag(_score(point | {key: value+1e-20j}))/1e-20)
            for key, value in point.items()}
