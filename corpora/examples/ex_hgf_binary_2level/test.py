"""Independent scalar recurrence; complex-step derivatives of the same law."""

import numpy as np

OUTCOMES = [1, 0, 1, 1, 0, 0]
CHOICES = np.array([1, 0, 1, 0, 1, 0])


def _score(point):
    omega, noise = point["omega_v"], point["log_noise_v"]
    mean, precision = 0.0, 1.0
    means = []
    for outcome in OUTCOMES:
        probability = 1/(1 + np.exp(-mean))
        prediction_variance = 1/precision + np.exp(omega)
        precision = 1/prediction_variance + probability*(1-probability)
        mean += (outcome-probability)/precision
        means.append(mean)
    logits = np.asarray(means)/np.exp(noise)
    likelihood = np.sum(CHOICES*logits - np.log1p(np.exp(logits)))
    prior = -0.5*(omega+2)**2 - 0.5*(noise/0.5)**2 - np.log(2*np.pi*0.5)
    return likelihood + prior


def oracle(point):
    return float(_score(point))


def grad_oracle(point):
    return {key: float(np.imag(_score(point | {key: value+1e-20j}))/1e-20)
            for key, value in point.items()}
