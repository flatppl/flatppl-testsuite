"""Analytic Exp(1) shared-variance mixture of three Normal observations."""
import numpy as np


def oracle(point):
    radius = np.linalg.norm(np.asarray(point["point"], dtype=float) - 1.5)
    return float(-np.sqrt(2)*radius - np.log(2*np.pi*radius))


def grad_oracle(point):
    delta = np.asarray(point["point"], dtype=float) - 1.5
    radius = np.linalg.norm(delta)
    return {"point": (-(np.sqrt(2)+1/radius)*delta/radius).tolist()}
