"""Independent oracle for ex_signal_background_counting.

Signal-plus-background rare-event search (the BAT.jl paper example) over 5
detector datasets: uniform priors on ``S``, ``sigma_B``, ``m_B``, ``lam``;
``B ~ iid(LogNormal(log(m_B) - sigma_B^2/2, sigma_B), 5)`` so ``E[B_i] =
m_B``; per-dataset counts ``Poisson(nu_B + nu_S)``; per-event energies a
two-component mixture ``w*Exponential(scale=lam) + (1-w)*Normal(100, 2)``
with ``w = nu_B / (nu_B + nu_S)``. The model spells the background as
``Exponential(1/lam)`` because FlatPPL's ``Exponential`` is
rate-parameterised, so the scipy scale is ``lam`` itself.
"""
import numpy as np
from scipy.stats import expon, lognorm, norm, poisson, uniform

_EXPOSURE = np.array([1.6, 1.3, 1.0, 0.7, 0.4])
_EFFICIENCY = np.array([0.5, 0.6, 0.7, 0.8, 0.9])
_COUNTS = np.array([13, 10, 10, 2, 2])
_EVENTS = [
    [32.29945024522565, 69.08580773988763, 25.440423777510436,
     75.61403615267723, 7.9267656229908585, 3.3354153534688638,
     20.670985493456392, 82.75167461000609, 136.75349754794027,
     19.976541400061382, 34.57232691499431, 52.32632653325007,
     4.75519376172241],
    [63.76604456320073, 4.602635553779128, 68.24639884216056,
     4.272813246216692, 49.33863066627453, 41.748541234687956,
     47.49935392156085, 37.52247790527123, 26.040570121268903,
     58.08667451701596],
    [58.822595816291454, 8.785379233761681, 55.58320441541379,
     63.51718319418958, 58.297943138713215, 40.37315193306145,
     7.875521943121733, 139.17144907577784, 100.79360983559722,
     98.9325347685139],
    [31.878682258259104, 21.28067021826852],
    [53.041794807908694, 98.55886779280806],
]


def oracle(point: dict) -> float:
    S, sigma_B = point["S"], point["sigma_B"]
    m_B, lam = point["m_B"], point["lam"]
    B = np.asarray(point["B"], dtype=float)

    lp = (uniform.logpdf(S, 0.0, 10.0)
          + uniform.logpdf(sigma_B, 0.1, 0.9)
          + uniform.logpdf(m_B, 1e-10, 20.0 - 1e-10)
          + uniform.logpdf(lam, 1e-10, 100.0 - 1e-10)
          + lognorm.logpdf(B, s=sigma_B,
                           scale=np.exp(np.log(m_B) - sigma_B ** 2 / 2)).sum())

    nu_B = _EXPOSURE * B
    nu_S = _EXPOSURE * _EFFICIENCY * S
    w = nu_B / (nu_B + nu_S)
    lp += poisson.logpmf(_COUNTS, nu_B + nu_S).sum()
    for i, events in enumerate(_EVENTS):
        ev = np.asarray(events)
        dens = w[i] * expon.pdf(ev, scale=lam) + (1 - w[i]) * norm.pdf(ev, 100.0, 2.0)
        lp += np.log(dens).sum()
    return float(lp)
