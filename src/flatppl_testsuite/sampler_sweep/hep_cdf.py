"""Independent §09 CDFs without a scipy.stats distribution counterpart.

The double-sided shape joins two scipy Crystal Ball half-densities at their
common peak height. Voigt integrates SciPy's Faddeeva profile, independently
of the engine's Gaussian-plus-Cauchy sampling construction.
"""

from __future__ import annotations

from functools import partial

import numpy as np
from scipy import integrate, special, stats


def _double_crystalball(
    x, peak, left_scale, right_scale, left_alpha, right_alpha, left_power, right_power
):
    left_weight = left_scale / stats.crystalball.pdf(0, left_alpha, left_power)
    right_weight = right_scale / stats.crystalball.pdf(0, right_alpha, right_power)
    mass = left_weight * stats.crystalball.cdf(
        0, left_alpha, left_power
    ) + right_weight * stats.crystalball.cdf(0, right_alpha, right_power)
    x = np.asarray(x)
    return (
        np.where(
            x <= peak,
            left_weight
            * stats.crystalball.cdf((x - peak) / left_scale, left_alpha, left_power),
            mass
            - right_weight
            * stats.crystalball.cdf((peak - x) / right_scale, right_alpha, right_power),
        )
        / mass
    )


def _bifurcated_normal(x, peak, left_scale, right_scale):
    x = np.asarray(x)
    return np.where(
        x <= peak,
        2 * left_scale * special.ndtr((x - peak) / left_scale),
        left_scale
        + right_scale
        - 2 * right_scale * special.ndtr((peak - x) / right_scale),
    ) / (left_scale + right_scale)


def _voigt(x, peak, width, sigma):
    x = np.asarray(x)
    radii, inverse = np.unique(np.abs(x - peak), return_inverse=True)
    # Integrate adjacent sorted radii, so even a far-tail value never sends
    # adaptive quadrature across a narrow peak. A vector KS request shares
    # each cumulative integral; there is no interpolation or frozen grid.
    cumulative = np.zeros(radii.size)
    lower = 0.0
    for i, upper in enumerate(radii):
        area = integrate.quad(
            special.voigt_profile,
            lower,
            upper,
            args=(sigma, width / 2),
            epsabs=2e-13,
            epsrel=2e-13,
        )[0]
        cumulative[i] = area + (cumulative[i - 1] if i else 0)
        lower = upper
    return 0.5 + np.sign(x - peak) * cumulative[inverse].reshape(x.shape)


def cdf(name, args):
    return partial(
        {
            "double_crystalball": _double_crystalball,
            "bifurcated_normal": _bifurcated_normal,
            "voigt": _voigt,
        }[name],
        **args,
    )
