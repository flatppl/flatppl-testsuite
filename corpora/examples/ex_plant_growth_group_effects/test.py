"""Independent oracle for ex_plant_growth_group_effects.

One-way group effects on R's ``PlantGrowth`` weights: ``weight_i ~
Normal(mu + effects[g_i], sigma)`` with ten plants in each of three groups.
"""
from scipy.stats import norm

_WEIGHTS = [
    4.17, 5.58, 5.18, 6.11, 4.50, 4.61, 5.17, 4.53, 5.33, 5.14,
    4.81, 4.17, 4.41, 3.59, 5.87, 3.83, 6.03, 4.89, 4.32, 4.69,
    6.31, 5.12, 5.54, 5.50, 5.37, 5.29, 4.92, 6.15, 5.80, 5.26,
]


def oracle(point: dict) -> float:
    mu, effects, sigma = point["mu"], point["effects"], point["sigma"]
    return float(sum(
        norm.logpdf(w, mu + effects[i // 10], sigma) for i, w in enumerate(_WEIGHTS)
    ))
