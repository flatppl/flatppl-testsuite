"""Independent Gaussian spline target in the documented native coordinates."""
import json
import math
from pathlib import Path

_DIRECTORY = Path(__file__).resolve().parent
_DATA = json.loads((_DIRECTORY / "data.json").read_text())
_MAP = json.loads((_DIRECTORY / "mapping.json").read_text())["stan_to_native"]


def evaluate(point):
    intercept, slope, *rest = point["theta"]
    z, scale, sigma = rest[:-2], rest[-2], rest[-1]
    # Z_native * Q = Z_Stan, hence Z_native = Z_Stan * Q^T.
    design = [[sum(row[j] * rotation[j] for j in range(len(z)))
               for rotation in _MAP] for row in _DATA["Zs_1_1"]]
    fixed = [row[0] for row in _DATA["Xs"]]
    smooth = [sum(x * coefficient for x, coefficient in zip(row, z, strict=True))
              for row in design]
    residual = [y - intercept - slope * x - scale * random
                for y, x, random in zip(_DATA["Y"], fixed, smooth, strict=True)]
    n = len(residual)
    log2pi = math.log(2 * math.pi)
    value = -0.5 * ((n + 2 + len(z)) * log2pi
                    + sum(r*r for r in residual) / sigma**2
                    + intercept**2 + slope**2 / 4 + sum(c*c for c in z))
    value -= n * math.log(sigma) + math.log(2) + scale + sigma
    gradient = [sum(residual) / sigma**2 - intercept,
                sum(r*x for r, x in zip(residual, fixed, strict=True)) / sigma**2 - slope/4]
    gradient.extend(scale * sum(r * row[j] for r, row in zip(residual, design, strict=True))
                    / sigma**2 - z[j] for j in range(len(z)))
    gradient.extend([sum(r*x for r, x in zip(residual, smooth, strict=True)) / sigma**2 - 1,
                     sum(r*r for r in residual) / sigma**3 - n/sigma - 1])
    return value, {"theta": gradient}
