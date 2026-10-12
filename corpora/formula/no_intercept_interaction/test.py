"""Absolute Gaussian score from the frozen R model matrix, not converter output."""
import json
import math
from pathlib import Path

_R = json.loads((Path(__file__).with_name("oracle.json")).read_text())


def oracle(point):
    theta = point["theta"]
    offset = int(_R["intercept"])
    beta, sigma = theta[offset:-1], theta[-1]
    intercept = theta[0] if offset else 0.0
    means = [intercept + math.fsum((x - center) * b
             for x, center, b in zip(row, _R["means"], beta))
             for row in _R["matrix"]]
    # Every parameter has a Normal(0,2) prior; sigma has the positive restriction.
    prior = math.fsum(-math.log(2 * math.sqrt(2 * math.pi)) - value ** 2 / 8
                      for value in theta) + math.log(2)
    likelihood = math.fsum(-math.log(sigma * math.sqrt(2 * math.pi))
                          - (y - mean) ** 2 / (2 * sigma ** 2)
                          for y, mean in zip(_R["response"], means))
    return prior + likelihood
