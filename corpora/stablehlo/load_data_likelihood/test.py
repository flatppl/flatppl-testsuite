"""Independent scipy oracle: iid/broadcast Normal likelihood over a linear
model, sum(norm.logpdf(y, loc=alpha + X @ beta, scale=sigma)). X (a column of
3-vectors) and y default to data.json (what the runner feeds the module); a
point may override either, which is how the reuse test scores a second
dataset."""
import json
from pathlib import Path

import numpy as np
from scipy.stats import norm

_DATA = Path(__file__).parent / "data.json"


def oracle(point: dict) -> float:
    data = json.loads(_DATA.read_text())
    x = np.asarray(point.get("x_data", data["x"]))
    y = np.asarray(point.get("y_data", data["y"]))
    means = point["alpha"] + x @ np.asarray(point["beta"])
    return float(np.sum(norm.logpdf(y, loc=means, scale=point["sigma"])))
