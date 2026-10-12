"""Independent categorical targets from captured brms design matrices."""
import numpy as np
from scipy import special, stats


def evaluate(reference, point):
    theta = np.asarray(point["theta"], dtype=float)
    data = reference["standata"]
    y = np.asarray(data["Y"]) - 1
    weights = np.asarray(data.get("weights", np.ones(len(y))))
    logits = np.zeros((len(y), data["ncat"]))
    designs = []
    for category, predictor in enumerate(reference["predictors"], start=1):
        name = predictor["name"]
        design = np.array(data[f"X_{name}"], dtype=float)
        if predictor["intercept"]:
            design[:, 1:] -= design[:, 1:].mean(axis=0)
        positions = predictor["positions"]
        logits[:, category] = design @ theta[positions] + data.get(f"offsets_{name}", 0)
        designs.append((category, design, positions))
    log_probability = logits - special.logsumexp(logits, axis=1)[:, None]
    score = np.dot(weights, log_probability[np.arange(len(y)), y])
    gradient = np.zeros(len(theta))
    for category, design, positions in designs:
        residual = (y == category) - np.exp(log_probability[:, category])
        gradient[positions] += design.T @ (weights * residual)
    for prior in reference["priors"]:
        index = prior["position"]
        if prior["kind"] == "normal":
            location, scale = prior["args"]
            score += stats.norm.logpdf(theta[index], loc=location, scale=scale)
            gradient[index] -= (theta[index] - location) / scale**2
        else:
            df, location, scale = prior["args"]
            delta = theta[index] - location
            score += stats.t.logpdf(theta[index], df=df, loc=location, scale=scale)
            gradient[index] -= (df + 1) * delta / (df * scale**2 + delta**2)
    return float(score), {"theta": gradient.tolist()}
