"""Each category contributes its count divided by its probability."""
def oracle_grad(point):
    probabilities = point["p"]
    observed = point["observed"]
    return {"p": [sum(k == i + 1 for k in observed)/p if p else 0.0
                  for i, p in enumerate(probabilities)]}
