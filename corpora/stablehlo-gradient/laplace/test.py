"""Independent analytic oracle: gradient of Laplace(location, scale) logdensity
at xobs w.r.t. location and scale. logpdf = -log(2*scale) - abs(x-location)/scale,
so d/dlocation = sign(x-location)/scale and
d/dscale = -1/scale + abs(x-location)/scale**2. The location derivative does
not exist at x=location. The gradient corpus uses differentiable points only.
"""


def _sign(v: float) -> float:
    if v == 0:
        raise ValueError("Laplace location derivative is undefined at x=location")
    return 1.0 if v > 0 else -1.0


def grad_oracle(point: dict) -> dict:
    location = point["location"]
    scale = point["scale"]
    x = point["xobs"]
    d = x - location
    dlocation = _sign(d) / scale
    dscale = -1.0 / scale + abs(d) / scale**2
    return {"location": dlocation, "scale": dscale}
