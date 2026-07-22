"""Independent analytic oracle: gradient of Laplace(location, scale) logdensity
at xobs w.r.t. location and scale. logpdf = -log(2*scale) - abs(x-location)/scale,
so d/dlocation = sign(x-location)/scale and
d/dscale = -1/scale + abs(x-location)/scale**2. abs(t) is non-differentiable at
t=0 (a kink hit exactly by the location=xobs=0.0 point); the emitted engine's
lowering resolves the tie deterministically as the `t>=0` branch, i.e.
sign(0)=+1, so this oracle matches that convention rather than the
symmetric-subgradient sign(0)=0."""


def _sign(v: float) -> float:
    return 1.0 if v >= 0 else -1.0


def grad_oracle(point: dict) -> dict:
    location = point["location"]
    scale = point["scale"]
    x = point["xobs"]
    d = x - location
    dlocation = _sign(d) / scale
    dscale = -1.0 / scale + abs(d) / scale**2
    return {"location": dlocation, "scale": dscale}
