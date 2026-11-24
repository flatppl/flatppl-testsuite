"""Independent scipy oracle: logdensity of `locscale(Dirichlet([2, 3, 4]), [1, 1, 1], 2.0)` at `y`.

§06 makes `locscale(m, shift, scale)` the map x -> scale * x + shift, and §07 `mul`
scales every cell by a scalar, so a scalar scale with a vector shift is the uniform
affine map `2.0 .* x .+ [1, 1, 1]`. §03 gives `Lebesgue(stdsimplex(n))` as the
coordinate Lebesgue measure of the n - 1 free coordinates, each scaled by the scale,
so the volume term is `(n - 1) * log|scale|`:

    log p(y) = log Dirichlet((y - 1) / 2; [2, 3, 4]) - 2 * log 2

At y = [1.4, 1.6, 2.0] (x = [0.2, 0.3, 0.5]) mpmath 1.4.1 at 50 digits gives
0.63657682907155101126 (logDirichlet 2.0228711901914416301, 2 * log 2
1.3862943611198906188), the frozen value.
"""
import math

from scipy.stats import dirichlet


def oracle(point: dict) -> float:
    x = [(v - 1.0) / 2.0 for v in point["y"]]
    return float(dirichlet.logpdf(x, [2.0, 3.0, 4.0])) - 2.0 * math.log(2.0)
