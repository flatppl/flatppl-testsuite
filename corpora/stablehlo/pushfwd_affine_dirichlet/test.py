"""Independent scipy oracle: logdensity of `pushfwd(fn(2.0 .* _ .+ 1.0), Dirichlet([2, 3, 4]))` at `y`.

§03 gives `Lebesgue(stdsimplex(n))` as the coordinate Lebesgue measure of the n - 1 free
coordinates. A uniform scale `a` scales each of them by `a`, so the volume term of
`a .* x .+ b` is `(n - 1) * log|a|`, not the ambient `n * log|a|`:

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
