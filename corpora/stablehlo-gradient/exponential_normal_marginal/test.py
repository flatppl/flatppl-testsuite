"""Independent analytic oracle: gradient of the exponentially modified Gaussian
log-density. With z = (lam sig^2 - y) / (sqrt(2) sig) and
log f = log(lam/2) - lam y + lam^2 sig^2 / 2 + log erfc(z), and
d log erfc(z) / dz = -2 / (sqrt(pi) erfcx(z)):
  d/dy   = -lam + k / (sqrt(2) sig)
  d/dlam = 1/lam - y + lam sig^2 - k sig / sqrt(2)
  d/dsig = lam^2 sig - k (lam / sqrt(2) + y / (sqrt(2) sig^2))
with k = 2 / (sqrt(pi) erfcx(z))."""
import math

from scipy.special import erfcx


def grad_oracle(point: dict) -> dict:
    y, lam, sig = point["t_y"], point["lam"], point["sig"]
    r2 = math.sqrt(2.0)
    z = (lam * sig * sig - y) / (r2 * sig)
    k = 2.0 / (math.sqrt(math.pi) * erfcx(z))
    return {
        "t_y": float(-lam + k / (r2 * sig)),
        "lam": float(1.0 / lam - y + lam * sig * sig - k * sig / r2),
        "sig": float(lam * lam * sig - k * (lam / r2 + y / (r2 * sig * sig))),
    }
