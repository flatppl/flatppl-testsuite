"""§09 Module polynomials defines physicist's Hermite polynomials.
H0=1, H1=2z, H(n+1)=2z*Hn-2n*H(n-1), so H3=8z^3-12z.
The probabilist's convention gives different means at both points."""
import math


def oracle(point: dict) -> float:
    z = point["z"]
    mu = 8 * z ** 3 - 12 * z
    return -0.5 * math.log(2 * math.pi) - 0.5 * (1 - mu) ** 2
