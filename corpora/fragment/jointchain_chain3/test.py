"""Independent oracle for frag_jointchain_chain3.

3-step jointchain `a -> b -> c`, each step `Normal(mu=<prev>, sigma=...)`,
joined via `jointchain(lawof(record(a=a)), k1, k2)` (§06); the joint
log-density at `record(a=0.3, b=0.7, c=1.1)` is the sum of all three
per-step log-pdfs (chain rule extended to 3 steps).
"""
from scipy.stats import norm


def oracle() -> float:
    return (
        norm.logpdf(0.3, loc=0.0, scale=1.0)
        + norm.logpdf(0.7, loc=0.3, scale=0.5)
        + norm.logpdf(1.1, loc=0.7, scale=0.25)
    )
