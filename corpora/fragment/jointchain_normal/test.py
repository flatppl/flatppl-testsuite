"""Independent oracle for frag_jointchain_normal.

`a ~ Normal(mu=0.0, sigma=1.0)`, `b | a ~ Normal(mu=a, sigma=0.5)` joined via
`jointchain(lawof(record(a=a)), k)` (§06); the joint log-density at
`record(a=0.3, b=0.7)` is the SUM of the marginal-of-a log-pdf and the
kernel's conditional log-pdf (chain rule, one step, no extra normalizing
constant).
"""
from scipy.stats import norm


def oracle() -> float:
    return norm.logpdf(0.3, loc=0.0, scale=1.0) + norm.logpdf(0.7, loc=0.3, scale=0.5)
