"""Independent oracle for frag_densityof_normal.

`d = densityof(lawof(record(a = draw(Normal(0, 1)))), record(a = 0.5))` is the
plain (non-log) standard-normal density at 0.5. Run offline via
`pixi run regen corpora/fragment/densityof_normal`; the harness only compares
the frozen value at test time.
"""
from scipy.stats import norm


def oracle():
    return float(norm.pdf(0.5, loc=0.0, scale=1.0))
