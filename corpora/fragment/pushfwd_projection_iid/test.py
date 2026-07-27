"""Independent oracle for frag_pushfwd_projection_iid.

`m` is a 3-way iid `Normal(0,1)` relabelled to fields a/b/c;
`pushfwd(fn(get(_, ["a", "c"])), m)` is the structural projection onto
coordinates a and c, dropping b. Because the components are independent, the
marginal log-density at `record(a=0.1, c=0.3)` is just the sum of the two
kept coordinates' log-pdfs (the dropped b integrates to 1).
"""
from scipy.stats import norm


def oracle() -> float:
    return norm.logpdf(0.1, 0, 1) + norm.logpdf(0.3, 0, 1)
