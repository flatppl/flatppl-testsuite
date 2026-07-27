"""Independent oracle for frag_jointchain_scalar.

Same maths as `jointchain_normal`'s oracle, but the jointchain is built over
a SCALAR variate (`lawof(a)` / `kernelof(Normal(...), a=a)`) rather than a
record, and scored at the vector point `[0.3, 0.7]` instead of a record --
exercises the scalar/vector-variate jointchain lowering path; numerically
identical to the record-valued version.
"""
from scipy.stats import norm


def oracle() -> float:
    return norm.logpdf(0.3, loc=0.0, scale=1.0) + norm.logpdf(0.7, loc=0.3, scale=0.5)
