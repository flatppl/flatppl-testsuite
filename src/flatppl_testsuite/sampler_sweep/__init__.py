"""The oracle-backed SAMPLER sweep.

The density side of this repo has `flatppl_testsuite.sweep` (854 probe rows,
`verdicts/density-sweep.json`). This package is its sampling counterpart: it
draws from every sampleable REGISTRY distribution and from a roster of measure
combinator wraps, and checks the empirical moments, the cross-coordinate
covariance, the goodness of fit and the reported total mass against CLOSED
FORMS.

Read `space.py` for the roster and `oracle.py` for where every closed form
comes from and how it was verified.
"""
