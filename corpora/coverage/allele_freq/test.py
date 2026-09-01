"""Independent oracle for cov_allele_freq (scipy).

Conjugate Dirichlet-Categorical: with class counts n = (14, 8, 8),

    lp(p) = Dirichlet(p | alpha).logpdf + sum_k n_k log p_k

The Dirichlet term uses scipy's chart-measure formula (normative per
the settled §03/§06/§08 ruling); on that term alone the engine and
scipy are same-source, so the categorical count term is the
independently checkable part.

The point keys are `query.flatppl`'s ABI input names, not the model's
draw names: the dir scores under Mode ABI, so one query, one point set
and one frozen vector serve both det-js and stablehlo.
"""
import numpy as np
from scipy import stats

ALPHA = [2.0, 3.0, 4.0]
COUNTS = np.array([14, 8, 8])


def oracle(point: dict) -> float:
    p = np.asarray(point["p_v"], dtype=float)
    return float(stats.dirichlet.logpdf(p, ALPHA) + (COUNTS * np.log(p)).sum())
