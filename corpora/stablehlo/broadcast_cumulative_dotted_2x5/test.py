"""Numpy closed form for cumsum.(vv) and cummax.(vv) over a vector of rows.

S and M are the row-wise cumulative sums and cumulative maxima of x. The output is
a * sum(S^2) / 2 + sum(M^2) / 2. d/da is sum(S^2) / 2. d/dx[r, i] is
a * sum_{k >= i} S[r, k], plus the sum of M[r, k] over the k whose running
maximum sits at i (the points have no ties).
"""

import numpy as np


def _parts(point):
    a = float(point["a"])
    x = np.asarray(point["x"], dtype=float)
    s = np.cumsum(x, axis=1)
    m = np.maximum.accumulate(x, axis=1)
    return a, x, s, m


def oracle(point):
    a, _, s, m = _parts(point)
    return float(a * np.sum(s * s) / 2 + np.sum(m * m) / 2)


def grad_oracle(point):
    a, x, s, m = _parts(point)
    gx = a * np.cumsum(s[:, ::-1], axis=1)[:, ::-1]
    for r in range(x.shape[0]):
        for k in range(x.shape[1]):
            gx[r, int(np.argmax(x[r, : k + 1]))] += m[r, k]
    return {"a": float(np.sum(s * s) / 2), "x": gx.tolist()}
