"""Independent reference: MvNormal(mu, cov) TIER-2 (multivariate fanout)
sample checks.

`stat()` freezes only the fields `sample_checks.py`'s dispatch actually reads:
  - `fanout_dim` (= len(mu)): `check_fanout_distribution`'s if/elif chain
    routes a truthy `fanout_dim` to `_check_mvnormal_fanout` (ahead of the
    scalar/discrete branches). The SAME field is read by
    `sample_stablehlo.py`'s runner (`stat.get("fanout_dim")`) to pick
    `samples_fanned_multivariate` over the elementwise `samples_fanned` --
    each `iid(MvNormal(...), 20000)` call's flat `@sample` return gets
    reshaped to `(-1, fanout_dim)` rows instead of scrambled flat.

`_check_mvnormal_fanout` itself takes its reference mean/covariance straight
from `params["mu"]`/`params["cov"]` (test.json's `params`, the same free
params the `.flatppl` query is called with) -- not from `stat` -- so `stat()`
does not duplicate them here.
"""
PARAMS = {"mu": [0.5, -0.3], "cov": [[1.2, 0.3], [0.3, 0.8]]}


def stat() -> dict:
    return {"fanout_dim": len(PARAMS["mu"])}
