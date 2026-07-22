"""Independent reference: Dirichlet(alpha) simplex fanout + independence
sample checks.

`stat()` freezes only the fields `sample_checks.py` actually reads:
  - `fanout_dim` (= len(alpha)): read by `sample_stablehlo.py`'s runner
    (`stat.get("fanout_dim")`) to pick `samples_fanned_multivariate` over the
    elementwise `samples_fanned` -- each `iid(Dirichlet(...), 20000)` call's
    flat `@sample` return gets reshaped to `(-1, fanout_dim)` simplex rows.
  - `fanout_simplex` (truthy): `check_fanout_distribution`'s if/elif chain
    checks this FIRST, routing to `_check_dirichlet_fanout` (simplex
    row-sum-to-1, per-component mean/variance vs the closed-form Dirichlet
    moments, and component-correlation vs `dirichlet_theo_corr`) ahead of the
    `fanout_dim` (mvnormal) branch.
  - `independence: "dirichlet"`: `check_independence` dispatches on
    `stat.get("independence")`; the `"dirichlet"` branch checks marginal-Beta
    KS per component, component-correlation vs `dirichlet_theo_corr`, and
    lag-1 autocorrelation. It runs on the SCALAR (non-fanned) `query.flatppl`
    draws -- one Dirichlet 3-vector per chained `@sample` call, stacked to
    shape `(draws, 3)` by `ex.samples` -- not the `iid`-fanned query.

Both `_check_dirichlet_fanout` and the `check_independence` dirichlet branch
take alpha/a0 straight from `params["alpha"]` (test.json's `params`, the same
free param the `.flatppl` query is called with) -- not from `stat` -- so
`stat()` does not duplicate it here.
"""
PARAMS = {"alpha": [2.0, 3.0, 5.0]}


def stat() -> dict:
    return {
        "fanout_dim": len(PARAMS["alpha"]),
        "fanout_simplex": True,
        "independence": "dirichlet",
    }
