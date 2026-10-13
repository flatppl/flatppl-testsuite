# posteriorDB targets

Ten models from [posteriorDB revision 91027f9](https://github.com/stan-dev/posteriordb/tree/91027f9cdd13b7e76283fec3f2898f59d4b83ec4).
Each directory contains the FlatPPL model, native JSON data, original Stan source,
a query, and three frozen density/gradient points. The models also live in
[`flatppl-examples/posteriordb`](https://github.com/flatppl/flatppl-examples/tree/main/posteriordb).

BridgeStan 2.9.0 evaluated the original Stan models with `propto=false` and
`jacobian=false`. Stan differentiates in its free coordinates even with that
flag. Divide each derivative by the scalar forward-chart derivative to obtain
the saved constrained gradient: one for reals, `x-lower` for a lower bound,
and `(x-lower)*(upper-x)/(upper-lower)` for a finite interval.

FlatPPL normalizes half-Normal and half-Cauchy priors. Consequently, add
`log(2)` to Stan's score for `arK`, `blr`, `eight_schools_centered`, and
`kidscore_interaction`, and `2*log(2)` for `gp_regr`. Proper Uniform priors on
fixed finite intervals add `-log(40) - 3*log(20)` for `GLM_Poisson_model` and
`-log(5)` for `Mh_model`. These constants leave the gradients and posterior
laws unchanged. Each manifest records the raw Stan scores and its offset.
Only Kidscore's unbounded coefficient priors retain `Flat` and its zero
density contribution.

All 30 oracle points passed an independent unconstraining check. A constrained
directional finite difference checked each model, with maximum scaled error
`5.37e-9`. The rate model also matched an analytic Beta-Binomial calculation.
These are density and differentiation checks, not convergence claims.

The unified tests check densities through Rust-to-JS and StableHLO.
`tests/test_posteriordb.py` checks JS's own measure/data path and full
StableHLO gradients in float32 and float64. No Stan installation is needed
to run these checks. To refresh the oracles, evaluate `reference.stan` with
the adjacent native data and the recorded BridgeStan flags, then apply the
coordinate and constant conventions above. Never generate expected values
with FlatPPL.

The imported Stan sources and data retain posteriorDB's [BSD license](LICENSE).
