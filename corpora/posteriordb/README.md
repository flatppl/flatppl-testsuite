# posteriorDB targets

Twenty-one posteriors from [posteriorDB revision 91027f9](https://github.com/stan-dev/posteriordb/tree/91027f9cdd13b7e76283fec3f2898f59d4b83ec4).
Each directory contains the FlatPPL model, native JSON data, original Stan source,
a query, and three frozen density/gradient points. Models with a worked-example
role also live in
[`flatppl-examples/posteriordb`](https://github.com/flatppl/flatppl-examples/tree/main/posteriordb).
The others exist only here, as engine tests.

BridgeStan 2.9.0 evaluated the original Stan models with `propto=false` and
`jacobian=false`. Stan differentiates in its free coordinates even with that
flag. Divide each derivative by the scalar forward-chart derivative to obtain
the saved constrained gradient: one for reals, `x-lower` for a lower bound,
and `(x-lower)*(upper-x)/(upper-lower)` for a finite interval.
For ordered means, solve the coupled chart: `g2=h2/(mu2-mu1)` and
`g1=h1-g2`, where `h` is Stan's free-coordinate gradient. Positive ordered
vectors use the same chart with `x1=exp(u1)`.

For a general chart `x=T(u)` with Jacobian `J`, the saved gradient is
`J (J'J)^-1 h`. On a simplex this is the tangent component: the gradient minus
its mean. Stan's simplex chart is `x=softmax(V u)`, where `V` has the
orthonormal Helmert columns `(1,...,1,-k,0,...,0)/sqrt(k(k+1))`. An ambient
gradient is not defined on the simplex, so `simplex_inputs` in a manifest lists
the inputs whose engine gradient is centered before the comparison.

FlatPPL normalizes half-Normal and half-Cauchy priors. Consequently, add
`log(2)` to Stan's score for `arK`, `blr`, `eight_schools_centered`,
`kidscore_interaction`, and `arma11`, `2*log(2)`
for `gp_regr`, and `3*log(2)` for `irt_2pl` and
`radon_variable_intercept_slope_noncentered`. Proper Uniform priors on
fixed finite intervals add `-log(40) - 3*log(20)` for `GLM_Poisson_model` and
`-log(5)` for `Mh_model`. The ordered mixture adds `3*log(2)`. Stan starts the
`hmm_example` forward recursion with weight one per state, so the uniform
initial distribution adds `-log(2)`. Its positive ordered means keep Stan's
truncated Normal densities without renormalization. These constants leave the
gradients and posterior laws unchanged. Each manifest records the raw Stan
scores and its offset.
Unbounded flat priors (Kidscore, `wells_dist`) use explicit Lebesgue measure and its
zero density contribution.

Every oracle point passed an independent unconstraining check in both
directions. A constrained directional finite difference checked each model.
The maximum scaled error was `1.68e-7` for the first 48 points and `1.79e-9`
for the 15 points that followed. The rate model also matched an analytic
Beta-Binomial calculation. These are density and differentiation checks, not
convergence claims.

The unified tests check densities through Rust-to-JS and StableHLO.
`tests/test_posteriordb.py` checks JS's own measure/data path and full
StableHLO gradients in float32 and float64. No Stan installation is needed
to run these checks. To refresh the oracles, evaluate `reference.stan` with
the adjacent native data and the recorded BridgeStan flags, then apply the
coordinate and constant conventions above. Never generate expected values
with FlatPPL.

## Excluded posteriors

`gp_pois_regr-gp_pois_regr` adds a jitter of `1e-10` to a nearly singular
squared-exponential covariance before its Cholesky factorization. The result
depends on factorization roundoff: float64 gradients differ from Stan's by up to
`5.8e-9` (absolute) and float32 values and gradients by up to 23%. float32
cannot represent the jitter. The JS densities match Stan.

The imported Stan sources and data retain posteriorDB's [BSD license](LICENSE).
