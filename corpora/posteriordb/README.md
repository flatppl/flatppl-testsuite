# posteriorDB targets

Thirty-six posteriors from [posteriorDB revision 91027f9](https://github.com/stan-dev/posteriordb/tree/91027f9cdd13b7e76283fec3f2898f59d4b83ec4).
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
`kidscore_interaction`, `kidscore_momhs`, `kidscore_momhsiq`, `kidscore_momiq`,
and `arma11`, `2*log(2)`
for `gp_regr`, and `3*log(2)` for `irt_2pl` and
`radon_variable_intercept_slope_noncentered`. Proper Uniform priors on
fixed finite intervals add `-log(40) - 3*log(20)` for `GLM_Poisson_model` and
`-log(5)` for `Mh_model`. The ordered mixture adds `3*log(2)`. Stan starts the
`hmm_example` forward recursion with weight one per state, so the uniform
initial distribution adds `-log(2)`. Its positive ordered means keep Stan's
truncated Normal densities without renormalization. These constants leave the
gradients and posterior laws unchanged. Each manifest records the raw Stan
scores and its offset.
Unbounded flat priors (the Kidscore, earnings, mesquite and `wells_dist`
regressions) use explicit Lebesgue measure and its zero density contribution.

Every oracle point passed an independent unconstraining check in both
directions. A constrained directional finite difference checked each model.
The maximum scaled error was `1.68e-7` for the first 48 points, `1.79e-9`
for the next 15, and `1.17e-9` for the 45 points after them. The rate model also matched an analytic
Beta-Binomial calculation. These are density and differentiation checks, not
convergence claims.

The unified tests check densities through Rust-to-JS and StableHLO.
`tests/test_posteriordb.py` checks JS's own measure/data path and full
StableHLO gradients in float32 and float64. No Stan installation is needed
to run these checks. To refresh the oracles, evaluate `reference.stan` with
the adjacent native data and the recorded BridgeStan flags, then apply the
coordinate and constant conventions above. Never generate expected values
with FlatPPL.

## Float64-only rows

A manifest may list `"precisions": ["f64"]`. `tests/test_posteriordb.py` then
checks StableHLO in float64 only. The JS checks do not change. A row qualifies
only when an independent float32 evaluation of the Stan formula misses the
float32 tolerance at an oracle point, in line with the engine's float32 error.
An engine float32 error clearly above the independent one is an engine defect.
The independent evaluation is plain NumPy float32 over 200 summation orders.

| Row | Point, component | Engine float32 error (M4; CI) | NumPy float32 error (median, max) | Tolerance |
|---|---|---:|---:|---:|
| `logearn_height` | 1, `beta[2]` | `0.028` | `0.027`, `0.033` | `0.0014` |
| `logearn_height_male` | 1, `beta[2]` | `0.0119`; `0.0125` | `0.0134`, `0.0157` | `0.0119` |
| `logearn_logheight_male` | 1, `beta[2]` | `0.0019` | `0.0019`, `0.0021` | `0.00058` |
| `kidscore_momiq` | 1, `beta[2]` | `0.00029`; `0.00030` | `0.00042`, `0.00048` | `0.00030` |

These rows regress on uncentered height (about 66), log height or maternal IQ
(about 100). Near the posterior mean the coefficient gradient is a small
difference of large float32 sums. CI runs on Linux x86-64; the M4 runs locally.

## Excluded posteriors

`gp_pois_regr-gp_pois_regr` adds a jitter of `1e-10` to a nearly singular
squared-exponential covariance before its Cholesky factorization. The result
depends on factorization roundoff. At the three oracle points, float64
gradients differ from Stan's by up to `5.8e-9` (absolute). float32 gradients
differ by up to 30%, or are NaN at one posterior draw. float32 cannot represent
the jitter, which comes from the Stan source. The JS densities match Stan.

The imported Stan sources and data retain posteriorDB's [BSD license](LICENSE).
