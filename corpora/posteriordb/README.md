# posteriorDB targets

Eighty-nine posteriors from [posteriorDB revision 91027f9](https://github.com/stan-dev/posteriordb/tree/91027f9cdd13b7e76283fec3f2898f59d4b83ec4).
Each directory contains the FlatPPL model, native JSON data, original Stan source,
a query, and three frozen density/gradient points. The seven `radon_all-*` rows
share one copy of their data in `_data/radon_all.json`. Models with a worked-example
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
`matrix_inputs` lists the inputs that are matrices rather than arrays of
arrays, so the JS checks write them with `rowstack`.

FlatPPL normalizes half-Normal and half-Cauchy priors. Consequently, add
`log(2)` to Stan's score for `arK`, `blr`, `eight_schools_centered`,
`kidscore_interaction`, `kidscore_momhs`, `kidscore_momhsiq`, `kidscore_momiq`,
and `arma11`, `2*log(2)`
for `gp_regr`, and `3*log(2)` for `irt_2pl` and
`radon_variable_intercept_slope_noncentered`. Among the `radon_mn-*` rows,
`radon_pooled` and `radon_county_intercept` add `log(2)` and the four other
noncentered models add `2*log(2)`. The `radon_all-*` rows use the same offsets
as their `radon_mn-*` twins, and the national varying intercept and slope model
adds `3*log(2)`. `bym2_offset_only` adds `log(2)`. Proper Uniform priors on
fixed finite intervals add `-log(40) - 3*log(20)` for `GLM_Poisson_model` and
`-log(5)` for `Mh_model` and `GLMM1_model`, `-2*log(100)` for
`radon_mod-radon_county`, `-3*log(100)` for `pilots` and `-5*log(100)` for
`election88_full`. `seeds_stanified_model`, `seeds_centered_model` and
`lsat_model` add `log(2)` for one half-Cauchy or half-Normal prior, and
`low_dim_gauss_mix_collapse` adds `2*log(2)`. The uniform five-component simplex
of `normal_mixture_k` has density 24 and its Uniform(0, 10) scales add
`-5*log(10)`. The ordered mixture adds `3*log(2)`. Stan starts the
`hmm_example` forward recursion with weight one per state, so the uniform
initial distribution adds `-log(2)`. Its positive ordered means keep Stan's
truncated Normal densities without renormalization. These constants leave the
gradients and posterior laws unchanged. Each manifest records the raw Stan
scores and its offset.
Unbounded flat priors (the Kidscore, earnings, mesquite, Sesame, wells, NES and
NES-logit regressions, the Kilpisjarvi residual scale, and the rat growth
scales, which enter through `jointchain` because the effects depend on them) use explicit Lebesgue
measure and its zero density contribution.

Every oracle point passed an independent unconstraining check in both
directions. A constrained directional finite difference checked each model.
The maximum scaled error was `1.68e-7` for the first 48 points, `1.79e-9`
for the next 15, `1.17e-9` for the following 45, `3.68e-9` for the next 39,
`7.68e-9` for the next 45, `1.04e-8` for the next 42, `7.5e-8` for the next
27, `1.72e-10` for the three `logearn_interaction` points, and `2.6e-10` for
the three `hierarchical_gp` points. Rows without reference draws take their
second and third points from a short adaptive HMC run on Stan's own density.
The rate model also matched an analytic Beta-Binomial calculation. These are
density and differentiation checks, not convergence claims.

The unified tests check densities through Rust-to-JS and StableHLO.
`tests/test_posteriordb.py` checks JS's own measure/data path and full
StableHLO gradients in float32 and float64. No Stan installation is needed
to run these checks. To refresh the oracles, evaluate `reference.stan` with
the adjacent native data and the recorded BridgeStan flags, then apply the
coordinate and constant conventions above. Never generate expected values
with FlatPPL.

## Float64-only rows

A manifest may list `"precisions": ["f64"]`. `tests/test_posteriordb.py` and
the unified StableHLO runner then check StableHLO in float64 only. The JS
checks do not change. A row qualifies only when an independent float32
evaluation of the Stan formula misses the float32 tolerance at an oracle
point, in line with the engine's float32 error. An engine float32 error
clearly above the independent one is an engine defect. The independent
evaluation is plain NumPy float32 over 200 summation orders. For
`hierarchical_gp` it is a JAX float32 gradient (`jax.grad`) of the Stan
formula over 50 observation orders, since its Cholesky adjoint has no hand
NumPy form.

| Row | Point, component | Engine float32 error (M4; CI) | NumPy float32 error (median, max) | Tolerance |
|---|---|---:|---:|---:|
| `logearn_height` | 1, `beta[2]` | `0.028` | `0.027`, `0.033` | `0.0014` |
| `logearn_height_male` | 1, `beta[2]` | `0.0119`; `0.0125` | `0.0134`, `0.0157` | `0.0119` |
| `logearn_logheight_male` | 1, `beta[2]` | `0.0019` | `0.0019`, `0.0021` | `0.00058` |
| `logearn_interaction` | 1, `beta[4]` | `0.0259` | `0.0259`, `0.0272` | `0.00041` |
| `hierarchical_gp` | 3, `length_GP_region_short` | `0.00315` | `0.00330`, `0.00338` (JAX) | `0.00050` |
| `kidscore_momiq` | 1, `beta[2]` | `0.00029`; `0.00030` | `0.00042`, `0.00048` | `0.00030` |
| `kidscore_momhsiq` | 1, `beta[3]` | `0.00017` | `0.00027`, `0.00034` | `0.00033` |
| `kilpisjarvi` | 2, `beta` | `0.174` | `0.169`, `0.177` | `0.0163` |
| `pilots` | 3, `sigma_a` | `0.00110` | `0.00112`, `0.00112` | `0.00051` |
| `election88_full` | 3, `beta[1]` | `0.00049` | `0.00042`, `0.00094` | `0.00046` |
| `radon_all-radon_pooled` | 1, `sigma_y` | `0.0019` | `0.0017`, `0.0023` | `0.0012` |
| `radon_all-radon_variable_slope_noncentered` | 3, `sigma_y` | `0.0015` | `0.0009`, `0.0032` | `0.0006` |
| `radon_all-radon_variable_intercept_slope_noncentered` | 3, `mu_alpha` | `0.00051` | `0.00052`, `0.00081` | `0.00048` |

These rows regress on uncentered height (about 66), its product with an
indicator, log height, maternal IQ (about 100) or the calendar year (about
2000), or they sum thousands of observations (`election88_full`, the
12573-home `radon_all` rows) or cancel large scale terms (`pilots`, where
sigma_a is 0.015) or factor a nearly singular covariance
(`hierarchical_gp`). Near the posterior mean the coefficient gradient is a
small difference of large float32 sums. CI runs on Linux x86-64; the M4 runs
locally.

## Excluded posteriors

`gp_pois_regr-gp_pois_regr` adds a jitter of `1e-10` to a nearly singular
squared-exponential covariance before its Cholesky factorization. The result
depends on factorization roundoff. At the three oracle points, float64
gradients differ from Stan's by up to `5.8e-9` (absolute). float32 gradients
differ by up to 30%, or are NaN at one posterior draw. float32 cannot represent
the jitter, which comes from the Stan source. The JS densities match Stan.

## Skipped posteriors

These posteriors have no row by design. Posteriors held on engine findings land later.

- ODE models: `lotka_volterra`, `one_comp_mm_elim_abs`, `sir`, `soil_incubation`.
  FlatPPL has no ODE solver.
- Cholesky correlation factors: `dogs_nonhierarchical`, `hier_2pl`,
  `kronecker_gp`. FlatPPL has no set or chart for them.
- A second parameterization of a ported model on the same data: the centered
  radon twins, `eight_schools_noncentered`, `log10earn_height`,
  `logearn_interaction_z`, `kidscore_interaction_c2`, `kidscore_interaction_z`,
  `wells_dist100_model`, `wells_interaction_c_model`.
- Exact duplicates: `covid19imperial_v3` on `ecdc0401` and `ecdc0501`. Its Stan
  file is byte-identical to `covid19imperial_v2`.
- Data over 5 MB: `mnist-nn_rbm1bJ100`, `prostate-logistic_regression_rhs`.
- No Stan source at this revision: `M0_data-M0_model_discrete`.

The imported Stan sources and data retain posteriorDB's [BSD license](LICENSE).
