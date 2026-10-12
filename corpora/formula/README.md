# Formula runtime oracles

Each `model.formula.json` is a typed converter input. The harness converts it at
test time, appends `query.flatppl`, and checks the resulting posterior. Generated
FlatPPL is deliberately not a fixture.

The thirteen brms models have paired density and `_gradient` directories. Densities
run through deterministic JS and StableHLO; gradients run through StableHLO and
Enzyme. Both use the same `theta` order recorded in `oracle.json`.

| Cases | Contract |
| --- | --- |
| `family_*` | Eight family defaults, centered predictors, link functions, normalized auxiliary priors, binomial trials |
| `expressions_offset` | `I(x^2)`, `log(z)`, and `offset(log(z))` with explicit Normal priors |
| `default_log_offset` | Gaussian log link, a negative response, and the offset adjustment to the default intercept prior |
| `slopes_correlated` | Non-centered intercept and slope, matrix standard Normals, half-Normal scales, and `LKJCholesky(2, 1)` |
| `slopes_independent` | Independent non-centered intercept and slope with the same group coding |
| `group_priors` | Two correlated groups, group/coefficient prior selectors, nonzero-location truncated Student-t scales, and distinct LKJ shapes |

The earlier interaction cases retain their R `model.matrix` oracles. The brms
cases additionally preserve actual normalized Stan sources, data, scores,
gradients, and exact tool versions in [oracles](oracles/README.md).

`test.py` regenerates from the shared independent SciPy densities and analytic
derivatives in `oracles/analytic.py`. It uses the captured brms matrix and prior
parameters, never FlatPPL output or frozen expected values. Run regeneration from
the complete checkout so this shared oracle remains available. Each case is
self-contained for runtime conversion and scoring.
