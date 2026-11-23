# Matched posteriorDB catalogue points

These 24 fixtures contain absolute float64 log densities and full constrained gradients at the three matched points for each posterior. Each record retains its original field names and shapes. The points are ordered as the declared input, the negative perturbation and the positive perturbation.

`expected` comes from BridgeStan 2.9.0's `log_density`, including the prior normalization below. `expected_grad` comes from `log_density_gradient`. Both calls use `propto=False` and `jacobian=False`. Each fixture records both raw Stan primal values, the original Stan and data SHA-256 digests, the FlatPPL model and query digests, and a digest for every input field at every point.

Stan returns derivatives in its free coordinates even when the density has no chart Jacobian. The fixtures use real, positive and unit-interval parameters. For a constrained value `x`, divide its free derivative by 1, `x` or `x * (1 - x)`, respectively. Regeneration checks both directions of these charts against BridgeStan.

FlatPPL normalizes each half-Normal, half-Cauchy or half-Student-t prior listed under `normalized_half_priors`. Every scalar, including each member of a vector prior, contributes `log(2)` above Stan's target. In the Imperial models the positive Normal(1, 0.1) prior on `ifr_noise` also contributes `-log(Phi(10))` per country. The Normal term for `mu` enters with its original log weight. Neither correction changes these derivatives.

The soil model has a linear two-pool ODE. Its fixture retains both values and the full gradient from the original RK45 call under `provenance.original_solver`. For the primary reference the same Stan ODE uses relative tolerance `1e-12`, absolute tolerance `1e-14` and a maximum of one million steps. Tightening the tolerances to `1e-13` and `1e-15` changed the three standalone scores by at most `1.27e-11` and the gradient components by at most `3.13e-10`. The Poisson GP retains posteriorDB's covariance jitter of `1e-10`. Its nearly singular Cholesky factor can move the final digits of the latent-field derivatives across platforms.

## Reproduce the values

The Stan programs and zipped JSON observations come from [posteriorDB at 91027f9](https://github.com/stan-dev/posteriordb/tree/91027f9cdd13b7e76283fec3f2898f59d4b83ec4/posterior_database). The FlatPPL source strings come from catalogue revision `79fbd188c3c1dcd0bbda8b93d2f8a1a25a0ffecf`. The [published measurements file](https://ppl.bench.siccha.de/data/measurements.json) has SHA-256 `f0233f8695b0ed9868cea3f0ee36522173205444f270e445ef5e5f97726a24d0`. Its 24 embedded models use `data.json` and the accompanying [record query](score.flatppl).

With a posteriorDB checkout at the specified revision, run the following from the testsuite root. `DB` is its `posterior_database` directory. The first command regenerates and compares each reference using the existing `bridge` environment. Pass `--write` to replace the frozen numerical values.

```sh
cases=corpora/posteriordb/catalogue
pixi run -e bridge python "$cases/regenerate.py" "$DB" .stan-cache/catalogue "$cases"/*.json
```

Use an environment containing the FlatPPL Python package and its JAX/Enzyme runtime for the public query check. It accepts dashboard snapshots with matching model, query and observation hashes, verifies the parameter hashes and compares the standalone score, AD primal and every derivative. The catalogue's existing comparison uses absolute and relative tolerances of `1e-8`. A GPU run uses the GPU backend and `--target gpu`.

```sh
curl -fL https://ppl.bench.siccha.de/data/measurements.json -o measurements.json
JAX_PLATFORMS=cpu python "$cases/check_flatppl.py" measurements.json "$DB" "$cases"/*.json
```
