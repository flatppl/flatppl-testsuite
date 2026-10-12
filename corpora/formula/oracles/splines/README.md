# Thin-plate Gaussian oracle

The paired `thinplate_gaussian` corpus cases convert `y ~ s(x, k=6)` on every
run. They retain a normalized actual brms 2.23.2 Stan model, its mgcv 1.9.4
design and the executed score and raw unconstrained gradient. Tool provenance
is shared with [the other brms oracles](../provenance.json).

The point order is `Intercept, bs[1], z[1:4], sds, sigma`. The first two and
last two coordinates equal Stan's. Native penalized coefficients satisfy
`z_native = Q * z_Stan`, with the exact signed permutation `Q` in `mapping.json`.
Thus `Z_native * Q = Z_Stan` and the standard Normal prior is unchanged.
For this fixture, the evaluated native basis agrees with that map to 6e-12;
no density offset is added. A future valid basis rotation must update the
explicit map, rather than the captured Stan score or the prior.

`jacobian=false` removes the density correction, but Stan still differentiates
its unconstrained coordinates. Divide the last two raw derivatives by `sds`
and `sigma`, then multiply the four `z` derivatives by `Q`. `score.json` retains
both the raw result and the constrained native mapping. `analytic.py` independently
uses the captured brms design, Normal and Exponential densities and analytic
derivatives. It reads neither FlatPPL output nor the frozen Stan score.

Regenerate source, data and prediction in a pinned R oracle environment:

```sh
Rscript --vanilla corpora/formula/oracles/splines/generate.R /tmp/spline-oracle
```

The script does not install packages, call FlatPPL or overwrite the corpus.
Compare its outputs with the committed files. The Stan source reproduces
byte-for-byte. The historical data passed through R's `dput`/`dget`; fresh
numeric values agree within 2e-16 rather than necessarily retaining the same
last bit. Keep the original data with its captured score. Reproduce the score using the
[Stan MCP procedure](../README.md), passing the saved `theta_unc` with
`propto=false, jacobian=false`. Offline density/gradient regeneration uses
each case's `test.py` through the existing testsuite `regen` command.

`prediction.json` comes from `s2rPred` on the retained training smooth,
including extrapolation; the five prediction rows are never refitted.
Permanent native API coverage lives in flatppl-rust's
`crates/formula/tests/splines.rs`: conversion, recipe JSON roundtrip and
response-free `prepare_data` compare the fixed column and full penalized
covariance to these same pinned mgcv/brms rules. The testsuite CLI corpus
checks training scores; it does not pretend that reconverting prediction
rows would test retained preprocessing.
