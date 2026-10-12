# Categorical targets

These three models were generated without edits by brms 2.23.2 at
[`c89c4c0`](https://github.com/paul-buerkner/brms/tree/c89c4c05ef52fefb4b78d5fb80df0cf6352f48b4)
with R 4.5.3. BridgeStan 2.9.0 / Stan 2.39.0 evaluated the normalized
targets with `propto=false, jacobian=false`. All coefficients are unconstrained
real values. Each `score.json` retains the raw Stan order, the query order,
parameter points and mapped gradients; `cases.json` records source/data hashes.

The first retained factor level is the reference category. `default` uses
centred category predictors and brms's Student-t intercept/Flat coefficient
priors. `weighted` combines a no-intercept category, a category-specific offset,
proper Normal priors and zero/fractional weights. `reordered` relabels categories
so their lexical predictor order differs from their response-level order.

`analytic.py` computes log-softmax likelihoods and closed-form coefficient
derivatives from the captured design matrices and prior parameters. It never
reads FlatPPL output or frozen expected scores. Ordinary regeneration needs
only the testsuite's existing NumPy/SciPy environment.

To regenerate the independent brms sources and data without changing the corpus:

```sh
Rscript --vanilla corpora/formula/oracles/categorical/generate.R /tmp/categorical-oracles
```

Compare the generated files with `*/model.stan` and `*/data.json`. Use the
usual Stan MCP compile/log-density operations with each `theta_unc` in
`score.json`; the explicit index permutation converts its raw gradients to
query order. No density or derivative offset is applied.
