# Example corpus

These directories vendor the current worked models from
[flatppl-examples](https://github.com/flatppl/flatppl-examples).
Each `test.json` records the source revision, path and SHA-256 hash.
The query declares its inputs and outputs explicitly. Independent NumPy/SciPy
oracles supply the frozen values and gradients.

The corpus covers the posterior examples, both HGF recurrences, the Dalitz
amplitude, the `minimal` predictive law, and the deterministic `aggregates`
showcase. Bayesian inference variants 3 and 4 include their two imported helper
modules alongside the source. The helper files have no separate case here.

Most cases run through both det-js and StableHLO. Two require StableHLO:

- Dalitz uses nested metric contractions and positional Cartesian-product
  membership that det-js does not yet execute.
- `minimal` uses explicit numerical integration over its shared latent variance.
  Its source also fixes the keyword-only call to `f_root`.

The per-case `engines` and `engine_limits` fields state these limits.
No example case uses `allow_skip`. The corpus roster tests pin every case and
engine, and gradient twins must retain identical model and query sources.

Run `pixi run test` for the full suite or `pixi run unified` for corpus cases.
Use `pixi run -e stablehlo regen corpora/examples/<case>` to regenerate a frozen
oracle after an intentional model change. The oracle must remain independent
of the implementation being tested.
