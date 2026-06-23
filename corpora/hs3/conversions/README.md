### HS3 Examples

Three models from Appendix A of the [HS3 paper](https://arxiv.org/abs/2606.01760)
([spec](https://hep-statistics-serialization-standard.github.io/)),
reproduced in FlatPPL and verified against ROOT/RooFit.

#### Models

**Gaussian** (`gaussian.hs3.json`) — single observation $x = 1.27$ from $\mathrm{Normal}(\mu, \sigma)$.

**Product** (`product.hs3.json`) — 10-point toy dataset from a normalized product of two Gaussians:

$$p(x) \propto \mathrm{Normal}(x;\,\mu_1,\sigma_1)\cdot\mathrm{Normal}(x;\,\mu_2,\sigma_2)$$

**HistFactory** (`histfactory.hs3.json`) — 2-bin likelihood with signal, two backgrounds, three
normalization systematics (`normsys`), and two Barlow–Beeston MC-stat parameters (`staterror`):

$$
L(\mu,\boldsymbol{\theta}) = \prod_k \mathrm{Poisson}\left(n_k \mid \mu s_k + b_k\right)
  \cdot \prod_i \mathrm{Normal}(\theta_i \mid 0,1)
  \cdot \prod_j \mathrm{Poisson}(\gamma_j \tau_j \mid \tau_j)
$$

ROOT omits $\sum_k \log n_k!$ from its NLL (HEP convention); FlatPPL keeps it. That offset is
data-only, so $\Delta(\log L)$ matches exactly.

#### Representations

- [`gaussian.hs3.json`](gaussian/gaussian.hs3.json) / [`gaussian.flatppl`](gaussian/gaussian.flatppl)
- [`product.hs3.json`](product/product.hs3.json) / [`product.flatppl`](product/product.flatppl)
- [`histfactory.hs3.json`](histfactory/histfactory.hs3.json) / [`histfactory.flatppl`](histfactory/histfactory.flatppl)

#### Verification

Each model ships a frozen ROOT 2ΔNLL vector in `<model>/expected.json` (check kind
`twice_delta_nll_points`). The harness scores the committed `<model>.flatppl` with the FlatPPL
engine over the same θ points and compares — the same score+compare loop as the `fixtures/`
corpus, with no hand-copied numbers. Run from the repo root:

```
pixi run hs3                 # score every fixture + conversion, print comparison tables
pixi run test                # the same checks, gated by pytest
```

The frozen vectors are **generated from ROOT**, not the sibling engine. To regenerate after a
converter change (requires ROOT ≥ 6.30):

```
pixi run -e root gen-conversions
```

| Script | Purpose |
|--------|---------|
| `gen_expected.py` | Regenerate each `expected.json` from the ROOT/RooFit oracle |
| `regen.py` | Regenerate each `<model>.flatppl` from `<model>.hs3.json` via the converter |
| `<model>/<model>_root.py` | Standalone ROOT/RooFit oracle table (ROOT ≥ 6.30) |
