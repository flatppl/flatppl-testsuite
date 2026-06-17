### HS3 Examples

Three models from Appendix A of the [HS3 paper](https://arxiv.org/abs/2606.01760)
([spec](https://hep-statistics-serialization-standard.github.io/)),
reproduced in FlatPPL and verified against ROOT/RooFit.

#### Models

**Gaussian** (`gaussian.hs3`) — single observation $x = 1.27$ from $\mathrm{Normal}(\mu, \sigma)$.

**Product** (`product.hs3`) — 10-point toy dataset from a normalized product of two Gaussians:

$$p(x) \propto \mathrm{Normal}(x;\,\mu_1,\sigma_1)\cdot\mathrm{Normal}(x;\,\mu_2,\sigma_2)$$

**HistFactory** (`histfactory.hs3`) — 2-bin likelihood with signal, two backgrounds, three
normalization systematics (`normsys`), and two Barlow–Beeston MC-stat parameters (`staterror`):

$$
L(\mu,\boldsymbol{\theta}) = \prod_k \mathrm{Poisson}\left(n_k \mid \mu s_k + b_k\right)
  \cdot \prod_i \mathrm{Normal}(\theta_i \mid 0,1)
  \cdot \prod_j \mathrm{Poisson}(\gamma_j \tau_j \mid \tau_j)
$$

ROOT omits $\sum_k \log n_k!$ from its NLL (HEP convention); FlatPPL keeps it. That offset is
data-only, so $\Delta(\log L)$ matches exactly.

#### Representations

- [`gaussian.hs3`](gaussian/gaussian.hs3) / [`gaussian.flatppl`](gaussian/gaussian.flatppl)
- [`product.hs3`](product/product.hs3) / [`product.flatppl`](product/product.flatppl)
- [`histfactory.hs3`](histfactory/histfactory.hs3) / [`histfactory.flatppl`](histfactory/histfactory.flatppl)

#### Reproduction

Requires [pixi](https://pixi.sh). From this directory:

```
pixi run repro
```

Clones [flatppl-js](https://github.com/flatppl/flatppl-js) into `~/.cache/flatppl-js` on first
run. ROOT oracle steps are skipped if PyROOT is unavailable.

| Script | Purpose |
|--------|---------|
| `repro_hs3.sh` | Driver: engine setup, JS harness, ROOT oracle |
| `repro_hs3_js.cjs` | FlatPPL JS checks against ROOT oracle values |
| `<model>/<model>_root.py` | ROOT/RooFit oracle (ROOT ≥ 6.30) |
