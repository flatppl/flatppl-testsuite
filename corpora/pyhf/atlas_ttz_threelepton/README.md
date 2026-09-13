# ATLAS ttZ, three-lepton fit

[ATLAS, arXiv:2103.12603](https://arxiv.org/abs/2103.12603),
Eur. Phys. J. C 81 (2021) 737.

Source: [HEPData likelihood archive](https://doi.org/10.17182/hepdata.100351.v2/r1),
[resource 2258588](https://www.hepdata.net/record/resource/2258588?view=true).
The complete `likelihood_ttbarZ_13TeV/3L_workspace.json` is unchanged.
This is the published three-lepton fit, not the combined three- and four-lepton fit.
It has three channels, three bins, and 197 parameters. No signal patch is needed.

The file-specific [rights record](https://api.datacite.org/dois/10.17182/hepdata.100351.v2%2Fr1)
declares [CC0-1.0](https://creativecommons.org/publicdomain/zero/1.0/legalcode).
`test.json` records the archive hash and member.

Nine points use suggested initial values and bounded POI, scalar-nuisance,
per-bin, and mixed shifts. Fixed parameters remain fixed. The oracle uses
`pyhf.set_backend("jax")` with no precision override. All scores are absolute
log-likelihoods at `atol=1e-9`, `rtol=0`.
