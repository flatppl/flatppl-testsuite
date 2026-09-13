# ATLAS four-top multilepton evidence

[ATLAS, arXiv:2007.14858](https://arxiv.org/abs/2007.14858),
Eur. Phys. J. C 80 (2020) 1085.

Source: [HEPData complete workspace](https://doi.org/10.17182/hepdata.100170.v1/r1),
[resource 2007736](https://www.hepdata.net/record/resource/2007736?view=true).
The published `workspace.json` has five channels, 31 bins, and 249 parameters.
All observations, samples, modifiers, and measurement settings remain unchanged.
No archive assembly or signal patch is needed.

The file-specific [rights record](https://api.datacite.org/dois/10.17182/hepdata.100170.v1%2Fr1)
declares [CC0-1.0](https://creativecommons.org/publicdomain/zero/1.0/legalcode).
`test.json` records the downloaded file's hash.

Eleven points use suggested initial values and bounded POI, scalar-nuisance,
template, per-bin, and mixed shifts. Fixed parameters remain fixed. The oracle
uses `pyhf.set_backend("jax")` with no precision override. All scores are absolute
log-likelihoods at `atol=1e-9`, `rtol=0`.
