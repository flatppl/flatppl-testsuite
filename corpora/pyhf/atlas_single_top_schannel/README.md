# ATLAS single-top s-channel

[ATLAS, arXiv:2209.08990](https://arxiv.org/abs/2209.08990),
JHEP 06 (2023) 191.

Source: [HEPData workspace.json](https://doi.org/10.17182/hepdata.133620.v1/r1),
downloaded directly from [resource 3386157](https://www.hepdata.net/record/resource/3386157?view=true).
The complete workspace retains all observations, samples, and modifiers.
No signal patch or model simplification is applied.

The file-specific [DataCite rights record](https://api.datacite.org/dois/10.17182/hepdata.133620.v1%2Fr1)
declares [CC0-1.0](https://creativecommons.org/publicdomain/zero/1.0/legalcode).
`test.json` records the source hash and license links.

The oracle uses pyhf 0.7.6, NumPy float64: the suggested initial point plus
POI, nuisance, template, per-bin, and mixed shifts within suggested bounds.
Fixed parameters remain fixed. All checks compare absolute log-likelihoods.
The original measurement label `cross-section_Measurement` remains in the
converted module's documentation, with a valid collision-free metadata binding.
