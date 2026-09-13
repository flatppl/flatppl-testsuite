# ATLAS top–photon FCNC: left-handed tuγ

[ATLAS, arXiv:2205.02537](https://arxiv.org/abs/2205.02537),
Phys. Lett. B 842 (2023) 137379.

Source: [HEPData workspace_tugamma_LH.json](https://doi.org/10.17182/hepdata.129959.v1/r1),
downloaded directly from [resource 2916834](https://www.hepdata.net/record/resource/2916834?view=true).
This is the complete unblinded three-channel workspace, not a fitted or simplified model.
All observations, samples, and modifiers remain unchanged.

The file-specific [DataCite rights record](https://api.datacite.org/dois/10.17182/hepdata.129959.v1%2Fr1)
declares [CC0-1.0](https://creativecommons.org/publicdomain/zero/1.0/legalcode).
`test.json` records the source hash and license links.

The oracle uses pyhf 0.7.6 with its default JAX backend: the suggested initial point plus
POI, nuisance, template, per-bin, and mixed shifts within suggested bounds.
The suggested signal strength is zero. Fixed parameters remain fixed.
All checks compare absolute log-likelihoods, including auxiliary constraints.

Signal-strength shifts use ±0.01, including in the mixed points. The generic
−0.5 step gives negative Poisson means and `NaN` in pyhf despite lying within
the suggested fit bounds. Both ±0.01 points are finite; nuisance steps remain
±0.5 widths and the template step remains ±1.5. No model term or tolerance changes.
