# ATLAS displaced vertex plus muon

[ATLAS, arXiv:2003.11956](https://arxiv.org/abs/2003.11956),
Phys. Rev. D 102 (2020) 032006.

Source: [HEPData Likelihoods.zip](https://doi.org/10.17182/hepdata.91760.v2/r1),
downloaded directly from [resource 2514540](https://www.hepdata.net/record/resource/2514540?view=true).
Selected complete workspace:
`Likelihoods/DVPlusMuConfig/SRMET_StopRHadron_1000_100.json`.
It also equals the supplied `SRMET_bkgonly.json` after applying the published
`SRMET_patchset.json` patch `StopRHadron_1000_100` with pyhf's digest check.
The zero observed count and all six parameters remain unchanged.

The file-specific [DataCite rights record](https://api.datacite.org/dois/10.17182/hepdata.91760.v2%2Fr1)
and [archive landing page](https://www.hepdata.net/record/resource/2514540?landing_page=true)
declare [CC0-1.0](https://creativecommons.org/publicdomain/zero/1.0/legalcode).
`test.json` records archive and workspace hashes, the member path, and license links.

The oracle uses pyhf 0.7.6, NumPy float64: the suggested initial point plus
POI, scalar-nuisance, per-bin, and mixed shifts within suggested bounds.
All checks compare absolute log-likelihoods, including auxiliary constraints.
