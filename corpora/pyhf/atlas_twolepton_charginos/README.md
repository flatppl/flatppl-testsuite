# ATLAS two-lepton chargino search

[ATLAS, arXiv:1908.08215](https://arxiv.org/abs/1908.08215),
Eur. Phys. J. C 80 (2020) 123.

Source: [HEPData likelihood archive](https://doi.org/10.17182/hepdata.89413.v4/r5),
downloaded from [resource 2917462](https://www.hepdata.net/record/resource/2917462?view=true).
The workspace applies `C1C1WW_patchset.json` entry `C1C1_WW_300_50` to
`bkgonly.json`. Pyhf verifies the published background digest.
All 39 channels and their observations, samples, and modifiers remain intact.

The archive-specific [DataCite rights record](https://api.datacite.org/dois/10.17182/hepdata.89413.v4%2Fr5)
declares [CC0-1.0](https://creativecommons.org/publicdomain/zero/1.0/legalcode).
`test.json` records the archive hash and selected signal patch.

The oracle uses pyhf 0.7.6's default JAX backend. Eleven points cover the suggested
initial values and bounded POI, scalar-nuisance, template, per-bin, and mixed shifts.
Luminosity remains free, and the measurement's fixed background scale stays fixed.
All checks compare absolute log-likelihoods, with `atol=1e-9` and `rtol=0`.
