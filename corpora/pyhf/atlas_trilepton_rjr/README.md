# ATLAS three-lepton RJ-mimic

[ATLAS, arXiv:1912.08479](https://arxiv.org/abs/1912.08479),
Phys. Rev. D 101 (2020) 072001.

Source: [HEPData likelihood archive](https://doi.org/10.17182/hepdata.91127.v2/r3),
downloaded from [resource 1404698](https://www.hepdata.net/record/resource/1404698?view=true).
The workspace applies `patchset.json` entry `ERJR_500p0_0p0` to `BkgOnly.json`,
the archive README's example. Pyhf verifies the published background digest.
All four channels and their observations, samples, and modifiers remain intact.

The archive-specific [DataCite rights record](https://api.datacite.org/dois/10.17182/hepdata.91127.v2%2Fr3)
declares [CC0-1.0](https://creativecommons.org/publicdomain/zero/1.0/legalcode).
`test.json` records the archive hash and selected signal patch.

The oracle uses pyhf 0.7.6's default JAX backend. Eleven points cover the suggested
initial values and bounded POI, scalar-nuisance, template, per-bin, and mixed shifts.
The measurement's fixed luminosity and background scale remain fixed.
All checks compare absolute log-likelihoods, with `atol=1e-9` and `rtol=0`.
