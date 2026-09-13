# ATLAS direct stau search: combined regions

Source: [ATLAS, arXiv:1911.06660](https://arxiv.org/abs/1911.06660),
Phys. Rev. D 101 (2020) 032009.
The workspace comes from the [HEPData likelihood archive](https://doi.org/10.17182/hepdata.92006.v2/r2), released under CC0.

`pyhf.json` applies patch `DS_400_40_Staus` from `Region-combined/patchset.json`
to `Region-combined/BkgOnly.json` using `pyhf.PatchSet.apply`, including its digest check.
It retains every observation and constraint: 5 channels, 5 bins, 120 parameters.

`test.json` records the archive hash, source paths, suggested parameter start,
ten shifts, and absolute log-likelihoods from pyhf 0.7.6 with its default JAX backend.
