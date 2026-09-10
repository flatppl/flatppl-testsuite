# ATLAS sbottom search: Region A

Source: [ATLAS, arXiv:1908.03122](https://arxiv.org/abs/1908.03122),
JHEP 12 (2019) 060.
The workspace comes from the [HEPData likelihood archive](https://doi.org/10.17182/hepdata.89408.v3/r2), released under CC0.

`pyhf.json` applies patch `sbottom_1300_850_60` from `RegionA/patchset.json`
to `RegionA/BkgOnly.json` using `pyhf.PatchSet.apply`, including its digest check.
It retains every observation and constraint: 2 channels, 6 bins, 64 parameters.

`test.json` records the archive hash, source paths, suggested parameter start,
ten shifts, and absolute log-likelihoods from pyhf 0.7.6, NumPy float64.
