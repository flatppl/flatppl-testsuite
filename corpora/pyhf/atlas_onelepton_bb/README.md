# ATLAS one-lepton plus bb search

Source: [ATLAS, arXiv:1909.09226](https://arxiv.org/abs/1909.09226),
Eur. Phys. J. C 80 (2020) 691.
Original data: [HEPData likelihood archive](https://doi.org/10.17182/hepdata.90607.v3/r3).
Files were retrieved from the [pinned SModelS mirror](https://github.com/SModelS/smodels-utils/tree/97eef9dc51d1b6fa6a258f6d92bb70c7670b458d/pyhfintegration/SUSY-2019-08_likelihoods).
The background digest matches the [published pyhf tutorial](https://pyhf.github.io/tutorial-dance-codas-2022/SerializationAndPatching.html).

`pyhf.json` applies patch `C1N2_Wh_hbb_700_400` from `patchset.json`
to `BkgOnly.json` using `pyhf.PatchSet.apply`, including its digest check.
It retains every observation and constraint: 8 channels, 14 bins, 126 parameters.

`test.json` records both file hashes, source paths, suggested parameter start,
ten shifts, and absolute log-likelihoods from pyhf 0.7.6 with its default JAX backend.
