# ATLAS boosted hadronic bosons: WW

Source: [ATLAS, arXiv:2108.07586](https://arxiv.org/abs/2108.07586),
Phys. Rev. D 104 (2021) 112010.
Original data: [HEPData record](https://doi.org/10.17182/hepdata.104458).
The archive was retrieved from the [pinned ML_LHClikelihoods mirror](https://github.com/hreyes91/ML_LHClikelihoods/blob/3cffc20c5979d760e6eba998fd5d96a98d502e5b/data/2108.07586.tar.gz).

`pyhf.json` applies patch `Sig_sig_SM_C1C1_WW_1000_0`
from `2108.07586/patchset_SM_C1C1_WW.json` to
`2108.07586/BGOnly_SM_C1C1_WW.json` using `pyhf.PatchSet.apply`, including its digest check.
It retains every observation and constraint: 2 channels, 2 bins, 82 parameters.

`test.json` records the archive hash, source paths, suggested parameter start,
ten shifts, and absolute log-likelihoods from pyhf 0.7.6, NumPy float64.
