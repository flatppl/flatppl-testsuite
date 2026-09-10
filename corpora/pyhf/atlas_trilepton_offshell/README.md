# ATLAS three-lepton search: off-shell WZ

Source: [ATLAS, arXiv:2106.01676](https://arxiv.org/abs/2106.01676),
Eur. Phys. J. C 81 (2021) 1118.
Original data: [HEPData record](https://doi.org/10.17182/hepdata.95751).
The archive was retrieved from the [pinned ML_LHClikelihoods mirror](https://github.com/hreyes91/ML_LHClikelihoods/blob/3cffc20c5979d760e6eba998fd5d96a98d502e5b/data/2106.01676.tar.gz).

`pyhf.json` applies patch `CN_WZ_200_100_harmonised_offshell`
from `2106.01676/offshell_winobino_plus_patchset.json` to
`2106.01676/bkg_offshell.json` using `pyhf.PatchSet.apply`, including its digest check.
It retains every observation and constraint: 33 channels, 33 bins, 353 parameters.

`test.json` records the archive hash, source paths, suggested parameter start,
ten shifts, and absolute log-likelihoods from pyhf 0.7.6, NumPy float64.
