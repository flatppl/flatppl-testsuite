# ATLAS displaced leptons: combined

Source: [ATLAS, arXiv:2011.07812](https://arxiv.org/abs/2011.07812),
Phys. Rev. Lett. 127 (2021) 051802.
Original data: [HEPData record](https://doi.org/10.17182/hepdata.98796).
The archive was retrieved from the [pinned ML_LHClikelihoods mirror](https://github.com/hreyes91/ML_LHClikelihoods/blob/3cffc20c5979d760e6eba998fd5d96a98d502e5b/data/2011.07812.tar.gz).

`pyhf.json` applies patch `DisplacedLeptons_Comb_1000_10`
from `2011.07812/Comb_patchset.json` to `2011.07812/Comb_bkgonly.json`
using `pyhf.PatchSet.apply`, including its digest check.
It retains every observation and constraint: 3 channels, 3 bins, 16 parameters.

`test.json` records the archive hash, source paths, suggested parameter start,
six shifts, and absolute log-likelihoods from pyhf 0.7.6 with its default JAX backend.
There are no template or per-bin parameters, so those shift groups add no points.
