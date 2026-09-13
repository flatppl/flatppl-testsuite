# ATLAS single-top plus photon: particle-level workspace

Source: [ATLAS, arXiv:2302.01283](https://arxiv.org/abs/2302.01283).
Original data: [HEPData record](https://doi.org/10.17182/hepdata.134244).
The archive was retrieved from the [pinned ML_LHClikelihoods mirror](https://github.com/hreyes91/ML_LHClikelihoods/blob/3cffc20c5979d760e6eba998fd5d96a98d502e5b/data/2302.01283.tar.gz).

`pyhf.json` contains `2302.01283/workspace_particleLevel.json` without a signal
patch. It retains every observation and constraint: 4 channels, 51 bins,
333 parameters. The archive's separate parton-level workspace is not used.

`test.json` records the archive hash, source path, suggested parameter start,
ten shifts, and absolute log-likelihoods from pyhf 0.7.6 with its default JAX backend.
