# ATLAS inclusive top-pair cross section

Source: [ATLAS, arXiv:2006.13076](https://arxiv.org/abs/2006.13076),
Phys. Lett. B 810 (2020) 135797.
Original data: [HEPData record](https://doi.org/10.17182/hepdata.95748).
The archive was retrieved from the [pinned ML_LHClikelihoods mirror](https://github.com/hreyes91/ML_LHClikelihoods/blob/3cffc20c5979d760e6eba998fd5d96a98d502e5b/data/2006.13076.tar.gz).

`pyhf.json` is `2006.13076/likelihood_ttbar_ljets_13TeV_inclusive.json`,
reserialized without changing its model or observations: 3 channels, 37 bins,
211 parameters. The released JSON uses Gaussian MC-stat constraints, unlike
the paper's Poisson fit constraints. This fixture tests the released JSON.

`test.json` records the archive hash, source path, suggested parameter start,
ten shifts, and absolute log-likelihoods from pyhf 0.7.6 with its default JAX backend.
Its absolute tolerance is `1e-8`, with no relative tolerance.
The [corpus README](../README.md) documents the independent arithmetic check
that justifies this exception for million-count Poisson terms.
