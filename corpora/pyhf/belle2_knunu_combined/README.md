# Belle II B+ to K+ neutrinos: combined ITA and HTA

Measurement: [Belle II, arXiv:2311.14647](https://arxiv.org/abs/2311.14647).
Full likelihood release: [Belle II, arXiv:2507.12393](https://arxiv.org/abs/2507.12393),
Phys. Rev. D 112 (2025) 092016.
Original data: [HEPData record](https://doi.org/10.17182/hepdata.166082).
The workspace was retrieved from the [pinned author repository](https://github.com/lorenzennio/knunu-reinterpretation-mini/blob/710e304614ae47c044ec34f86c97b82b1f1ce956/scripts/data/combined_likelihood.json).

`pyhf.json` contains that complete published combined likelihood, not the
repository's derived WET reinterpretation model. No signal patch is applied.
It retains every observation and constraint: 5 channels, 30 bins, 232 parameters.

`test.json` records the retrieved file hash, source path, suggested parameter
start, ten shifts, and absolute log-likelihoods from pyhf 0.7.6 with its default JAX backend.
The retrieval hash identifies the author-copy bytes; it does not assert
independent byte identity with the unavailable upstream download.
