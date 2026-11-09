"""Every pyhf channel scores `hepphys.ContinuedPoisson`, integer or fractional.

`corpora/pyhf/asimov_fractional_counts` holds an integer channel `ca` beside a
mixed integer/fractional channel `cb`. The converter emits
`hepphys.ContinuedPoisson` for both: pyhf and ROOT score every count with the
continuous lgamma extension, which spec 09 defines, while spec 08 gives core
`Poisson` the domain `integers`.

Its frozen `logpdf_points` row catches one direction only. Emitting `Poisson`
for `cb` makes the density `-inf`, so that regression fails the row. Emitting
`Poisson` for `ca` moves NO number: the two densities agree on a whole number.
That direction needs this structural check.
"""
from __future__ import annotations

import re
from pathlib import Path

from flatppl_testsuite.formats.hs3.importer import convert

_FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "corpora" / "pyhf" / "asimov_fractional_counts" / "pyhf.json"
)

# `<channel>_model = functionof(<head>.(<channel>_expected))`
_MODEL = re.compile(
    r"^(\w+)_model = functionof\((.+?)\.\(\1_expected\)\)$", re.M
)


def test_every_channel_scores_the_continued_poisson():
    heads = dict(_MODEL.findall(convert(_FIXTURE, source_format="pyhf")))
    assert heads == {
        "ca": "hepphys.ContinuedPoisson",
        "cb": "hepphys.ContinuedPoisson",
    }
