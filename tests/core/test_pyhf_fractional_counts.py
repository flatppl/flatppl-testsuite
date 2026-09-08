"""The fractional-count rule is PER CHANNEL, and the numeric row only half sees it.

`corpora/pyhf/asimov_fractional_counts` holds an integer channel `ca` beside a
mixed integer/fractional channel `cb`, so the converter must emit `Poisson` for
one and `hepphys.ContinuedPoisson` for the other from a single workspace.

Its frozen `logpdf_points` row catches one direction only. Flipping `cb` back to
`Poisson` makes the density `-inf` (spec 08 gives `Poisson` the support
`nonnegintegers`), so that regression fails the row. Flipping `ca` to
`ContinuedPoisson` moves NO number: the two densities agree on a whole number
(spec 09 continues the Poisson factorial with the gamma function), so a converter
that scored every channel with the continued form would pass the numeric gate
while dropping `Poisson` from the emitted corpus entirely. That direction needs a
structural check.
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


def test_the_count_distribution_is_chosen_per_channel():
    heads = dict(_MODEL.findall(convert(_FIXTURE, source_format="pyhf")))
    assert heads == {
        "ca": "Poisson",
        "cb": "hepphys.ContinuedPoisson",
    }
