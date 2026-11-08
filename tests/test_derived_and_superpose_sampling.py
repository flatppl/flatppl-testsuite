"""Sampling a derived value and a normalized superposition (§06, §07 `rand`, §13).

`rand(lawof(z), s)` for `z = 2x + 1` with `x ~ Binomial(10, q)` samples `x` and
returns `z`. `rand(normalize(superpose(weighted(3, Poisson(r)), Poisson(2))), s)`
picks the first component with probability 3/4. Each check compares the draws
with the exact pmf: the mean and variance, and the frequency of each value with
enough mass, within five standard errors."""

import math
from collections import Counter

import numpy as np
import pytest
from scipy import stats

from flatppl_testsuite.unified import stablehlo_exec as ex
from tests.test_unified import _gate_engine

N = 20000

DERIVED = """q = elementof(interval(0.0, 1.0))
x ~ Binomial(10, q)
z = x * 2 + 1
s = rnginit(0)
d, s2 = rand(lawof(z), s)
inputs = (s, q)
outputs = (d, s2)
"""

SUPERPOSE = """r = elementof(posreals)
m = superpose(weighted(3.0, Poisson(r)), Poisson(2.0))
s = rnginit(0)
d, s2 = rand(normalize(m), s)
inputs = (s, r)
outputs = (d, s2)
"""


def _check_against_pmf(draws, pmf):
    """Moments and per-value frequencies of integer draws against an exact pmf."""
    support = np.arange(0, 200)
    probabilities = np.array([pmf(k) for k in support])
    mean = float(probabilities @ support)
    variance = float(probabilities @ (support - mean) ** 2)
    assert abs(draws.mean() - mean) < 5 * math.sqrt(variance / len(draws))
    fourth = float(probabilities @ (support - mean) ** 4)
    assert abs(draws.var() - variance) < 5 * math.sqrt((fourth - variance**2) / len(draws))
    counts = Counter(draws.astype(int).tolist())
    assert set(counts) <= {int(k) for k in support[probabilities > 0]}
    for k, p in zip(support, probabilities):
        if p * len(draws) > 50:
            assert abs(counts.get(int(k), 0) / len(draws) - p) < 5 * math.sqrt(p * (1 - p) / len(draws)), k


def _draws(tmp_path, text, args):
    query = tmp_path / "query.flatppl"
    query.write_text(text)
    source = ex.emit(query, "sample", restrict_enzyme_compatible=False)
    return ex.samples(source, N, args).astype(float)


@pytest.mark.stablehlo_only
def test_a_derived_value_follows_its_pushforward_law(tmp_path):
    _gate_engine("stablehlo")
    draws = _draws(tmp_path, DERIVED, [0.3])
    _check_against_pmf(draws, lambda k: stats.binom.pmf((k - 1) / 2, 10, 0.3) if k % 2 == 1 else 0.0)


@pytest.mark.stablehlo_only
def test_a_normalized_superpose_draws_components_by_mass(tmp_path):
    _gate_engine("stablehlo")
    draws = _draws(tmp_path, SUPERPOSE, [5.0])
    _check_against_pmf(draws, lambda k: 0.75 * stats.poisson.pmf(k, 5.0) + 0.25 * stats.poisson.pmf(k, 2.0))
