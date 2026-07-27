"""At least one hs3 model has its ABSOLUTE log-density pinned, not just 2DeltaNLL.

Every numeric check in `corpora/hs3/` is a `twice_delta_nll*` quantity: a
difference against a reference point, with a 0.0 at that point by construction.
Those are deliberately offset-invariant -- it is what lets HistFactory's ROOT
`Sum log(n_k!)` convention drop out -- but it means a converter or engine change
that shifted EVERY hs3 log-density by the same constant would pass all 19 checks.

The legacy suite had exactly one absolute check (`test_score.py`'s
`-1.7253885332`) and it was deleted with the gates, leaving nothing to constrain
absolute normalisation.

The oracle here is closed form, derived independently of any engine: the gaussian
conversion's data is a single observation, so

    logdensityof(obs, record(mu = 0, sigma = 1)) = log N(1.27 | 0, 1)
                                                 = -0.5*log(2*pi) - 0.5*1.27**2

Only the gaussian conversion is covered. `product` normalises a Gaussian product
whose normalising integral's domain is ambiguous from the source, and
`histfactory` carries the ROOT factorial-convention offset that 2DeltaNLL exists
to cancel -- for both, a confidently-wrong oracle would be worse than none, so
they stay uncovered and are called out rather than guessed at.
"""
from __future__ import annotations

import math
from pathlib import Path

import pytest
from scipy.stats import norm

from flatppl_testsuite.unified import detjs_exec as ex

_CORPORA = Path(__file__).resolve().parents[2] / "corpora"
_GAUSSIAN = _CORPORA / "hs3" / "conversions" / "gaussian"

# The single observation in the fixture's dataset, and the nominal point the
# model's own `log_likelihood` binding evaluates at.
_X = 1.27
_MU = 0.0
_SIGMA = 1.0


def _oracle() -> float:
    """Closed-form, independent of any FlatPPL engine."""
    return -0.5 * math.log(2.0 * math.pi) - 0.5 * ((_X - _MU) / _SIGMA) ** 2


def test_the_oracle_agrees_with_scipy():
    """Two independent routes to the same number, so a slip in either is caught."""
    assert _oracle() == pytest.approx(
        float(norm.logpdf(_X, loc=_MU, scale=_SIGMA)), abs=1e-12
    )


def test_the_oracle_reproduces_the_historical_frozen_value():
    """The value the deleted legacy check asserted, re-derived rather than copied."""
    assert _oracle() == pytest.approx(-1.7253885332046727, abs=1e-12)


@pytest.mark.skipif(not ex.engine_available(), reason="det-js path unavailable")
def test_gaussian_conversion_absolute_logdensity_matches_the_oracle():
    """The model binds `log_likelihood = logdensityof(obs, record(mu=0, sigma=1))`
    already, so this scores that binding directly -- an ABSOLUTE density, with no
    reference point subtracted."""
    got = ex.score_binding(_GAUSSIAN / "gaussian.flatppl", "log_likelihood")
    want = _oracle()
    assert got == pytest.approx(want, abs=1e-9, rel=1e-9), (
        f"absolute log-density drifted: got {got!r}, closed-form oracle {want!r}. "
        "Every other hs3 check is offset-invariant, so this is the only one that "
        "would notice a constant shift."
    )


@pytest.mark.skipif(not ex.engine_available(), reason="det-js path unavailable")
def test_an_offset_shift_would_be_caught():
    """Guards the guard: a constant shift -- the exact regression every
    twice_delta_nll check is blind to -- must fail this comparison."""
    got = ex.score_binding(_GAUSSIAN / "gaussian.flatppl", "log_likelihood")
    shifted = got + 0.5
    assert shifted != pytest.approx(_oracle(), abs=1e-9, rel=1e-9)
