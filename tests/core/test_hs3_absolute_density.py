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

Two of the three conversions are covered: `gaussian` here and `product` below.

`histfactory` is NOT covered, and the honest reason is effort, not soundness -- an
earlier draft of this docstring claimed the ROOT `Sum log(n_k!)` convention made
it unsound to check absolutely, which is wrong: that convention only matters when
comparing against ROOT, not against an independently written Poisson-product
oracle. It is simply not yet done.
"""
from __future__ import annotations

import math
from pathlib import Path

import pytest
from scipy.stats import norm

from flatppl_testsuite.unified import detjs_exec as ex

_CORPORA = Path(__file__).resolve().parents[2] / "corpora"
_GAUSSIAN = _CORPORA / "hs3" / "conversions" / "gaussian"

# The nominal point the model's own `log_likelihood` binding evaluates at.
_MU = 0.0
_SIGMA = 1.0


def _gaussian_observation() -> float:
    """The single observation, READ from the model rather than restated -- so a
    future dataset edit fails this test instead of inviting an oracle 'fix'."""
    import re
    src = (_GAUSSIAN / "gaussian.flatppl").read_text()
    m = re.search(r"obs_gaussian_channel = table\(x = \[([^\]]*)\]\)", src)
    assert m, "could not read obs_gaussian_channel from the model"
    xs = [float(v) for v in m.group(1).split(",") if v.strip()]
    assert len(xs) == 1, f"expected a single observation, model has {len(xs)}"
    return xs[0]


_X = _gaussian_observation()


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


# --- product conversion: a second absolute anchor, covering normalize/logweighted --
#
# Reviewed claim that this was untestable ("the normalising integral's domain is
# ambiguous") was WRONG, and is corrected here. `prod = normalize(logweighted(x ->
# logdensityof(g2, x), g1))` normalises over g1's OWN base measure, i.e. R --
# `toy_domain`/`default_domain` are separate metadata bindings that never enter the
# measure expression. So Z is the Gaussian-product identity in closed form:
#
#     Z = \int N(x|mu1,s1) N(x|mu2,s2) dx = N(mu1 | mu2, sqrt(s1^2 + s2^2))
#
# This anchor matters more than the gaussian one: it covers the
# normalize/logweighted composition, which is where a constant-offset regression
# (a dropped or doubled log-normaliser) is most likely to hide, and every other
# hs3 check would be blind to it.
_PRODUCT = _CORPORA / "hs3" / "conversions" / "product"
_P = {"mu1": 0.0, "sigma1": 1.0, "mu2": 1.0, "sigma2": 2.0}


def _product_data() -> list[float]:
    """Read the dataset from the model rather than restating it, so a future
    dataset edit cannot silently invite an oracle 'fix'."""
    import re
    src = (_PRODUCT / "product.flatppl").read_text()
    block = re.search(r"toy = table\((.*?)\n\)", src, re.S).group(1)
    return [float(v) for v in re.findall(r"-?\d+\.\d+(?:[eE][-+]?\d+)?", block)]


def _product_oracle() -> float:
    xs = _product_data()
    log_z = float(norm.logpdf(
        _P["mu1"], loc=_P["mu2"],
        scale=math.sqrt(_P["sigma1"] ** 2 + _P["sigma2"] ** 2),
    ))
    return sum(
        float(norm.logpdf(x, _P["mu1"], _P["sigma1"]))
        + float(norm.logpdf(x, _P["mu2"], _P["sigma2"]))
        - log_z
        for x in xs
    )


def test_product_dataset_is_the_expected_size():
    assert len(_product_data()) == 10, "iid(prod, 10) expects a 10-row toy table"


@pytest.mark.skipif(not ex.engine_available(), reason="det-js path unavailable")
def test_product_conversion_absolute_logdensity_matches_the_oracle():
    got = ex.score_binding(_PRODUCT / "product.flatppl", "log_likelihood")
    want = _product_oracle()
    assert got == pytest.approx(want, abs=1e-9, rel=1e-9), (
        f"absolute log-density drifted: got {got!r}, closed-form oracle {want!r}. "
        "This is the only check that would notice a dropped or doubled "
        "log-normaliser in the normalize/logweighted path."
    )
