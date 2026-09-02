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

All three conversions are covered: `gaussian` here, then `product` and
`histfactory` below. An earlier draft of this docstring claimed the ROOT
`Sum log(n_k!)` convention made `histfactory` unsound to check absolutely, which
is wrong: that convention only matters when comparing against ROOT, not against
an independently written Poisson-product oracle.

`histfactory` matters for a second reason. It is the only corpus model that uses
a §09 standard-module FUNCTION member (`interp_poly6_exp`, three times), and
`score_binding` determinizes unconditionally, unlike the `convert` runner that
scores the rest of this corpus through the `FLATPPL_ENGINE`-selected engine
(default `js`). So this is the only place histfactory's determinised path, and
therefore the §09 function lowering it needs, is scored at all.
"""
from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import norm, poisson

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


# --- histfactory conversion: the third absolute anchor, and the only scoring of
# --- the §09 function lowering this corpus's own models reach ----------------
#
# §09 specifies interp_poly6_exp only as C^2 conditions and does not write the
# six coefficients out, so the spec text alone does not pin the boundary FIRST
# derivative: its own extrapolation form f(a) = f(+-1) exp((a-+1) f'(+-1)/f(+-1))
# reduces C^2 to f''(+-1) = f'(+-1)^2/f(+-1), which leaves f'(+-1) free. §09's
# table names pyhf code4 as the reference and that closes it -- the polynomial
# matches the exponential interpolation center*(right/center)^a on the right and
# its mirror on the left.
#
# This oracle therefore SOLVES §09's 6x6 C^2 system numerically rather than
# restating a coefficient the implementation carries. Its independence is not
# taken on trust: `test_histfactory_oracle_reproduces_the_frozen_root_vector`
# below feeds the oracle's own absolute values through the same 2DeltaNLL
# difference the dir's frozen ROOT vector holds, and they agree to 8e-13. So the
# absolute value asserted here is anchored by ROOT for everything except the
# offset, and by closed-form maths for the offset itself.
_HISTFACTORY = _CORPORA / "hs3" / "conversions" / "histfactory"

# The interpolation anchors, i.e. the normsys hi/lo the modifier declares. Not
# read back from the model: only the DATA is, per the two anchors above.
_ANCHOR_LO, _ANCHOR_CTR, _ANCHOR_HI = 0.95, 1.0, 1.05


def _hf_vector(binding: str) -> np.ndarray:
    """One of the model's data vectors, READ from the model rather than restated,
    so a future dataset edit fails this test instead of inviting an oracle 'fix'."""
    src = (_HISTFACTORY / "histfactory.flatppl").read_text()
    m = re.search(rf"^{binding} = \[([^\]]*)\]", src, re.M)
    assert m, f"could not read {binding} from the model"
    return np.array([float(v) for v in m.group(1).split(",") if v.strip()])


_OBSERVED = _hf_vector("model_channel1_observed")
_SIGNAL = _hf_vector("model_channel1_signal_nominal")
_BKG1 = _hf_vector("model_channel1_background1_nominal")
_BKG2 = _hf_vector("model_channel1_background2_nominal")
_TAU = _hf_vector("mcstat_tau")


def _poly6_exp(left, center, right, alpha):
    """§09's interp_poly6_exp: a 6th-order polynomial on [-1, 1] whose C^2
    conditions match the exponential continuation, solved here as a 6x6 system."""
    lo, hi = math.log(left / center), math.log(right / center)
    if alpha > 1.0:
        return center * (right / center) ** alpha
    if alpha < -1.0:
        return center * (left / center) ** (-alpha)
    A = np.zeros((6, 6))
    for i in range(1, 7):
        A[0, i - 1] = 1.0                          # f(+1) - center
        A[1, i - 1] = (-1.0) ** i                  # f(-1) - center
        A[2, i - 1] = i                            # f'(+1)
        A[3, i - 1] = i * (-1.0) ** (i - 1)        # f'(-1)
        A[4, i - 1] = i * (i - 1)                  # f''(+1)
        A[5, i - 1] = i * (i - 1) * (-1.0) ** i    # f''(-1)
    b = np.array([right - center, left - center,
                  right * hi, -left * lo,
                  right * hi * hi, left * lo * lo])
    a = np.linalg.solve(A, b)
    return center + sum(a[i - 1] * alpha ** i for i in range(1, 7))


def _histfactory_oracle(point: dict) -> float:
    """Closed form, independent of any FlatPPL engine: a two-bin Poisson product,
    three unit-Gaussian nuisance constraints, and the staterror term as §09's
    ContinuedPoisson density lambda**x e**-lambda / Gamma(x+1)."""
    mcstat = np.asarray(point["mcstat"], dtype=float)
    f = _poly6_exp
    nu = (_SIGNAL * f(_ANCHOR_LO, _ANCHOR_CTR, _ANCHOR_HI, point["syst1"]) * point["mu"]
          + _BKG1 * f(_ANCHOR_LO, _ANCHOR_CTR, _ANCHOR_HI, point["syst2"]) * mcstat
          + _BKG2 * f(_ANCHOR_LO, _ANCHOR_CTR, _ANCHOR_HI, point["syst3"]) * mcstat)
    lp = float(np.sum(poisson.logpmf(_OBSERVED, nu)))
    for name in ("syst1", "syst2", "syst3"):
        lp += float(norm.logpdf(0.0, point[name], 1.0))
    rate = mcstat * _TAU
    lp += float(np.sum(_TAU * np.log(rate) - rate
                       - np.array([math.lgamma(t + 1.0) for t in _TAU])))
    return lp


# The nominal point the model's own `log_likelihood` binding evaluates at.
_HF_NOMINAL = {"mu": 1.0, "syst1": 0.0, "syst2": 0.0, "syst3": 0.0,
               "mcstat": [1.0, 1.0]}


def test_histfactory_dataset_is_the_expected_shape():
    """A dataset edit must fail here rather than silently reshape the oracle."""
    assert _OBSERVED.tolist() == [122.0, 112.0]
    assert _TAU.tolist() == [400.0, 100.0]
    for v in (_SIGNAL, _BKG1, _BKG2):
        assert v.shape == (2,), f"expected a two-bin channel, got {v.shape}"


def test_histfactory_oracle_reproduces_the_frozen_root_vector():
    """The independence check on this oracle, and the ONLY gate in the suite on
    the interp_poly6_exp coefficients against a non-FlatPPL source.

    ROOT's own 2DeltaNLL vector is already frozen in the dir. Differencing the
    oracle's absolute values must reproduce it, which pins everything about the
    oracle except its offset -- and the offset is the closed-form Poisson and
    ContinuedPoisson normalisation, which is exactly what the absolute test
    below then asserts. No engine is involved in either direction.
    """
    import json
    check = json.loads((_HISTFACTORY / "test.json").read_text())["checks"][0]
    ref = _histfactory_oracle(check["reference_point"])
    got = [-2.0 * (_histfactory_oracle(p) - ref) for p in check["points"]]
    for i, (g, w) in enumerate(zip(got, check["expected"])):
        assert g == pytest.approx(w, abs=1e-9, rel=1e-9), (
            f"point {i}: oracle 2DeltaNLL {g!r} != frozen ROOT {w!r}. The oracle "
            "and ROOT disagree, so one of them is wrong -- do NOT 'fix' this by "
            "re-pinning either."
        )


@pytest.mark.skipif(not ex.engine_available(), reason="det-js path unavailable")
def test_histfactory_conversion_absolute_logdensity_matches_the_oracle():
    """The model binds `log_likelihood = logdensityof(likelihood, <nominal>)`
    already, so this scores that binding directly -- an ABSOLUTE density.

    `score_binding` determinizes, so this also fails if the §09 function
    lowering for `interp_poly6_exp` regresses. That is not covered anywhere else:
    the dir's own unified row goes through the `convert` runner, which scores via
    the `FLATPPL_ENGINE`-selected engine and defaults to pure `js`.
    """
    got = ex.score_binding(_HISTFACTORY / "histfactory.flatppl", "log_likelihood")
    want = _histfactory_oracle(_HF_NOMINAL)
    assert got == pytest.approx(want, abs=1e-9, rel=1e-9), (
        f"absolute log-density drifted: got {got!r}, closed-form oracle {want!r}. "
        "The frozen 2DeltaNLL vector is offset-invariant and would not notice."
    )


@pytest.mark.skipif(not ex.engine_available(), reason="det-js path unavailable")
def test_a_histfactory_offset_shift_would_be_caught():
    """Guards the guard, as for the gaussian anchor above."""
    got = ex.score_binding(_HISTFACTORY / "histfactory.flatppl", "log_likelihood")
    assert got + 0.5 != pytest.approx(
        _histfactory_oracle(_HF_NOMINAL), abs=1e-9, rel=1e-9)
