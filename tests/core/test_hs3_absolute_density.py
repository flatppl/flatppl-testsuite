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

`histfactory`'s oracle is the one that lives in its own directory's `test.py`
rather than in this file, because the dir's `"stablehlo"` block freezes absolute
per-point values that `regen` never refreezes -- see the comment above it.

`histfactory` matters for a second reason. It is the only corpus model that uses
a §09 standard-module FUNCTION member (`interp_poly6_exp`, three times), and
`score_binding` determinizes it against a closed-form oracle. The `convert`
runner that drives the dir's own row determinizes too since 2026-09-02, but it
compares against the frozen ROOT 2DeltaNLL vector, which is offset-invariant.
So this is still the only place the §09 function lowering is checked in an
ABSOLUTE density.
"""
from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import norm, poisson

from flatppl_testsuite.unified import detjs_exec as ex
from flatppl_testsuite.unified.loader import load_test_module

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
# The oracle lives in the dir's own `test.py`, which is where it has to live:
# the dir's `"stablehlo"` block carries its own frozen `expected`, `regen` never
# refreezes a block, and `test_engine_override_rows.py` re-derives that block
# from `test.py::logdensity`. A second copy of the formula in this file is
# exactly how the two would drift. `test.py` carries the derivation of §09's
# interp_poly6_exp C^2 system it solves.
#
# Its independence is not taken on trust:
# `test_histfactory_oracle_reproduces_the_frozen_root_vector` below feeds the
# oracle's own absolute values through the same 2DeltaNLL difference the dir's
# frozen ROOT vector holds, and they agree to 8e-13. So the absolute value
# asserted here is anchored by ROOT for everything except the offset, and by
# closed-form maths for the offset itself.
_HISTFACTORY = _CORPORA / "hs3" / "conversions" / "histfactory"
_HF = load_test_module(_HISTFACTORY)
_histfactory_oracle = _HF.oracle


# The nominal point the model's own `log_likelihood` binding evaluates at.
_HF_NOMINAL = {"mu": 1.0, "syst1": 0.0, "syst2": 0.0, "syst3": 0.0,
               "mcstat": [1.0, 1.0]}


def test_histfactory_dataset_is_the_expected_shape():
    """A dataset edit must fail here rather than silently reshape the oracle."""
    assert _HF.OBSERVED.tolist() == [122.0, 112.0]
    assert _HF.TAU.tolist() == [400.0, 100.0]
    for v in (_HF.SIGNAL, _HF.BKG1, _HF.BKG2):
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
    lowering for `interp_poly6_exp` regresses. The dir's own unified row
    determinizes as well, but against an offset-invariant 2DeltaNLL vector, so
    this is the only ABSOLUTE check on that lowering.
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


def test_the_stablehlo_row_freezes_this_oracles_absolute_values():
    """The dir's `"stablehlo"` override block scores the SAME query the det-js
    `convert` case does, but absolutely rather than as a 2DeltaNLL difference,
    so the vector it freezes must be this oracle's own -- not a value read back
    off an engine.

    The row runs only in the `stablehlo` pixi environment. This check is what
    keeps its frozen numbers honest from the default environment, where nothing
    else looks at them.
    """
    import json
    body = json.loads((_HISTFACTORY / "test.json").read_text())
    row = body["stablehlo"]
    check = body["checks"][0]
    assert row["points"] == check["points"], (
        "the StableHLO row must score the same grid as the frozen ROOT check, "
        "so the two verdicts move together"
    )
    assert row["inputs"] == ["mu", "syst1", "syst2", "syst3", "mcstat"], (
        "the ABI order the query declares -- `test.py::logdensity` takes its "
        "positional arguments in exactly this order"
    )
    # The per-point VALUES are re-derived by
    # `test_engine_override_rows.py::test_a_block_expected_still_matches_the_dirs_own_oracle`,
    # which runs for every dir whose engine block carries its own `expected`.
    # This test owns the grid and the ABI instead.


def test_the_stablehlo_query_scores_the_models_own_likelihood_root():
    """`emit_concat` hard-codes `query.flatppl`, so the ABI query lives in its
    own file rather than in the model. It must still be the model's own root
    binding at the model's own record shape -- a query that scored a different
    binding would freeze a number the det-js case never sees."""
    q = (_HISTFACTORY / "query.flatppl").read_text()
    assert "logdensityof(likelihood, record(" in q, f"unexpected query:\n{q}"
    assert "outputs = (lp)" in q, f"query must name the ABI outputs:\n{q}"
    model = (_HISTFACTORY / "histfactory.flatppl").read_text()
    for name in ("mu", "syst1", "syst2", "syst3", "mcstat"):
        assert re.search(rf"^{name} = elementof\(", model, re.M), (
            f"{name} must already be a free boundary in the model, so the query "
            "only names the ABI"
        )
