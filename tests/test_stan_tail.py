"""Which Stan `_lccdf` and `_lcdf` functions stay exact far in the tail.

The Stan emitter writes `T[lo, hi]` with both ends finite through a helper
that subtracts two tail probabilities, and the `tail` flag of each law in
flatppl-rust `crates/stan/distributions.ron` names which pair that helper may
use: `lccdf` (gamma, exponential, lognormal, weibull, rayleigh, pareto,
inv_gamma, chi_square, double_exponential, poisson) or the mirrored `lcdf`
pair (`mirror_lcdf`: normal). This test pins that choice against BridgeStan's
Stan Math. Each flagged function must match the oracle to `RTOL` at a moderate
and a far tail point.

The laws without a flag are recorded with what Stan returns today: `-inf` or
a finite value off the oracle. A Stan upgrade that makes one of them exact
fails its case, so the flag table can grow.

Oracles are closed forms or scipy special functions (`log_ndtr`,
`gammainc`, `gammaincc`, `betainc` through `stats`). Each value was checked
against mpmath at 50 digits: the largest relative error is 3.6e-16.

Run it in the ``bridge`` environment, which needs no `flatppl` binary:
``pixi run -e bridge stan-tail``.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
from scipy import special, stats

from tests.test_unified import _unavailable

pytestmark = [
    pytest.mark.stan_only,
    pytest.mark.filterwarnings("ignore:Loading a shared object"),
]

RTOL = 1e-14
EXACT, NEG_INF, OFF = "exact", "-inf", "off the oracle"

# (Stan call with `x` or `k` as the argument, argument, oracle, expected outcome)
_FLAGGED_LCCDF = [
    *[("gamma_lccdf(x | 2, 2)", x, math.log1p(2 * x) - 2 * x, EXACT) for x in (5.0, 300.0)],
    *[("exponential_lccdf(x | 2)", x, -2 * x, EXACT) for x in (5.0, 300.0)],
    *[("double_exponential_lccdf(x | 0, 1)", x, math.log(0.5) - x, EXACT) for x in (5.0, 300.0)],
    *[("lognormal_lccdf(x | 0, 1)", x, float(special.log_ndtr(-math.log(x))), EXACT)
      for x in (10.0, 1e8)],
    *[("pareto_lccdf(x | 1, 2)", x, -2 * math.log(x), EXACT) for x in (10.0, 1e8)],
    *[("inv_gamma_lccdf(x | 2, 1)", x, math.log(special.gammainc(2, 1 / x)), EXACT)
      for x in (10.0, 1e8)],
    *[("weibull_lccdf(x | 2, 1)", x, -x * x, EXACT) for x in (3.0, 20.0)],
    *[("rayleigh_lccdf(x | 1)", x, -x * x / 2, EXACT) for x in (3.0, 30.0)],
    *[("chi_square_lccdf(x | 3)", x, math.log(special.gammaincc(1.5, x / 2)), EXACT)
      for x in (20.0, 300.0)],
    *[("poisson_lccdf(k | 3)", k, math.log(special.gammainc(k + 1, 3)), EXACT) for k in (10, 60)],
]

_FLAGGED_MIRROR_LCDF = [
    ("normal_lcdf(x | 0, 1)", x, float(special.log_ndtr(x)), EXACT)
    for x in (-5.0, -30.0, -300.0, -1e8, -1e12)
]

# The emitter writes `lcdf(L - 1)` for the lower end of a discrete `T[L, U]`,
# so this value must be -inf at L = 0.
_DISCRETE_LOWER_END = [
    ("poisson_lcdf(k | 3)", -1, -math.inf, EXACT),
    ("binomial_lcdf(k | 100, 0.1)", -1, -math.inf, EXACT),
    ("neg_binomial_2_lcdf(k | 3, 2)", -1, -math.inf, EXACT),
]

# Laws without a `tail` flag, with what BridgeStan 2.9.0 returns today.
_UNFLAGGED = [
    ("normal_lccdf(x | 0, 1)", 30.0, float(special.log_ndtr(-30.0)), NEG_INF),
    ("beta_lccdf(x | 2, 3)", 0.999999,
     3 * math.log1p(-0.999999) + math.log1p(3 * 0.999999), NEG_INF),
    ("logistic_lccdf(x | 0, 1)", 300.0, -300.0 - math.log1p(math.exp(-300.0)), NEG_INF),
    ("gumbel_lccdf(x | 0, 1)", 300.0, math.log(-math.expm1(-math.exp(-300.0))), NEG_INF),
    ("neg_binomial_2_lccdf(k | 3, 2)", 200, float(stats.nbinom(2, 0.4).logsf(200)), NEG_INF),
    ("binomial_lccdf(k | 100, 0.1)", 80, float(stats.binom(100, 0.1).logsf(80)), NEG_INF),
    ("student_t_lccdf(x | 3, 0, 1)", 1e8, float(stats.t(3).logsf(1e8)), OFF),
    ("cauchy_lccdf(x | 0, 1)", 1e12, math.log(math.atan(1e-12) / math.pi), OFF),
    ("frechet_lccdf(x | 2, 1)", 1e8, math.log(-math.expm1(-1e-16)), OFF),
    ("logistic_lcdf(x | 0, 1)", -1e8, -1e8 - math.log1p(math.exp(-1e8)), NEG_INF),
    ("student_t_lcdf(x | 3, 0, 1)", -1e8, float(stats.t(3).logcdf(-1e8)), OFF),
    ("student_t_lcdf(x | 3, 0, 1)", -1e12, float(stats.t(3).logcdf(-1e12)), NEG_INF),
    ("cauchy_lcdf(x | 0, 1)", -1e8, math.log(math.atan(1e-8) / math.pi), OFF),
    ("cauchy_lcdf(x | 0, 1)", -1e12, math.log(math.atan(1e-12) / math.pi), OFF),
]

CASES = _FLAGGED_LCCDF + _FLAGGED_MIRROR_LCDF + _DISCRETE_LOWER_END + _UNFLAGGED
_CALLS = sorted({call for call, *_ in CASES})
_PROGRAM = "data {\n  int which;\n  real x;\n  int k;\n}\nmodel {\n" + "".join(
    f"  if (which == {i}) target += {call};\n" for i, call in enumerate(_CALLS, 1)
) + "}\n"


def _label(call: str, arg) -> str:
    return call.replace("(x |", f"({arg:g} |").replace("(k |", f"({arg} |")


def _outcome(got: float, want: float) -> str:
    if got == want or (math.isfinite(want) and abs(got - want) <= RTOL * abs(want)):
        return EXACT
    return NEG_INF if got == -math.inf else OFF


@pytest.fixture(scope="module")
def tail_model():
    try:
        import bridgestan
    except ImportError:
        _unavailable("stan", "the Stan tail test needs `bridgestan` (the `bridge` env)")
    from flatppl_testsuite.unified.runners.logdensity_stan import compile_stan

    so = str(compile_stan(_PROGRAM))

    def log_density(call: str, arg) -> float:
        data = {"which": _CALLS.index(call) + 1, "x": 0.0, "k": 0}
        data["k" if "(k |" in call else "x"] = arg
        model = bridgestan.StanModel(so, json.dumps(data))
        return model.log_density(np.zeros(0), propto=False, jacobian=False)

    return log_density


@pytest.mark.parametrize("call,arg,want,expect", CASES,
                         ids=[_label(c, a) for c, a, *_ in CASES])
def test_stan_tail(tail_model, call, arg, want, expect):
    got = tail_model(call, arg)
    outcome = _outcome(got, want)
    if outcome == expect:
        return
    rel = abs(got - want) / abs(want) if math.isfinite(want) and math.isfinite(got) else math.inf
    detail = f"Stan {got!r}, oracle {want!r}, relative error {rel:.1e} (RTOL {RTOL})"
    if expect == EXACT:
        pytest.fail(f"{_label(call, arg)} is no longer exact: {detail}. "
                    "Check its `tail` flag in flatppl-rust crates/stan/distributions.ron.")
    pytest.fail(f"{_label(call, arg)} changed from {expect!r} to {outcome!r}: {detail}. "
                "Update this table, and a law that is now exact may take a `tail` flag.")
