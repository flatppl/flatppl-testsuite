"""The Stan runner's rule for a point that Stan rejects.

Stan Math raises where a law has no density, and a rejection means density
0. So `log_density` raising passes a point frozen at -inf when a law rejects
its arguments, and a point frozen as an error record when the rejection is of
the class that record names. Every other exception fails the point: an index
or size error at a -inf point, any exception at a finite point, and any
exception before `log_density` (data, constraints).

Hand-written programs, scored through the runner's `_score_points`.

Run it in the ``bridge`` environment: ``pixi run -e bridge stan-reject``.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from flatppl_testsuite.scoring.result import NUMERIC_MISMATCH, STAN_REJECTED, UNSCOREABLE
from tests.test_unified import _unavailable

pytestmark = [
    pytest.mark.stan_only,
    pytest.mark.filterwarnings("ignore:Loading a shared object"),
]

PROGRAM = """data {
  int which;
  real<upper=10> y;
  int i;
  int n;
  real sigma;
}
parameters {
  real<lower=0> s;
}
model {
  if (which == 1) target += rayleigh_lpdf(y | 2.0);
  if (which == 2) {
    vector[3] v = [1, 2, 3]';
    target += v[i];
  }
  if (which == 3) target += normal_lpdf([1, 2, 3]' | rep_vector(0, n), 1);
  if (which == 4) reject("custom rejection");
  if (which == 5) target += normal_lpdf(y | 0, sigma);
  if (which == 6) target += normal_lpdf(y | 0, s);
}
"""

RAYLEIGH_15 = math.log(1.5 / 4) - 1.5 ** 2 / 8  # rayleigh_lpdf(1.5 | 2)
OUTSIDE = {"error": "outside its declared value set"}
INDEX = {"error": "index out of range"}

# (case, point, frozen expected, wanted status, wanted tag)
CASES = [
    ("law rejection at -inf", dict(which=1, y=-1.0), "-inf", "passed", STAN_REJECTED),
    ("law rejection at a finite value", dict(which=1, y=-1.0), RAYLEIGH_15, "failed", UNSCOREABLE),
    ("law rejection at its error record", dict(which=5, sigma=0.0), OUTSIDE, "passed", STAN_REJECTED),
    ("law rejection at an unknown error record", dict(which=1, y=-1.0), {"error": "other"},
     "failed", UNSCOREABLE),
    ("index error at -inf", dict(which=2, i=5), "-inf", "failed", UNSCOREABLE),
    ("index error at its error record", dict(which=2, i=5), INDEX, "passed", STAN_REJECTED),
    ("index error at the law error record", dict(which=2, i=0), OUTSIDE, "failed", UNSCOREABLE),
    ("law rejection at the index error record", dict(which=1, y=-1.0), INDEX, "failed", UNSCOREABLE),
    ("size error in a law at -inf", dict(which=3, n=2), "-inf", "failed", UNSCOREABLE),
    ("reject() at -inf", dict(which=4), "-inf", "failed", UNSCOREABLE),
    ("data constraint at -inf", dict(which=1, y=11.0), "-inf", "failed", UNSCOREABLE),
    ("parameter constraint at -inf", dict(which=6, s=-1.0), "-inf", "failed", UNSCOREABLE),
    ("value match", dict(which=1, y=1.5), RAYLEIGH_15, "passed", ""),
    ("finite wrong value", dict(which=1, y=1.5), RAYLEIGH_15 + 1e-6, "failed", NUMERIC_MISMATCH),
    ("value at an error record", dict(which=1, y=1.5), OUTSIDE, "failed", NUMERIC_MISMATCH),
]


@pytest.fixture(scope="module")
def runner():
    try:
        import bridgestan  # noqa: F401
    except ImportError:
        _unavailable("stan", "the Stan rejection test needs `bridgestan` (the `bridge` env)")
    from flatppl_testsuite.unified.runners import logdensity_stan

    return logdensity_stan


@pytest.mark.parametrize("case,point,expected,status,tag", CASES, ids=[c[0] for c in CASES])
def test_stan_rejection_rule(runner, case, point, expected, status, tag):
    point = {"which": 1, "y": 1.0, "i": 1, "n": 3, "sigma": 1.0, "s": 1.0, **point}
    fields = ["which", "y", "i", "n", "sigma"]
    manifest = {
        "parameters": [{"stan": "s", "field": "s", "input": "s"}],
        "data": [{"input": f, "stan": f, "type": "int"} for f in fields],
    }
    routes = runner._routes(manifest, [*fields, "s"], {f: f for f in [*fields, "s"]},
                            lambda p: p.get("input"))
    data = {f: point[f] for f in fields}
    [result] = runner._score_points("probe", "logdensity", PROGRAM, data, routes, [point],
                                    [expected], {"atol": 1e-9, "rtol": 1e-9})
    assert (result.status, result.tag) == (status, tag), result.message
    if tag == STAN_REJECTED:
        assert result.message.startswith("rejected (")
    if expected == {"error": "other"}:
        assert "'other'" in result.message, result.message


# Each Stan Math message form measured on BridgeStan 2.9.0, with the class the
# runner gives it. A Stan upgrade that changes a form fails here.
MESSAGES = [
    ("target += rayleigh_lpdf(-1.0 | 2.0);", "law"),
    ("target += chi_square_lpdf(-1.0 | 1.0);", "law"),
    ("target += normal_lpdf(0.9 | 0.5, 0.0);", "law"),
    ("target += poisson_lpmf(3 | -0.5);", "law"),
    ("target += poisson_lpmf(-1 | 2.5);", "law"),
    ("target += categorical_lpmf(2 | [0.5, 0.6, 0.1]');", "law"),
    ("target += categorical_lpmf(4 | [0.2, 0.3, 0.5]');", "law"),
    ("target += categorical_lpmf(0 | [0.2, 0.3, 0.5]');", "law"),
    ("target += gamma_lpdf(0.8 | -1.0, 2.0);", "law"),
    ("target += multi_normal_lpdf([0, 0]' | [0, 0]', [[1, 2], [2, 1]]);", "law"),
    ("target += gamma_lcdf(-0.5 | 2, 2);", "law"),
    ("target += exponential_lccdf(-1 | 2);", "law"),
    ("{ vector[3] v = [1, 2, 3]'; target += v[5]; }", "index"),
    ("{ array[3] real a = {1.0, 2.0, 3.0}; target += a[0]; }", "index"),
    ("{ array[3] vector[2] r; for (j in 1:3) r[j] = [0.5, 0.5]'; "
     "array[2] int ix = {1, 5}; target += sum(r[ix][1]); }", "index"),
    ("{ matrix[3, 2] m = rep_matrix(1, 3, 2); target += sum(m[4]); }", "index"),
    ("target += normal_lpdf([1, 2, 3]' | [0, 0]', 1);", None),
    ("target += multi_normal_lpdf([0, 0, 0]' | [0, 0]', [[1, 0], [0, 1]]);", None),
    ("target += normal_lpdf(not_a_number() | 0, 1);", None),
    ("{ vector[3] a = [1, 2, 3]'; vector[2] b = [1, 2]'; target += sum(a + b); }", None),
    ("{ matrix[2, 3] a = rep_matrix(1, 2, 3); vector[2] b = [1, 2]'; target += sum(a * b); }",
     None),
    ("{ vector[3] a; a = [1, 2]'; target += sum(a); }", None),
    ("reject(\"custom message\");", None),
    ("target += sum(cholesky_decompose([[1, 2], [2, 1]]));", None),
]
MESSAGE_PROGRAM = "data {\n  int which;\n}\nmodel {\n" + "".join(
    f"  if (which == {n}) {call}\n" for n, (call, _) in enumerate(MESSAGES, 1)
) + "}\n"


@pytest.mark.parametrize("which,call,cls", [(n, c, k) for n, (c, k) in enumerate(MESSAGES, 1)],
                         ids=[c for c, _ in MESSAGES])
def test_stan_message_class(runner, which, call, cls):
    import bridgestan

    so = str(runner.compile_stan(MESSAGE_PROGRAM))
    model = bridgestan.StanModel(so, json.dumps({"which": which}))
    with pytest.raises(RuntimeError) as raised:
        model.log_density(np.zeros(0), propto=False, jacobian=False)
    message = str(raised.value)
    got = ("law" if runner.LAW_REJECTION.search(message)
           else "index" if runner.INDEX_REJECTION.search(message) else None)
    assert not (runner.LAW_REJECTION.search(message) and runner.INDEX_REJECTION.search(message))
    assert got == cls, message
