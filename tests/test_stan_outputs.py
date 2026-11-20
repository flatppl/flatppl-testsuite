"""The Stan runner's switch from the `binding` program to the `outputs` program.

When no point input binds a parameter of the `binding` program (the query
builds that parameter from its inputs), the runner scores the `outputs`
program, whose parameters are the ABI inputs. Any other mapping error fails,
a refusal of `outputs` counts as any refusal, and a `binding` program that
binds every parameter never emits `outputs`.

`run` on a hand-written case, with `emit` replaced by hand-written Stan
programs and manifests per target. BridgeStan scores the programs.

Run it in the ``bridge`` environment: ``pixi run -e bridge stan-reject``.
"""

from __future__ import annotations

import json
import math

import pytest

from flatppl_testsuite.scoring.result import (
    EMIT_REFUSED, NUMERIC_MISMATCH, STAN_OUTPUTS, UNSCOREABLE,
)
from flatppl_testsuite.unified import loader
from tests.stan_verdict import stan_verdict
from tests.test_unified import _unavailable

pytestmark = [
    pytest.mark.stan_only,
    pytest.mark.filterwarnings("ignore:Loading a shared object"),
]

MODEL = """a ~ Normal(0.0, 1.0)
b ~ Normal(0.0, 1.0)
Pi_post = lawof(record(a = a, b = b))
"""
QUERY = """theta = elementof(cartpow(reals, 2))
inputs = theta
outputs = logdensityof(Pi_post, record(a = theta[1], b = theta[2]))
"""
# A second ABI input that neither program can receive.
QUERY_EXTRA = """theta = elementof(cartpow(reals, 2))
c = elementof(reals)
inputs = (theta, c)
outputs = logdensityof(Pi_post, record(a = theta[1], b = theta[2]))
"""

BINDING_STAN = """parameters {
  real a;
  real b;
}
model {
  a ~ normal(0, 1);
  b ~ normal(0, 1);
}
"""
OUTPUTS_STAN = """parameters {
  vector[2] theta;
}
model {
  theta ~ normal(0, 1);
}
"""
UNBOUND = {"parameters": [{"stan": "a", "field": "a"}, {"stan": "b", "field": "b"}]}
BOUND = {"parameters": [{"stan": "a", "field": "a", "input": "theta", "index": 1},
                        {"stan": "b", "field": "b", "input": "theta", "index": 2}]}
OUTPUTS = {"parameters": [{"stan": "theta", "field": "theta", "input": "theta"}]}

THETA = [0.3, -0.2]
VALUE = -math.log(2 * math.pi) - (0.3 ** 2 + 0.2 ** 2) / 2


@pytest.fixture(scope="module")
def runner():
    try:
        import bridgestan  # noqa: F401
    except ImportError:
        _unavailable("stan", "the Stan outputs test needs `bridgestan` (the `bridge` env)")
    from flatppl_testsuite.unified.runners import logdensity_stan

    return logdensity_stan


def _run(runner, monkeypatch, tmp_path, programs, *, query=QUERY, body=None):
    """Run the case; ``programs`` maps a target to (Stan code, manifest) or an
    exception. Returns (results, the targets emitted, in order)."""
    (tmp_path / "model.flatppl").write_text(MODEL)
    (tmp_path / "query.flatppl").write_text(query)
    emitted = []

    def fake_emit(dir, target, model_name="model.flatppl", query_name="query.flatppl"):
        emitted.append(target)
        program = programs[target]
        if isinstance(program, Exception):
            raise program
        code, manifest = program
        return code, {}, json.loads(json.dumps(manifest))

    monkeypatch.setattr(runner, "emit", fake_emit)
    body = {"test_type": "logdensity", "engines": ["stan"], "binding": "Pi_post",
            "inputs": ["theta"], "points": [{"theta": THETA}], "expected": [VALUE],
            "tolerance": {"atol": 1e-12, "rtol": 0}, **(body or {})}
    spec = loader.TestSpec(dir=tmp_path, test_type="logdensity", engines=["stan"], body=body)
    return runner.run(spec, tmp_path), emitted


def test_unbound_parameter_scores_outputs(runner, monkeypatch, tmp_path):
    results, emitted = _run(runner, monkeypatch, tmp_path, {
        "Pi_post": (BINDING_STAN, UNBOUND), "outputs": (OUTPUTS_STAN, OUTPUTS)})
    assert emitted == ["Pi_post", "outputs"]
    [r] = results
    assert (r.status, r.tag) == ("passed", STAN_OUTPUTS), r.message
    assert "binding parameter(s) ['a', 'b']" in r.message


def test_outputs_failure_keeps_its_tag_and_names_the_switch(runner, monkeypatch, tmp_path):
    results, _ = _run(runner, monkeypatch, tmp_path, {
        "Pi_post": (BINDING_STAN, UNBOUND), "outputs": (OUTPUTS_STAN, OUTPUTS)},
        body={"expected": [VALUE + 1e-6]})
    [r] = results
    assert (r.status, r.tag) == ("failed", NUMERIC_MISMATCH), r.message
    assert r.message.startswith("outputs program: no point input binds")


def test_bound_binding_never_emits_outputs(runner, monkeypatch, tmp_path):
    results, emitted = _run(runner, monkeypatch, tmp_path, {
        "Pi_post": (BINDING_STAN, BOUND), "outputs": AssertionError("outputs emitted")})
    assert emitted == ["Pi_post"]
    [r] = results
    assert (r.status, r.tag) == ("passed", ""), r.message


def test_other_mapping_error_does_not_switch(runner, monkeypatch, tmp_path):
    results, emitted = _run(runner, monkeypatch, tmp_path, {
        "Pi_post": (BINDING_STAN, BOUND), "outputs": (OUTPUTS_STAN, OUTPUTS)},
        query=QUERY_EXTRA,
        body={"inputs": ["theta", "c"], "points": [{"theta": THETA, "c": 1.0}]})
    assert emitted == ["Pi_post"]
    [r] = results
    assert (r.status, r.tag) == ("failed", UNSCOREABLE), r.message
    assert "reach neither a Stan parameter" in r.message


def test_unbound_outputs_target_does_not_switch_again(runner, monkeypatch, tmp_path):
    results, emitted = _run(runner, monkeypatch, tmp_path, {
        "outputs": (BINDING_STAN, UNBOUND)}, body={"binding": "outputs"})
    assert emitted == ["outputs"]
    [r] = results
    assert (r.status, r.tag) == ("failed", UNSCOREABLE), r.message
    assert "no point input bound to Stan parameter(s) ['a', 'b']" in r.message


@pytest.mark.parametrize("refusal", [
    "stan: not yet supported: `x` is not supported by the Stan emitter yet",
    "stan: no Stan form: `x` is a draw the target marginalizes",
])
def test_refused_outputs_counts_as_any_refusal(runner, monkeypatch, tmp_path, refusal):
    results, emitted = _run(runner, monkeypatch, tmp_path, {
        "Pi_post": (BINDING_STAN, UNBOUND), "outputs": runner.StanRefused(refusal)})
    assert emitted == ["Pi_post", "outputs"]
    [r] = results
    assert (r.status, r.tag) == ("skipped", EMIT_REFUSED), r.message
    assert "binding parameter(s) ['a', 'b']" in r.message and refusal in r.message
    # The corpora accept both refusal kinds; posteriorDB accepts only "not yet".
    with pytest.raises(pytest.skip.Exception):
        stan_verdict(results, skips=(runner.NOT_YET, "no Stan form"))
    if runner.NOT_YET in refusal:
        with pytest.raises(pytest.skip.Exception):
            stan_verdict(results, skips=(runner.NOT_YET,))
    else:
        with pytest.raises(AssertionError):
            stan_verdict(results, skips=(runner.NOT_YET,))


def test_ill_formed_outputs_emit_fails(runner, monkeypatch, tmp_path):
    results, _ = _run(runner, monkeypatch, tmp_path, {
        "Pi_post": (BINDING_STAN, UNBOUND),
        "outputs": runner.StanEmitError("flatppl stan exited 1: parse error")})
    [r] = results
    assert (r.status, r.tag) == ("failed", UNSCOREABLE), r.message
    assert "binding parameter(s) ['a', 'b']" in r.message and "parse error" in r.message
