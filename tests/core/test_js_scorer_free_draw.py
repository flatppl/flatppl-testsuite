"""The JS scorers refuse a score that reads a free draw.

§13 "Output reduction": draw nodes take their values from the explicit point,
unless marginalized out. A density whose measure reads a draw outside every
lawof has no single value; the JS engine evaluates it per atom (decisions-log
2026-10-21: no implicit marginal). The scorers materialise one atom, so without
the check they reported one random conditional density as the score. They now
refuse with the wording of the flatppl-rust determiniser. Legal spellings
(lawof, a kernelof likelihood, a posterior) still score their closed forms.

The walk does not trust the engine's phases: a call to a callable that
captures a draw is marked fixed, and scored one random atom. The scorers also
refuse any model the engine refuses statically.
"""
from __future__ import annotations

import json
import math
import shutil
import subprocess
from pathlib import Path

import pytest

from flatppl_testsuite.config import CONFIG
from flatppl_testsuite.scoring import worker

pytestmark = pytest.mark.skipif(
    not (CONFIG.flatppl_js_dir / "packages" / "engine" / "index.ts").exists()
    or shutil.which(CONFIG.node_bin) is None and not Path(CONFIG.node_bin).exists(),
    reason="requires Node and a flatppl-js checkout",
)

_SHARED = ("z = draw(Normal(mu = 0.4, sigma = 1.0))\n"
           "M = joint(f1 = Normal(mu = z, sigma = 1.0), f2 = Normal(mu = z, sigma = 1.5))\n")


def _model(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "m.flatppl"
    path.write_text(text)
    return path


def _logn(x: float, mu: float, var: float) -> float:
    return -0.5 * math.log(2 * math.pi * var) - (x - mu) ** 2 / (2 * var)


def test_score_js_refuses_a_measure_that_reads_a_free_draw(tmp_path):
    with pytest.raises(RuntimeError) as raised:
        worker.score_js_file(_model(tmp_path, _SHARED), "M", "record(f1 = 0.5, f2 = 0.7)")
    msg = str(raised.value)
    assert "the density query in `M` reads draw `z`" in msg
    assert "neither in its point nor integrated out by a `lawof`" in msg
    assert "put `z` in the point, or score `lawof(...)` of the value instead" in msg


def test_score_js_names_an_inline_draw(tmp_path):
    model = _model(tmp_path, "M = Normal(mu = draw(Normal(mu = 0.0, sigma = 1.0)), sigma = 1.0)\n")
    with pytest.raises(RuntimeError) as raised:
        worker.score_js_file(model, "M", "0.5")
    assert "the density query in `M` reads an inline `draw`" in str(raised.value)


def test_score_binding_refuses_a_free_draw_density_value(tmp_path):
    model = _model(tmp_path, "z = draw(Normal(mu = 0.0, sigma = 1.0))\n"
                             "lp = logdensityof(Normal(mu = z, sigma = 1.0), 0.5)\n")
    with pytest.raises(RuntimeError) as raised:
        worker.score_binding_file(model, "lp")
    assert "the density query in `lp` reads draw `z`" in str(raised.value)


@pytest.mark.parametrize("text, binding, theta, expected", [
    # lawof(y) with y ~ Normal(z, 1), z ~ Normal(0, 1): y ~ Normal(0, 2).
    ("z = draw(Normal(mu = 0.0, sigma = 1.0))\ny = draw(Normal(mu = z, sigma = 1.0))\nM = lawof(y)\n",
     "M", "0.5", _logn(0.5, 0.0, 2.0)),
    # A kernelof likelihood integrates its ancestor draw: obs | theta ~ Normal(theta, 2).
    ("theta = elementof(reals)\nz = draw(Normal(mu = 0.0, sigma = 1.0))\n"
     "y = draw(Normal(mu = theta + z, sigma = 1.0))\nK = kernelof(y, theta = theta)\nL = likelihoodof(K, 0.5)\n",
     "L", "record(theta = 0.2)", _logn(0.5, 0.2, 2.0)),
    # A joint law: log N(0.1; 0, 1) + log N(0.3; 0.1, 1).
    ("a ~ Normal(mu = 0.0, sigma = 1.0)\nb ~ Normal(mu = a, sigma = 1.0)\nM = lawof(record(a = a, b = b))\n",
     "M", "record(a = 0.1, b = 0.3)", _logn(0.1, 0.0, 1.0) + _logn(0.3, 0.1, 1.0)),
])
def test_score_js_still_scores_marginals_and_likelihoods(tmp_path, text, binding, theta, expected):
    assert worker.score_js_file(_model(tmp_path, text), binding, theta) == pytest.approx(expected, abs=1e-12)


def test_score_binding_still_scores_a_density_through_lawof(tmp_path):
    model = _model(tmp_path, "z = draw(Normal(mu = 0.0, sigma = 1.0))\n"
                             "lp = logdensityof(lawof(z), 0.1) + 1.0\n")
    assert worker.score_binding_file(model, "lp") == pytest.approx(_logn(0.1, 0.0, 1.0) + 1.0, abs=1e-12)


_CORE = Path(__file__).resolve().parents[2] / "src" / "flatppl_testsuite" / "scoring" / "score_core.cjs"
_CAPTURED = [
    # A lambda, a user function and a functionof, each capturing `beta`.
    "beta ~ Normal(0, 1)\nf = fn(Normal(mu = beta * _, sigma = 1))\nxs = [1.0, 2.0]\nM = broadcast(f, xs)\n",
    "beta ~ Normal(0, 1)\ng(x) = Normal(mu = beta * x, sigma = 1)\nM = g(2.0)\n",
    "beta ~ Normal(0, 1)\nt = elementof(reals)\nh = functionof(Normal(mu = beta * t, sigma = 1), t = t)\nM = h(2.0)\n",
]


def _walk(source: str, binding: str) -> dict:
    """The walk over the engine's post-lift graph, whatever the engine's diagnostics and phases."""
    script = (
        "const core = require(process.argv[1]);"
        "const e = core.loadEngine(process.argv[2]);"
        "const p = e.processSource(process.argv[3]);"
        "const b = e.orchestrator.buildDerivations(p.linkedBindings, {});"
        "process.stdout.write(JSON.stringify({ draws: core.freeDraws(b, process.argv[4]) }));"
    )
    engine = CONFIG.flatppl_js_dir / "packages" / "engine"
    out = subprocess.run([CONFIG.node_bin, "-e", script, str(_CORE), str(engine), source, binding],
                         capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


@pytest.mark.parametrize("text", _CAPTURED)
def test_the_walk_finds_a_draw_captured_by_a_callable(text):
    walk = _walk(text, "M")
    assert walk["draws"] == ["beta"]


@pytest.mark.parametrize("text", _CAPTURED)
def test_score_js_refuses_a_static_error(tmp_path, text):
    with pytest.raises(RuntimeError) as raised:
        worker.score_js_file(_model(tmp_path, text), "M", "0.5" if "g(" in text or "h(" in text else "[0.5, 0.5]")
    msg = str(raised.value)
    assert "the JS engine refuses the model scored by `M` statically" in msg
    assert "A reified callable cannot capture a draw" in msg


def test_score_binding_refuses_a_static_error_off_the_scored_path(tmp_path):
    model = _model(tmp_path, "beta ~ Normal(0, 1)\ng(x) = Normal(mu = beta * x, sigma = 1)\nlp = 1.0\n")
    with pytest.raises(RuntimeError) as raised:
        worker.score_binding_file(model, "lp")
    assert "the JS engine refuses the model scored by `lp` statically" in str(raised.value)


def test_a_boundary_input_is_not_a_free_draw(tmp_path):
    text = "theta ~ Normal(0, 1)\nh = functionof(Normal(mu = theta, sigma = 1), theta = theta)\nM = h(0.2)\n"
    assert _walk(text, "M")["draws"] == []
    assert worker.score_js_file(_model(tmp_path, text), "M", "0.5") == pytest.approx(_logn(0.5, 0.2, 1.0), abs=1e-12)


def test_a_module_draw_is_named_as_written():
    script = (
        "const core = require(process.argv[1]);"
        "const bindings = new Map([['M', { type: 'call', ir: { kind: 'ref', ns: 'self', name: 'm$z' } }],"
        " ['m$z', { type: 'draw', ir: null }]]);"
        "try { core.refuseFreeDraw({ bindings }, 'M', 'M'); } catch (e) { process.stdout.write(e.message); }"
    )
    out = subprocess.run([CONFIG.node_bin, "-e", script, str(_CORE)], capture_output=True, text=True, check=True)
    assert "reads draw `m.z`" in out.stdout
    assert "put `m.z` in the point" in out.stdout


def test_score_batch_refuses_a_free_draw():
    rows = worker.score_batch_rows(
        ["z = draw(Normal(mu = 0.0, sigma = 1.0))\nlp = logdensityof(Normal(mu = z, sigma = 1.0), 0.5)\n"], "lp")
    assert rows[0]["ok"] is False
    assert "the density query in `lp` reads draw `z`" in rows[0]["error"]
