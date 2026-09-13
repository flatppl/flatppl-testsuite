"""Exercise the real compiled, batched IREE likelihood path."""

import math
import subprocess

import pytest

from flatppl_testsuite.unified import iree_exec
from tests.test_unified import _gate_engine


@pytest.mark.iree_only
def test_iree_compiles_and_invokes_once_for_all_model_points(tmp_path, monkeypatch):
    _gate_engine("iree")
    import iree.runtime as runtime

    model = tmp_path / "model.flatppl"
    model.write_text(
        "mu = elementof(reals)\nshift = elementof(cartpow(reals, 2))\n"
        "M = likelihoodof(iid(Normal(mu + sum(shift), 1.0), 3), [0.1, 0.2, 0.3])\n"
    )
    emissions, invocations = [], []
    run, invoke = subprocess.run, runtime.VmContext.invoke

    def tracked_run(command, *args, **kwargs):
        if len(command) > 1 and command[1] == "stablehlo":
            emissions.append(command)
        return run(command, *args, **kwargs)

    def tracked_invoke(context, *args, **kwargs):
        invocations.append(True)
        return invoke(context, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", tracked_run)
    monkeypatch.setattr(runtime.VmContext, "invoke", tracked_invoke)
    points = [
        {"mu": -0.4, "shift": [-0.2, 0.3]},
        {"shift": [0.8, -0.1], "mu": 1.2},
        {"mu": -0.4, "shift": [-0.2, 0.3]},
    ]
    scores = iree_exec.log_density_points(model, "M", points)
    expected = [
        -1.5 * math.log(2 * math.pi)
        - 0.5 * sum((x - p["mu"] - sum(p["shift"]))**2 for x in (0.1, 0.2, 0.3))
        for p in points
    ]
    assert [score.value for score in scores] == pytest.approx(expected, rel=0, abs=1e-12)
    assert len(emissions) == 1
    assert len(invocations) == 1
