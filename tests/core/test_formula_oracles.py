"""Formula oracle regeneration must preserve independent Stan scores."""

import hashlib
import json
from pathlib import Path
import runpy

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2] / "corpora" / "formula"
CASES = json.loads((ROOT / "oracles" / "cases.json").read_text())
EVALUATE = runpy.run_path(str(ROOT / "oracles" / "analytic.py"))["evaluate"]


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["name"])
def test_analytic_density_and_gradient_match_stan(case):
    directory = ROOT / case["name"]
    paired = directory.with_name(directory.name + "_gradient")
    reference = json.loads((directory / "oracle.json").read_text())
    density = json.loads((directory / "test.json").read_text())
    gradient = json.loads((paired / "test.json").read_text())
    assert density["points"] == gradient["points"]
    for point, expected, expected_grad in zip(
        density["points"], density["expected"], gradient["expected_grad"], strict=True,
    ):
        value, derivative = EVALUATE(reference, point)
        assert value == pytest.approx(expected, abs=1e-10, rel=0)
        np.testing.assert_allclose(
            derivative["theta"], expected_grad["theta"], atol=1e-10, rtol=0,
        )
        for current in (directory, paired):
            module = runpy.run_path(str(current / "test.py"))
            assert module["oracle"](point) == pytest.approx(value, abs=1e-12, rel=0)
            np.testing.assert_allclose(
                module["grad_oracle"](point)["theta"], derivative["theta"],
                atol=1e-12, rtol=0,
            )


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["name"])
def test_paired_converter_inputs_retain_stan_provenance(case):
    directory = ROOT / case["name"]
    paired = directory.with_name(directory.name + "_gradient")
    for filename in ("model.formula.json", "query.flatppl", "oracle.json", "test.py"):
        assert (directory / filename).read_bytes() == (paired / filename).read_bytes()
    reference = json.loads((directory / "oracle.json").read_text())
    source = ROOT / "oracles" / case["name"] / "model.stan"
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    assert digest == case["stan_sha256"] == reference["source"]["stan_sha256"]
    assert reference["coordinates"] == case["coordinates"]
    for current in (directory, paired):
        test = json.loads((current / "test.json").read_text())
        assert test["convert"] == {"source": "model.formula.json", "from": "formula"}
        for point in test["points"]:
            assert len(point["theta"]) == len(case["coordinates"])
        assert not (current / "model.flatppl").exists()
