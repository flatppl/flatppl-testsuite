"""Categorical regeneration retains normalized Stan targets and category order."""
import hashlib
import json
from pathlib import Path
import runpy

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2] / "corpora" / "formula"
ORACLES = ROOT / "oracles" / "categorical"
CASES = json.loads((ORACLES / "cases.json").read_text())


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["name"])
def test_categorical_analytic_targets_match_normalized_stan(case):
    directory = ROOT / case["name"]
    paired = directory.with_name(directory.name + "_gradient")
    capture = ORACLES / case["capture"]
    scores = json.loads((capture / "score.json").read_text())
    density = json.loads((directory / "test.json").read_text())
    gradient = json.loads((paired / "test.json").read_text())
    assert density["points"] == gradient["points"]
    for file in ("model.formula.json", "query.flatppl", "oracle.json", "test.py"):
        assert (directory / file).read_bytes() == (paired / file).read_bytes()
    for file, key in (("model.stan", "stan_sha256"), ("data.json", "data_sha256")):
        assert hashlib.sha256((capture / file).read_bytes()).hexdigest() == case[key]
    for index, point in enumerate(density["points"]):
        raw = scores["points"][index]
        mapped = np.asarray(raw["gradient"])[scores["query_from_stan"]]
        assert raw["query_point"] == point["theta"]
        assert raw["lp"] == pytest.approx(density["expected"][index], abs=1e-12, rel=0)
        np.testing.assert_allclose(mapped, gradient["expected_grad"][index]["theta"], atol=1e-12, rtol=0)
        for current in (directory, paired):
            oracle = runpy.run_path(str(current / "test.py"))
            assert oracle["oracle"](point) == pytest.approx(raw["lp"], abs=1e-11, rel=0)
            np.testing.assert_allclose(oracle["grad_oracle"](point)["theta"], mapped, atol=1e-11, rtol=0)
