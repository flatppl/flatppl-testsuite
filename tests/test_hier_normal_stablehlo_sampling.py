"""`corpora/sample/hier_normal` sampled on StableHLO.

The model has no free parameter, and with `outputs = draws` and no `inputs` its
`@sample` takes only the random state. The dir's `sample_stats` checks (means,
variances, the shared-ancestor covariance and the variance of a difference) are
read from its `test.json` and scored on `n_samples` chained StableHLO draws from the
model's own `rnginit` state, as the det-js runner scores them on its seed sweep. The
dir's own `stablehlo` engine block remains its density check."""

import json
from pathlib import Path

import numpy as np
import pytest

from flatppl_testsuite.unified import stablehlo_exec as ex
from tests.test_unified import _gate_engine

DIR = Path(__file__).resolve().parents[1] / "corpora/sample/hier_normal"


@pytest.mark.stablehlo_only
def test_hier_normal_samples_match_their_closed_form_statistics(tmp_path):
    _gate_engine("stablehlo")
    spec = json.loads((DIR / "test.json").read_text())
    query = tmp_path / "query.flatppl"
    query.write_text((DIR / spec["model"]).read_text() + "\noutputs = draws\n")
    source = ex.emit(query, "sample", restrict_enzyme_compatible=False)

    jax, jnp, hlo_call = ex._jax()
    step = jax.jit(lambda key: hlo_call(key, source=source.replace("@sample(", "@main(")))
    # Start from the model's own `s = rnginit(...)`, never a hand-written state.
    seed_query = tmp_path / "seed.flatppl"
    seed_query.write_text((DIR / spec["model"]).read_text() + "\noutputs = s\n")
    seed_source = ex.emit(seed_query, "logdensity", dtype="f64", restrict_enzyme_compatible=False)
    key = jax.jit(ex.call_with_status(seed_source))()[0][0]
    rows = []
    for _ in range(spec["n_samples"]):
        *values, key = step(key)
        rows.append([float(np.asarray(v)) for v in values[: len(spec["bindings"])]])
    fields = dict(zip(spec["bindings"], np.array(rows).T))

    checked = 0
    for check in spec["checks"]:
        if check["kind"] != "sample_stats":
            continue
        stat = check["stat"]
        if stat in ("mean", "var"):
            x = fields[check["field"]]
            got = x.mean() if stat == "mean" else x.var(ddof=1)
        else:
            a, b = (fields[f] for f in check["fields"])
            got = np.cov(a, b, ddof=1)[0, 1] if stat == "cov" else (a - b).var(ddof=1)
        assert abs(got - check["expected"]) <= check["atol"], (check["id"], got, check["expected"])
        checked += 1
    assert checked == 8
