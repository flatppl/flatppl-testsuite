"""StableHLO fallback-coverage: a model declaring neither `inputs` nor
`outputs` must be REFUSED (exit 3), not silently emitted through some legacy
last-public-binding heuristic.

Ported from the deleted `corpora/examples/gate_stablehlo.py`'s
`check_legacy_no_abi_refused` (design doc "Fallback + migration"): the
concrete-point query convention every check used before the `inputs`/`outputs`
ABI landed has no fallback left — it must refuse outright. This only needs the
`flatppl` CLI and a tiny inline no-ABI model, so it lives in tests/core/
rather than threading through the examples corpus or an executor.
"""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import pytest

from flatppl_testsuite.unified import stablehlo_exec as ex

pytestmark = pytest.mark.skipif(
    not ex.binary_supports_stablehlo(),
    reason="requires a `flatppl` binary built with the `stablehlo` feature (set FLATPPL_BIN)",
)

_NO_ABI_MODEL = (
    'flatppl_compat = "0.1"\n'
    "mu = elementof(reals)\n"
    "sigma = elementof(posreals)\n"
    "g = Normal(mu = mu, sigma = sigma)\n"
    "score = logdensityof(g, 1.27)\n"
)


def test_no_abi_model_refused():
    with tempfile.TemporaryDirectory() as td:
        model = Path(td) / "no_abi.flatppl"
        model.write_text(_NO_ABI_MODEL)
        proc = subprocess.run(
            [str(ex.flatppl_bin()), "stablehlo", str(model), "--mode", "logdensity"],
            capture_output=True, text=True,
        )
    assert proc.returncode == 3, (
        f"expected exit 3 (refused), got {proc.returncode}: stderr={proc.stderr!r}"
    )
    assert "no inputs/outputs ABI declared" in proc.stderr, (
        f"refusal message changed: stderr={proc.stderr!r}"
    )
