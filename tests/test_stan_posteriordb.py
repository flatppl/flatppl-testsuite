"""The Stan emitter's round trip over posteriorDB.

`flatppl stan` turns each case's `binding` into a Stan program, BridgeStan
scores it at the case's frozen points (`propto=False, jacobian=False`), and the
value must equal `expected`: posteriorDB's own Stan value plus the README's
`log_density_offset`. A program that fails to emit, compile or match fails.
Only a "not yet supported" refusal (exit 3) skips, and the skip names it, so
`-rs` lists what the emitter does not write yet.

Run it in the ``bridge`` environment with a binary built with `--features stan`:
``FLATPPL_BIN=<.../release/flatppl> pixi run -e bridge stan-posteriordb``.
Without `bridgestan` or the `stan` verb every case skips with that reason, or
fails when `FLATPPL_REQUIRE_ENGINES` names `stan`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from flatppl_testsuite.scoring.result import EMIT_REFUSED
from flatppl_testsuite.unified.harness import run_test_dir
from flatppl_testsuite.unified.runners.logdensity_stan import NOT_YET
from tests.test_unified import _gate_engine

pytestmark = [
    pytest.mark.stan_only,
    # Cases with the same Stan text share one cached library, which cannot
    # change because its path is the hash of that text.
    pytest.mark.filterwarnings("ignore:Loading a shared object"),
]

CASES = sorted(
    p.parent for p in (Path(__file__).parents[1] / "corpora" / "posteriordb").glob("*/test.json")
)


def stan_verdict(results, skips: tuple[str, ...]) -> None:
    """Fail on any failure, or on a skip other than an emitter refusal naming
    one of ``skips``; else skip with the refusal."""
    assert results, "no results"
    bad = [r for r in results if r.status == "failed"
           or (r.status == "skipped"
               and (r.tag != EMIT_REFUSED or not any(s in r.message for s in skips)))]
    assert not bad, "\n".join(f"{r.check_id}: {r.tag} {r.message}" for r in bad)
    refused = [r for r in results if r.status == "skipped"]
    if refused:
        pytest.skip(refused[0].message)


@pytest.mark.parametrize("case", CASES, ids=[c.name for c in CASES])
def test_stan_matches_expected(case: Path):
    _gate_engine("stan")
    # Every posteriorDB model has a Stan source, so "no Stan form" is a defect.
    stan_verdict(run_test_dir(case, engines=["stan"]), skips=(NOT_YET,))
