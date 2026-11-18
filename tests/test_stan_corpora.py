"""The Stan emitter's round trip over every corpus but posteriorDB.

Each logdensity case is scored against its frozen `expected`, and each pyhf
convert case against pyhf's absolute `Model.logpdf`. Cases need not list
`stan` among their `engines`: the emitter grows each slice, and a case it
cannot write yet skips with the refusal ("not yet supported" or "no Stan
form"), so `-rs` lists what is left. Any other outcome that is not a match
fails, including a program that compiles but scores a different value.

Run it in the ``bridge`` environment with a binary built with `--features stan`:
``FLATPPL_BIN=<.../release/flatppl> pixi run -e bridge stan-corpora``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from flatppl_testsuite.unified.harness import run_test_dir
from flatppl_testsuite.unified.loader import discover_test_dirs, merged_body
from flatppl_testsuite.unified.runners.logdensity_stan import NOT_YET
from tests.stan_verdict import stan_verdict
from tests.test_unified import _gate_engine

pytestmark = [
    pytest.mark.stan_only,
    pytest.mark.filterwarnings("ignore:Loading a shared object"),
]

_CORPORA = Path(__file__).parents[1] / "corpora"


def _scored(dir: Path) -> bool:
    body = merged_body(json.loads((dir / "test.json").read_text()), "stan")
    if body["test_type"] == "logdensity":
        return True
    return body["test_type"] == "convert" and body.get("fixture_kind") == "pyhf"


CASES = [d for d in discover_test_dirs(_CORPORA)
         if d.relative_to(_CORPORA).parts[0] != "posteriordb" and _scored(d)]


@pytest.mark.parametrize("case", CASES, ids=[str(c.relative_to(_CORPORA)) for c in CASES])
def test_stan_matches_expected(case: Path):
    _gate_engine("stan")
    stan_verdict(run_test_dir(case, engines=["stan"]), skips=(NOT_YET, "no Stan form"))
