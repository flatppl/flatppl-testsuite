"""Tests for flatppl_testsuite.oracle — skipped when oracle environments are absent."""

from __future__ import annotations

import os
import shutil

import pytest

from flatppl_testsuite.formats.hs3.engines import run_oracle

# The live ROOT/pyHS3 oracles need their heavy pixi envs; CI sets FLATPPL_NO_ORACLE
# to run the FlatPPL engine against the frozen, oracle-verified values only.
_no_oracle = pytest.mark.skipif(
    shutil.which("pixi") is None or bool(os.environ.get("FLATPPL_NO_ORACLE")),
    reason="oracle disabled (FLATPPL_NO_ORACLE) or pixi missing",
)


@_no_oracle
def test_root_oracle_rf101():
    try:
        vec = run_oracle("roofit", "rf101_basics")
    except RuntimeError as e:
        pytest.skip(f"root env unavailable: {e}")
    assert len(vec) == 5


@_no_oracle
def test_pyhs3_oracle_rf101():
    try:
        vec = run_oracle("pyhs3", "rf101_basics")
    except RuntimeError as e:
        pytest.skip(f"pyhs3 env unavailable: {e}")
    assert len(vec) == 5


def test_run_oracle_bad_backend():
    with pytest.raises(ValueError, match="unsupported oracle backend"):
        run_oracle("nonexistent", "rf101_basics")
