"""Tests for flatppl_testsuite.oracle — skipped when oracle environments are absent."""

from __future__ import annotations

import shutil

import pytest

from flatppl_testsuite.formats.hs3.engines import run_oracle


@pytest.mark.skipif(shutil.which("pixi") is None, reason="pixi required")
def test_root_oracle_rf101():
    try:
        vec = run_oracle("roofit", "rf101_basics")
    except RuntimeError as e:
        pytest.skip(f"root env unavailable: {e}")
    assert len(vec) == 5


@pytest.mark.skipif(shutil.which("pixi") is None, reason="pixi required")
def test_pyhs3_oracle_rf101():
    try:
        vec = run_oracle("pyhs3", "rf101_basics")
    except RuntimeError as e:
        pytest.skip(f"pyhs3 env unavailable: {e}")
    assert len(vec) == 5


def test_run_oracle_bad_backend():
    with pytest.raises(ValueError, match="unsupported oracle backend"):
        run_oracle("nonexistent", "rf101_basics")
