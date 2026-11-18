"""The pass, skip or fail verdict shared by the Stan round-trip tests."""

from __future__ import annotations

import pytest

from flatppl_testsuite.scoring.result import EMIT_REFUSED


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
