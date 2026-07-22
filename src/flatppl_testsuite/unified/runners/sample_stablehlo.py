"""Runner: test_type=sample, engine=stablehlo.

Concatenate model.flatppl + query.flatppl into one module, emit StableHLO
(`flatppl stablehlo --mode sample`), execute the threaded-key `@sample` and
dispatch to `sample_checks` for whichever checks `test.json` lists. The
fan-out check instead concatenates model.flatppl + query_iid.flatppl (the
`iid(K, m)`-batched query). ABI param order comes from `inputs` (defaults to
`params`' dict order, mirroring `logdensity_stablehlo.py`'s `inputs` usage).
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from flatppl_testsuite.scoring.result import CheckResult, UNSCOREABLE
from flatppl_testsuite.unified import sample_checks as checks
from flatppl_testsuite.unified import stablehlo_exec as ex
from flatppl_testsuite.unified.loader import TestSpec

_SCALAR_CHECKS = {"distribution", "independence", "key_reproducibility", "key_advance"}


def _concat(model_text: str, query_text: str) -> str:
    return model_text.rstrip() + "\n" + query_text.lstrip()


def _emit(dir: Path, query_name: str) -> str:
    model = (dir / "model.flatppl").read_text()
    query = (dir / query_name).read_text()
    src_text = _concat(model, query)
    with tempfile.NamedTemporaryFile("w", suffix=".flatppl", delete=False) as f:
        f.write(src_text)
        tmp = Path(f.name)
    try:
        return ex.emit(tmp, "sample")
    finally:
        tmp.unlink(missing_ok=True)


def run(spec: TestSpec, dir: Path) -> list[CheckResult]:
    tid = dir.name
    body = spec.body
    params: dict = body["params"]
    n_draws = int(body["draws"])
    key = tuple(int(k) for k in body["key"])
    active = set(body["checks"])
    stat: dict = body["stat"]
    tol = {**checks.DEFAULT_TOLERANCES, **body.get("tolerance", {})}
    arg_names = body.get("inputs") or list(params)
    arg_values = [params[name] for name in arg_names]

    results: list[CheckResult] = []

    # --- scalar `@sample` (model.flatppl + query.flatppl) ---
    needs_scalar = active & _SCALAR_CHECKS
    src = None
    if needs_scalar:
        try:
            src = _emit(dir, "query.flatppl")
        except ex.EmitRefused as e:
            for check_id in sorted(needs_scalar):
                results.append(CheckResult(tid, check_id, "failed", UNSCOREABLE,
                                           f"emit refused: {e}"))
            needs_scalar = set()

    if "distribution" in needs_scalar:
        xs = ex.samples(src, n_draws, arg_values, key)
        results.append(checks.check_distribution(tid, xs, stat, **tol))

    if "independence" in needs_scalar:
        xs = ex.samples(src, n_draws, arg_values, key)
        results.append(checks.check_independence(tid, xs, stat, params, **tol))

    if "key_reproducibility" in needs_scalar:
        v1, k1 = ex.sample_call(src, key, arg_values)
        v2, k2 = ex.sample_call(src, key, arg_values)
        results.append(checks.check_key_reproducibility(tid, v1, k1, v2, k2))

    if "key_advance" in needs_scalar:
        keys = [np.asarray(key, dtype=np.uint64)]
        cur = key
        for _ in range(5):
            _, cur = ex.sample_call(src, cur, arg_values)
            keys.append(np.asarray(cur))
        results.append(checks.check_key_advance(tid, keys))

    # --- fanout `@sample` (model.flatppl + query_iid.flatppl) ---
    if "fanout_distribution" in active:
        try:
            fanout_src = _emit(dir, "query_iid.flatppl")
        except ex.EmitRefused as e:
            results.append(CheckResult(tid, "fanout_distribution", "failed", UNSCOREABLE,
                                       f"emit refused: {e}"))
        else:
            fanout_dim = stat.get("fanout_dim")
            if fanout_dim:
                xs = ex.samples_fanned_multivariate(fanout_src, n_draws, fanout_dim, arg_values, key)
            else:
                xs = ex.samples_fanned(fanout_src, n_draws, arg_values, key)
            results.append(checks.check_fanout_distribution(tid, xs, stat, params, **tol))

    return results
