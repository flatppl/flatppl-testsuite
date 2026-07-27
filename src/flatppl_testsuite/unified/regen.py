"""Offline oracle -> frozen `expected` (logdensity) / `stat` (sample). Runs
each dir's test.py.oracle(point) or test.py.stat() and writes the result into
test.json. The ONLY place oracles execute; the harness at test time only
compares (sample's `stat['distribution']` recipe is a deterministic scipy
lookup, not oracle computation — see sample_checks.py's module docstring).

`test_type == "sample"` directories come in two `stat()` shapes:

* the stablehlo-sample shape -- `stat()` returns a single KS-test recipe
  (`{"distribution": {...}, "discrete": bool}`), frozen wholesale under the
  top-level `stat` key (`sample_checks.py` reconstructs a live scipy frozen
  distribution from it at test time).
* the per-check shape (e.g. `corpora/sample/hier_normal`) -- `stat()` returns
  `{check_id: {"expected": ..., "atol": ...}}` and each entry is merged into
  the matching `checks[i]` (by `id`) in place; there is no top-level `stat`
  key for this shape because the frozen values already live inline on each
  check (mean/var/cov per binding), not as one recipe.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

from flatppl_testsuite.unified.loader import load_test_module


def _json_safe(v):
    """A freshly computed oracle value, encoded so it survives JSON.

    `json.dumps` emits bare `Infinity`/`-Infinity`/`NaN` by default, which are
    not valid JSON (RFC 8259 defines no such literals). Python and `jq` accept
    them, but a strict parser rejects them -- including JavaScript's
    `JSON.parse`, and this repo scores through a Node engine. So a non-finite
    value is frozen in the same STRING form the corpus already uses and
    `detjs_exec.parse_expected` already reads back (e.g. fragment's `trunc_out`,
    a truncation gate scored outside its support: density exactly 0, log-density
    `-inf`). Recurses so vector/gradient/stat payloads are covered too."""
    if isinstance(v, float):
        if math.isnan(v):
            return "nan"
        if math.isinf(v):
            return "inf" if v > 0 else "-inf"
        return v
    if isinstance(v, list):
        return [_json_safe(x) for x in v]
    if isinstance(v, tuple):
        return [_json_safe(x) for x in v]
    if isinstance(v, dict):
        return {k: _json_safe(x) for k, x in v.items()}
    return v


def _write(dir: Path, raw: dict) -> None:
    """Write `test.json`. `allow_nan=False` makes any non-finite that escaped
    `_json_safe` raise instead of silently emitting an invalid literal."""
    (dir / "test.json").write_text(
        json.dumps(raw, indent=2, allow_nan=False) + "\n"
    )


def regen_dir(dir: Path) -> list[float] | list[dict] | dict:
    dir = Path(dir)
    raw = json.loads((dir / "test.json").read_text())
    test_type = raw.get("test_type")
    if test_type == "convert":
        raise NotImplementedError(
            f"{dir}: regen does not support test_type 'convert' -- the hs3 corpus's "
            "frozen twice_delta_nll vectors are ROOT/RooFit oracle values, regenerated "
            "in the separate `root` pixi env by subprocessing a ROOT/RooFit run over "
            "the HS3SUITE checkout (see corpora/hs3/conversions/gen_expected.py and "
            "the `pixi run -e root ...` oracle path), not by this offline test.py "
            "oracle harness. Do not add a generic fallback here."
        )
    mod = load_test_module(dir)
    if test_type == "sample":
        stat = mod.stat()
        if isinstance(stat, dict) and "distribution" in stat:
            raw["stat"] = _json_safe(stat)
        else:
            for check in raw["checks"]:
                update = stat.get(check["id"])
                if update:
                    check.update(_json_safe(update))
        _write(dir, raw)
        return stat
    if test_type == "gradient":
        expected_grad = [mod.grad_oracle(pt) for pt in raw["points"]]
        raw["expected_grad"] = _json_safe(expected_grad)
        _write(dir, raw)
        return expected_grad
    if "points" not in raw:
        # Mode A (det-js logdensity, no `points`): a single reference point
        # baked into the model/oracle itself -- test.py::oracle() takes no
        # arguments and freezes one scalar.
        expected_scalar = float(mod.oracle())
        raw["expected"] = _json_safe(expected_scalar)
        _write(dir, raw)
        return [expected_scalar]
    # Mode B (det-js logdensity, `points` present): test.py::oracle(point) is
    # called once per point, in order, and the whole list is frozen.
    expected = [float(mod.oracle(pt)) for pt in raw["points"]]
    raw["expected"] = _json_safe(expected)
    _write(dir, raw)
    return expected


_KIND_BY_TEST_TYPE = {"sample": "stat keys", "gradient": "expected_grad entries"}


def main(argv: list[str] | None = None) -> int:
    dirs = [Path(a) for a in (argv if argv is not None else sys.argv[1:])]
    for d in dirs:
        test_type = json.loads((d / "test.json").read_text()).get("test_type")
        result = regen_dir(d)
        n = len(result)
        kind = _KIND_BY_TEST_TYPE.get(test_type, "expected values")
        print(f"{d}: wrote {n} {kind}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
