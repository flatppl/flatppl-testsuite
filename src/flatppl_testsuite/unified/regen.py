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
import sys
from pathlib import Path

from flatppl_testsuite.unified.loader import load_test_module


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
            raw["stat"] = stat
        else:
            for check in raw["checks"]:
                update = stat.get(check["id"])
                if update:
                    check.update(update)
        (dir / "test.json").write_text(json.dumps(raw, indent=2) + "\n")
        return stat
    if test_type == "gradient":
        expected_grad = [mod.grad_oracle(pt) for pt in raw["points"]]
        raw["expected_grad"] = expected_grad
        (dir / "test.json").write_text(json.dumps(raw, indent=2) + "\n")
        return expected_grad
    if "points" not in raw:
        # Mode A (det-js logdensity, no `points`): a single reference point
        # baked into the model/oracle itself -- test.py::oracle() takes no
        # arguments and freezes one scalar.
        expected_scalar = float(mod.oracle())
        raw["expected"] = expected_scalar
        (dir / "test.json").write_text(json.dumps(raw, indent=2) + "\n")
        return [expected_scalar]
    # Mode B (det-js logdensity, `points` present): test.py::oracle(point) is
    # called once per point, in order, and the whole list is frozen.
    expected = [float(mod.oracle(pt)) for pt in raw["points"]]
    raw["expected"] = expected
    (dir / "test.json").write_text(json.dumps(raw, indent=2) + "\n")
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
