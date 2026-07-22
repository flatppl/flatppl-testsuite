"""Offline oracle -> frozen `expected` (logdensity) / `stat` (sample). Runs
each dir's test.py.oracle(point) or test.py.stat() and writes the result into
test.json. The ONLY place oracles execute; the harness at test time only
compares (sample's `stat['distribution']` recipe is a deterministic scipy
lookup, not oracle computation — see sample_checks.py's module docstring)."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _load_module(dir: Path):
    spec = importlib.util.spec_from_file_location(f"_oracle_{dir.name}", dir / "test.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def regen_dir(dir: Path) -> list[float] | dict:
    dir = Path(dir)
    raw = json.loads((dir / "test.json").read_text())
    mod = _load_module(dir)
    if raw.get("test_type") == "sample":
        stat = mod.stat()
        raw["stat"] = stat
        (dir / "test.json").write_text(json.dumps(raw, indent=2) + "\n")
        return stat
    expected = [float(mod.oracle(pt)) for pt in raw["points"]]
    raw["expected"] = expected
    (dir / "test.json").write_text(json.dumps(raw, indent=2) + "\n")
    return expected


def main(argv: list[str] | None = None) -> int:
    dirs = [Path(a) for a in (argv if argv is not None else sys.argv[1:])]
    for d in dirs:
        result = regen_dir(d)
        n = len(result) if isinstance(result, list) else len(result)
        kind = "expected values" if isinstance(result, list) else "stat keys"
        print(f"{d}: wrote {n} {kind}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
