"""Offline oracle -> frozen `expected`. Runs each dir's test.py.oracle(point)
and writes the resulting list into test.json's `expected`. The ONLY place
oracles execute; the harness at test time only compares."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _load_oracle(dir: Path):
    spec = importlib.util.spec_from_file_location(f"_oracle_{dir.name}", dir / "test.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.oracle


def regen_dir(dir: Path) -> list[float]:
    dir = Path(dir)
    raw = json.loads((dir / "test.json").read_text())
    oracle = _load_oracle(dir)
    expected = [float(oracle(pt)) for pt in raw["points"]]
    raw["expected"] = expected
    (dir / "test.json").write_text(json.dumps(raw, indent=2) + "\n")
    return expected


def main(argv: list[str] | None = None) -> int:
    dirs = [Path(a) for a in (argv if argv is not None else sys.argv[1:])]
    for d in dirs:
        vals = regen_dir(d)
        print(f"{d}: wrote {len(vals)} expected values")
    return 0


if __name__ == "__main__":
    sys.exit(main())
