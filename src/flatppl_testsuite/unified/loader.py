"""Load + validate a per-test `test.json`."""
from __future__ import annotations

import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

KNOWN_TEST_TYPES = {"logdensity", "sample", "gradient", "convert"}  # extend as runners land


@dataclass(frozen=True)
class TestSpec:
    dir: Path
    test_type: str
    engines: list[str]
    body: dict  # the whole parsed test.json


def load_test(dir: Path) -> TestSpec:
    raw = json.loads((Path(dir) / "test.json").read_text())
    tt = raw.get("test_type")
    if tt is None:
        raise ValueError(f"{dir}: test.json missing required key 'test_type'")
    if tt not in KNOWN_TEST_TYPES:
        raise ValueError(f"{dir}: unknown test_type {tt!r} (known: {sorted(KNOWN_TEST_TYPES)})")
    engines = raw.get("engines")
    if not isinstance(engines, list) or not engines:
        raise ValueError(f"{dir}: test.json 'engines' must be a non-empty list")
    return TestSpec(dir=Path(dir), test_type=tt, engines=list(engines), body=raw)


def discover_test_dirs(root: Path) -> list[Path]:
    return sorted(p.parent for p in Path(root).rglob("test.json"))


def load_test_module(dir: Path) -> ModuleType:
    """Dynamically load a test directory's `test.py` (its `oracle` /
    `grad_oracle` / `stat` / `logdensity` functions, per test_type). The one
    shared loader: `regen.py` uses it offline to freeze values into
    `test.json`, and a runner may also use it at test time when a check has
    no frozen scalar to compare against and must evaluate the directory's own
    oracle live (e.g. `sample_detjs.py`'s `density_consistency` check)."""
    dir = Path(dir)
    spec = importlib.util.spec_from_file_location(f"_testmod_{dir.name}", dir / "test.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
