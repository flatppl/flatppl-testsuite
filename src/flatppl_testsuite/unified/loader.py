"""Load + validate a per-test `test.json`."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

KNOWN_TEST_TYPES = {"logdensity"}  # extend as runners land (gradient, sample, convert)


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
