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


def merged_body(body: dict, engine: str) -> dict:
    """`body` with its per-engine override block for `engine` applied on top.

    A test dir describes ONE model, but the two paths can need different
    scoring shapes for the same query. `corpora/fragment/superpose` is the
    case: det-js scores the model's own `lp = logdensityof(m, 0.5)` binding
    directly (Mode A, no `points`), while the StableHLO emitter refuses a
    module with no `inputs`/`outputs` ABI and therefore needs the same point
    as an ABI argument. A `"stablehlo": { "inputs": …, "points": … }` block
    supplies exactly that, and the det-js case keeps the body it always had --
    so adding a StableHLO row to an existing dir cannot move the det-js
    verdict.

    Shallow merge, on purpose: `tolerance` is replaced whole, since the f32
    band the StableHLO path needs has nothing to do with the 1e-9 band the
    det-js path holds to, and a deep merge would leave the det-js `atol` in
    place where it means nothing.

    `expected` deliberately stays at the top level in the dirs that use this:
    ONE frozen oracle value gates both paths, which is the whole point of
    scoring the same query on both."""
    override = body.get(engine)
    if not isinstance(override, dict):
        return body
    return {**body, **override}


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
    for engine in engines:
        override_tt = merged_body(raw, engine).get("test_type")
        if override_tt not in KNOWN_TEST_TYPES:
            raise ValueError(
                f"{dir}: engine {engine!r} override sets unknown test_type "
                f"{override_tt!r} (known: {sorted(KNOWN_TEST_TYPES)})"
            )
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
