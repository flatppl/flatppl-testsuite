"""`scripts/engine-pins.py` is the CI contract: the commits it prints are the
commits CI builds, and the two sweep gates then demand exactly those. So it is
tested against the gates' OWN readers, not against a second read of the same
JSON path -- a re-implementation would agree with the script while both pointed
at the wrong table.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "engine-pins.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("engine_pins", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _fake_tables(root: Path, commit) -> None:
    (root / "verdicts").mkdir()
    for name, key in (("density-sweep.json", "determinizer_commit"),
                      ("sampler-sweep.json", "engine_commit")):
        (root / "verdicts" / name).write_text(
            json.dumps({"metadata": {key: commit}, "rows": []}))


def test_the_printed_pins_are_the_commits_the_two_gates_demand():
    from flatppl_testsuite.sampler_sweep import table as sampler_table
    from flatppl_testsuite.sweep import table as density_table

    out = subprocess.run([sys.executable, str(SCRIPT)],
                         capture_output=True, text=True, check=True)
    printed = dict(line.split("=", 1) for line in out.stdout.split())

    density_meta = density_table.load_metadata(density_table.DEFAULT_PATH)
    sampler_meta, _rows = sampler_table.load(sampler_table.DEFAULT_PATH)
    assert printed == {
        "FLATPPL_RUST_REF": density_meta["determinizer_commit"],
        "FLATPPL_JS_REF": sampler_meta["engine_commit"],
    }


def test_an_unrecorded_pin_fails_instead_of_naming_some_other_build(tmp_path):
    """`unknown` is what a table frozen without provenance records, and a branch
    name is what the unpinned build used to resolve. Either one emitted as a ref
    builds a commit the frozen rows do not describe, and the gate then reports a
    mismatch that reads as the engine's fault."""
    mod = _load_script()
    for i, bad in enumerate(("unknown", "main", "95c897cf", "", None)):
        root = tmp_path / f"case{i}"
        root.mkdir()
        _fake_tables(root, bad)
        mod.ROOT = root
        lines, problems = mod.resolve()
        assert not lines and len(problems) == 2, f"{bad!r} was accepted as a pin"


def test_a_missing_table_is_reported_not_skipped(tmp_path):
    mod = _load_script()
    mod.ROOT = tmp_path
    lines, problems = mod.resolve()
    assert not lines
    assert all("no such table" in p for p in problems), problems
