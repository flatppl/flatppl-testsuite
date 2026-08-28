"""`scripts/repin.py`'s blocking path, which is the property that matters: a
re-pin must never be how a moved verdict gets frozen.

Exercised against a throwaway git repo holding copies of the real tables, with
a stub "regen" in place of the engines -- so the control flow (regen, diff,
restore, commit) is tested without a sweep, and `preflight` is stubbed out
because its subject is the live engines rather than this flow.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "repin.py"
TABLE_PATHS = (Path("verdicts/sampler-sweep.json"), Path("verdicts/density-sweep.json"))

# A stub regen: rewrite the table in place, applying the mutation named by argv.
_STUB = """
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
d = json.loads(p.read_text())
d["metadata"]["generated_at"] = "2030-01-01T00:00:00+00:00"
if sys.argv[2] == "flip-a-row":
    row = next(r for r in d["rows"] if r["outcome"] in ("DRAWS", "LOWERS"))
    row["outcome"] = "REFUSES"
    row["marker"] = "not-implemented"
if sys.argv[2] == "change-the-seed":
    d["metadata"]["seed"] = 1
p.write_text(json.dumps(d, indent=1) + "\\n")
"""


def _load_repin():
    spec = importlib.util.spec_from_file_location("repin", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    (repo / "verdicts").mkdir(parents=True)
    for rel in TABLE_PATHS:
        (repo / rel).write_bytes((ROOT / rel).read_bytes())
    for args in (("init", "-q"), ("config", "user.email", "t@t"),
                 ("config", "user.name", "t"), ("add", "-A"),
                 ("commit", "-qm", "tables")):
        subprocess.run(("git", "-C", str(repo)) + args, check=True,
                       capture_output=True)
    return repo


def _wire(mod, repo: Path, mutation: str, *, only: str | None = None):
    """Point the module at `repo` and replace each regen with the stub."""
    mod.ROOT = repo
    mod.TABLES = tuple(
        (name, rel, (sys.executable, "-c", _STUB, str(repo / rel),
                     mutation if only in (None, name) else "none"))
        for name, rel in (("sampler", TABLE_PATHS[0]), ("density", TABLE_PATHS[1]))
    )
    saved = {}
    keep = repo.parent / "saved"
    keep.mkdir()
    for name, rel, _cmd in mod.TABLES:
        copy = keep / rel.name
        copy.write_bytes((repo / rel).read_bytes())
        saved[name] = copy
    mod.preflight = lambda: saved
    return saved


def _head(repo: Path) -> str:
    return subprocess.run(("git", "-C", str(repo), "log", "--format=%s", "-1"),
                          capture_output=True, text=True, check=True).stdout.strip()


def test_a_moved_verdict_blocks_the_repin_and_the_tables_are_restored(tmp_path, capsys):
    mod = _load_repin()
    repo = _repo(tmp_path)
    saved = _wire(mod, repo, "flip-a-row", only="sampler")
    before = {rel: (repo / rel).read_bytes() for rel in TABLE_PATHS}

    assert mod.main() == 1, "a row that flipped to REFUSES was accepted as a re-pin"
    for rel in TABLE_PATHS:
        assert (repo / rel).read_bytes() == before[rel], f"{rel} was left regenerated"
    assert _head(repo) == "tables", "a blocked re-pin still committed"
    err = capsys.readouterr().err
    assert "outcome DRAWS -> REFUSES" in err, err
    assert "nothing committed" in err
    assert saved  # the copies are what the restore came from


def test_a_changed_seed_blocks_even_with_every_row_identical(tmp_path):
    """The draw count, seed and sigma define what the table measures. Moving one
    while the rows happen to match is a redefinition, not a re-pin."""
    mod = _load_repin()
    repo = _repo(tmp_path)
    _wire(mod, repo, "change-the-seed", only="sampler")
    assert mod.main() == 1
    assert _head(repo) == "tables"


def test_metadata_only_movement_commits_with_the_fixed_message(tmp_path):
    mod = _load_repin()
    repo = _repo(tmp_path)
    _wire(mod, repo, "none")
    assert mod.main() == 0
    assert _head(repo) == "verdicts: re-pin sweep tables to current engines"
    changed = subprocess.run(
        ("git", "-C", str(repo), "show", "--name-only", "--format=", "HEAD"),
        capture_output=True, text=True, check=True).stdout.split()
    assert sorted(changed) == sorted(str(p) for p in TABLE_PATHS)


def test_nothing_to_commit_is_not_an_error(tmp_path):
    """A re-pin against the engines already pinned is a no-op, not a failure."""
    mod = _load_repin()
    repo = _repo(tmp_path)
    # A stub that writes the file back byte-identical.
    mod.ROOT = repo
    mod.TABLES = tuple((name, rel, (sys.executable, "-c", "pass"))
                       for name, rel in (("sampler", TABLE_PATHS[0]),
                                         ("density", TABLE_PATHS[1])))
    keep = repo.parent / "saved"
    keep.mkdir()
    saved = {}
    for name, rel, _cmd in mod.TABLES:
        copy = keep / rel.name
        copy.write_bytes((repo / rel).read_bytes())
        saved[name] = copy
    mod.preflight = lambda: saved
    assert mod.main() == 0
    assert _head(repo) == "tables"


def test_the_committed_tables_still_carry_the_metadata_a_repin_may_move(tmp_path):
    """`_REPINNABLE_METADATA` is an allow-list, so a renamed metadata key would
    silently turn into blocking drift. Checked against the real tables."""
    mod = _load_repin()
    for rel in TABLE_PATHS:
        meta = json.loads((ROOT / rel).read_text())["metadata"]
        moving = {"generated_at"} | {k for k in meta if k.endswith("_commit")}
        missing = moving - mod._REPINNABLE_METADATA
        assert not missing, f"{rel}: {missing} would block every re-pin"


@pytest.mark.parametrize("name", ["sampler", "density"])
def test_gate_drift_is_silent_when_a_table_is_compared_against_itself(name):
    mod = _load_repin()
    rel = TABLE_PATHS[0] if name == "sampler" else TABLE_PATHS[1]
    assert mod.gate_drift(name, ROOT / rel, ROOT / rel) == []
