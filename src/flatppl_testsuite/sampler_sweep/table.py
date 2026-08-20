"""Verdict rows, the frozen table, and the outcome classification.

VERDICT VOCABULARY. Three outcomes per row, mirroring the density sweep's
`sweep/classify.py` `Outcome` enum (`LOWERS` / `REFUSES` / `MALFORMED`) in shape
but not in words: `LOWERS` is a determiniser verb and this sweep does not run the
determiniser, so the drawing outcome is named for what it is.

    DRAWS     — the engine produced `n` finite draws. The checks then run, and
                each has its own passed/failed/skipped.
    REFUSES   — the engine declined, with a message matching a DELIBERATE
                refusal (see REFUSAL_PATTERNS). Not a defect on its own: a
                refusal can be correct (sampling a `weighted` measure is
                genuinely intractable) or a carded gap. What matters is that the
                refusal SET is frozen, so an over-refusal or a newly-admitted
                shape shows up as a table diff.
    MALFORMED — the engine threw something that is not a recognised refusal, or
                returned non-finite draws. Always a defect.

A DRAWS row with a failing check is the third signal and the one this gate
exists for: a number the engine was willing to produce and got wrong.

WHY THE PATTERNS LIVE IN PYTHON. The driver reports only `THREW` plus the
message. Classifying here keeps the pattern list and its rationale in one
reviewable place instead of splitting the judgement across two languages.
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path

from flatppl_testsuite.config import CONFIG
from flatppl_testsuite.sampler_sweep import checks as C
from flatppl_testsuite.sampler_sweep import engine, space

DEFAULT_PATH = Path(__file__).resolve().parents[3] / "verdicts" / "sampler-sweep.json"


class Outcome(str, Enum):
    DRAWS = "DRAWS"
    REFUSES = "REFUSES"
    MALFORMED = "MALFORMED"


# Substrings that mark a message as a DELIBERATE refusal rather than a crash.
# Each is paired with the shape that produces it, so the list can be audited
# against the engine rather than grown by whatever a run happened to print.
REFUSAL_PATTERNS: tuple[tuple[str, str], ...] = (
    ("is not supported",
     "`iid` over a record measure at >1 atoms — an explicit unsupported-shape guard"),
    ("cannot be sampled",
     "`weighted` in draw position — a weighted measure is not a probability measure"),
    ("is intractable",
     "the weighted/logweighted/bayesupdate family — genuinely intractable to sample"),
    ("requires reweighting",
     "`normalize` of a measure not provably normalized"),
    ("not implemented",
     "a carded gap in the sampler, refused rather than guessed"),
    ("no sampler",
     "a density-only distribution's sampler stub"),
    ("Undefined variable",
     "the construct does not exist in this engine at all (e.g. ksuperpose)"),
    ("not a measure expression we can sample",
     "an inline combinator the sampler walker does not handle"),
)


def _marker(message: str) -> str:
    """A short, stable tag for the reason a row refused.

    Frozen in the table so a CHANGED refusal reason is a diff, not just a
    still-refusing row. Mirrors the density sweep's `marker` field.
    """
    for pat, _why in REFUSAL_PATTERNS:
        if pat in message:
            return pat.replace(" ", "-")
    return "unclassified"


def classify(draws: engine.Draws) -> tuple[Outcome, str]:
    if draws.status == "DRAWS":
        return Outcome.DRAWS, ""
    if draws.status == "NONFINITE":
        return Outcome.MALFORMED, "nonfinite-draws"
    if draws.status == "THREW":
        for pat, _why in REFUSAL_PATTERNS:
            if pat in draws.error:
                return Outcome.REFUSES, _marker(draws.error)
        return Outcome.MALFORMED, "unrecognised-throw"
    raise RuntimeError(f"driver reported unknown status {draws.status!r} for {draws.id}")


@dataclass
class Row:
    probe_id: str
    family: str
    wrap: str
    outcome: str
    n: int
    k: int
    marker: str | None = None
    error: str | None = None
    checks: list[dict] = field(default_factory=list)
    note: str = ""

    @property
    def failed(self) -> list[dict]:
        return [c for c in self.checks if c["status"] == "failed"]

    @property
    def worst_sigma(self) -> float | None:
        vals = [c["sigma"] for c in self.checks
                if c["status"] == "failed" and c.get("sigma") is not None]
        return max(vals) if vals else None


def evaluate(probe: space.Probe, draws: engine.Draws) -> Row:
    """Turn one probe's draws into a verdict row."""
    outcome, marker = classify(draws)
    row = Row(probe_id=probe.id, family=probe.family, wrap=probe.wrap,
              outcome=outcome.value, n=probe.n_draws, k=probe.k,
              marker=marker or None, note=probe.note)
    if outcome is not Outcome.DRAWS:
        row.error = draws.error or None
        return row

    results: list[C.Check] = []
    for i in range(draws.k):
        results.append(C.check_mean(i, draws.mean(i), probe.mean, probe.var, draws.n))
        results.append(C.check_var(i, draws.var(i), probe.var, probe.fourth, draws.n))
        if i > 0:
            results.append(C.check_cov(i, draws.cov0(i), probe.cov, probe.var, draws.n))
    results.append(C.check_ks(list(draws.ks_sample), probe.ks, len(draws.ks_sample)))
    results.append(C.check_totalmass(draws.log_totalmass, probe.logtotalmass))

    row.checks = [
        {"name": c.name, "status": c.status, "detail": c.detail,
         "got": _num(c.got), "want": _num(c.want), "band": _num(c.band),
         "sigma": _num(c.sigma), "fallback": c.fallback}
        for c in results
    ]
    return row


def _num(v):
    """inf/nan round-trip as strings, matching `sweep/table.py`'s convention."""
    if v is None:
        return None
    f = float(v)
    if f != f:
        return "nan"
    if f == float("inf"):
        return "inf"
    if f == float("-inf"):
        return "-inf"
    return f


def sweep(*, engine_dir: Path | None = None, probes=None) -> list[Row]:
    """Draw and evaluate the whole space (or a given subset)."""
    probes = list(probes if probes is not None else space.enumerate_probes())
    drawn = engine.run(probes, seed=space.SEED, ks_subsample=space.KS_SUBSAMPLE,
                       engine_dir=engine_dir)
    return [evaluate(p, drawn[p.id]) for p in probes]


# ---------------------------------------------------------------------------
# Provenance. A frozen table is only meaningful against the engine that produced
# it, so the engine commit is recorded and the gate checks it before diffing —
# the same guard `sweep/table.py::check_provenance` applies on the density side.
# ---------------------------------------------------------------------------
def engine_commit() -> str:
    try:
        out = subprocess.run(["git", "-C", str(CONFIG.flatppl_js_dir), "rev-parse", "HEAD"],
                             capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except Exception:  # noqa: BLE001 — provenance is best-effort, never fatal
        return "unknown"


def store(rows: list[Row], path: Path = DEFAULT_PATH) -> None:
    import datetime

    counts: dict[str, int] = {}
    for r in rows:
        counts[r.outcome] = counts.get(r.outcome, 0) + 1
    payload = {
        "metadata": {
            "generated_at": datetime.datetime.now(datetime.timezone.utc)
            .replace(microsecond=0).isoformat(),
            "engine_commit": engine_commit(),
            "seed": space.SEED,
            "n_draws": space.N_DRAWS,
            "ks_subsample": space.KS_SUBSAMPLE,
            "sigma": C.SIGMA,
            "probe_count": len(rows),
            "outcome_counts": counts,
            "failing_rows": sorted(r.probe_id for r in rows if r.failed),
        },
        "rows": [asdict(r) for r in rows],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1, sort_keys=False) + "\n")


def load(path: Path = DEFAULT_PATH) -> tuple[dict, dict[str, Row]]:
    if not path.exists():
        return {}, {}
    payload = json.loads(path.read_text())
    rows = {}
    for d in payload["rows"]:
        rows[d["probe_id"]] = Row(**d)
    return payload.get("metadata", {}), rows


def diff(expected: dict[str, Row], actual: dict[str, Row]) -> list[str]:
    """Every way a live sweep can disagree with the frozen table.

    The five signals mirror the density gate's, translated to the sample path:

      DRAWS but a check failed that the table records as passing
                                    -> a wrong number, or a newly wrong one
      DRAWS where the table REFUSES -> newly admitted; needs an oracle verdict
      REFUSES where the table DRAWS -> a regression, or an over-refusal
      a changed refusal marker      -> the reason moved; re-read it
      MALFORMED anywhere            -> always a defect
    """
    problems: list[str] = []
    for pid in sorted(set(expected) | set(actual)):
        e, a = expected.get(pid), actual.get(pid)
        if e is None:
            problems.append(f"{pid}: in the live sweep but not the table (new probe — refreeze)")
            continue
        if a is None:
            problems.append(f"{pid}: in the table but not the live sweep (removed probe — refreeze)")
            continue
        if e.outcome != a.outcome:
            problems.append(f"{pid}: outcome {e.outcome} -> {a.outcome}"
                            + (f" ({a.error})" if a.error else ""))
            continue
        if a.outcome == Outcome.REFUSES.value and e.marker != a.marker:
            problems.append(f"{pid}: refusal marker {e.marker!r} -> {a.marker!r}")
            continue
        was = {c["name"]: c["status"] for c in e.checks}
        for c in a.checks:
            before = was.get(c["name"])
            if before is None:
                problems.append(f"{pid}: new check {c['name']} (refreeze)")
            elif before != c["status"]:
                problems.append(
                    f"{pid}: check {c['name']} {before} -> {c['status']}: {c['detail']}")
    return problems
