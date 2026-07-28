"""The committed verdict table: one row per probe, frozen to disk, diffed
against a live rerun by the CI gate.

A row's `outcome`/`value`/`marker` come straight from `classify()` (Task 3);
its `oracle`/`oracle_unvalidated` come from `true_logpdf()` (Task 2), computed
for EVERY probe regardless of outcome -- a REFUSES row still carries the
value the density SHOULD have, so the day it starts lowering there is already
a number to check it against, not a fresh derivation.

`spec_justified` is decided from the PROBE'S OWN STRUCTURE (spelling, wrap
kind), not from `classify()`'s `marker` text: the ONE guard this probe space's
`record`-spelling REFUSES trace to is §04 "Calling conventions"' auto-splat
correspondence rule, applied to a scalar map (`pushfwd`/`affine`/`locscale`)
over a record variate (verified against a pinned binary: `refuse user-call
...: value must be a record`, `refuse divide ...: value must be a record`,
`refuse Lit(Real(2.0)) ...: locscale base measure variate domain is not
confirmed scalar or vector`, and the pushfwd-log/sqrt domain guard refusing
because a record obscures the support it would otherwise confirm). Keying off
`marker` instead would be fragile: `_marker`'s word-slice is dominated by the
generic `refuse X (node NodeId(N)):` preamble and a gensym'd node id, so
semantically-identical refusals can carry different marker strings across
regens. A refusal that does NOT match this structural predicate is an
unrecognized, unreviewed refusal -- `spec_justified = False`, a tracked
over-refusal, not a pass. (Which refusals a given determinizer commit actually
PRODUCES is a moving target -- e.g. a determinizer that can't yet lower a
second `logdensityof` query against one measure refuses every
`pinned_earlier`/`pinned_later` probe outright, regardless of spelling, and
those correctly show up as unjustified over-refusals here, not this guard.)

`known_defect` marks a LOWERS row whose value does not match its oracle AND
whose (base, wrap, spelling) matches a defect this sweep has actually
investigated (see `_known_defect_reason` below) -- never merely "this row
happens to mismatch". An unrecognized mismatch stays unflagged, so `diff()`
still reports it: regenerating the table is not a way to make a defect green.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from flatppl_testsuite.config import CONFIG
from flatppl_testsuite.scoring.compare import compare_scalar
from flatppl_testsuite.sweep.classify import Outcome, classify
from flatppl_testsuite.sweep.oracle import OracleUnsupported, true_logpdf
from flatppl_testsuite.sweep.space import (
    BASES,
    ORDERINGS,
    WRAPS,
    Probe,
    Wrap,
    _point_for,
    _supported,
    enumerate_probes,
)

DEFAULT_PATH = Path(__file__).resolve().parents[3] / "verdicts" / "density-sweep.json"

_TOLERANCE = {"atol": 1e-9, "rtol": 1e-9}

# The guard behind every REFUSES this probe space produces (verified: 126/126
# REFUSES in a full sweep are `spelling == "record"` wrapped in one of these
# three kinds -- see the module docstring). §04 "Calling conventions"'
# auto-splat rule: a scalar map's argument names must correspond to a
# record's field names, or the call is a static error; `affine`/`locscale`
# desugar to the same scalar-map shape, and `pushfwd`'s log/sqrt domain guard
# also refuses through a record because the record obscures the support it
# would otherwise confirm.
_AUTO_SPLAT_REFUSAL_WRAPS = {"pushfwd", "affine", "locscale"}


@dataclass(frozen=True)
class Row:
    probe_id: str
    outcome: str
    oracle: float | None
    value: float | None
    marker: str | None
    spec_justified: bool | None
    oracle_unvalidated: bool
    known_defect: bool = False
    known_defect_reason: str | None = None


def _spec_justified(probe: Probe, outcome: str) -> bool | None:
    if outcome != Outcome.REFUSES.value:
        return None
    wrap = probe.wraps[0]
    if probe.spelling == "record" and wrap.kind in _AUTO_SPLAT_REFUSAL_WRAPS:
        return True
    return False  # an unrecognized refusal: a tracked over-refusal, not a pass


def _known_defect_reason(probe: Probe) -> str | None:
    """Investigated, cited defects only -- see the module docstring. Both were
    found and confirmed by hand against the pinned `c570844` binary's emitted
    FlatPDL; neither is fixed here (that is out of scope for this module)."""
    wrap = probe.wraps[0]
    if probe.spelling == "record" and wrap.kind == "truncate":
        return ("truncate's containment gate compares the WHOLE record variate "
                "against the interval instead of the scalar field it wraps -- "
                "emitted as `record(x = ...) in interval(lo, hi)`, which is "
                "always false, so the density is always -inf regardless of the "
                "query point")
    if (probe.base.kind == "poisson" and wrap.kind == "pushfwd"
            and wrap.args[0] == "sqrt"
            and probe.spelling in ("direct", "stochastic_node")):
        return ("pushfwd(sqrt, Poisson) feeds the float-roundtrip preimage "
                "(y*y, e.g. 2.0000000000000004) straight into "
                "builtin_logdensityof(Poisson, ...) with no integer snap, so "
                "it lands off-lattice and scores -inf instead of the correct pmf")
    if probe.base.kind == "poisson" and (
            wrap.kind in ("affine", "locscale")
            or (wrap.kind == "pushfwd" and wrap.args[0] == "exp")):
        # §06 line 28: counting measure is not distorted by a bijection, so a
        # discrete base should carry NO log-volume term (see oracle.py's own
        # `_is_discrete` zeroing). Verified emitted FlatPDL subtracts one
        # anyway: `builtin_logdensityof(Poisson, ..., log(y)) - log(y)` for
        # pushfwd(exp); `builtin_logdensityof(Poisson, ..., (y-b)/a) -
        # log(abs(a))` for affine/locscale. `pushfwd(neg)` is unaffected
        # (its volume element is 0, so subtracting it is a no-op) -- verified
        # numerically correct, not merely assumed.
        return ("emits a spurious log-volume subtraction for the discrete "
                "Poisson base under a continuous bijection (affine/locscale/"
                "pushfwd(exp)) -- the counting measure isn't distorted by a "
                "bijection, so no volume term should be subtracted at all")
    return None


def _row_for(probe: Probe) -> Row:
    v = classify(probe)
    try:
        oracle_val = true_logpdf(probe)
        oracle_unvalidated = False
    except OracleUnsupported:
        oracle_val = None
        oracle_unvalidated = True

    known_defect = False
    known_defect_reason = None
    if v.outcome == Outcome.LOWERS and v.value is not None and oracle_val is not None:
        try:
            compare_scalar(v.value, oracle_val, _TOLERANCE)
        except AssertionError:
            reason = _known_defect_reason(probe)
            if reason is not None:
                known_defect = True
                known_defect_reason = reason

    return Row(
        probe_id=probe.id,
        outcome=v.outcome.value if isinstance(v.outcome, Outcome) else v.outcome,
        oracle=oracle_val,
        value=v.value,
        marker=v.marker,
        spec_justified=_spec_justified(probe, v.outcome),
        oracle_unvalidated=oracle_unvalidated,
        known_defect=known_defect,
        known_defect_reason=known_defect_reason,
    )


# --------------------------------------------------------------------------
# The CI slice
# --------------------------------------------------------------------------

# What the slice fixes/excludes, relative to the full space -- named here
# once so `_slice_probes` and the committed table's metadata cannot drift
# apart (the metadata is written FROM this constant, not hand-copied). Text,
# not a value list: `spelling`/`base` are no longer cleanly "excluded", since
# `_KNOWN_DEFECT_COVERAGE` below punches a couple of probes back in specifically
# so the two known defects are regression-visible to the CI gate.
SLICE_EXCLUDED_AXES = {
    "spelling": ("excludes stochastic_node entirely; excludes record except "
                 "one truncate probe (base=normal) covering the known "
                 "record-containment-gate defect"),
    "base": ("excludes beta entirely; excludes poisson except pushfwd(sqrt) "
             "at spelling=direct, covering the known float-roundtrip defect"),
    "consumer": "fixed False (True excluded)",
}
SLICE_DESCRIPTION = (
    "every (wrap, ordering) pair, spelling='direct', consumer=False; "
    "base='normal' except pushfwd(log)/pushfwd(sqrt), which normal's domain "
    "guard refuses -- those use 'gamma' instead (see space._supported); plus "
    "a handful of extra probes so the two known defects are inside the "
    "slice (see SLICE_EXCLUDED_AXES / _KNOWN_DEFECT_COVERAGE)"
)

# Minimal punch-through of the excluded spelling/base axes: enough probes,
# across all three orderings, to make each currently-known defect (see
# `_known_defect_reason`) visible to the fast CI slice rather than only to a
# `--full` run. Not a general record/poisson slice -- one shape per defect.
_KNOWN_DEFECT_COVERAGE = [
    dict(base_kind="normal", wrap=Wrap("truncate", (0.0, "inf")), spelling="record"),
    dict(base_kind="poisson", wrap=Wrap("pushfwd", ("sqrt",)), spelling="direct"),
]


def _slice_probes() -> list[Probe]:
    by_kind = {b.kind: b for b in BASES}
    normal, gamma = by_kind["normal"], by_kind["gamma"]
    out: list[Probe] = []
    for wrap in WRAPS:
        base = normal if _supported(normal, wrap) else gamma
        wname = wrap.kind + ("_" + "_".join(str(a) for a in wrap.args) if wrap.args else "")
        for ordering in ORDERINGS:
            pid = f"{base.kind}.{wname}.direct.{ordering}.noconsumer"
            out.append(Probe(
                id=pid, base=base, wraps=(wrap,), spelling="direct",
                ordering=ordering, consumer=False, point=_point_for(base, wrap),
            ))
    for extra in _KNOWN_DEFECT_COVERAGE:
        base = by_kind[extra["base_kind"]]
        wrap = extra["wrap"]
        spelling = extra["spelling"]
        wname = wrap.kind + ("_" + "_".join(str(a) for a in wrap.args) if wrap.args else "")
        for ordering in ORDERINGS:
            pid = f"{base.kind}.{wname}.{spelling}.{ordering}.noconsumer"
            out.append(Probe(
                id=pid, base=base, wraps=(wrap,), spelling=spelling,
                ordering=ordering, consumer=False, point=_point_for(base, wrap),
            ))
    return out


def sweep(slice_only: bool = False) -> list[Row]:
    probes = _slice_probes() if slice_only else enumerate_probes()
    return [_row_for(p) for p in probes]


# --------------------------------------------------------------------------
# Persistence -- ±inf/nan do not survive JSON, so they round-trip as the
# strings "inf"/"-inf"/"nan", matching `detjs_exec.parse_expected`.
# --------------------------------------------------------------------------

def _dump_num(x: float | None):
    if x is None:
        return None
    if math.isnan(x):
        return "nan"
    if math.isinf(x):
        return "inf" if x > 0 else "-inf"
    return x


def _load_num(v):
    return None if v is None else float(v)


def save(path: Path, rows: list[Row], *, commit: str | None = None) -> None:
    """Write `rows` (sorted by `probe_id`, so a diff is readable) plus
    metadata documenting the determinizer commit this table was generated
    against and what the CI slice excludes -- both have to be visible in the
    committed artifact itself, not just in a report that can drift from it."""
    ordered = sorted(rows, key=lambda r: r.probe_id)
    counts = {o: sum(1 for r in ordered if r.outcome == o)
              for o in (Outcome.LOWERS.value, Outcome.REFUSES.value, Outcome.MALFORMED.value)}
    doc = {
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "determinizer_commit": commit or "unknown",
            "probe_count": len(ordered),
            "outcome_counts": counts,
            "ci_slice": {
                "description": SLICE_DESCRIPTION,
                "excludes": SLICE_EXCLUDED_AXES,
            },
        },
        "rows": [
            {
                "probe_id": r.probe_id,
                "outcome": r.outcome,
                "oracle": _dump_num(r.oracle),
                "value": _dump_num(r.value),
                "marker": r.marker,
                "spec_justified": r.spec_justified,
                "oracle_unvalidated": r.oracle_unvalidated,
                "known_defect": r.known_defect,
                "known_defect_reason": r.known_defect_reason,
            }
            for r in ordered
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2) + "\n")


def load(path: Path) -> dict[str, Row]:
    if not path.exists():
        return {}
    doc = json.loads(path.read_text())
    rows = doc["rows"] if isinstance(doc, dict) and "rows" in doc else doc
    out = {}
    for r in rows:
        row = Row(
            probe_id=r["probe_id"],
            outcome=r["outcome"],
            oracle=_load_num(r.get("oracle")),
            value=_load_num(r.get("value")),
            marker=r.get("marker"),
            spec_justified=r.get("spec_justified"),
            oracle_unvalidated=r.get("oracle_unvalidated", False),
            known_defect=r.get("known_defect", False),
            known_defect_reason=r.get("known_defect_reason"),
        )
        out[row.probe_id] = row
    return out


def load_metadata(path: Path) -> dict:
    if not path.exists():
        return {}
    doc = json.loads(path.read_text())
    return doc.get("metadata", {}) if isinstance(doc, dict) else {}


# --------------------------------------------------------------------------
# Provenance -- which determinizer commit is actually running, vs which one
# the committed table claims to have been generated against. A `diff()` full
# of "REFUSES where the table LOWERS" against a DIFFERENT binary is not a
# regression, it's noise: query-ordering probes on this branch lower via a
# bare-`lawof` value-law path that main doesn't have yet, so they correctly
# refuse there. Checked once, up front, before any per-probe comparison.
# --------------------------------------------------------------------------

_COMMIT_SIDECAR_NAME = "flatppl-rust.commit"


def resolved_commit() -> str | None:
    """The commit the CURRENTLY CONFIGURED `flatppl` binary was built from.

    Two sources, in order: `FLATPPL_RUST_COMMIT` (an explicit override, for a
    hand-built binary with no sidecar file -- e.g. a scratch build used to
    regenerate this table), then the sidecar `scripts/setup.sh` writes next
    to the binary it installs (`<install-root>/flatppl-rust.commit`, sibling
    of `.crates2.json`). Neither may exist -- FLATPPL_BIN can point at a
    co-development sibling build that never went through `pixi run setup` --
    and that unknown provenance must be REPRESENTABLE (`None`), not a crash.
    """
    env = os.environ.get("FLATPPL_RUST_COMMIT")
    if env and env.strip():
        return env.strip()
    sidecar = CONFIG.flatppl_bin.resolve().parent.parent / _COMMIT_SIDECAR_NAME
    if not sidecar.exists():
        return None
    text = sidecar.read_text().strip()
    return text or None


def check_provenance(path: Path) -> str | None:
    """`None` if the live binary's commit matches the committed table's
    `determinizer_commit`; otherwise ONE explanatory message. Deliberately
    not a per-probe diff -- every per-probe divergence is uninterpretable
    once the binaries differ. Unknown provenance on EITHER side is ALSO a
    failure, not a pass: an unverifiable gate must not report green.
    """
    table_commit = load_metadata(path).get("determinizer_commit")
    live_commit = resolved_commit()
    if not table_commit or table_commit == "unknown":
        return ("the committed table's determinizer_commit is unrecorded "
                "(\"unknown\") -- regenerate with a known commit before trusting the gate")
    if not live_commit:
        return (f"table was generated against determinizer {table_commit}, but this "
                f"run's flatppl binary has no recorded provenance (no "
                f"{_COMMIT_SIDECAR_NAME} next to it, and FLATPPL_RUST_COMMIT unset) -- "
                f"regenerate with `pixi run sweep-regen`, or check out {table_commit}")
    if live_commit != table_commit:
        return (f"table was generated against determinizer {table_commit}, this run "
                f"used {live_commit} -- regenerate with `pixi run sweep-regen`, or "
                f"check out {table_commit}")
    return None


# --------------------------------------------------------------------------
# The diff
# --------------------------------------------------------------------------

def diff(expected: dict[str, Row], actual: dict[str, Row]) -> list[str]:
    """One line per divergence. Only checked from `actual`'s side: when
    `actual` comes from `sweep(slice_only=True)` it is, BY CONSTRUCTION, a
    subset of the committed (full-space) table's probe ids, so a probe
    present in `expected` only is the slice simply not covering that axis --
    documented in the table's metadata, not a divergence. A probe present in
    `actual` only (not found in `expected` at all) IS a divergence: either the
    committed table is stale, or the slice/id scheme drifted from it.
    """
    problems: list[str] = []
    for pid, a in sorted(actual.items()):
        e = expected.get(pid)
        if e is None:
            problems.append(f"{pid}: not in the committed table (run `pixi run sweep-regen`)")
            continue

        if a.outcome == Outcome.MALFORMED.value:
            problems.append(f"{pid}: MALFORMED (marker={a.marker}) -- always a defect")
        elif a.outcome == Outcome.LOWERS.value and e.outcome == Outcome.REFUSES.value:
            problems.append(f"{pid}: newly LOWERS where the table REFUSES -- needs an oracle value")
        elif a.outcome == Outcome.REFUSES.value and e.outcome == Outcome.LOWERS.value:
            problems.append(f"{pid}: REFUSES where the table LOWERS -- a regression, or an over-refusal")
        elif a.outcome != e.outcome:
            problems.append(f"{pid}: outcome changed: table={e.outcome} live={a.outcome}")

        if (a.outcome == Outcome.LOWERS.value and not a.known_defect
                and a.value is not None and a.oracle is not None):
            try:
                compare_scalar(a.value, a.oracle, _TOLERANCE)
            except AssertionError as err:
                problems.append(f"{pid}: LOWERS but value != oracle: {err}")
    return problems
