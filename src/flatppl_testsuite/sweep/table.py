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
correspondence rule, applied to a scalar map over a record variate. Three
`space.Wrap` kinds reach it: `pushfwd` (a bare map), `locscale` (a named §06
construct that IS itself `pushfwd(x -> scale*x+shift, m)` per spec), and
`affine` -- this sweep's OWN axis label, not a FlatPPL construct, for the
lambda-pushfwd SPELLING of that same affine map (see `space.WRAPS`). All
three desugar to one scalar map applied to the base's variate (verified
against a pinned binary: `refuse user-call ...: value must be a record`,
`refuse divide ...: value must be a record`, `refuse Lit(Real(2.0)) ...:
locscale base measure variate domain is not confirmed scalar or vector`, and
the pushfwd-log/sqrt domain guard refusing because a record obscures the
support it would otherwise confirm). Keying off
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
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from flatppl_testsuite.config import CONFIG
from flatppl_testsuite.scoring.compare import compare_scalar
from flatppl_testsuite.sweep.classify import Outcome, classify
from flatppl_testsuite.sweep.oracle import (
    OracleUnsupported,
    simplex_chart_to_hausdorff_offset,
    true_logpdf,
)
from flatppl_testsuite.sweep.space import (
    BASES,
    ORDERINGS,
    WRAPS,
    Probe,
    Wrap,
    _point_for,
    _supported,
    _vector_point_for,
    _wrap_name,
    enumerate_probes,
    is_vector_base,
    vector_shapes,
)

DEFAULT_PATH = Path(__file__).resolve().parents[3] / "verdicts" / "density-sweep.json"

_TOLERANCE = {"atol": 1e-9, "rtol": 1e-9}

# The guard behind every REFUSES this probe space's `record` spelling
# produces -- see the module docstring. §04 "Calling conventions"' auto-splat
# rule: a scalar map's argument names must correspond to a record's field
# names, or the call is a static error. `pushfwd`, `locscale`, and this
# sweep's `affine` label (a SPELLING of `pushfwd`, not a separate construct --
# see `space.WRAPS`) all apply such a scalar map, so all three trip the same
# guard; `pushfwd`'s log/sqrt domain guard also refuses through a record
# because the record obscures the support it would otherwise confirm.
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
    # An UNRESOLVED SPEC-WORDING question behind this row's reference measure. Not a
    # doubtful value: `spec_wording_pending` says the number is the one the project
    # requires (numerical parity with Stan/NumPyro/scipy) while two normative
    # sections word its reference measure incompatibly, so what will move is the
    # spec text, not the row. `oracle_alt_reading` records what the OTHER wording
    # would imply, so a future ruling can be checked mechanically instead of
    # re-derived. See `_spec_wording_pending`.
    spec_wording_pending: bool = False
    spec_wording_note: str | None = None
    oracle_alt_reading: float | None = None


def _spec_justified(probe: Probe, outcome: str) -> bool | None:
    if outcome != Outcome.REFUSES.value:
        return None
    wrap = probe.wraps[0]
    if probe.spelling == "record" and wrap.kind in _AUTO_SPLAT_REFUSAL_WRAPS:
        return True
    if probe.spelling == "record" and wrap.kind == "truncate":
        # The determiniser now refuses a truncation set whose space provably
        # mismatches the variate (record against a scalar `interval`). §06's
        # ν(A) = M(A ∩ S) made -inf the correct zero-measure density, so the
        # refusal is a diagnostic for an ill-typed restriction (§07 `in`
        # requires x's type to match S's element type), not a capability gap.
        # Same shape the oracle declines (`oracle.py`, OracleUnsupported).
        return True
    if is_vector_base(probe.base) and wrap.kind == "truncate":
        # The SAME set-kind check, reached from the other side: a scalar
        # `interval` (§03 — a set of reals) against a vector variate. The
        # determiniser names both spaces and refuses
        # (`density.rs::refuse_truncation_set_kind_mismatch`), and the oracle
        # declines the shape for the same §06 reason it declines the record one.
        # Conformant, not a gap.
        #
        # As everywhere in this function, the refusal's MARKER is deliberately not
        # consulted, so this returns True for a refusal of this SHAPE whatever
        # reason the determiniser gave — a refusal for an unrelated reason would
        # still read `spec_justified`. That is the stated policy, not an oversight
        # (see the module docstring), and the `aa1cdcb` diagnostics rewording
        # vindicated it: keying on the message would have silently reclassified 84
        # conformant refusals when only the prose changed. The cost is this blind
        # spot; `tests/sweep/test_vector_arms.py` covers it from the other side by
        # asserting the emitted arm.
        return True
    return False  # an unrecognized refusal: a tracked over-refusal, not a pass


_DIRICHLET_WORDING_NOTE = (
    "§08's Dirichlet formula is normalised against the CHART measure "
    "dx_1...dx_{n-1}, but §03 \"Standard simplex\" and §06 \"Lebesgue\" define "
    "`Lebesgue(stdsimplex(n))` as SURFACE AREA on the embedded simplex, i.e. the "
    "(n-1)-dimensional Hausdorff measure, whose area element is "
    "sqrt(n) dx_1...dx_{n-1}. The two readings are log sqrt(n) apart "
    "(0.5493061443340549 at n = 3). This row's `oracle` and `value` are the CHART "
    "reading, which is required for numerical parity with Stan, NumPyro and scipy "
    "and will not move; `oracle_alt_reading` is what the §03/§06 wording implies. "
    "What is unresolved is which wording flatppl-design settles on -- tracked as an "
    "open spec question in flatppl-dev/measure-algebra-audit.md"
)


def _spec_wording_pending(probe: Probe,
                          chart: float | None) -> tuple[str, float | None] | None:
    """`(note, alternative_reading)` when this probe's reference measure is subject to
    an unresolved spec-wording question, else `None`.

    Keyed on the probe's own structure, like every other classifier here, never on
    engine or refusal text. Only `dirichlet` qualifies: §08's `Multinomial` density is
    w.r.t. `iid(Counting(integers), k)`, and no section words a counting reference
    two ways.

    `chart` is the oracle value the CALLER already computed for this probe (`None`
    where the oracle withheld one), threaded in rather than recomputed — one
    `true_logpdf` per probe, and no way for the row's `oracle` and its
    `oracle_alt_reading` to be derived from two different evaluations. The
    alternative reading needs a finite value to shift; a withheld, refused or
    infinite row still carries the note with `None`, which keeps the open question
    visible on that row.
    """
    if probe.base.kind != "dirichlet":
        return None
    if chart is None or not math.isfinite(chart):
        return (_DIRICHLET_WORDING_NOTE, None)
    (alpha,) = probe.base.params
    offset = simplex_chart_to_hausdorff_offset(len(alpha))
    return (_DIRICHLET_WORDING_NOTE, chart - offset)


def _known_defect_reason(probe: Probe) -> str | None:
    """Investigated, cited defects only -- see the module docstring. Both were
    found and confirmed by hand against the pinned `c570844` binary's emitted
    FlatPDL. Both are since fixed in the determiniser (the truncate shape now
    refuses at lowering; the discrete preimage is snapped to the lattice), so
    these entries are regression tripwires: a row matching one again means the
    fix regressed."""
    wrap = probe.wraps[0]
    if probe.spelling == "record" and wrap.kind == "truncate":
        # NOT "the gate should have compared the field". §06's ν(A) = M(A ∩ S)
        # makes -inf the correct density of the zero measure here, because a
        # scalar `interval` (§03) and a record variate are disjoint spaces, and
        # no spec rule restricts a record's field by a scalar set -- §04's
        # auto-splat is a calling convention and does not reach `truncate`'s
        # second argument. The defect is that an ill-typed truncation reads as a
        # computation: a modelling error should be a static refusal, which is
        # what the same engine does for a record variate under `pushfwd`. The
        # oracle therefore supplies no value for this shape (see `oracle.py`).
        return ("truncate accepts a truncation set whose space does not match "
                "the measure's variate -- `record(x = ...) in interval(lo, hi)` "
                "is always false, so it silently yields the zero measure's -inf "
                "instead of refusing an ill-typed restriction")
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
        # Three wrap kinds, one scalar map: `pushfwd(exp)`, this sweep's
        # `affine` label (the lambda-pushfwd SPELLING of `x -> a*x+b`, not a
        # separate FlatPPL construct -- see `space.WRAPS`), and `locscale`
        # (§06's named construct for that same map). §06 line 28: counting
        # measure is not distorted by a bijection, so a discrete base should
        # carry NO log-volume term (see oracle.py's own `_is_discrete`
        # zeroing). Verified emitted FlatPDL subtracts one anyway:
        # `builtin_logdensityof(Poisson, ..., log(y)) - log(y)` for
        # pushfwd(exp); `builtin_logdensityof(Poisson, ..., (y-b)/a) -
        # log(abs(a))` for the affine-map spellings. `pushfwd(neg)` is
        # unaffected (its volume element is 0, so subtracting it is a no-op)
        # -- verified numerically correct, not merely assumed. (The returned
        # string below still says "affine/locscale/pushfwd(exp)" -- matching
        # the wording already baked into the committed table's
        # `known_defect_reason` field; reworded here in the comment only, not
        # the persisted string, so this edit causes no diff against what is
        # already on disk.)
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
    if v.outcome == Outcome.LOWERS and v.value is not None:
        # A mismatch against the oracle is the usual trigger. A LOWERS row with NO
        # oracle is the other one: a shape the oracle deliberately refuses to value
        # because the model is ill-typed still carries a defect -- the determiniser
        # returned a number for it. `_known_defect_reason` returns None for every
        # shape it does not recognise, so this admits only investigated ones.
        mismatched = oracle_val is None
        if not mismatched:
            try:
                compare_scalar(v.value, oracle_val, _TOLERANCE)
            except AssertionError:
                mismatched = True
        if mismatched:
            reason = _known_defect_reason(probe)
            if reason is not None:
                known_defect = True
                known_defect_reason = reason

    wording = _spec_wording_pending(probe, oracle_val)

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
        spec_wording_pending=wording is not None,
        spec_wording_note=wording[0] if wording else None,
        oracle_alt_reading=wording[1] if wording else None,
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
    "vector_family": ("one probe per (base, wrap) shape space.vector_shapes() "
                      "generates, at spelling='direct' ordering='single'; the "
                      "shapes flatppl-js cannot evaluate are not in the family "
                      "at all (space._ENGINE_BLOCKED) and are pinned instead by "
                      "tests/sweep/test_vector_arms.py"),
}
SLICE_DESCRIPTION = (
    "every (wrap, ordering) pair, spelling='direct', consumer=False; "
    "base='normal' except pushfwd(log)/pushfwd(sqrt), which normal's domain "
    "guard refuses -- those use 'gamma' instead (see space._supported); plus "
    "a handful of extra probes so the two known defects are inside the "
    "slice (see SLICE_EXCLUDED_AXES / _KNOWN_DEFECT_COVERAGE); plus one probe "
    "per vector-family (base, wrap) shape, so every vector arm the family "
    "covers is inside the fast gate rather than only in a --full run"
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
        wname = _wrap_name(wrap)
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
        wname = _wrap_name(wrap)
        for ordering in ORDERINGS:
            pid = f"{base.kind}.{wname}.{spelling}.{ordering}.noconsumer"
            out.append(Probe(
                id=pid, base=base, wraps=(wrap,), spelling=spelling,
                ordering=ordering, consumer=False, point=_point_for(base, wrap),
            ))
    # One probe per vector-family SHAPE, not per probe: the family's spelling and
    # ordering axes repeat arms the scalar slice already gates, while the (base,
    # wrap) shape is what decides which vector arm the determiniser emits. Every
    # shape in the family is therefore in the fast gate.
    for base, wrap in vector_shapes():
        pid = f"{base.kind}.{_wrap_name(wrap)}.direct.single.noconsumer"
        out.append(Probe(
            id=pid, base=base, wraps=(wrap,), spelling="direct",
            ordering="single", consumer=False, point=_vector_point_for(base, wrap),
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
            # The determiniser decides LOWERS vs REFUSES; the ENGINE decides
            # MALFORMED and every recorded value, so a table naming only the
            # determiniser is half-pinned.
            "engine_commit": engine_commit() or "unknown",
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
                "spec_wording_pending": r.spec_wording_pending,
                "spec_wording_note": r.spec_wording_note,
                "oracle_alt_reading": _dump_num(r.oracle_alt_reading),
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
            spec_wording_pending=r.get("spec_wording_pending", False),
            spec_wording_note=r.get("spec_wording_note"),
            oracle_alt_reading=_load_num(r.get("oracle_alt_reading")),
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


def engine_commit() -> str | None:
    """The commit of the flatppl-js checkout the scorer loads, or `None` when the
    directory is absent or is not a git checkout."""
    d = CONFIG.flatppl_js_dir
    if not d.exists():
        return None
    try:
        out = subprocess.run(["git", "-C", str(d), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None if out.returncode == 0 else None


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

    # The engine side, for the same reason. A missing engine directory is the
    # loudest case and the one that actually misleads: every gated probe then
    # classifies MALFORMED, which reads as 16 determiniser defects rather than as
    # an absent scorer. The sibling default (`../flatppl-js`) resolves off the
    # repo root, so it does NOT exist when the harness runs from a worktree --
    # set `FLATPPL_JS_DIR` there.
    if not CONFIG.flatppl_js_dir.exists():
        return (f"the configured engine directory does not exist "
                f"({CONFIG.flatppl_js_dir}) -- every gated probe would classify "
                f"MALFORMED for want of a scorer; set FLATPPL_JS_DIR")
    # A DIFFERING engine commit is deliberately not a failure, unlike a differing
    # determiniser commit. `setup.sh` pins the engine at `FLATPPL_JS_REF`, default
    # `main`, so it advances on every unrelated flatppl-js merge; failing here would
    # be a standing false alarm rather than a signal. The commit is recorded in the
    # table's metadata so a reader can see which engine produced the values, and a
    # verdict that actually moved still surfaces through `diff()`.
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
