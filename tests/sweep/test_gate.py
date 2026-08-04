"""The gate: the sweep's live verdicts must match the committed table.

Four signals, per the spec:
  LOWERS but value != oracle   -> a wrong number (the class this exists to find)
  LOWERS where table REFUSES   -> newly admitted; needs an oracle value
  REFUSES where table LOWERS   -> a regression, or an over-refusal
  MALFORMED anywhere           -> always a defect
"""
import pytest

from flatppl_testsuite.config import CONFIG
from flatppl_testsuite.sweep import table

pytestmark = pytest.mark.skipif(
    not CONFIG.flatppl_bin.exists(), reason="needs the flatppl binary"
)


def test_the_running_binary_matches_the_table_it_is_compared_against():
    """Checked BEFORE any per-probe comparison, and failing on its own with
    ONE message: a per-probe diff against a DIFFERENT determinizer is noise,
    not signal (this is what actually broke CI -- 22 phantom
    "REFUSES where the table LOWERS" rows, all query-ordering probes whose
    lowering path only exists on a branch, not on the main build CI runs).
    Unknown provenance on either side also fails here, deliberately, rather
    than letting an unverifiable comparison report green.
    """
    problem = table.check_provenance(table.DEFAULT_PATH)
    assert problem is None, problem


def test_live_sweep_matches_the_committed_table():
    expected = table.load(table.DEFAULT_PATH)
    assert expected, "no committed verdict table — run `pixi run sweep-regen`"
    actual = {r.probe_id: r for r in table.sweep(slice_only=True)}
    problems = table.diff(expected, actual)
    assert not problems, "sweep diverged from the committed table:\n" + "\n".join(problems)


def test_the_table_records_no_malformed_and_no_wrong_numbers():
    """A committed table containing a MALFORMED row, or a LOWERS row whose value
    disagrees with its oracle, is a determiniser defect that has been frozen
    rather than reported. Regenerating the table must never be a way to make a
    defect green."""
    rows = table.load(table.DEFAULT_PATH).values()
    malformed = [r.probe_id for r in rows if r.outcome == "MALFORMED"]
    assert not malformed, f"MALFORMED rows frozen in the table: {malformed}"


def test_every_dirichlet_row_carries_the_open_wording_question():
    """The §08-vs-§03/§06 reference-measure inconsistency has to be visible in the
    COMMITTED artifact, not only in `measure-algebra-audit.md`. A Dirichlet row whose
    `oracle` and `value` agree looks like clean independent corroboration; it is not,
    because the oracle and the engine both transcribe §08's formula, so neither can
    detect a `log sqrt(n)` error in the reference measure. The flag is what stops the
    next reader taking that agreement at face value.

    The values themselves are settled — numerical parity with Stan, NumPyro and scipy
    requires §08's formula — so this is `spec_wording_pending`, not a defect and not
    an unvalidated oracle.
    """
    rows = table.load(table.DEFAULT_PATH).values()
    dirichlet = [r for r in rows if r.probe_id.startswith("dirichlet.")]
    assert dirichlet, "no Dirichlet rows in the committed table"
    unflagged = [r.probe_id for r in dirichlet if not r.spec_wording_pending]
    assert not unflagged, (
        "Dirichlet rows recorded without the open reference-measure wording "
        f"question: {unflagged}")
    for r in dirichlet:
        assert r.spec_wording_note, f"{r.probe_id}: flagged with no reason recorded"
    # Rows with a finite oracle must carry the alternative reading, exactly log sqrt(3)
    # below it, so a future spec ruling can be applied mechanically.
    import math as _math
    for r in dirichlet:
        if r.oracle is None or not _math.isfinite(r.oracle):
            continue
        assert r.oracle_alt_reading is not None, (
            f"{r.probe_id}: has an oracle value but no alternative reading")
        gap = r.oracle - r.oracle_alt_reading
        assert abs(gap - 0.5 * _math.log(3)) < 1e-12, (
            f"{r.probe_id}: the two readings are {gap} apart, expected log sqrt(3)")


def test_no_row_outside_the_dirichlet_base_is_flagged_spec_wording_pending():
    """Scope the flag, so it keeps meaning something. Only §08's Dirichlet reference
    measure is worded two ways; a flag that spread would stop being a signal."""
    rows = table.load(table.DEFAULT_PATH).values()
    stray = [r.probe_id for r in rows
             if r.spec_wording_pending and not r.probe_id.startswith("dirichlet.")]
    assert not stray, f"spec_wording_pending leaked onto unrelated rows: {stray}"


def test_the_table_flags_no_unreviewed_wrong_numbers():
    """The preceding test's name promises "no wrong numbers", but its body only
    checks MALFORMED -- a LOWERS row IS allowed to disagree with its oracle,
    provided `known_defect` says so and names why (see `table._known_defect_reason`).
    An UNFLAGGED mismatch is exactly the defect this whole module exists to
    surface, so it fails here rather than being silently frozen alongside the
    two investigated defects.
    """
    from flatppl_testsuite.scoring.compare import compare_scalar

    rows = table.load(table.DEFAULT_PATH).values()
    unflagged = []
    for r in rows:
        if r.outcome != "LOWERS" or r.known_defect:
            continue
        if r.value is None or r.oracle is None:
            continue
        try:
            compare_scalar(r.value, r.oracle, {"atol": 1e-9, "rtol": 1e-9})
        except AssertionError:
            unflagged.append(r.probe_id)
    assert not unflagged, f"unreviewed wrong numbers frozen in the table: {unflagged}"
