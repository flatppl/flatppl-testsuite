"""The gate: the sweep's live verdicts must match the committed table.

Five signals:
  LOWERS but value != oracle   -> a wrong number (the class this exists to find)
  LOWERS but oracle withheld   -> a number for a shape no spec rule values
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


def test_dirichlet_rows_record_the_coordinate_measure_value():
    """A plain regression check where an open-question flag used to be.

    §08's Dirichlet formula is normalised against the coordinate measure
    `dx_1 ... dx_{n-1}`, which §03 "Standard simplex" and §06 "Lebesgue" now name
    explicitly as what `Lebesgue(stdsimplex(n))` denotes -- and §03 states that the
    surface (Hausdorff) measure "is larger by the factor sqrt(n)". Those rows briefly
    carried a `spec_wording_pending` flag while §03/§06 said the opposite; the spec
    edit settled it in favour of the value the sweep already shipped, so the flag is
    gone and this pins the number instead.

    A row here reading 1.4735650458573875 rather than 2.0228711901914425 would mean
    something applied the sqrt(n) surface-measure correction the spec now rules out.
    """
    rows = table.load(table.DEFAULT_PATH).values()
    bare = [r for r in rows if r.probe_id.startswith("dirichlet.identity.")]
    assert bare, "no bare Dirichlet rows in the committed table"
    for r in bare:
        assert r.oracle == pytest.approx(2.0228711901914425, abs=1e-12), (
            f"{r.probe_id}: oracle {r.oracle} is not the coordinate-measure value")
        assert r.value == pytest.approx(2.0228711901914425, abs=1e-12), (
            f"{r.probe_id}: engine value {r.value} is not the coordinate-measure value")


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


def test_the_table_freezes_no_value_the_oracle_withheld():
    """The companion signal to the test above, which its `r.oracle is None` skip
    cannot see: a LOWERS row for a shape the ORACLE REFUSED TO VALUE.

    Those are not the same defect. A wrong number disagrees with a known truth; this
    one is a number for a shape no spec rule gives a density to at all, so there is
    nothing to disagree with and the comparison above silently passes it.

    Found by the shared-latent family's `singular` shape: the pre-#137 determiniser
    LOWERS `logdensityof(joint(lawof(y), lawof(y)), [0.5, 0.7])` to a finite value,
    while §06 "Singular joints" says the joint law "has no density w.r.t. the product
    reference measure". `known_defect` still excuses an investigated shape, same as
    for a wrong number.
    """
    rows = table.load(table.DEFAULT_PATH).values()
    unvalued = [r.probe_id for r in rows
                if r.outcome == "LOWERS" and not r.known_defect
                and r.value is not None and r.oracle is None]
    assert not unvalued, (
        "rows frozen with a value the oracle withheld: " + str(unvalued))
