"""Unit tests for the table module's pure logic -- persistence, the diff, and
the spec_justified/known_defect predicates -- kept separate from `test_gate.py`
(which needs the real binary) so these run with no engine present."""
import math

from flatppl_testsuite.sweep import table
from flatppl_testsuite.sweep.space import Base, Probe, Wrap


def _row(probe_id="p", outcome="LOWERS", oracle=-1.0, value=-1.0, marker=None,
         spec_justified=None, oracle_unvalidated=False, known_defect=False,
         known_defect_reason=None):
    return table.Row(probe_id=probe_id, outcome=outcome, oracle=oracle, value=value,
                      marker=marker, spec_justified=spec_justified,
                      oracle_unvalidated=oracle_unvalidated, known_defect=known_defect,
                      known_defect_reason=known_defect_reason)


# --------------------------------------------------------------------------
# save/load round-trip, including the ±inf/nan strings
# --------------------------------------------------------------------------

def test_save_load_round_trips_ordinary_rows(tmp_path):
    rows = [_row("b", value=2.0, oracle=2.0), _row("a", value=1.0, oracle=1.0)]
    path = tmp_path / "t.json"
    table.save(path, rows, commit="deadbeef")
    loaded = table.load(path)
    assert set(loaded) == {"a", "b"}
    assert loaded["a"].value == 1.0
    assert loaded["b"].oracle == 2.0


def test_save_orders_rows_by_probe_id_for_a_readable_diff(tmp_path):
    rows = [_row("z"), _row("a"), _row("m")]
    path = tmp_path / "t.json"
    table.save(path, rows)
    import json
    doc = json.loads(path.read_text())
    assert [r["probe_id"] for r in doc["rows"]] == ["a", "m", "z"]


def test_infinities_and_nan_round_trip_as_the_documented_strings(tmp_path):
    rows = [_row("inf_row", value=math.inf, oracle=math.inf),
            _row("neginf_row", value=-math.inf, oracle=-math.inf),
            _row("nan_row", value=math.nan, oracle=math.nan)]
    path = tmp_path / "t.json"
    table.save(path, rows)
    import json
    doc = json.loads(path.read_text())
    by_id = {r["probe_id"]: r for r in doc["rows"]}
    assert by_id["inf_row"]["value"] == "inf"
    assert by_id["neginf_row"]["value"] == "-inf"
    assert by_id["nan_row"]["value"] == "nan"

    loaded = table.load(path)
    assert loaded["inf_row"].value == math.inf
    assert loaded["neginf_row"].oracle == -math.inf
    assert math.isnan(loaded["nan_row"].value)


def test_load_of_a_missing_path_is_an_empty_table(tmp_path):
    assert table.load(tmp_path / "nope.json") == {}


def test_metadata_records_the_commit_and_the_slice_exclusions(tmp_path):
    path = tmp_path / "t.json"
    table.save(path, [_row()], commit="c570844")
    meta = table.load_metadata(path)
    assert meta["determinizer_commit"] == "c570844"
    assert meta["ci_slice"]["excludes"] == table.SLICE_EXCLUDED_AXES


# --------------------------------------------------------------------------
# The Outcome-as-str-Enum landmine (see classify.py / the sweep review):
# `str(Outcome.LOWERS)` is "Outcome.LOWERS", not "LOWERS" -- a table writer
# that used `str()` or an f-string on the raw outcome would freeze the wrong
# text. `_row_for` normalizes via `.value` before a Row is ever built; this
# pins that normalization against the landmine directly, with no binary
# needed (a bare Outcome member, no classify() call).
# --------------------------------------------------------------------------

def test_saved_outcome_is_the_plain_value_not_the_enum_repr(tmp_path):
    from flatppl_testsuite.sweep.classify import Outcome

    row = table.Row(probe_id="p", outcome=Outcome.LOWERS, oracle=None, value=None,
                     marker=None, spec_justified=None, oracle_unvalidated=False)
    path = tmp_path / "t.json"
    table.save(path, [row])
    import json
    saved = json.loads(path.read_text())["rows"][0]["outcome"]
    assert saved == "LOWERS"
    assert saved != "Outcome.LOWERS"


# --------------------------------------------------------------------------
# spec_justified: structural, not marker-text-based (see table.py's docstring
# -- every REFUSES this space produces is `record` spelling wrapped in
# pushfwd/affine/locscale; anything else is an unreviewed over-refusal).
# --------------------------------------------------------------------------

def _probe(base_kind="normal", wrap_kind="pushfwd", wrap_args=("exp",), spelling="record"):
    return Probe(id="t", base=Base(base_kind, (0.0, 1.0)),
                 wraps=(Wrap(wrap_kind, wrap_args),), spelling=spelling,
                 ordering="single", consumer=False, point=0.5)


def test_record_pushfwd_refusal_is_spec_justified():
    p = _probe(wrap_kind="pushfwd", spelling="record")
    assert table._spec_justified(p, "REFUSES") is True


def test_record_affine_and_locscale_refusals_are_spec_justified():
    assert table._spec_justified(_probe(wrap_kind="affine", wrap_args=(2.0, 1.0),
                                         spelling="record"), "REFUSES") is True
    assert table._spec_justified(_probe(wrap_kind="locscale", wrap_args=(1.0, 2.0),
                                         spelling="record"), "REFUSES") is True


def test_a_refusal_outside_the_known_auto_splat_shape_is_not_justified():
    p = _probe(wrap_kind="truncate", wrap_args=(0.0, "inf"), spelling="direct")
    assert table._spec_justified(p, "REFUSES") is False


def test_spec_justified_is_none_when_the_outcome_is_not_refuses():
    p = _probe()
    assert table._spec_justified(p, "LOWERS") is None
    assert table._spec_justified(p, "MALFORMED") is None


# --------------------------------------------------------------------------
# known_defect_reason: the two investigated defects, structurally, and
# nothing else -- an uninvestigated mismatch must stay unflagged (see
# test_gate.py::test_the_table_flags_no_unreviewed_wrong_numbers).
# --------------------------------------------------------------------------

def test_record_truncate_is_a_known_defect():
    p = _probe(wrap_kind="truncate", wrap_args=(0.0, "inf"), spelling="record")
    assert table._known_defect_reason(p) is not None


def test_poisson_pushfwd_sqrt_direct_is_a_known_defect():
    p = Probe(id="t", base=Base("poisson", (3.0,)), wraps=(Wrap("pushfwd", ("sqrt",)),),
              spelling="direct", ordering="single", consumer=False, point=1.4142135623730951)
    assert table._known_defect_reason(p) is not None


def test_poisson_pushfwd_sqrt_record_is_not_flagged():
    """The record spelling of this combo REFUSES (the auto-splat guard fires
    before density evaluation is ever reached), so it never reaches the
    known-defect predicate's caller with a mismatch -- but the predicate
    itself is scoped to (direct, stochastic_node) only, so it must say no
    even if asked about the record spelling directly."""
    p = Probe(id="t", base=Base("poisson", (3.0,)), wraps=(Wrap("pushfwd", ("sqrt",)),),
              spelling="record", ordering="single", consumer=False, point=1.4142135623730951)
    assert table._known_defect_reason(p) is None


def test_an_ordinary_probe_is_not_a_known_defect():
    p = _probe(wrap_kind="identity", wrap_args=(), spelling="direct")
    assert table._known_defect_reason(p) is None


# --------------------------------------------------------------------------
# diff(): the four signals, plus the "one side only" asymmetry (see
# diff()'s own docstring for why expected-only is not a divergence).
# --------------------------------------------------------------------------

def test_diff_is_empty_when_both_sides_agree():
    e = {"p": _row("p", outcome="LOWERS", value=-1.0, oracle=-1.0)}
    a = {"p": _row("p", outcome="LOWERS", value=-1.0, oracle=-1.0)}
    assert table.diff(e, a) == []


def test_diff_flags_a_wrong_number():
    e = {"p": _row("p", outcome="LOWERS", value=-1.0, oracle=-1.0)}
    a = {"p": _row("p", outcome="LOWERS", value=-2.0, oracle=-1.0)}
    problems = table.diff(e, a)
    assert len(problems) == 1 and "value != oracle" in problems[0]


def test_diff_does_not_flag_a_known_defects_wrong_number():
    e = {"p": _row("p", outcome="LOWERS", value=-math.inf, oracle=-1.0, known_defect=True)}
    a = {"p": _row("p", outcome="LOWERS", value=-math.inf, oracle=-1.0, known_defect=True)}
    assert table.diff(e, a) == []


def test_diff_flags_newly_lowers_where_the_table_refuses():
    e = {"p": _row("p", outcome="REFUSES", value=None, oracle=-1.0)}
    a = {"p": _row("p", outcome="LOWERS", value=-1.0, oracle=-1.0)}
    problems = table.diff(e, a)
    assert len(problems) == 1 and "newly LOWERS" in problems[0]


def test_diff_flags_refuses_where_the_table_lowers():
    e = {"p": _row("p", outcome="LOWERS", value=-1.0, oracle=-1.0)}
    a = {"p": _row("p", outcome="REFUSES", value=None, oracle=-1.0)}
    problems = table.diff(e, a)
    assert len(problems) == 1 and "REFUSES where the table LOWERS" in problems[0]


def test_diff_flags_malformed_unconditionally_even_if_both_sides_agree():
    e = {"p": _row("p", outcome="MALFORMED", value=None, oracle=None, marker="residual-user-call")}
    a = {"p": _row("p", outcome="MALFORMED", value=None, oracle=None, marker="residual-user-call")}
    problems = table.diff(e, a)
    assert len(problems) == 1 and "MALFORMED" in problems[0]


def test_diff_flags_a_probe_present_only_in_actual():
    a = {"new": _row("new")}
    problems = table.diff({}, a)
    assert len(problems) == 1 and "not in the committed table" in problems[0]


def test_diff_does_not_flag_a_probe_present_only_in_expected():
    """`actual` from `sweep(slice_only=True)` is, by construction, a strict
    subset of the committed full-space table -- that is the slice working as
    designed, not a divergence."""
    e = {"only_in_full_space": _row("only_in_full_space")}
    assert table.diff(e, {}) == []
