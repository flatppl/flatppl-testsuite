"""Unit tests for the table module's pure logic -- persistence, the diff, and
the spec_justified/known_defect predicates -- kept separate from `test_gate.py`
(which needs the real binary) so these run with no engine present."""
import math
from types import SimpleNamespace

import pytest

from flatppl_testsuite.sweep import table
from flatppl_testsuite.sweep.oracle import true_logpdf
from flatppl_testsuite.sweep.space import (
    Base,
    Probe,
    SharedLatentProbe,
    Wrap,
    shared_latent_shapes,
)


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


def _poisson_probe(wrap_kind, wrap_args, spelling="direct"):
    return Probe(id="t", base=Base("poisson", (3.0,)), wraps=(Wrap(wrap_kind, wrap_args),),
                 spelling=spelling, ordering="single", consumer=False, point=1.0)


def test_poisson_pushfwd_exp_affine_locscale_are_known_defects():
    """The missing discrete-no-volume-term defect: verified directly against
    main (9ac4ae76) -- pushfwd(exp)/affine/locscale over Poisson each emit a
    spurious `- log(...)` volume subtraction that a discrete base should
    never carry (§06 line 28's counting measure)."""
    assert table._known_defect_reason(_poisson_probe("pushfwd", ("exp",))) is not None
    assert table._known_defect_reason(_poisson_probe("affine", (2.0, 1.0))) is not None
    assert table._known_defect_reason(_poisson_probe("locscale", (1.0, 2.0))) is not None


def test_poisson_pushfwd_neg_is_not_a_known_defect():
    """`neg`'s volume element is 0, so subtracting it is a no-op -- verified
    numerically correct on main, not merely assumed exempt."""
    assert table._known_defect_reason(_poisson_probe("pushfwd", ("neg",))) is None


def test_the_volume_term_defect_is_scoped_to_poisson():
    """A continuous base's volume term IS supposed to be subtracted -- this
    defect is specifically about the counting measure not being distorted,
    so it must not fire for a continuous base."""
    assert table._known_defect_reason(_probe(base_kind="normal", wrap_kind="pushfwd",
                                              wrap_args=("exp",), spelling="direct")) is None


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


def _shared(shape, spelling="record_law", n=2):
    return SharedLatentProbe(id=f"shared.{shape}.n{n}.{spelling}.none", shape=shape,
                             n=n, spelling=spelling, latent_query="none",
                             point=(0.5, 0.7))


def test_a_singular_joints_refusal_is_spec_justified():
    """§06 "Singular joints": the joint law "has no density w.r.t. the product
    reference measure ... a density query is a static error where statically
    detectable, and is otherwise refused by the engine." Refusing is conformance."""
    assert table._spec_justified(_shared("singular"), "REFUSES") is True
    assert table._spec_justified(_shared("singular", "joint_pos"), "REFUSES") is True


@pytest.mark.parametrize("shape", ["fan", "chain", "disjoint"])
def test_every_other_shared_latent_refusal_is_an_over_refusal(shape):
    """Not a default falling through. §06 "Equivalent record law" gives all three
    equivalent spellings a density, §06 `iid` gives the product measure, and §06's
    contrast sentence gives the constructor joint the product of its marginals —
    every one closed-form, with the oracle carrying the value. So a refusal is a
    tracked capability gap.

    `chain` is the expected occupant if any: #131 lowered the FAN arm.
    """
    assert table._spec_justified(_shared(shape), "REFUSES") is False


def test_a_shared_latent_probe_is_never_a_flagged_known_defect():
    """This sweep has investigated no defect in the family, so nothing there may be
    frozen as an excused mismatch — an unflagged one fails the gate, which is the
    right outcome for a wrong number nobody has looked at."""
    for shape in ("fan", "chain", "disjoint", "singular"):
        for spelling in ("record_law", "joint_kw", "joint_pos"):
            assert table._known_defect_reason(_shared(shape, spelling)) is None


def test_the_slice_covers_every_shared_latent_shape_and_spelling():
    """The fast gate must see every (shape, spelling) pair, or a joint arm is only
    checked by a `--full` run that CI does not do."""
    ids = {p.id for p in table._slice_probes()}
    for shape, spelling in shared_latent_shapes():
        assert f"shared.{shape}.n2.{spelling}.none" in ids, (
            f"{shape}/{spelling} is outside the CI slice")


def test_diff_flags_a_value_the_oracle_withheld():
    """The signal `test_diff_flags_a_wrong_number`'s guard cannot reach.

    Its comparison is conditioned on `oracle is not None`, so before this check a
    LOWERS row for a shape the oracle refused to value produced NO problem line —
    which is how a pre-#137 determiniser's finite answer for
    `joint(lawof(y), lawof(y))` (§06: no density) would have survived a regen.

    Both sides LOWERS on purpose: that is the FROZEN state, the one that was silent.
    The transitional state is covered by
    `test_the_first_appearance_of_such_a_row_was_already_reported` below — the two
    together are what make the scoping claim in `diff`'s comment checkable.
    """
    e = {"p": _row("p", outcome="LOWERS", value=-1.0, oracle=None)}
    a = {"p": _row("p", outcome="LOWERS", value=-1.0, oracle=None)}
    problems = table.diff(e, a)
    assert len(problems) == 1 and "withholds any value" in problems[0]


def test_the_first_appearance_of_such_a_row_was_already_reported():
    """The honest scope of the check above: while the table still says REFUSES, the
    pre-existing `newly LOWERS where the table REFUSES` branch fires, so a
    determiniser answering a no-density shape for the FIRST time was never silent.

    Pinned so the comment in `diff` cannot overstate what the new signal added — the
    gap it closes is the frozen row, not the first sighting.
    """
    e = {"p": _row("p", outcome="REFUSES", value=None, oracle=None,
                   spec_justified=True)}
    a = {"p": _row("p", outcome="LOWERS", value=-1.0, oracle=None)}
    problems = table.diff(e, a)
    assert any("newly LOWERS" in p for p in problems), problems
    assert any("withholds any value" in p for p in problems), problems


def test_diff_does_not_flag_a_withheld_value_on_an_investigated_shape():
    """`known_defect` excuses it, exactly as it excuses a wrong number — the point
    is that an UNINVESTIGATED one is reported, not that the check cannot be
    satisfied."""
    e = {"p": _row("p", outcome="LOWERS", value=-1.0, oracle=None, known_defect=True)}
    a = {"p": _row("p", outcome="LOWERS", value=-1.0, oracle=None, known_defect=True)}
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


# --------------------------------------------------------------------------
# resolved_commit / check_provenance -- what actually broke CI (comparing a
# table generated against one determinizer to a run of a different one, which
# produced 22 phantom per-probe diffs no probe-level fix could explain).
# --------------------------------------------------------------------------

def test_resolved_commit_prefers_the_env_override(monkeypatch):
    monkeypatch.setenv("FLATPPL_RUST_COMMIT", "deadbeef")
    assert table.resolved_commit() == "deadbeef"


def test_resolved_commit_falls_back_to_the_setup_sh_sidecar(monkeypatch, tmp_path):
    monkeypatch.delenv("FLATPPL_RUST_COMMIT", raising=False)
    root = tmp_path / "install-root"
    (root / "bin").mkdir(parents=True)
    binary = root / "bin" / "flatppl"
    binary.write_text("")
    (root / "flatppl-rust.commit").write_text("cafebabe\n")
    monkeypatch.setattr(table, "CONFIG", SimpleNamespace(flatppl_bin=binary))
    assert table.resolved_commit() == "cafebabe"


def test_resolved_commit_is_none_with_no_env_and_no_sidecar(monkeypatch, tmp_path):
    """Representable as unknown, not a crash -- e.g. FLATPPL_BIN pointed at a
    co-development sibling build that never went through `pixi run setup`."""
    monkeypatch.delenv("FLATPPL_RUST_COMMIT", raising=False)
    root = tmp_path / "install-root"
    (root / "bin").mkdir(parents=True)
    binary = root / "bin" / "flatppl"
    binary.write_text("")
    monkeypatch.setattr(table, "CONFIG", SimpleNamespace(flatppl_bin=binary))
    assert table.resolved_commit() is None


def _present_engine(monkeypatch, tmp_path, commit="eng111"):
    """Pin the engine side so a determiniser-provenance test does not depend on
    whether an engine checkout happens to sit beside the running repo -- it does
    not, when the harness runs from a worktree."""
    d = tmp_path / "flatppl-js"
    d.mkdir(exist_ok=True)
    monkeypatch.setattr(table, "CONFIG", SimpleNamespace(flatppl_js_dir=d))
    monkeypatch.setattr(table, "engine_commit", lambda: commit)
    return d


def test_check_provenance_passes_when_the_live_commit_matches_the_table(monkeypatch, tmp_path):
    _present_engine(monkeypatch, tmp_path)
    path = tmp_path / "t.json"
    table.save(path, [_row()], commit="abc123")
    monkeypatch.setattr(table, "resolved_commit", lambda: "abc123")
    assert table.check_provenance(path) is None


def test_check_provenance_fails_with_one_message_on_a_mismatch(monkeypatch, tmp_path):
    _present_engine(monkeypatch, tmp_path)
    path = tmp_path / "t.json"
    table.save(path, [_row()], commit="abc123")
    monkeypatch.setattr(table, "resolved_commit", lambda: "def456")
    problem = table.check_provenance(path)
    assert problem is not None
    assert "abc123" in problem and "def456" in problem


def test_check_provenance_fails_when_the_table_commit_is_unknown(monkeypatch, tmp_path):
    _present_engine(monkeypatch, tmp_path)
    path = tmp_path / "t.json"
    table.save(path, [_row()], commit=None)
    monkeypatch.setattr(table, "resolved_commit", lambda: "abc123")
    assert table.check_provenance(path) is not None


def test_check_provenance_fails_when_the_live_commit_is_unresolvable(monkeypatch, tmp_path):
    """Unknown provenance on the LIVE side is also a failure, not a pass --
    an unverifiable gate must not report green."""
    _present_engine(monkeypatch, tmp_path)
    path = tmp_path / "t.json"
    table.save(path, [_row()], commit="abc123")
    monkeypatch.setattr(table, "resolved_commit", lambda: None)
    problem = table.check_provenance(path)
    assert problem is not None
    assert "abc123" in problem


def test_check_provenance_fails_when_the_engine_directory_is_absent(monkeypatch, tmp_path):
    """The case that misleads rather than merely failing: with no scorer, every
    gated probe classifies MALFORMED, which reads as a batch of determiniser
    defects. Measured: 16 phantom MALFORMED rows, all `truncate`, from running
    the sweep in a worktree where the `../flatppl-js` sibling default does not
    resolve."""
    path = tmp_path / "t.json"
    d = _present_engine(monkeypatch, tmp_path)
    table.save(path, [_row()], commit="abc123")
    monkeypatch.setattr(table, "resolved_commit", lambda: "abc123")
    d.rmdir()
    problem = table.check_provenance(path)
    assert problem is not None
    assert "FLATPPL_JS_DIR" in problem


def test_a_differing_engine_commit_is_recorded_but_does_not_fail_the_gate(monkeypatch, tmp_path):
    """Unlike the determiniser pin. `setup.sh` pins the engine at `FLATPPL_JS_REF`,
    default `main`, so it advances on every unrelated flatppl-js merge -- failing
    here would be a standing false alarm. The commit is recorded so a reader knows
    which engine produced the values."""
    path = tmp_path / "t.json"
    _present_engine(monkeypatch, tmp_path, commit="eng111")
    table.save(path, [_row()], commit="abc123")
    assert table.load_metadata(path)["engine_commit"] == "eng111"
    monkeypatch.setattr(table, "resolved_commit", lambda: "abc123")
    monkeypatch.setattr(table, "engine_commit", lambda: "eng222")
    assert table.check_provenance(path) is None
