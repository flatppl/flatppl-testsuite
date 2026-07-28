"""One fixture per outcome, so the classifier is pinned in all three directions."""
import pytest

from flatppl_testsuite.config import CONFIG
from flatppl_testsuite.sweep import classify as classify_mod
from flatppl_testsuite.sweep.classify import Outcome, classify_source

pytestmark = pytest.mark.skipif(
    not CONFIG.flatppl_bin.exists(), reason="needs the flatppl binary"
)

_PLAIN_SOURCE = "m = Normal(mu = 0.0, sigma = 1.0)\nlp = logdensityof(m, 0.5)\n"


def test_a_plain_density_lowers_and_scores():
    v = classify_source("m = Normal(mu = 0.0, sigma = 1.0)\nlp = logdensityof(m, 0.5)\n", "lp")
    assert v.outcome == Outcome.LOWERS
    assert abs(v.value - -1.0439385332046727) < 1e-9


def test_an_unsupported_construct_refuses_with_a_marker():
    # A two-draw derived value law: §06 case 3, a static error by default.
    src = ("x1 = draw(Normal(mu = 0.0, sigma = 1.0))\n"
           "x2 = draw(Normal(mu = 0.0, sigma = 1.0))\n"
           "d = x1 - x2\n"
           "lp = logdensityof(lawof(d), 0.3)\n")
    v = classify_source(src, "lp")
    assert v.outcome == Outcome.REFUSES
    assert v.marker, "a refusal must carry a marker for the table to diff on"
    assert v.value is None


def test_a_record_pushfwd_whose_map_does_not_match_the_variate_refuses():
    """Not the brief's original MALFORMED fixture -- that one (a whole-record
    map spelled `r -> get(r, "y1") - get(r, "y2")`) now refuses via §04's
    auto-splat guard, verified directly against this binary:
    `determinize: refuse pushfwd (node NodeId(27)): pushfwd's map and the
    record variate of its base measure do not correspond`.

    This is the SAME guard reached through a shape the probe space's own
    `record` spelling actually generates -- a single-field record law wrapped
    in `pushfwd`, where the map (`exp`) takes a bare scalar, not the record --
    confirming the shape refuses cleanly rather than surviving as a `%call`:
    `determinize: refuse user-call (node NodeId(17)): value must be a record`.

    See `classify.py`'s module docstring for why no reachable end-to-end
    `MALFORMED` case exists against this binary: the surface printer is
    head-kind-blind, so a surviving `%call` (if one existed) would not even
    be visible in `determinize -o`'s output.
    """
    src = ("mb = Normal(mu = 0.0, sigma = 1.0)\n"
           "x = draw(mb)\n"
           "m = pushfwd(exp, lawof(record(x = x)))\n"
           "lp = logdensityof(m, record(x = 1.6487212707001282))\n")
    v = classify_source(src, "lp")
    assert v.outcome == Outcome.REFUSES
    assert v.marker


def test_a_scorer_crash_on_emitted_output_is_malformed_not_a_hard_abort(monkeypatch):
    """`determinize` exits 0 (the source above is unremarkable), but the JS
    scorer throws evaluating what it emitted -- e.g. a builtin fed the wrong
    shape, per the design note in `classify.py`'s module docstring. That is a
    determinizer defect (`MALFORMED`), not the "scorer crash fails the run
    loudly" infrastructure case, so it must not propagate past `classify_source`.

    No end-to-end fixture in this probe space currently reaches this against
    the pinned binary (0 crashes over the full 738-probe sweep) -- the
    `score_binding` failure mode is exercised directly here instead, the same
    way `test_classify_malformed_detector.py` pins `_RESIDUAL_CALL` against
    synthetic text rather than a real CLI run.
    """
    def _boom(model, binding):
        raise RuntimeError("score_flatpdl failed: score_flatpdl: log expects a number, got object")

    monkeypatch.setattr(classify_mod, "score_binding", _boom)
    v = classify_source(_PLAIN_SOURCE, "lp")
    assert v.outcome == Outcome.MALFORMED
    assert v.marker.startswith("crash:")
    assert v.value is None


def test_a_genuine_determinize_failure_inside_score_binding_still_fails_loudly(monkeypatch):
    """The OTHER `RuntimeError` shape `score_binding` can raise -- its own
    internal `determinize` call failing -- is infrastructure, not a probe
    verdict, and must still propagate rather than being swallowed into
    `MALFORMED` alongside the scorer-crash case."""
    def _boom(model, binding):
        raise RuntimeError("determinize failed: something is badly wrong")

    monkeypatch.setattr(classify_mod, "score_binding", _boom)
    with pytest.raises(RuntimeError, match="determinize failed"):
        classify_source(_PLAIN_SOURCE, "lp")
