"""One fixture per outcome, so the classifier is pinned in all three directions."""
import pytest

from flatppl_testsuite.config import CONFIG
from flatppl_testsuite.sweep.classify import Outcome, classify_source

pytestmark = pytest.mark.skipif(
    not CONFIG.flatppl_bin.exists(), reason="needs the flatppl binary"
)


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
