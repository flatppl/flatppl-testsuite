"""Direct tests of the `MALFORMED` detector against synthetic text.

No reachable end-to-end `MALFORMED` case exists against the pinned binary --
see `classify.py`'s module docstring and `test_classify.py`'s record-pushfwd
case, both of which land on `REFUSES` instead. Worse: even if one did exist,
`determinize -o` prints surface FlatPPL text re-derived from the determinized
module, and the surface printer renders a residual `CallHead::User` call
exactly like any other call (`flatppl-dev/TODO-flatppl-rust.md`, "a CLI text
comparison cannot see it" -- a FlatPIR `(%call log 0.5)` prints as
`v = log(0.5)` and re-parses to a builtin call). So `_RESIDUAL_CALL` can never
match real CLI output; the only way to pin it against ITS OWN logic is
synthetic text, in both directions, run with no binary required.
"""
from flatppl_testsuite.sweep.classify import _RESIDUAL_CALL


def test_residual_call_pattern_matches_a_surviving_user_call():
    assert _RESIDUAL_CALL.search("out = (%call f 0.5)\n")


def test_residual_call_pattern_does_not_match_real_flatpdl_text():
    # Shape lifted from an actual `pushfwd` density lowering: deterministic
    # ops plus a `builtin_*` primitive, no `%call` anywhere.
    text = (
        "lp = sub(builtin_logdensityof(Normal, record(mu = 0.0, sigma = 1.0), "
        "log(record(a = 1.0))), log(record(a = 1.0)))\n"
    )
    assert not _RESIDUAL_CALL.search(text)


def test_residual_call_pattern_does_not_match_an_ordinary_user_function_call():
    # A ordinary, successfully-reduced user call prints indistinguishably from
    # a builtin call (see the module docstring) -- neither should ever trip
    # the detector, since the detector's whole job is a text pattern, not
    # semantic head-kind inspection.
    assert not _RESIDUAL_CALL.search("f = x -> x * 2.0\nlp = f(0.5)\n")
