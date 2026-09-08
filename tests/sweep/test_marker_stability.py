"""The refusal marker is frozen in `verdicts/density-sweep.json`, so it must
not carry a `tempfile` random component. Before this was normalised, every
`pixi run repin` rewrote ~140 rows with no verdict change, which made a real
marker move indistinguishable from noise.

The two messages here are verbatim `flatppl determinize` stderr, captured from
the release binary at flatppl-rust `f1e794f`, differing only in the temp path.
"""
from flatppl_testsuite.sweep.classify import _crash_marker, _marker

_REFUSAL = (
    "determinize: refuse `2` in `m` "
    "(/var/folders/45/b8pzgqds26123bklmfvc1l_40000gp/T/{tmp}/probe.flatppl:4:41): "
    "locscale base measure variate domain is not confirmed scalar or vector "
    "— refuse rather than guess the affine form\n"
)


def test_two_runs_of_the_same_refusal_share_one_marker():
    a = _marker(_REFUSAL.format(tmp="tmpkirmvsxj"))
    b = _marker(_REFUSAL.format(tmp="tmprkvscgon"))
    assert a == b


def test_the_marker_names_the_reason_not_the_path():
    m = _marker(_REFUSAL.format(tmp="tmpkirmvsxj"))
    assert "tmp" not in m
    assert "folders" not in m
    assert m == "`2`:refuse-flatppl-locscale-base-measure-variate"


def test_a_crash_marker_drops_the_temp_path_too():
    msg = (
        "score_flatpdl failed: score_flatpdl: cannot read /tmp/tmpab12cd34/"
        "probe.flatpdl.flatppl: log expects a number, got object"
    )
    m = _crash_marker(msg)
    assert "tmp" not in m
    assert "ab12cd34" not in m


def test_a_refusal_with_no_path_is_untouched():
    m = _marker("determinize: refuse divide: value must be a record\n")
    assert m == "divide::refuse-divide-value-must-record"
