"""The pluggable-engine seam: registry, selection, and the record serialiser.

These pin the contract a new FlatPPL engine plugs into (subclass FlatpplEngine +
register_engine) without invoking any engine — no Node, no scorer subprocess.
"""

import pytest

from flatppl_testsuite.scoring.engine import (
    FlatpplEngine,
    JsScoreEngine,
    get_engine,
    register_engine,
    render_record,
)


def test_default_engine_is_js():
    assert get_engine().name == "js"
    assert isinstance(get_engine("js"), JsScoreEngine)


def test_env_selects_engine(monkeypatch):
    monkeypatch.setenv("FLATPPL_ENGINE", "js")
    assert get_engine().name == "js"


def test_unknown_engine_raises():
    with pytest.raises(KeyError, match="unknown FlatPPL engine"):
        get_engine("nope")


def test_register_engine_roundtrips():
    class Dummy(FlatpplEngine):
        name = "dummy-test-engine"
        def log_density(self, model, binding, theta):  # pragma: no cover - never called
            return 0.0

    register_engine(Dummy())
    assert get_engine("dummy-test-engine").name == "dummy-test-engine"


def test_render_record_int_float_list():
    # ints -> N.0, floats verbatim, lists -> arrays.
    assert render_record({"n": 3}) == "record(n = 3.0)"
    assert render_record({"x": [1, 2.5]}) == "record(x = [1.0, 2.5])"
