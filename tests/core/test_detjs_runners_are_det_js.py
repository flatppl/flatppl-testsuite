"""A det-js-labelled runner must score through det-js, whatever the environment says.

`corpora/<...>/test.json`'s `engines` list picks the RUNNER, and a runner named
`*_detjs` is the harness's claim that its rows exercise `flatppl determinize` ->
`score_flatpdl.cjs`. `convert_detjs` broke that claim: it reached scoring through
`scoring.engine.get_engine()`, which reads `FLATPPL_ENGINE` and defaults to
`"js"`. So under a plain `pixi run test` all 8 `corpora/hs3` rows scored in pure
JS and never determinized -- provable by pointing `FLATPPL_BIN` at a wrapper that
exits nonzero on `determinize`, which left all 8 rows green. The det-js label was
false for a whole corpus with no failing test.

Two guards, because either alone is escapable:

* the BEHAVIOURAL one runs each det-js scoring entry point with `FLATPPL_ENGINE`
  set to `js` and `get_engine` booby-trapped, so a re-introduced lookup raises;
* the STATIC one parses each det-js module and fails if `get_engine` appears as
  a real identifier, or `"FLATPPL_ENGINE"` as a non-docstring string, catching a
  lookup on a branch the behavioural test does not happen to walk. It reads the
  AST rather than the text so these modules stay free to EXPLAIN the lookup they
  must not perform.

Neither guard needs a `flatppl` binary or a Node engine: the det-js scorer itself
is stubbed, since what is under test is WHICH scorer gets called.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from flatppl_testsuite.scoring import engine as engine_mod
from flatppl_testsuite.suites import hs3_import
from flatppl_testsuite.unified import detjs_exec
from flatppl_testsuite.unified.loader import load_test
from flatppl_testsuite.unified.runners import convert_detjs

_SRC = Path(__file__).resolve().parents[2] / "src" / "flatppl_testsuite"
_CORPORA = Path(__file__).resolve().parents[2] / "corpora"
_CONVERSION = _CORPORA / "hs3" / "conversions" / "gaussian"

# The modules a det-js-labelled runner scores through. `suites/hs3_import` is
# here because `convert_detjs` delegates its two numeric check kinds to it, so a
# lookup hidden one call deep is the same defect one call up.
_DET_JS_MODULES = [
    _SRC / "unified" / "runners" / "convert_detjs.py",
    _SRC / "unified" / "runners" / "logdensity_detjs.py",
    _SRC / "unified" / "runners" / "sample_detjs.py",
    _SRC / "unified" / "detjs_exec.py",
    _SRC / "suites" / "hs3_import.py",
    _SRC / "scoring" / "flatppl_engine.py",
]


@pytest.fixture
def no_engine_lookup(monkeypatch):
    """`FLATPPL_ENGINE=js` plus a `get_engine` that refuses to answer."""
    def trapped(*a, **k):
        raise AssertionError(
            "a det-js runner reached scoring.engine.get_engine(), so the "
            "environment can steer it to another engine -- score through "
            "unified.detjs_exec instead"
        )

    monkeypatch.setenv("FLATPPL_ENGINE", "js")
    monkeypatch.setattr(engine_mod, "get_engine", trapped)
    return trapped


def test_every_det_js_module_exists():
    """A renamed or deleted module must not silently stop being guarded."""
    missing = [str(p) for p in _DET_JS_MODULES if not p.exists()]
    assert not missing, f"_DET_JS_MODULES names path(s) that do not exist: {missing}"


def test_the_convert_runner_scores_through_detjs_not_the_environment(
        no_engine_lookup, monkeypatch):
    """`convert`/det-js on a real corpus dir, with the det-js scorer stubbed.

    A recorded call proves the runner chose det-js; the booby-trapped
    `get_engine` proves it did not consult the environment. The stub returns a
    constant, so the numeric compare fails -- that is expected and irrelevant
    here, only WHICH scorer ran is under test.
    """
    calls = []

    def stub(model, binding, theta):
        calls.append((Path(model).name, binding))
        return -1.0

    monkeypatch.setattr(detjs_exec, "log_density_at", stub)

    convert_detjs.run(load_test(_CONVERSION), _CONVERSION)

    assert calls, (
        "the convert/det-js runner scored no point through unified.detjs_exec"
    )


def test_score_scan_and_score_points_require_a_scorer():
    """Neither helper may default to one, since a default is what the
    environment lookup hid behind."""
    body = json.loads((_CONVERSION / "test.json").read_text())
    check = next(c for c in body["checks"] if c["kind"] == "twice_delta_nll_points")

    with pytest.raises(TypeError):
        hs3_import.score_points(_CONVERSION / body["model"], check)

    with pytest.raises(TypeError):
        hs3_import.score_scan({}, _CONVERSION / "hs3.json", check)


def _live_names_and_strings(path: Path) -> tuple[set[str], set[str]]:
    """Every identifier a module actually uses, and every string it evaluates.

    Docstrings are excluded from the strings, so a module can name the
    environment variable while explaining that it must not read it.
    """
    tree = ast.parse(path.read_text())
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                             ast.AsyncFunctionDef)):
            body = getattr(node, "body", None)
            if (body and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                docstrings.add(id(body[0].value))

    names: set[str] = set()
    strings: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.alias):
            names.add(node.name.rsplit(".", 1)[-1])
            if node.asname:
                names.add(node.asname)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if id(node) not in docstrings:
                strings.add(node.value)
    return names, strings


@pytest.mark.parametrize("path", _DET_JS_MODULES, ids=lambda p: p.name)
def test_no_det_js_module_performs_the_engine_lookup(path):
    """The static half: no live `get_engine`, no live `"FLATPPL_ENGINE"`."""
    names, strings = _live_names_and_strings(path)
    hits = sorted(
        ({"get_engine"} & names) | ({"FLATPPL_ENGINE"} & strings)
    )
    assert not hits, (
        f"{path.name} uses {hits}; a det-js scoring path must not reach the "
        "environment-selected engine -- score through unified.detjs_exec"
    )


def test_the_static_guard_actually_catches_a_lookup(tmp_path):
    """Guards the guard: a module that does the lookup must be caught, and a
    module that only DESCRIBES it must not."""
    live = tmp_path / "live.py"
    live.write_text(
        "import os\n"
        "from flatppl_testsuite.scoring.engine import get_engine\n"
        "def f():\n"
        "    return get_engine(os.environ.get('FLATPPL_ENGINE'))\n"
    )
    names, strings = _live_names_and_strings(live)
    assert "get_engine" in names
    assert "FLATPPL_ENGINE" in strings

    prose = tmp_path / "prose.py"
    prose.write_text('"""Never call get_engine; never read FLATPPL_ENGINE."""\n')
    names, strings = _live_names_and_strings(prose)
    assert "get_engine" not in names
    assert "FLATPPL_ENGINE" not in strings
