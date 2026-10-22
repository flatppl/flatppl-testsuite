"""Harness configuration: generic FlatPPL tooling paths.

Provides paths to the FlatPPL converter binary, the JS engine checkout,
the Node runtime, and the generic JS scorer.  Corpus-specific config
(e.g. the HS3 corpus root and manifest) lives in the relevant corpus module.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PARENT = _REPO_ROOT.parent


@dataclass(frozen=True)
class Config:
    # The flatppl converter binary, installed by `pixi run setup`.
    flatppl_bin: Path = Path(os.environ.get("FLATPPL_BIN", _REPO_ROOT / ".pixi-bin" / "bin" / "flatppl"))
    # The flatppl-js checkout whose packages/engine is loaded by the scorer.
    flatppl_js_dir: Path = Path(os.environ.get("FLATPPL_JS_DIR", _PARENT / "flatppl-js"))
    # The flatppl-examples checkout; its models live at
    # `examples_dir / "examples" / "<file>.flatppl"` (see corpora/examples/).
    examples_dir: Path = Path(os.environ.get("FLATPPL_EXAMPLES_DIR", _PARENT / "flatppl-examples"))
    # Node 24, provided by pixi on PATH.
    node_bin: str = os.environ.get("NODE_BIN", "node")
    # Generic JS scorer for any FlatPPL model; used by scoring/flatppl_engine.py.
    scorer: Path = _REPO_ROOT / "src" / "flatppl_testsuite" / "scoring" / "score_js.cjs"
    # One-shot scorer for a deterministic binding in a determinized FlatPDL model;
    # the worker runs the same core (tests that drive Node directly use this).
    flatpdl_scorer: Path = _REPO_ROOT / "src" / "flatppl_testsuite" / "scoring" / "score_flatpdl.cjs"
    # One-shot seed-sweep scorer for a determinized FlatPDL sample-path model
    # (one `rnginit([...])` byte-vector substituted per seed); the worker runs the
    # same core for scoring/engine.py's sample_sweep().
    sample_sweep_scorer: Path = (
        _REPO_ROOT / "src" / "flatppl_testsuite" / "scoring" / "sample_sweep.cjs"
    )
    # The persistent worker every det-js scoring request goes through: it loads the
    # engine once and runs the scorers above as in-process requests, so a run starts
    # one Node process instead of one per case (scoring/worker.py).
    score_worker: Path = _REPO_ROOT / "src" / "flatppl_testsuite" / "scoring" / "score_worker.cjs"
    # One-shot batch scorer for an ABI query's `outputs` binding across many
    # already-bound FlatPDL sources; the worker runs the same core for
    # unified/detjs_exec.py's score_abi_points().
    flatpdl_batch_scorer: Path = (
        _REPO_ROOT / "src" / "flatppl_testsuite" / "scoring" / "score_flatpdl_batch.cjs"
    )


CONFIG = Config()
