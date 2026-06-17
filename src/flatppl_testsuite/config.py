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
    # Node 24, provided by pixi on PATH.
    node_bin: str = os.environ.get("NODE_BIN", "node")
    # Generic JS scorer for any FlatPPL model; used by scoring/flatppl_engine.py.
    scorer: Path = _REPO_ROOT / "src" / "flatppl_testsuite" / "scoring" / "score_js.cjs"


CONFIG = Config()
