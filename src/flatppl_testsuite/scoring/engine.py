"""Pluggable FlatPPL engine abstraction.

The harness scores FlatPPL models through a *selectable* engine, so alternative
FlatPPL engines (a second JS build, a future native scorer, ...) can be tested by
registering and selecting them — the rest of the harness (suites, comparison
tables, 2DeltaNLL) is engine-agnostic.

Select the active engine with the ``FLATPPL_ENGINE`` env var (default ``"js"``).
Add an engine by subclassing ``FlatpplEngine`` and calling ``register_engine``.
"""
from __future__ import annotations

import os
import subprocess
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path

from ..config import CONFIG


def render_record(theta: dict) -> str:
    """Serialise a Python dict to a FlatPPL ``record(...)`` literal.

    Int-valued numbers are emitted as ``N.0``; lists/tuples become arrays.
    """
    def lit(v):
        if isinstance(v, (list, tuple)):
            return "[" + ", ".join(lit(x) for x in v) + "]"
        return f"{v}.0" if isinstance(v, int) else repr(float(v))
    return "record(" + ", ".join(f"{k} = {lit(v)}" for k, v in theta.items()) + ")"


class FlatpplEngine(ABC):
    """Scores one FlatPPL model: ``logdensityof(<binding>, <theta record>)``."""

    name: str

    @abstractmethod
    def log_density(self, model: Path, binding: str, theta: dict) -> float:
        """Return logdensityof(binding, theta) for the FlatPPL model at `model`."""


class JsScoreEngine(FlatpplEngine):
    """flatppl-js, via the ``score_js.cjs`` single-point scorer (Node 24)."""

    name = "js"

    def log_density(self, model: Path, binding: str, theta: dict) -> float:
        proc = subprocess.run(
            [CONFIG.node_bin, str(CONFIG.scorer), str(model), binding,
             render_record(theta), "--engine",
             str(CONFIG.flatppl_js_dir / "packages" / "engine")],
            capture_output=True, text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"score_js failed: {proc.stderr.strip()}")
        return float(proc.stdout.strip())


class DeterminizeRefused(Exception):
    """``flatppl determinize`` refused a construct (exit 3).

    Not a hard error: it means the model (or the appended score query) uses a
    construct outside the determiniser's current density fragment — e.g. a
    continuous-latent marginal with no closed form. Callers should treat this
    like ``SkipUnimplemented``/``CONVERT_SKIP`` (tag ``DETERMINIZE_SKIP``), not
    fail the run.
    """


class DetJsScoreEngine(FlatpplEngine):
    """convert-free det-js path.

    Appends ``__score__ = logdensityof(binding, theta)`` to the model, runs
    ``flatppl determinize`` to lower the whole thing to the deterministic
    FlatPDL profile (eliminating the measure layer), then evaluates the
    resulting ``__score__`` binding via ``score_flatpdl.cjs``. Raises
    ``DeterminizeRefused`` if the determiniser can't legalize the model
    (exit 3); any other nonzero exit from either subprocess is a hard
    ``RuntimeError``.
    """

    name = "det-js"

    def log_density(self, model: Path, binding: str, theta: dict) -> float:
        src = model.read_text() + f"\n__score__ = logdensityof({binding}, {render_record(theta)})\n"
        with tempfile.NamedTemporaryFile(
            suffix=".flatppl", mode="w", delete=False
        ) as tf:
            tf.write(src)
            in_path = Path(tf.name)
        out_path = in_path.with_suffix(".flatpdl.flatppl")
        try:
            det = subprocess.run(
                [str(CONFIG.flatppl_bin), "determinize", str(in_path), "-o", str(out_path)],
                capture_output=True, text=True,
            )
            if det.returncode == 3:
                raise DeterminizeRefused(det.stderr.strip())
            if det.returncode != 0:
                raise RuntimeError(f"determinize failed: {det.stderr.strip()}")
            proc = subprocess.run(
                [
                    CONFIG.node_bin, str(CONFIG.flatpdl_scorer), str(out_path), "__score__",
                    "--engine", str(CONFIG.flatppl_js_dir / "packages" / "engine"),
                ],
                capture_output=True, text=True,
            )
            if proc.returncode != 0:
                raise RuntimeError(f"score_flatpdl failed: {proc.stderr.strip()}")
            return float(proc.stdout.strip())
        finally:
            in_path.unlink(missing_ok=True)
            out_path.unlink(missing_ok=True)


_REGISTRY: dict[str, FlatpplEngine] = {}


def register_engine(engine: FlatpplEngine) -> None:
    """Register a FlatPPL engine under its ``name`` (later registration wins)."""
    _REGISTRY[engine.name] = engine


def get_engine(name: str | None = None) -> FlatpplEngine:
    """Return the selected engine (``FLATPPL_ENGINE`` env, default ``"js"``)."""
    name = name or os.environ.get("FLATPPL_ENGINE", "js")
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"unknown FlatPPL engine {name!r}; registered: {sorted(_REGISTRY)}"
        ) from None


register_engine(JsScoreEngine())
register_engine(DetJsScoreEngine())
