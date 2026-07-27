"""Score FlatPPL through the convert-free det-js path.

The unified counterpart of `stablehlo_exec`: the ONLY place the unified
runners touch the det-js engine. Everything here delegates to
`scoring/engine.py`, which owns the `flatppl determinize` -> `score_flatpdl.cjs`
subprocess pair; this module exists so runners import one stable surface
rather than reaching into the legacy scoring package directly.
"""
from __future__ import annotations

import re
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path

from flatppl_testsuite.config import CONFIG
from flatppl_testsuite.scoring.engine import (  # noqa: F401  (re-exported)
    DeterminizeRefused,
    DetJsScoreEngine,
    sample_sweep,
    score_binding,
)

_ENGINE = DetJsScoreEngine()


def log_density_at(model: Path, binding: str, theta: dict) -> float:
    """`logdensityof(binding, theta)` via the theta-splice path (Mode B)."""
    return _ENGINE.log_density(model, binding, theta)


def parse_expected(v):
    """Frozen expected value -> float. `±inf`/`nan` cannot round-trip through
    JSON, so they are stored as the STRINGS "inf"/"-inf"/"nan" (e.g.
    fragment's trunc_out, whose density outside the support is exactly 0).
    `float()` already parses those strings natively, so this is just a
    documented single entry point for runners to call."""
    return float(v)


# --- the inputs/outputs ABI path -------------------------------------------
#
# The StableHLO backend consumes the ABI natively: it compiles ONE module whose
# `func.func` takes the point as runtime arguments. det-js has no compiled
# signature to satisfy, so it consumes the same ABI differently -- it binds each
# declared input to a literal and evaluates the `outputs` binding. That keeps ONE
# query.flatppl serving both engines against one frozen oracle, replacing the
# older path where det-js appended its own `__score__ = logdensityof(...)` and so
# scored different source than StableHLO did for the same test.
#
# Why bind at source level rather than through the engine: flatppl-js's
# `buildDerivations` produces no derivation for a binding that depends on an
# unvalued `elementof` free param (verified -- `outputs`, `inputs` and the param
# itself are all absent from `derivations`, while every other binding is
# present), so there is nothing to seed a runtime value into. Substituting the
# param's RHS makes the whole module derivable with no engine change.

# `name = elementof(<anything>)` as a top-level binding, captured so only the
# RHS is replaced (keeps the binding's own doc-comment and indentation).
def _elementof_rhs(name: str) -> re.Pattern[str]:
    return re.compile(
        r"^(\s*" + re.escape(name) + r"\s*=\s*)elementof\s*\([^\n]*\)",
        re.M,
    )

_INPUTS_BINDING = re.compile(r"^\s*inputs\s*=\s*([^\n]+)$", re.M)


def _literal(v) -> str:
    """A Python value as FlatPPL source. Lists become `[...]` (a vector input
    such as eight_schools' 8-element `theta`); ints are written as floats so a
    real-domain param does not silently become an integer literal."""
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_literal(x) for x in v) + "]"
    if isinstance(v, bool):
        raise TypeError(f"unexpected bool ABI input value {v!r}")
    return repr(float(v))


def abi_input_names(flatpdl_src: str) -> list[str]:
    """The binding names listed in a module's own `inputs`, in ABI order.

    Read off the module rather than assumed, because a query's input bindings
    need not be named after the point's fields: they are usually `t_<field>`
    (prefixed to avoid colliding with the model's own bindings), but a query
    whose model already declares its point coordinates as `elementof` reuses
    those bindings directly. Tuple order IS the ABI order (flatppl-design
    "Determinization" -> "Signature: `inputs` and `outputs`")."""
    m = _INPUTS_BINDING.search(flatpdl_src)
    if m is None:
        raise ValueError("module declares no `inputs` binding (not an ABI module)")
    rhs = m.group(1).strip()
    if rhs.startswith("("):
        rhs = rhs[1:rhs.rindex(")")] if ")" in rhs else rhs[1:]
    return [t.strip() for t in rhs.split(",") if t.strip()]


def determinize_abi(model: Path, query: Path) -> str:
    """Concatenate `model` + `query` and determinize, returning FlatPDL text.

    Concatenation (not `load_module`) matches how the StableHLO runner builds
    the same module, so both engines score byte-identical source. Done ONCE per
    model: the result is reused across every point, unlike the theta-splice path
    which re-determinizes per point."""
    src = model.read_text().rstrip() + "\n" + query.read_text().lstrip()
    with tempfile.TemporaryDirectory() as tmp:
        in_path = Path(tmp) / "abi.flatppl"
        in_path.write_text(src)
        out_path = Path(tmp) / "abi.flatpdl.flatppl"
        det = subprocess.run(
            [str(CONFIG.flatppl_bin), "determinize", str(in_path), "-o", str(out_path)],
            capture_output=True, text=True,
        )
        if det.returncode == 3:
            raise DeterminizeRefused(det.stderr.strip())
        if det.returncode != 0:
            raise RuntimeError(f"determinize failed: {det.stderr.strip()}")
        return out_path.read_text()


def score_abi_points(
    model: Path, query: Path, fields: list[str], points: list[dict]
) -> list[float]:
    """Score an ABI query module at each point, in order.

    `fields` names the point-dict keys in ABI order (the `inputs` key of a test
    dir's `test.json`); it is zipped positionally with the module's own `inputs`
    binding names, since the two orders are the same ABI order by construction.
    Raises `DeterminizeRefused` if the module is outside the determiniser's
    density fragment."""
    flatpdl = determinize_abi(model, query)
    names = abi_input_names(flatpdl)
    if len(names) != len(fields):
        raise ValueError(
            f"ABI arity mismatch: module `inputs` lists {names} "
            f"but test.json declares fields {fields}"
        )

    out: list[float] = []
    for pt in points:
        src = flatpdl
        for name, field in zip(names, fields):
            if field not in pt:
                raise ValueError(f"point {pt} has no value for ABI field {field!r}")
            pat = _elementof_rhs(name)
            src, n = pat.subn(lambda m: m.group(1) + _literal(pt[field]), src, count=1)
            if n != 1:
                raise ValueError(
                    f"could not bind ABI input {name!r}: no top-level "
                    f"`{name} = elementof(...)` binding in the determinized module"
                )
        with tempfile.TemporaryDirectory() as tmp:
            bound = Path(tmp) / "bound.flatpdl.flatppl"
            bound.write_text(src)
            out.append(_score_flatpdl_binding(bound, "outputs"))
    return out


def _score_flatpdl_binding(flatpdl_path: Path, binding: str) -> float:
    """Evaluate one deterministic binding of an already-determinized module."""
    proc = subprocess.run(
        [
            CONFIG.node_bin, str(CONFIG.flatpdl_scorer), str(flatpdl_path), binding,
            "--engine", str(CONFIG.flatppl_js_dir / "packages" / "engine"),
        ],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"score_flatpdl failed: {proc.stderr.strip()}")
    return float(proc.stdout.strip())


@lru_cache(maxsize=1)
def engine_available() -> bool:
    """True if both subprocess halves of the det-js path are resolvable."""
    return Path(CONFIG.flatppl_bin).exists() and Path(CONFIG.flatpdl_scorer).exists()
