"""Score FlatPPL through the convert-free det-js path.

The unified counterpart of `stablehlo_exec`: the ONLY place the unified
runners touch the det-js engine. Everything here delegates to
`scoring/engine.py`, which owns the `flatppl determinize` -> `score_flatpdl.cjs`
subprocess pair; this module exists so runners import one stable surface
rather than reaching into the legacy scoring package directly.
"""
from __future__ import annotations

import json
import re
import subprocess
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from flatppl_testsuite.config import CONFIG
from flatppl_testsuite.scoring.engine import (  # noqa: F401  (re-exported)
    DeterminizeRefused,
    DetJsScoreEngine,
    sample_sweep,
    score_binding,
    temporary_source_dir,
)

_ENGINE = DetJsScoreEngine()


def log_density_at(model: Path, binding: str, theta: dict) -> float:
    """`logdensityof(binding, theta)` via the theta-splice path (Mode B)."""
    return _ENGINE.log_density(model, binding, theta)


def log_density_points(model: Path, binding: str, points: list[dict]) -> list["PointScore"]:
    """Determinize one parameterized density, then score all model points.

    Broadcast the reified density over point indices in one JS evaluation.
    All points must have the same fields and input shapes.
    """
    if not points:
        return []
    fields = list(points[0])
    domains = {field: _input_set(points[0][field]) for field in fields}
    for point in points:
        if set(point) != set(fields) or any(
            _input_set(point[field]) != domains[field] for field in fields
        ):
            raise ValueError("model points must have matching fields, shapes, and element kinds")
    prefix = "__point_"
    model_source = model.read_text()
    while prefix in model_source:
        prefix = "_" + prefix
    names = [f"{prefix}{i}__" for i in range(len(fields))]
    score, function, result = (prefix + suffix for suffix in ("score__", "fn__", "result__"))
    declarations = "\n".join(
        f"{name} = elementof({domains[field]})" for name, field in zip(names, fields)
    )
    record = ", ".join(f"{field} = {name}" for name, field in zip(names, fields))
    flatpdl = _determinize(
        model, f"{declarations}\n{score} = logdensityof({binding}, record({record}))\n", [score, *names],
    )
    columns = "\n".join(
        f"{name}column = {_literal([point[field] for point in points])}"
        for name, field in zip(names, fields)
    )
    index = prefix + "index__"
    # Iterate only points. Broadcasting nested columns directly also iterates
    # their inner event axes, whereas indexing preserves each complete value.
    def bind_point(m: re.Match[str]) -> str:
        name = m.group(1).split("=", 1)[0].strip()
        return f"{m.group(1)}get({name}column, {index})\n"

    flatpdl, count = _input_rhs(names).subn(bind_point, flatpdl)
    if count != len(names):
        raise ValueError("determinization lost a point input")
    flatpdl += (
        f"\n{columns}\n{index} = elementof(integers)\n"
        f"{function} = functionof({score}, {index} = {index})\n"
        f"{result} = broadcast({function}, "
        f"{index} = [{', '.join(str(i) for i in range(1, len(points) + 1))}])\n"
    )
    return _score_flatpdl_batch([flatpdl], result, vector_size=len(points))


def _input_set(value) -> str:
    """Preserve scalar versus nested-array inputs without baking their values."""
    if isinstance(value, (list, tuple)):
        elem = _input_set(value[0]) if value else "reals"
        if any(_input_set(v) != elem for v in value):
            raise ValueError("model inputs must be rectangular arrays of one element kind")
        return f"cartpow({elem}, {len(value)})"
    return "booleans" if isinstance(value, bool) else "reals"


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

# Canonical full syntax puts top-level binding names at column zero and indents
# continued expressions. Consume the complete RHS, including wrapped defaults.
# §13 also promotes external and derived fixed inputs.
def _input_rhs(names: list[str]) -> re.Pattern[str]:
    return re.compile(
        r"^((?:" + "|".join(re.escape(name) for name in names) + r")[ \t]*=[ \t]*)(.*?)"
        r"(?=^\w+[ \t]*(?:=|~|:)|\Z)",
        re.M | re.S,
    )


def _literal(v) -> str:
    """A Python value as FlatPPL source. Lists become `[...]` (a vector input
    such as eight_schools' 8-element `theta`); ints are written as floats so a
    real-domain param does not silently become an integer literal."""
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_literal(x) for x in v) + "]"
    if isinstance(v, bool):
        return "true" if v else "false"
    return repr(float(v))


def abi_input_names(flatpdl_src: str) -> list[str]:
    """The binding names listed in a module's own `inputs`, in ABI order.

    Read off the module rather than assumed, because a query's input bindings
    need not be named after the point's fields: they are usually `t_<field>`
    (prefixed to avoid colliding with the model's own bindings), but a query
    whose model already declares its point coordinates as `elementof` reuses
    those bindings directly. Tuple order IS the ABI order (flatppl-design
    "Determinization" -> "Signature: `inputs` and `outputs`")."""
    m = _input_rhs(["inputs"]).search(flatpdl_src)
    if m is None:
        raise ValueError("module declares no `inputs` binding (not an ABI module)")
    rhs = m.group(2).strip()
    if rhs.startswith("("):
        rhs = rhs[1:rhs.rindex(")")] if ")" in rhs else rhs[1:]
    return [t.strip() for t in rhs.split(",") if t.strip()]


def determinize_abi(model: Path, query: Path) -> str:
    """Concatenate `model` + `query` and determinize, returning FlatPDL text.

    Concatenation (not `load_module`) matches how the StableHLO runner builds
    the same module, so both engines score byte-identical source. Done ONCE per
    model: the result is reused across every point, unlike the theta-splice path
    which re-determinizes per point."""
    return _determinize(model, query.read_text(), ["inputs", "outputs"])


def _determinize(model: Path, query: str, roots: list[str]) -> str:
    src = model.read_text().rstrip() + "\n" + query.lstrip()
    # §04 resolves relative module/data paths from the containing source file.
    # Keep the concatenated input beside the model, as emit_concat does.
    with tempfile.NamedTemporaryFile(
        "w", suffix=".flatppl", prefix=".abi-", dir=temporary_source_dir(model, src)
    ) as inp, tempfile.TemporaryDirectory() as tmp:
        inp.write(src)
        inp.flush()
        in_path = Path(inp.name)
        out_path = Path(tmp) / "abi.flatpdl.flatppl"
        det = subprocess.run(
            [str(CONFIG.flatppl_bin), "determinize", str(in_path),
             *[arg for root in roots for arg in ("--keep", root)], "-o", str(out_path)],
            capture_output=True, text=True,
        )
        if det.returncode == 3:
            raise DeterminizeRefused(det.stderr.strip())
        if det.returncode != 0:
            raise RuntimeError(f"determinize failed: {det.stderr.strip()}")
        # Canonical block docs can contain examples that look like bindings.
        # They carry no runtime meaning and must not intercept substitution.
        lines = []
        in_doc = False
        for line in out_path.read_text().splitlines():
            if line.startswith("%%%"):
                in_doc = not in_doc
            elif not in_doc and not line.startswith("%"):
                lines.append(line)
        return "\n".join(lines)


@dataclass
class PointScore:
    """One point's score or evaluation error.

    A failed vector evaluation marks every point in that vector as failed.
    Nonfinite numeric results remain individual values, not evaluation errors.
    """
    value: float | None
    error: str | None = None


def score_abi_points(
    model: Path, query: Path, fields: list[str], points: list[dict]
) -> list[PointScore]:
    """Score an ABI query module at each point, in order.

    `fields` names the point-dict keys in ABI order (the `inputs` key of a test
    dir's `test.json`); it is zipped positionally with the module's own `inputs`
    binding names, since the two orders are the same ABI order by construction.
    Raises `DeterminizeRefused` if the module is outside the determiniser's
    density fragment. Determinizes once, then batches all points' `outputs`
    evaluation into ONE Node process (see `_score_flatpdl_batch`) instead of
    spawning one process per point."""
    flatpdl = determinize_abi(model, query)
    names = abi_input_names(flatpdl)
    if len(names) != len(fields):
        raise ValueError(
            f"ABI arity mismatch: module `inputs` lists {names} "
            f"but test.json declares fields {fields}"
        )

    sources: list[str] = []
    for pt in points:
        src = flatpdl
        for name, field in zip(names, fields):
            if field not in pt:
                raise ValueError(f"point {pt} has no value for ABI field {field!r}")
            pat = _input_rhs([name])
            src, n = pat.subn(lambda m: m.group(1) + _literal(pt[field]) + "\n", src, count=1)
            if n != 1:
                raise ValueError(
                    f"could not bind ABI input {name!r}: no top-level "
                    f"`{name} = ...` binding in the determinized module"
                )
        sources.append(src)

    return _score_flatpdl_batch(sources, "outputs")


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


def _score_flatpdl_batch(
    sources: list[str], binding: str, *, vector_size: int | None = None,
) -> list[PointScore]:
    """Evaluate `binding` in each of `sources`, in ONE Node process -- the
    `sample_sweep.cjs` pattern applied to the ABI-point path (see
    `scoring/score_flatpdl_batch.cjs`'s header). Pays the ~0.3s engine-load
    cost once for the whole batch instead of once per point.

    A source that fails to evaluate comes back as `{"ok": false, "error":
    ...}` from the batch script rather than a nonzero exit, so one bad point
    surfaces as that source's `PointScore.error`. With `vector_size`, evaluate
    one source and return one row per vector element; a source-level error
    fails the entire vector, while nonfinite scores remain individual values.
    """
    with tempfile.TemporaryDirectory() as tmp:
        sources_path = Path(tmp) / "sources.json"
        sources_path.write_text(json.dumps(sources))
        proc = subprocess.run(
            [
                CONFIG.node_bin, str(CONFIG.flatpdl_batch_scorer), str(sources_path), binding,
                "--engine", str(CONFIG.flatppl_js_dir / "packages" / "engine"),
                *(["--vector-size", str(vector_size)] if vector_size is not None else []),
            ],
            capture_output=True, text=True,
        )
        if proc.returncode != 0:
            # The batch process itself failed (bad args, engine unresolvable,
            # etc.) -- distinct from a per-point failure, which the script
            # reports as an {"ok": false, ...} row instead of a nonzero exit.
            raise RuntimeError(f"score_flatpdl_batch failed: {proc.stderr.strip()}")
        rows = json.loads(proc.stdout)

    expected_rows = vector_size if vector_size is not None else len(sources)
    if len(rows) != expected_rows:
        raise RuntimeError(
            f"score_flatpdl_batch returned {len(rows)} rows, expected {expected_rows}"
        )

    out: list[PointScore] = []
    for row in rows:
        # `row.get(...)` rather than `row["value"]`/truthy `ok`: a missing or
        # null `value` (e.g. a dropped-by-JSON.stringify `undefined`, or any
        # future serialization surprise) degrades to this point's own error
        # instead of KeyError/TypeError escaping the whole batch.
        if row.get("ok") and row.get("value") is not None:
            out.append(PointScore(value=float(row["value"]), error=None))
        else:
            out.append(PointScore(value=None, error=row.get("error") or "score_flatpdl_batch row missing a value"))
    return out


@lru_cache(maxsize=1)
def engine_available() -> bool:
    """True if both subprocess halves of the det-js path are resolvable."""
    return Path(CONFIG.flatppl_bin).exists() and Path(CONFIG.flatpdl_scorer).exists()
