"""Runners: test_type=logdensity and test_type=convert (pyhf), engine=stan.

Emit a Stan program with `flatppl stan` (the opt-in `stan` cargo feature),
compile it with BridgeStan, and compare its log density with the frozen
oracle value under the shared comparator (`scoring/compare.py`).

The density is evaluated at `propto=False, jacobian=False`: every constant of
every `~` term is kept, and no change-of-variables term is added, so the value
is the density of the constrained parameters FlatPPL scores. For posteriorDB
the target is `expected` (posteriorDB's Stan value plus the README's
`log_density_offset`), not the raw Stan value, because the emitter writes
half-priors as `T[L, U]`.

logdensity has the two point modes of `logdensity_detjs`:

* Mode ABI (`points`): model + query is one module, and the target is the
  dir's `binding`, else the ABI `outputs`. Each ABI input of the point goes to
  the Stan parameter the manifest binds to it (its `input`, or its `field` for
  a kernel input), or to the Stan data entry of the same name (a §13 fixed
  input the program reads as data). The dir's `inputs` name the point fields
  in ABI order where they differ from the ABI names.
* Mode A (no `points`): the model alone ends in the scored binding at a fixed
  point, so the program has no parameters and one value.

convert scores a pyhf workspace (`fixture_kind: "pyhf"`): convert it, emit
each `logpdf_points` check's binding, map each point field to the Stan
parameter whose manifest `field` it is, and compare with pyhf's absolute
`Model.logpdf`.

Exit 3 that names "not yet supported" or "no Stan form" is a refusal
(EMIT_REFUSED, a skip). Every other exit, a compile or data error, and an
input the program cannot receive are failures.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from flatppl_testsuite.config import CONFIG
from flatppl_testsuite.formats.hs3.importer import convert
from flatppl_testsuite.scoring.compare import compare_scalar
from flatppl_testsuite.scoring.result import (
    CheckResult, EMIT_REFUSED, NUMERIC_MISMATCH, UNSCOREABLE,
)
from flatppl_testsuite.unified.detjs_exec import abi_input_names, parse_expected
from flatppl_testsuite.unified.loader import TestSpec

_REPO_ROOT = Path(__file__).resolve().parents[4]

# Compiled programs, keyed by Stan text and BridgeStan version, so a rerun
# compiles only what changed.
CACHE = Path(os.environ.get("FLATPPL_STAN_CACHE", _REPO_ROOT / ".stan-cache"))

NOT_YET = "not yet supported"
_REFUSALS = (NOT_YET, "no Stan form")


def binary_supports_stan() -> bool:
    """True if the configured binary has the feature-gated `stan` verb."""
    try:
        proc = subprocess.run(
            [str(CONFIG.flatppl_bin), "stan", "--help"], capture_output=True, text=True
        )
    except OSError:
        return False
    return proc.returncode == 0


class StanRefused(RuntimeError):
    """`flatppl stan` exited 3 with a refusal ("not yet supported", "no Stan form")."""


class StanEmitError(RuntimeError):
    """`flatppl stan` failed in any other way."""


def _emit_file(path: Path, target: str, label: str):
    """Emit ``target`` of the module at ``path``; return (stan text, data, manifest)."""
    with tempfile.TemporaryDirectory(prefix="flatppl-stan-") as out:
        out = Path(out)
        proc = subprocess.run(
            [str(CONFIG.flatppl_bin), "stan", str(path), "--target", target,
             "-o", str(out / "model.stan"),
             "--data", str(out / "data.json"),
             "--manifest", str(out / "manifest.json")],
            capture_output=True, text=True,
        )
        # Name the case, not the temporary file, in messages.
        err = proc.stderr.strip().replace(str(path), label)
        if proc.returncode == 3:
            refusal = next((line for line in err.splitlines()
                            if any(r in line for r in _REFUSALS)), None)
            if refusal is not None:
                raise StanRefused(refusal)
        if proc.returncode != 0:
            raise StanEmitError(f"flatppl stan exited {proc.returncode}: {err}")
        return (
            (out / "model.stan").read_text(),
            json.loads((out / "data.json").read_text()),
            json.loads((out / "manifest.json").read_text()),
        )


def emit(dir: Path, binding: str, model_name: str = "model.flatppl",
         query_name: str | None = "query.flatppl"):
    """Emit ``binding`` of the model (with its query, unless ``query_name`` is
    None) as one module; return (stan text, data, manifest).

    The module is written inside ``dir`` because the CLI resolves a relative
    `load_data` source against the model file's own location."""
    src = (dir / model_name).read_text()
    label = f"{dir.name}/{model_name}"
    if query_name is not None:
        src = src.rstrip() + "\n" + (dir / query_name).read_text().lstrip()
        label += f"+{query_name}"
    with tempfile.NamedTemporaryFile(
        "w", suffix=".flatppl", prefix=".concat-", dir=dir, delete=False
    ) as f:
        f.write(src)
        combined = Path(f.name)
    try:
        return _emit_file(combined, binding, label)
    finally:
        combined.unlink(missing_ok=True)


def compile_stan(code: str) -> Path:
    """The BridgeStan shared object for ``code``, compiled once per text."""
    import bridgestan

    key = hashlib.sha256(f"{bridgestan.__version__}\n{code}".encode()).hexdigest()[:16]
    root = CACHE / key
    so = root / "model_model.so"
    if not so.exists():
        root.mkdir(parents=True, exist_ok=True)
        stan = root / "model.stan"
        stan.write_text(code)
        built = Path(bridgestan.compile_model(str(stan)))
        if built != so:
            shutil.copy2(built, so)
    return so


def _stan_value(v):
    """Stan JSON has no booleans."""
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, list):
        return [_stan_value(x) for x in v]
    return v


class _Unmappable(ValueError):
    """The program cannot receive a point: a mapping defect, reported as failed."""


def _routes(manifest: dict, data: dict, names: list[str], key_of: dict[str, str],
            name_of_param) -> tuple[dict[str, str], dict[str, str]]:
    """Split the point fields into Stan parameters and Stan data entries.

    Returns ({stan parameter: point key}, {data entry: point key})."""
    params: dict[str, str] = {}
    unmapped = []
    for p in manifest["parameters"]:
        name = name_of_param(p)
        if name is None or name not in key_of:
            unmapped.append(p["stan"])
        else:
            params[p["stan"]] = key_of[name]
    if unmapped:
        raise _Unmappable(f"no point input bound to Stan parameter(s) {unmapped}")
    bound = set(params.values())
    as_data = {n: key_of[n] for n in names if key_of[n] not in bound and n in data}
    missing = [n for n in names if key_of[n] not in bound and n not in data]
    if missing:
        raise _Unmappable(f"point input(s) {missing} reach neither a Stan parameter nor data")
    return params, as_data


def _score_points(tid: str, check: str, code: str, data: dict, routes, points: list[dict],
                  expected: list, tol: dict) -> list[CheckResult]:
    """Score each point under the program and compare with its expected value."""
    import bridgestan

    try:
        so = compile_stan(code)
    except Exception as e:  # a compile error is a defect of the emitted program
        return [CheckResult(tid, check, "failed", UNSCOREABLE, f"BridgeStan: {e}")]
    params, as_data = routes
    models: dict[str, bridgestan.StanModel] = {}
    results: list[CheckResult] = []
    for i, (pt, raw) in enumerate(zip(points, expected)):
        cid = f"{check}[{i}]"
        want = parse_expected(raw)
        try:
            point_data = {**data, **{n: _stan_value(pt[k]) for n, k in as_data.items()}}
            constrained = {s: _stan_value(pt[k]) for s, k in params.items()}
            data_json = json.dumps(point_data)
            if data_json not in models:
                models[data_json] = bridgestan.StanModel(str(so), data_json)
            model = models[data_json]
            theta = model.param_unconstrain_json(json.dumps(constrained))
            got = model.log_density(theta, propto=False, jacobian=False)
        except Exception as e:  # a data or evaluation error of the emitted program
            results.append(CheckResult(tid, cid, "failed", UNSCOREABLE, f"point {pt}: {e}"))
            continue
        try:
            compare_scalar(got, want, tol)
        except AssertionError as e:
            results.append(CheckResult(
                tid, cid, "failed", NUMERIC_MISMATCH, f"Stan {got!r} vs expected {want!r}: {e}"
            ))
            continue
        results.append(CheckResult(tid, cid, "passed"))
    return results


def run(spec: TestSpec, dir: Path) -> list[CheckResult]:
    tid = dir.name
    body = spec.body
    model_name = body.get("model", "model.flatppl")
    tol = body.get("tolerance", {})
    tol = {"atol": tol.get("atol", 1e-9), "rtol": tol.get("rtol", 1e-9)}
    points = body.get("points")
    expected = body["expected"]
    if not isinstance(expected, list):
        expected = [expected]
    abi_mode = points is not None
    if not abi_mode:
        points = [{}]
    if len(expected) != len(points):
        return [CheckResult(tid, "logdensity", "failed", UNSCOREABLE,
                            f"{len(points)} points but {len(expected)} expected values")]
    query = "query.flatppl" if abi_mode and (dir / "query.flatppl").exists() else None
    try:
        code, data, manifest = emit(
            dir, body.get("binding", "outputs"), model_name=model_name, query_name=query
        )
    except StanRefused as e:
        return [CheckResult(tid, "logdensity", "skipped", EMIT_REFUSED, str(e))]
    except StanEmitError as e:
        return [CheckResult(tid, "logdensity", "failed", UNSCOREABLE, str(e))]

    try:
        if abi_mode:
            src = (dir / model_name).read_text()
            if query is not None:
                src += "\n" + (dir / query).read_text()
            names = abi_input_names(src)
            keys = body.get("inputs") or names
            if len(keys) != len(names):
                raise _Unmappable(f"test.json `inputs` {keys} do not match the ABI {names}")
            # An input absent from the point is a `load_data` source, which the
            # emitter already wrote into the data.
            key_of = {n: k for n, k in zip(names, keys) if k in points[0]}
            routes = _routes(
                manifest, data, list(key_of), key_of,
                lambda p: p.get("input") or (p["field"] if p["field"] in key_of else None),
            )
        else:
            routes = _routes(manifest, data, [], {}, lambda p: None)
    except _Unmappable as e:
        return [CheckResult(tid, "logdensity", "failed", UNSCOREABLE, str(e))]
    return _score_points(tid, "logdensity", code, data, routes, points, expected, tol)


def run_convert(spec: TestSpec, dir: Path) -> list[CheckResult]:
    tid = dir.name
    body = spec.body
    if body.get("fixture_kind") != "pyhf":
        raise ValueError(f"{dir}: the stan engine scores only pyhf convert cases")
    source = dir / body.get("source", "pyhf.json")
    results: list[CheckResult] = []
    for check in body["checks"]:
        if check["kind"] != "logpdf_points":
            continue
        cid = check["id"]
        try:
            src = convert(source, source_format="pyhf")
        except Exception as e:  # noqa: BLE001  (the det-js row owns converter verdicts)
            results.append(CheckResult(tid, cid, "failed", UNSCOREABLE, f"convert: {e}"))
            continue
        with tempfile.TemporaryDirectory(prefix="flatppl-stan-pyhf-") as tmp:
            path = Path(tmp) / "model.flatppl"
            path.write_text(src)
            try:
                code, data, manifest = _emit_file(path, check["binding"], f"{tid}/model.flatppl")
            except StanRefused as e:
                results.append(CheckResult(tid, cid, "skipped", EMIT_REFUSED, str(e)))
                continue
            except StanEmitError as e:
                results.append(CheckResult(tid, cid, "failed", UNSCOREABLE, str(e)))
                continue
        points = check["points"]
        names = list(points[0])
        try:
            routes = _routes(manifest, data, names, {n: n for n in names},
                             lambda p: p["field"] if p["field"] in names else None)
        except _Unmappable as e:
            results.append(CheckResult(tid, cid, "failed", UNSCOREABLE, str(e)))
            continue
        results.extend(_score_points(tid, cid, code, data, routes, points,
                                     check["expected"], check["tolerance"]))
    return results
