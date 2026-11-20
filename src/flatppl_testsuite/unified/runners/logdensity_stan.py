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
  dir's `binding`, else the ABI `outputs`. The frozen value is that of
  `outputs`, so a `binding` must be what `outputs` scores. All points have
  the same fields, and an ABI input outside them must be a `load_data`
  source, which the emitter writes into the data. Each ABI input of the point
  goes to the Stan parameters whose manifest `input` names it, whole or as
  the `index` part of it. When a parameter of the `binding` program has no
  `input` (the query builds it from the inputs, for example with `rowstack`),
  the runner scores the `outputs` program instead, whose parameters are the
  ABI inputs. A point passed this way has tag STAN_OUTPUTS, and a refusal of
  `outputs` counts as any refusal. Any other mapping error fails.
  Otherwise it goes to the Stan data entry the manifest's `data` list names
  for it (a §13 fixed input the program reads as data). The dir's `inputs`
  name the point fields in ABI order where they differ from the ABI names.
* Mode A (no `points`): the model alone ends in the scored binding at a fixed
  point, so the program has no parameters and one value.

convert scores a pyhf workspace (`fixture_kind: "pyhf"`): convert it, emit
each `logpdf_points` check's binding, map each point field to the Stan
parameter whose manifest `field` it is, and compare with pyhf's absolute
`Model.logpdf`.

A point frozen at -inf or as an error record (`{"error": text}`) passes when
`log_density` itself raises the Stan rejection that the frozen value names:
Stan rejects where a law has no density, which is density 0. The result is
`passed` with tag STAN_REJECTED, and its message holds the rejection class
and the Stan message. `_REJECTIONS` holds the classes and the pass table. Any
other exception fails the point.

Exit 3 that names "not yet supported" or "no Stan form" is a refusal
(EMIT_REFUSED, a skip). Every other exit, a compile or data error, and an
input the program cannot receive are failures.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from flatppl_testsuite.config import CONFIG
from flatppl_testsuite.formats.hs3.importer import convert
from flatppl_testsuite.scoring.compare import compare_scalar
from flatppl_testsuite.scoring.result import (
    CheckResult, EMIT_REFUSED, NUMERIC_MISMATCH, STAN_OUTPUTS, STAN_REJECTED, UNSCOREABLE,
)
from flatppl_testsuite.unified.detjs_exec import _input_rhs, abi_input_names, parse_expected
from flatppl_testsuite.unified.loader import TestSpec
from flatppl_testsuite.unified.stablehlo_exec import load_data_bindings

_REPO_ROOT = Path(__file__).resolve().parents[4]

# Compiled programs, keyed by Stan text and BridgeStan version, so a rerun
# compiles only what changed.
CACHE = Path(os.environ.get("FLATPPL_STAN_CACHE", _REPO_ROOT / ".stan-cache"))

NOT_YET = "not yet supported"
_REFUSALS = (NOT_YET, "no Stan form")

# Stan rejections that meet a frozen -inf or error record. BridgeStan 2.9.0
# reports every `log_density` exception as RuntimeError("log_density() failed
# with exception: Exception: <what()>"), so the class comes from the Stan Math
# message form. The forms were measured on BridgeStan 2.9.0 and its Stan Math;
# tests/test_stan_reject.py pins each one.
LAW_REJECTION = re.compile(
    # A law's support or parameter check, not its size or NaN checks.
    r"Exception: \w+_(?:lpdf|lpmf|lcdf|lccdf|cdf): .*?"
    r"(?: is (?!nan,)[^,]+, but must be | is not (?:a valid|positive definite|symmetric))"
)
INDEX_REJECTION = re.compile(r"index -?\d+ out of range; expecting index to be between")
# Frozen value -> the rejection class that meets it. The error texts are the
# StableHLO runtime statuses the corpora freeze.
_REJECTIONS = {
    "-inf": ("law", LAW_REJECTION),
    "outside its declared value set": ("law", LAW_REJECTION),
    "index out of range": ("index", INDEX_REJECTION),
}


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


class _UnboundParameter(_Unmappable):
    """A Stan parameter that no point input binds."""

    def __init__(self, params: list[str]):
        super().__init__(f"no point input bound to Stan parameter(s) {params}")
        self.params = params


def _tolerance(body: dict) -> dict[str, float]:
    tol = body.get("tolerance", {})
    return {"atol": tol.get("atol", 1e-9), "rtol": tol.get("rtol", 1e-9)}


def _check_scores_binding(src: str, binding: str) -> None:
    """Fail unless the ABI `outputs` scores `binding`.

    The frozen value is that of `outputs`, but the emitter takes the density
    target (`binding`) and does not yet take a literal record point as target."""
    if binding == "outputs":
        return
    rhs = _rhs(src, "outputs")
    alias = re.fullmatch(r"\(?\s*(\w+)\s*\)?", rhs)
    if alias and alias.group(1) == binding:
        return
    if alias:
        rhs = _rhs(src, alias.group(1))
    if not re.match(rf"logdensityof\(\s*{re.escape(binding)}\s*,", rhs):
        raise _Unmappable(f"`outputs` does not score `logdensityof({binding}, …)`")


def _rhs(src: str, name: str) -> str:
    m = _input_rhs([name]).search(src)
    if m is None:
        raise _Unmappable(f"the module has no binding `{name}`")
    return m.group(2).strip()


def _routes(manifest: dict, names: list[str], key_of: dict[str, str], name_of_param):
    """Split the point fields into Stan parameters and Stan data entries.

    A parameter may be an index or a slice of a point field (manifest
    `index`). A fixed input reaches the data only through the manifest's
    `data` list. Returns ({stan parameter: (point key, index)},
    {Stan data name: point key})."""
    params: dict[str, tuple[str, int | list[int] | None]] = {}
    unmapped = []
    for p in manifest["parameters"]:
        name = name_of_param(p)
        if name is None or name not in key_of:
            unmapped.append(p["stan"])
        else:
            params[p["stan"]] = (key_of[name], p.get("index"))
    if unmapped:
        raise _UnboundParameter(unmapped)
    # A data entry outside the point (a `load_data` source) keeps its emitted value.
    as_data = {d["stan"]: key_of[d["input"]]
               for d in manifest.get("data", []) if d["input"] in key_of}
    bound = {k for k, _ in params.values()} | set(as_data.values())
    missing = [n for n in names if key_of[n] not in bound]
    if missing:
        raise _Unmappable(f"point input(s) {missing} reach neither a Stan parameter "
                          "nor a manifest data entry")
    return params, as_data


def _project(value, index):
    """The part of a point value a parameter takes: all of it, `value[k]`, or
    the vector `value[[k, …]]` (FlatPPL indices are 1-based)."""
    if index is None:
        return value
    ks = [index] if isinstance(index, int) else index
    if not isinstance(value, list) or any(not 1 <= k <= len(value) for k in ks):
        raise _Unmappable(f"index {index} is outside the point value {value!r}")
    return value[index - 1] if isinstance(index, int) else [value[k - 1] for k in ks]


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
        error = raw.get("error") if isinstance(raw, dict) else None
        want = None if error is not None else parse_expected(raw)
        try:
            point_data = {**data, **{n: _stan_value(pt[k]) for n, k in as_data.items()}}
            constrained = {s: _stan_value(_project(pt[k], ix)) for s, (k, ix) in params.items()}
            data_json = json.dumps(point_data)
            if data_json not in models:
                models[data_json] = bridgestan.StanModel(str(so), data_json)
            model = models[data_json]
            theta = model.param_unconstrain_json(json.dumps(constrained))
        except Exception as e:  # a data or constraint error of the emitted program
            results.append(CheckResult(tid, cid, "failed", UNSCOREABLE, f"point {pt}: {e}"))
            continue
        try:
            got = model.log_density(theta, propto=False, jacobian=False)
        except Exception as e:
            results.append(_rejection(tid, cid, pt, str(e).strip(), want, error))
            continue
        if error is not None:
            results.append(CheckResult(
                tid, cid, "failed", NUMERIC_MISMATCH,
                f"point {pt}: Stan {got!r} vs expected an error containing {error!r}",
            ))
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


def _rejection(tid: str, cid: str, pt: dict, message: str, want: float | None,
               error: str | None) -> CheckResult:
    """The verdict for a point where `log_density` raised ``message``."""
    key = error if error is not None else ("-inf" if want == -math.inf else None)
    if key is None:
        return CheckResult(tid, cid, "failed", UNSCOREABLE, f"point {pt}: {message}")
    if key not in _REJECTIONS:
        return CheckResult(tid, cid, "failed", UNSCOREABLE,
                           f"point {pt}: no Stan rejection class meets the frozen error "
                           f"{error!r}; Stan raised: {message}")
    name, pattern = _REJECTIONS[key]
    if pattern.search(message):
        return CheckResult(tid, cid, "passed", STAN_REJECTED, f"rejected ({name}): {message}")
    return CheckResult(tid, cid, "failed", UNSCOREABLE,
                       f"point {pt}: the frozen {key!r} needs a {name} rejection; "
                       f"Stan raised: {message}")


def run(spec: TestSpec, dir: Path) -> list[CheckResult]:
    tid = dir.name
    body = spec.body
    model_name = body.get("model", "model.flatppl")
    tol = _tolerance(body)
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
    query = "query.flatppl" if abi_mode else None
    if abi_mode and not (dir / query).exists():
        return [CheckResult(tid, "logdensity", "failed", UNSCOREABLE,
                            "`points` without query.flatppl")]
    target = body.get("binding", "outputs")
    try:
        code, data, manifest = emit(dir, target, model_name=model_name, query_name=query)
    except StanRefused as e:
        return [CheckResult(tid, "logdensity", "skipped", EMIT_REFUSED, str(e))]
    except StanEmitError as e:
        return [CheckResult(tid, "logdensity", "failed", UNSCOREABLE, str(e))]

    note = None
    try:
        if abi_mode:
            src = (dir / model_name).read_text() + "\n" + (dir / query).read_text()
            if "binding" in body:
                _check_scores_binding(src, body["binding"])
            names = abi_input_names(src)
            keys = body.get("inputs") or names
            if len(keys) != len(names):
                raise _Unmappable(f"test.json `inputs` {keys} do not match the ABI {names}")
            fields = set(points[0])
            if any(set(pt) != fields for pt in points):
                raise _Unmappable("the points do not all have the same fields")
            # The emitter writes a `load_data` source into the data itself.
            sources = load_data_bindings(dir, model_name=model_name)
            dropped = [n for n, k in zip(names, keys) if k not in fields]
            unknown = [n for n in dropped if n not in sources]
            if unknown:
                raise _Unmappable(f"ABI input(s) {unknown} are neither in the point "
                                  "nor a `load_data` source")
            key_of = {n: k for n, k in zip(names, keys) if k in fields}
            try:
                routes = _routes(manifest, list(key_of), key_of, lambda p: p.get("input"))
            except _UnboundParameter as e:
                if target == "outputs":
                    raise
                # The query builds the target's point from its inputs, so the
                # `outputs` program takes exactly those inputs as parameters.
                note = f"outputs program: no point input binds binding parameter(s) {e.params}"
                try:
                    code, data, manifest = emit(
                        dir, "outputs", model_name=model_name, query_name=query
                    )
                except StanRefused as r:
                    return [CheckResult(tid, "logdensity", "skipped", EMIT_REFUSED,
                                        f"{note}; {r}")]
                except StanEmitError as r:
                    return [CheckResult(tid, "logdensity", "failed", UNSCOREABLE,
                                        f"{note}; {r}")]
                routes = _routes(manifest, list(key_of), key_of, lambda p: p.get("input"))
        else:
            routes = _routes(manifest, [], {}, lambda p: None)
    except ValueError as e:  # _Unmappable, or a module with no ABI `inputs`
        msg = str(e) if note is None else f"{note}; {e}"
        return [CheckResult(tid, "logdensity", "failed", UNSCOREABLE, msg)]
    results = _score_points(tid, "logdensity", code, data, routes, points, expected, tol)
    if note is not None:
        for r in results:
            if r.status == "passed" and not r.tag:
                r.tag, r.message = STAN_OUTPUTS, note
            else:  # a rejection keeps STAN_REJECTED, a failure its own tag
                r.message = f"{note}; {r.message}"
    return results


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
            routes = _routes(manifest, names, {n: n for n in names},
                             lambda p: p["field"] if p["field"] in names else None)
        except _Unmappable as e:
            results.append(CheckResult(tid, cid, "failed", UNSCOREABLE, str(e)))
            continue
        results.extend(_score_points(tid, cid, code, data, routes, points,
                                     check["expected"], _tolerance(check)))
    return results
