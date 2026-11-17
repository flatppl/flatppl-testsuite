"""Runner: test_type=logdensity, engine=stan.

Concatenate model.flatppl + query.flatppl into one module, emit a Stan program
with `flatppl stan` (the opt-in `stan` cargo feature), compile it with
BridgeStan, and compare its log density at each authored point with the frozen
`expected`, within the dir's float64 tolerance
(`|got - want| <= atol + rtol * |want|`).

The density is evaluated at `propto=False, jacobian=False`: every constant of
every `~` term is kept, and no change-of-variables term is added, so the value
is the density of the constrained parameters FlatPPL scores. The emitter
writes half-priors as `T[L, U]`, so the target is `expected` (posteriorDB's
Stan value plus the README's `log_density_offset`), not the raw Stan value.

The emitter's parameter manifest maps each Stan parameter to the query input
bound to it by `outputs = logdensityof(target, record(field = input, ...))`, so
a point (input name -> value) becomes a constrained parameter record without
the runner knowing FlatPPL.

Only exit 3 with "not yet supported" is a skip (EMIT_REFUSED): a Stan form
exists but the emitter does not write it yet. Every other refusal or exit code
is a failure.
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
from flatppl_testsuite.scoring.result import (
    CheckResult, EMIT_REFUSED, NUMERIC_MISMATCH, UNSCOREABLE,
)
from flatppl_testsuite.unified.detjs_exec import parse_expected
from flatppl_testsuite.unified.loader import TestSpec

_REPO_ROOT = Path(__file__).resolve().parents[4]

# Compiled programs, keyed by Stan text and BridgeStan version, so a rerun
# compiles only what changed.
CACHE = Path(os.environ.get("FLATPPL_STAN_CACHE", _REPO_ROOT / ".stan-cache"))

_NOT_YET = "not yet supported"


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
    """`flatppl stan` exited 3 with "not yet supported"."""


class StanEmitError(RuntimeError):
    """`flatppl stan` failed in any other way."""


def emit(dir: Path, binding: str, model_name: str = "model.flatppl",
         query_name: str = "query.flatppl"):
    """Emit ``binding`` of model + query as one module; return (stan text,
    data, manifest).

    The combined module is written inside ``dir`` because the CLI resolves a
    relative `load_data` source against the model file's own location."""
    model = (dir / model_name).read_text()
    query = (dir / query_name).read_text()
    with tempfile.TemporaryDirectory(prefix="flatppl-stan-") as out:
        out = Path(out)
        with tempfile.NamedTemporaryFile(
            "w", suffix=".flatppl", prefix=".concat-", dir=dir, delete=False
        ) as f:
            f.write(model.rstrip() + "\n" + query.lstrip())
            combined = Path(f.name)
        try:
            proc = subprocess.run(
                [str(CONFIG.flatppl_bin), "stan", str(combined), "--target", binding,
                 "-o", str(out / "model.stan"),
                 "--data", str(out / "data.json"),
                 "--manifest", str(out / "manifest.json")],
                capture_output=True, text=True,
            )
        finally:
            combined.unlink(missing_ok=True)
        # Name the case, not the temporary file, in messages.
        err = proc.stderr.strip().replace(str(combined), f"{dir.name}/{model_name}+{query_name}")
        if proc.returncode == 3 and _NOT_YET in err:
            raise StanRefused(next(line for line in err.splitlines() if _NOT_YET in line))
        if proc.returncode != 0:
            raise StanEmitError(f"flatppl stan exited {proc.returncode}: {err}")
        return (
            (out / "model.stan").read_text(),
            json.loads((out / "data.json").read_text()),
            json.loads((out / "manifest.json").read_text()),
        )


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


def run(spec: TestSpec, dir: Path) -> list[CheckResult]:
    import bridgestan

    tid = dir.name
    body = spec.body
    points: list[dict] = body["points"]
    expected = body["expected"]
    if not isinstance(expected, list):
        expected = [expected]
    tol = body.get("tolerance", {})
    atol = tol.get("atol", 1e-9)
    rtol = tol.get("rtol", 1e-9)
    if len(expected) != len(points):
        return [CheckResult(tid, "logdensity", "failed", UNSCOREABLE,
                            f"{len(points)} points but {len(expected)} expected values")]
    try:
        code, data, manifest = emit(
            dir, body["binding"], model_name=body.get("model", "model.flatppl")
        )
    except StanRefused as e:
        return [CheckResult(tid, "logdensity", "skipped", EMIT_REFUSED, str(e))]
    except StanEmitError as e:
        return [CheckResult(tid, "logdensity", "failed", UNSCOREABLE, str(e))]
    unmapped = [p["stan"] for p in manifest["parameters"] if "input" not in p]
    if unmapped:
        return [CheckResult(tid, "logdensity", "failed", UNSCOREABLE,
                            f"no query input bound to Stan parameter(s) {unmapped}")]
    try:
        model = bridgestan.StanModel(str(compile_stan(code)), json.dumps(data))
    except Exception as e:  # a compile or data error is a defect of the emitted program
        return [CheckResult(tid, "logdensity", "failed", UNSCOREABLE, f"BridgeStan: {e}")]

    results: list[CheckResult] = []
    for i, (pt, want) in enumerate(zip(points, expected)):
        want = parse_expected(want)
        constrained = {p["stan"]: pt[p["input"]] for p in manifest["parameters"]}
        try:
            theta = model.param_unconstrain_json(json.dumps(constrained))
            got = model.log_density(theta, propto=False, jacobian=False)
        except Exception as e:
            results.append(CheckResult(tid, f"logdensity[{i}]", "failed", UNSCOREABLE, str(e)))
            continue
        if abs(got - want) <= atol + rtol * abs(want):
            results.append(CheckResult(tid, f"logdensity[{i}]", "passed"))
        else:
            results.append(CheckResult(
                tid, f"logdensity[{i}]", "failed", NUMERIC_MISMATCH,
                f"Stan {got!r} vs expected {want!r} (|diff| {abs(got - want):.3e}, "
                f"atol {atol}, rtol {rtol})",
            ))
    return results
