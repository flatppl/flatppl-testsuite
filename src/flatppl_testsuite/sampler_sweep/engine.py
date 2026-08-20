"""Drive `scoring/sampler_sweep.cjs`: one Node process for the whole roster.

The whole space goes over in a single job file and comes back in a single JSON
reply. A subprocess per row would pay the engine's module load (~200 ms of
Node's type-stripping over the engine's TypeScript) once per row instead of once
per sweep, which is most of the wall clock at this roster size.

ENGINE RESOLUTION. `CONFIG.flatppl_js_dir` is passed explicitly as
`--engine <dir>/packages/engine`, not left to the environment. `pixi.toml`'s
`[activation.env]` can clobber an inline `FLATPPL_JS_DIR=... pixi run ...`
override, so an env-only handoff silently falls back to the default sibling
checkout — the gotcha recorded in this repo's CLAUDE.md. Passing the flag is the
non-clobbered path, exactly as `score_js.cjs --engine <dir>` is.
"""
from __future__ import annotations

import json
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from flatppl_testsuite.config import CONFIG

DRIVER = Path(__file__).resolve().parent.parent / "scoring" / "sampler_sweep.cjs"


@dataclass(frozen=True)
class Draws:
    """One row's raw return from the driver."""

    id: str
    status: str
    """`DRAWS`, `THREW`, or `NONFINITE` — the driver's vocabulary, before
    `table.classify` turns a THREW into REFUSES or MALFORMED."""
    n: int = 0
    k: int = 1
    sum: tuple[float, ...] = ()
    sumsq: tuple[float, ...] = ()
    cross: tuple[float, ...] = ()
    ks_sample: tuple[float, ...] = ()
    log_totalmass: float | None = None
    error: str = ""
    ms: int = 0

    def mean(self, i: int) -> float:
        return self.sum[i] / self.n

    def var(self, i: int) -> float:
        """Population variance (ddof = 0); see checks.py on why the bias is moot."""
        m = self.mean(i)
        return self.sumsq[i] / self.n - m * m

    def cov0(self, i: int) -> float:
        """Sample covariance of coordinate 0 with coordinate i, from raw sums."""
        return self.cross[i] / self.n - self.mean(0) * self.mean(i)


def run(probes, *, seed: int, ks_subsample: int, engine_dir: Path | None = None) -> dict[str, Draws]:
    """Draw every probe in one Node process. Keyed by probe id."""
    engine = Path(engine_dir or CONFIG.flatppl_js_dir) / "packages" / "engine"
    if not (engine / "index.ts").exists():
        raise RuntimeError(f"no flatppl-js engine at {engine} (set FLATPPL_JS_DIR)")

    job = {
        "seed": seed,
        "ksSubsample": ks_subsample,
        "probes": [
            {"id": p.id, "source": p.source, "binding": p.binding, "n": p.n_draws,
             "k": p.k, "field": p.field}
            for p in probes
        ],
    }

    with tempfile.TemporaryDirectory() as tmp:
        job_path = Path(tmp) / "job.json"
        job_path.write_text(json.dumps(job))
        proc = subprocess.run(
            [CONFIG.node_bin, str(DRIVER), str(job_path), "--engine", str(engine)],
            capture_output=True, text=True,
        )
    if proc.returncode != 0:
        raise RuntimeError(
            f"sampler_sweep.cjs exited {proc.returncode}:\n{proc.stderr.strip()}")

    payload = json.loads(proc.stdout)
    out: dict[str, Draws] = {}
    for r in payload["results"]:
        out[r["id"]] = Draws(
            id=r["id"], status=r["status"], n=r.get("n", 0), k=r.get("k", 1),
            sum=tuple(r.get("sum") or ()), sumsq=tuple(r.get("sumsq") or ()),
            cross=tuple(r.get("cross") or ()),
            ks_sample=tuple(r.get("ksSample") or ()),
            log_totalmass=r.get("logTotalmass"),
            error=r.get("error", ""), ms=r.get("ms", 0),
        )
    missing = {p.id for p in probes} - set(out)
    if missing:
        raise RuntimeError(f"driver returned no row for: {sorted(missing)}")
    return out
