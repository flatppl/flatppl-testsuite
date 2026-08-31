"""Drive `scoring/sampler_sweep.cjs`: one Node process for the whole roster.

The whole space goes over in a single job file and comes back in a single JSON
reply. A subprocess per row would pay the engine's module load (~200 ms of
Node's type-stripping over the engine's TypeScript) once per row instead of once
per sweep, which is most of the wall clock at this roster size.

ENGINE RESOLUTION. The resolved directory is passed explicitly as
`--engine <dir>/packages/engine`, never left to the environment. `pixi.toml`'s
`[activation.env]` can clobber an inline `FLATPPL_JS_DIR=... pixi run ...`
override, so an env-only handoff silently falls back to the default sibling
checkout — the gotcha recorded in this repo's CLAUDE.md. Passing the flag is the
non-clobbered path, exactly as `score_js.cjs --engine <dir>` is.

`CONFIG.flatppl_js_dir` alone is NOT enough, and the failure is not
hypothetical. `CONFIG` resolves the engine as a sibling of the testsuite repo
root; from inside a git worktree that root is `flatppl-testsuite/.worktrees/<branch>`,
so the sibling is `flatppl-testsuite/.worktrees/flatppl-js`. A stale flatppl-js
clone is parked at exactly that path today (`e9803b6`, 2026-08-05, against a
main of `af0bab9`), and it is a REAL checkout, so no existence check catches it —
the sweep would silently draw from a months-old engine.

A flatppl-js checkout inside `flatppl-testsuite/.worktrees/` is another repo's
tree parked in this repo's worktree directory, and is structurally never the
engine a testsuite worktree means to load. So `resolve_engine_dir` rejects that
shape and falls back to the WORKSPACE ROOT's `flatppl-js` — the workspace root
being the directory holding the sibling `flatppl-*` repos, which is what
`flatppl-testsuite/..` means from a normal checkout and what
`flatppl-testsuite/.worktrees/<branch>/../../..` means from a worktree.

HOW the path was chosen is recorded in the frozen table's metadata alongside the
engine commit, so a reader can always tell which engine produced the numbers.
The path itself is NOT recorded: that file is tracked, and an absolute local path
churns per machine and diverges on CI. The commit identifies the engine, and the
resolution kind explains how it was found; the resolved path is printed at
runtime, where it is useful and costs no churn.
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from flatppl_testsuite.config import CONFIG

DRIVER = Path(__file__).resolve().parent.parent / "scoring" / "sampler_sweep.cjs"

# The sibling repos that identify the FlatPPL workspace root.
_WORKSPACE_MARKERS = ("flatppl-design", "flatppl-js", "flatppl-testsuite")


def _is_engine(d: Path) -> bool:
    return (d / "packages" / "engine" / "index.ts").exists()


def _parked_under_worktrees(d: Path) -> bool:
    """True when `d` sits inside some repo's `.worktrees/` directory."""
    return ".worktrees" in d.resolve().parts


def _workspace_root() -> Path | None:
    """Walk up from this file looking for the directory holding the sibling repos."""
    for parent in Path(__file__).resolve().parents:
        if all((parent / m).is_dir() for m in _WORKSPACE_MARKERS):
            return parent
    return None


def _config_default_path() -> Path:
    """What `CONFIG.flatppl_js_dir` resolves to when `FLATPPL_JS_DIR` is unset.

    Recomputed rather than read, so a set-but-identical env var can be told apart
    from a deliberate override. This matters under pixi: `[activation.env]` sets
    `FLATPPL_JS_DIR=${FLATPPL_JS_DIR:-$PIXI_PROJECT_ROOT/../flatppl-js}`, so the
    variable is ALWAYS set and "is it in the environment" cannot distinguish an
    operator's choice from the fallback.
    """
    return Path(__file__).resolve().parents[3].parent / "flatppl-js"


def resolve_engine_dir(explicit: Path | str | None = None) -> tuple[Path, str]:
    """The flatppl-js checkout to draw from, plus HOW it was chosen.

    The second element is a resolution KIND, not a path — it goes into the frozen
    table's metadata, which is tracked, so it must not carry a local absolute
    path that churns per machine. Callers that want the path print the first
    element next to it.

    A DELIBERATE choice always wins — an explicit argument, or a
    `FLATPPL_JS_DIR` pointing somewhere other than the computed default. Pointing
    the sweep at a specific checkout is a legitimate thing to do (bisecting, or
    testing an unmerged engine branch), so this must not be second-guessed; the
    provenance gate is what catches a stale one, not this function.

    A deliberate choice that holds no engine RAISES rather than falling through
    to the default. Silently substituting the workspace root would hand back a
    different engine than the operator named, so a typo'd path would report green
    against a tree nobody asked for.

    Only the ACCIDENTAL default is filtered: an unset (or default-valued)
    `FLATPPL_JS_DIR` resolving under a `.worktrees/` directory means the sibling
    lookup landed on another repo's parked tree, and the workspace root is
    preferred instead.
    """
    cfg = Path(CONFIG.flatppl_js_dir)
    is_default = cfg.resolve() == _config_default_path().resolve()
    ws = _workspace_root()

    deliberate: list[tuple[Path, str]] = []
    if explicit:
        deliberate.append((Path(explicit), "explicit argument"))
    if not is_default:
        deliberate.append((cfg, "FLATPPL_JS_DIR override"))
    for path, why in deliberate:
        if _is_engine(path):
            return path, why
    if deliberate:
        named = "; ".join(f"{p} ({w})" for p, w in deliberate)
        raise RuntimeError(
            f"no flatppl-js engine at the checkout you named: {named}. "
            f"Each must hold packages/engine/index.ts. Fix the path, or unset "
            f"FLATPPL_JS_DIR to fall back to the workspace default.")

    candidates: list[tuple[Path, str]] = []
    if not _parked_under_worktrees(cfg):
        candidates.append((cfg, "CONFIG default (sibling of the repo root)"))
    if ws:
        candidates.append((ws / "flatppl-js", "workspace root"))
    # Last resort: the parked default, so an unusual layout still runs rather
    # than hard-failing — with the reason recorded so it is never a silent choice.
    if _parked_under_worktrees(cfg):
        candidates.append((cfg, "CONFIG default — parked under .worktrees, last resort"))

    for path, why in candidates:
        if _is_engine(path):
            return path, why
    tried = "; ".join(f"{p} ({w})" for p, w in candidates) or "nothing"
    raise RuntimeError(f"no flatppl-js engine found — tried: {tried}")


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
    latent_mean: float | None = None
    """Self-normalised WEIGHTED mean of the probe's `latent` binding, or None
    when the probe names no latent. Weighted because the defect class it exists
    for moves the weights and not the atom positions -- see the driver's
    header."""
    latent_n_eff: float | None = None
    """Effective sample size of the same weights, `1 / sum(w_i^2)`. The band on
    `latent_mean` is `sqrt(prior variance / n_eff)`: the variance is the
    closed-form oracle, the ESS is the only part the run supplies."""
    latent_cov: float | None = None
    """Self-normalised WEIGHTED covariance of the `latent` binding with the
    variate's coordinate 0, under the same weights. The only moment that sees a
    mixing weight decoupled from the variate -- both marginals stay correct."""
    moment_denom: float | None = None
    """What `sum`/`sumsq`/`cross` must be divided by: `n` for a raw sum, `1` for
    an already-normalised WEIGHTED one (a `weighted_variate` probe). None means
    `n`, so a driver that predates the field still reads correctly."""
    variate_n_eff: float | None = None
    """Effective sample size of the weights the variate moments were taken
    under. Bands them the way `latent_n_eff` bands the latent's mean; None on a
    row whose moments are unweighted, which bands with `n`."""
    error: str = ""
    ms: int = 0

    @property
    def denom(self) -> float:
        return self.n if self.moment_denom is None else self.moment_denom

    def mean(self, i: int) -> float:
        return self.sum[i] / self.denom

    def var(self, i: int) -> float:
        """Population variance (ddof = 0); see checks.py on why the bias is moot."""
        m = self.mean(i)
        return self.sumsq[i] / self.denom - m * m

    def cov0(self, i: int) -> float:
        """Sample covariance of coordinate 0 with coordinate i, from raw sums."""
        return self.cross[i] / self.denom - self.mean(0) * self.mean(i)


def run(probes, *, seed: int, ks_subsample: int, engine_dir: Path | None = None) -> dict[str, Draws]:
    """Draw every probe in one Node process. Keyed by probe id."""
    root, _why = resolve_engine_dir(engine_dir)
    engine = root / "packages" / "engine"

    job = {
        "seed": seed,
        "ksSubsample": ks_subsample,
        "probes": [
            {"id": p.id, "source": p.source, "binding": p.binding, "n": p.n_draws,
             "k": p.k, "field": p.field, "latent": p.latent,
             "weightedVariate": p.weighted_variate}
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
            latent_mean=r.get("latentMean"), latent_n_eff=r.get("latentNEff"),
            latent_cov=r.get("latentCov"),
            moment_denom=r.get("momentDenom"), variate_n_eff=r.get("variateNEff"),
            error=r.get("error", ""), ms=r.get("ms", 0),
        )
    missing = {p.id for p in probes} - set(out)
    if missing:
        raise RuntimeError(f"driver returned no row for: {sorted(missing)}")
    return out
