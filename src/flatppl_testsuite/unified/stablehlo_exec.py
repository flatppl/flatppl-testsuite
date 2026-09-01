#!/usr/bin/env python3
"""Execute emitted StableHLO under Enzyme-JAX.

``enzyme_ad.jax.hlo_call(*args, source=<stablehlo text>)`` imports a StableHLO
module (entry ``@main``) as a jit-able + differentiable JAX callable. This
module is the *only* place the gate touches the executor; everything else
compares its output to the independent scipy oracle.

Facts about the wheel that drive the API here:

* the call MUST run under ``jax.jit`` (Enzyme raises "must be JIT'ed"
  otherwise), so both value and gradient are wrapped in ``jax.jit``;
* the emitter names the entry ``@logdensity`` / ``@sample``, but ``hlo_call``
  binds ``@main`` — so we rename the entry symbol before handing the text over.
* ``@sample`` is a THREADED-KEY pure function (spec §07's `rand(rstate, m) ->
  (value, new_rstate)` contract): it takes a leading ``%key: tensor<2xui64>``
  and returns ``(value, new_key)``. A ``tensor<2xui64>`` argument silently
  truncates to ``uint32`` unless ``jax.config.update("jax_enable_x64", True)``
  runs BEFORE the key array is constructed — ``hlo_call`` then asserts on the
  dtype mismatch. Reproducibility (same key -> same draw) replaces the old
  ABI's reliance on XLA's per-call stateless nondeterminism.

Everything is single-precision (``tensor<f32>``): the emitted modules are f32,
so oracle comparisons use an f32-appropriate tolerance. The key itself is
``uint64`` regardless (dtype-independent, per the emitter's `MlirTy::Key`).
"""
from __future__ import annotations

import os
import re
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path

import numpy as np


def flatppl_bin() -> Path:
    """The `flatppl` binary to emit with. Must be built with the `stablehlo`
    feature (the subcommand is feature-gated). Taken from FLATPPL_BIN, else the
    setup-installed `.pixi-bin/bin/flatppl` (which will NOT have the subcommand
    unless the pinned rev carries it — hence the explicit env var)."""
    env = os.environ.get("FLATPPL_BIN")
    if env:
        return Path(env)
    # This file lives at <repo>/src/flatppl_testsuite/unified/stablehlo_exec.py,
    # so the repo root is parents[3] (parents[2] is `src`). Getting this wrong
    # silently breaks the `.pixi-bin` fallback → binary_supports_stablehlo()
    # returns False → the whole unified suite skips when FLATPPL_BIN is unset.
    repo = Path(__file__).resolve().parents[3]
    return repo / ".pixi-bin" / "bin" / "flatppl"


class EmitRefused(RuntimeError):
    """`flatppl stablehlo` refused to emit (exit 3) — a determiniser/emitter
    legalisation refusal, surfaced verbatim."""


def emit(model_path: Path, mode: str) -> str:
    """Emit StableHLO text for ``model_path`` in ``mode`` (logdensity|sample),
    with the entry symbol renamed to ``@main`` for ``hlo_call``."""
    proc = subprocess.run(
        [str(flatppl_bin()), "stablehlo", str(model_path), "--mode", mode],
        capture_output=True, text=True,
    )
    if proc.returncode == 3:
        raise EmitRefused(proc.stderr.strip() or proc.stdout.strip())
    if proc.returncode != 0:
        raise RuntimeError(
            f"flatppl stablehlo exited {proc.returncode} for {model_path} "
            f"({mode}):\n{proc.stderr}"
        )
    text = proc.stdout
    return re.sub(r"@(logdensity|sample)\b", "@main", text)


def emit_concat(
    dir: Path,
    mode: str,
    query_name: str = "query.flatppl",
    model_name: str = "model.flatppl",
) -> str:
    """Emit ``model_name`` + ``query_name`` concatenated into one module.

    ``model_name`` is a parameter because the fragment and bayesian_inference
    corpora name the model after the test id (`superpose.flatppl`), not
    `model.flatppl` — so a StableHLO row can be added to those dirs against
    the file the det-js case already scores, with no second copy of the model
    to drift.

    The combined temporary module is written INSIDE ``dir`` because the CLI
    resolves relative `load_data("x.csv", ...)` sources against the model
    file's own location — a temp file in the system temp dir would break
    every load_data fixture at emit time."""
    model = (dir / model_name).read_text()
    query = (dir / query_name).read_text()
    src_text = model.rstrip() + "\n" + query.lstrip()
    with tempfile.NamedTemporaryFile(
        "w", suffix=".flatppl", prefix=".concat-", dir=dir, delete=False
    ) as f:
        f.write(src_text)
        tmp = Path(f.name)
    try:
        return emit(tmp, mode)
    finally:
        tmp.unlink(missing_ok=True)


def load_data_bindings(dir: Path, model_name: str = "model.flatppl") -> dict[str, Path]:
    """``name -> source path`` for the `load_data` bindings of a fixture
    (textual scan; corpus fixtures use literal relative sources). The emitter
    never opens the source (§13: shape from the declared valueset) — the
    harness loads it and feeds it as the runtime argument."""
    src = "".join((dir / f).read_text() for f in (model_name, "query.flatppl"))
    return {m[1]: dir / m[2]
            for m in re.finditer(r'^\s*(\w+)\s*=\s*load_data\(\s*"([^"]+)"', src, re.M)}


def data_columns(path: Path) -> list[list]:
    """A data file's columns, in source order — the feed order, since a table
    input destructures into one tensor arg per column in declared order and
    fixtures declare their columns in source order. JSON struct-of-arrays
    (vector cells allowed) or header-row CSV (scalar cells)."""
    if path.suffix == ".json":
        import json

        return list(json.loads(path.read_text()).values())
    import polars as pl

    df = pl.read_csv(path)
    return [df[c].to_list() for c in df.columns]


# --- JAX/Enzyme plumbing (imported lazily so the module is importable, and the
# gate's manifest/oracle are usable, in an env without jax/enzyme). ---
def _jax():
    import jax
    import jax.numpy as jnp
    from enzyme_ad.jax import hlo_call

    # MUST happen before any tensor<2xui64> key array is constructed, else it
    # silently truncates to uint32 and hlo_call asserts on the dtype mismatch
    # (see the module docstring). Idempotent, so safe to call on every import.
    jax.config.update("jax_enable_x64", True)
    return jax, jnp, hlo_call


DEFAULT_KEY = (0, 0)


def _to_arg(jnp, v):
    """A Python float / list / nested list -> an f32 JAX array of the shape the
    emitted func arg expects (0-d for a scalar). A plain SCALAR Python ``int``
    becomes an i32 array instead (a list of ints still routes to float32,
    same as any other list) -- the emitter lowers an ``elementof(posintegers)``
    ABI arg (e.g. Binomial's `n`) to `tensor<i32>`, and `hlo_call` asserts on a
    dtype mismatch, not just a shape one."""
    dtype = np.int32 if isinstance(v, int) and not isinstance(v, bool) else np.float32
    return jnp.asarray(np.asarray(v, dtype=dtype))


def value(src: str, arg_values: list) -> float:
    """Execute ``@main`` at ``arg_values`` and return the scalar result."""
    jax, jnp, hlo_call = _jax()
    args = [_to_arg(jnp, v) for v in arg_values]

    def f(*a):
        return hlo_call(*a, source=src)[0]

    return float(jax.jit(f)(*args))


def gradient(src: str, arg_values: list, argnums: list[int]) -> list:
    """``jax.grad`` of ``@main`` w.r.t. the arguments in ``argnums`` — the HMC
    path. Returns one entry per requested argnum (a float for a scalar arg, a
    list for a vector arg), matching the finite-difference oracle's shape."""
    jax, jnp, hlo_call = _jax()
    args = [_to_arg(jnp, v) for v in arg_values]
    argnums_t = tuple(argnums)

    def f(*a):
        return hlo_call(*a, source=src)[0].sum()

    g = jax.jit(jax.grad(f, argnums=argnums_t))(*args)
    out = []
    for gi in g:
        arr = np.asarray(gi)
        out.append(float(arr) if arr.ndim == 0 else arr.tolist())
    return out


@lru_cache(maxsize=64)
def _jitted_sample(src: str):
    """The `jax.jit`-compiled ``(key, *free_params) -> (value, new_key)``
    callable for ``src``, cached by source text. Chaining calls a threaded
    `@sample` many times (reproducibility/advance/distribution checks all
    do); without this cache each call would define + trace + XLA-compile a
    fresh closure, which is the classic "jit inside a loop" antipattern and
    makes a 100k-draw chain many times slower than compiling once."""
    jax, _, hlo_call = _jax()

    def f(k, *a):
        return hlo_call(k, *a, source=src)

    return jax.jit(f)


def sample_call(
    src: str, key: tuple | np.ndarray, arg_values: list | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Execute a threaded-key ``@sample`` ONCE at ``key`` (+ any free params).

    Returns ``(value, new_key)`` as numpy arrays — the raw building block for
    reproducibility/advance/chaining/distribution checks. ``key`` is a
    length-2 ``(lo, hi)`` pair (or an existing ``tensor<2xui64>``-shaped
    array, e.g. a previous call's ``new_key``, to chain draws)."""
    _, jnp, _ = _jax()
    jit_f = _jitted_sample(src)
    key_arr = jnp.asarray(np.asarray(key, dtype=np.uint64))
    args = [_to_arg(jnp, v) for v in (arg_values or [])]
    value, new_key = jit_f(key_arr, *args)
    return np.asarray(value), np.asarray(new_key)


def samples(
    src: str, n: int, arg_values: list | None = None, key: tuple = DEFAULT_KEY
) -> np.ndarray:
    """Draw ``n`` independent realisations of a threaded-key ``@sample`` by
    CHAINING the key forward: call 1 uses ``key``, call ``i+1`` uses call
    ``i``'s returned ``new_key`` — the reproducible-ABI replacement for the
    old approach of calling one stateless jitted function ``n`` times and
    relying on XLA's per-call nondeterminism. One call = one draw (scalar or
    length-``k`` vector variate). Returns shape ``(n,)`` or ``(n, k)``."""
    draws = []
    cur = key
    for _ in range(n):
        v, cur = sample_call(src, cur, arg_values)
        draws.append(v)
    return np.stack(draws)


def samples_fanned(
    src: str, n: int, arg_values: list | None = None, key: tuple = DEFAULT_KEY
) -> np.ndarray:
    """Like `samples`, but for a Tier-1 FANNED ``@sample`` (``iid(K, m)``)
    whose single call already returns an ``[m]`` batch of iid draws from ONE
    ``rng_bit_generator`` advance. Chains calls (each advancing the key once)
    until at least ``n`` draws are collected, then trims to exactly ``n``."""
    draws: list[np.ndarray] = []
    cur = key
    total = 0
    while total < n:
        v, cur = sample_call(src, cur, arg_values)
        flat = v.reshape(-1)
        draws.append(flat)
        total += flat.size
    return np.concatenate(draws)[:n]


def samples_fanned_multivariate(
    src: str, n: int, d: int, arg_values: list | None = None, key: tuple = DEFAULT_KEY
) -> np.ndarray:
    """Like `samples_fanned`, but for a Tier-2 MULTIVARIATE fanned ``@sample``
    (``iid(MvNormal(mu, cov), m)``) whose single call returns an ``[m, d]``
    batch of iid d-vectors from ONE ``rng_bit_generator`` advance. Each call's
    draw is reshaped to ``(-1, d)`` — preserving the per-row d-vector, unlike
    `samples_fanned`'s elementwise flatten, which would scramble rows across
    components — and stacked along axis 0 until at least ``n`` rows are
    collected, then trimmed to exactly ``n``. Returns shape ``(n, d)``."""
    draws: list[np.ndarray] = []
    cur = key
    total = 0
    while total < n:
        v, cur = sample_call(src, cur, arg_values)
        rows = np.asarray(v).reshape(-1, d)
        draws.append(rows)
        total += rows.shape[0]
    return np.concatenate(draws, axis=0)[:n]


@lru_cache(maxsize=1)
def executor_available() -> bool:
    """True if jax + enzyme_ad import (so the gate can run at all)."""
    try:
        import jax  # noqa: F401
        from enzyme_ad.jax import hlo_call  # noqa: F401

        return True
    except Exception:
        return False


def binary_supports_stablehlo() -> bool:
    """True if the configured binary exposes the (feature-gated) `stablehlo`
    subcommand."""
    b = flatppl_bin()
    if not b.exists():
        return False
    try:
        proc = subprocess.run(
            [str(b), "stablehlo", "--help"], capture_output=True, text=True
        )
    except OSError:
        return False
    return proc.returncode == 0
