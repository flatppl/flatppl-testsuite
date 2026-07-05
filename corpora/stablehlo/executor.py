#!/usr/bin/env python3
"""Execute emitted StableHLO under Enzyme-JAX.

``enzyme_ad.jax.hlo_call(*args, source=<stablehlo text>)`` imports a StableHLO
module (entry ``@main``) as a jit-able + differentiable JAX callable. This
module is the *only* place the gate touches the executor; everything else
compares its output to the independent scipy oracle.

Two facts about the wheel drive the API here:

* the call MUST run under ``jax.jit`` (Enzyme raises "must be JIT'ed"
  otherwise), so both value and gradient are wrapped in ``jax.jit``;
* the emitter names the entry ``@logdensity`` / ``@sample``, but ``hlo_call``
  binds ``@main`` — so we rename the entry symbol before handing the text over.

Everything is single-precision (``tensor<f32>``): the emitted modules are f32,
so oracle comparisons use an f32-appropriate tolerance.
"""
from __future__ import annotations

import os
import re
import subprocess
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
    repo = Path(__file__).resolve().parents[2]
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


# --- JAX/Enzyme plumbing (imported lazily so the module is importable, and the
# gate's manifest/oracle are usable, in an env without jax/enzyme). ---
def _jax():
    import jax
    import jax.numpy as jnp
    from enzyme_ad.jax import hlo_call

    return jax, jnp, hlo_call


def _to_arg(jnp, v):
    """A Python float / list / nested list -> an f32 JAX array of the shape the
    emitted func arg expects (0-d for a scalar)."""
    return jnp.asarray(np.asarray(v, dtype=np.float32))


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


def samples(src: str, n: int, arg_values: list | None = None) -> np.ndarray:
    """Draw ``n`` independent realisations of ``@sample``.

    The emitted ``@sample`` bakes the seed and lowers each draw to a
    nondeterministic ``stablehlo.rng``; XLA advances its state per execution,
    so calling the jitted function ``n`` times yields ``n`` independent draws
    (verified: repeated calls differ). Returns an array of shape ``(n,)`` for a
    scalar variate or ``(n, k)`` for a length-``k`` vector variate."""
    jax, jnp, hlo_call = _jax()
    args = [_to_arg(jnp, v) for v in (arg_values or [])]

    def f(*a):
        return hlo_call(*a, source=src)[0]

    jf = jax.jit(f)
    draws = [np.asarray(jf(*args)) for _ in range(n)]
    return np.stack(draws)


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
