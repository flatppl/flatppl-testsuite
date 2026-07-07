#!/usr/bin/env python3
"""StableHLO numeric-EXECUTION gate.

Emits StableHLO from the LOCAL `flatppl` binary (`FLATPPL_BIN`, built with the
`stablehlo` feature), runs it under Enzyme-JAX, and checks the emitted density
modules as NUMBERS and GRADIENTS — not just structurally — against the
INDEPENDENT scipy oracle (`oracle.py`, frozen into each fixture's
`expected.json` by `gen_expected.py`). Per fixture, up to seven checks:

  logdensity_value            emitted @logdensity, executed, vs frozen scipy
                              value (f32, |Δ| < 1e-4)
  logdensity_gradient         jax.grad (Enzyme) w.r.t. θ vs the frozen central
                              finite-difference of scipy (|Δ| < 1e-3) — the
                              HMC path
  sample_distribution         N=100k draws of the threaded-key @sample (key
                              chained call-to-call) vs scipy (KS / moments)
  sample_independence         Beta + Dirichlet: separate internal rng streams
                              are independent (non-degenerate; lag-1 autocorr
                              ≈ 0; Dirichlet component correlations match the
                              theoretical simplex values, NOT +1)
  sample_key_reproducibility  same %key -> bit-identical (value, new_key)
  sample_key_advance          chaining a short run of draws never repeats a
                              key (a mechanical property of a counter-based
                              rng_bit_generator, checked independently of
                              whether the DECODED value happens to coincide —
                              common for a low-cardinality discrete dist)
  sample_fanout_distribution  `iid(K, n)` fanned draw vs scipy, built by
                              chaining the key across `[n]`-batched calls.
                              Tier 1 (Normal/Exponential/Uniform, straight-
                              line): KS + moments vs scipy. Tier 2 batched
                              rejection (Gamma/Beta/StudentT): same KS +
                              moments check, over the `[n]` batch produced by
                              the masked-`while` rejection loop. Tier 2 batched
                              multivariate (MvNormal, `[n, d]`): per-component
                              mean AND full sample covariance vs mu/cov (a
                              wrong `dot_general` contraction would show up
                              as a wrong covariance, not just a wrong mean).

Plus one standalone check not tied to a distribution fixture:

  chaining_independent_draws  two SEPARATE destructured `rand`s over the
                              IDENTICAL Normal(0,1) kernel, chained
                              `d1,s2=rand(k,·); d2,s3=rand(s2,·)` — the second
                              draw must differ from the first for a shared
                              key (a threading bug that reused the first
                              draw's bits for the second would show up as
                              d1 == d2)

    FLATPPL_BIN=/path/to/target/release/flatppl pixi run -e stablehlo shlo

Only a real MISMATCH (executed number/gradient/distribution/key-threading
outside tolerance) trips a nonzero exit — an emitter/executor refusal is
reported as SKIP.
"""
from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import oracle  # noqa: E402
import executor  # noqa: E402

N_SAMPLES = 100_000
# Fan-out distribution check: fewer total draws (still ample for a KS test)
# since each call already costs a full jit dispatch AND the fixture chains
# `[fanout_n]`-batched calls (`FANOUT_N // fx.fanout_n` calls, not N_SAMPLES).
FANOUT_N = 20_000
VALUE_ATOL = 1e-4
GRAD_ATOL = 1e-3
KS_MAX = 0.02
MOMENT_RTOL = 0.03
CORR_ABS_MAX = 0.05  # for a genuinely-zero correlation (lag-1 autocorrelation)


@dataclass
class CheckResult:
    test_id: str
    check_id: str
    status: str  # "passed" | "skipped" | "failed"
    message: str = ""
    worst: float = field(default=float("nan"))


def _arg_index(fx: oracle.Fixture, name: str) -> int:
    return list(fx.params).index(name)


# ---------------------------------------------------------------------------
# Per-mode checks
# ---------------------------------------------------------------------------
def check_value(fx, expected) -> CheckResult:
    try:
        src = executor.emit(HERE / fx.key / f"{fx.key}.flatppl", "logdensity")
        got = executor.value(src, fx.param_values())
    except executor.EmitRefused as e:
        return CheckResult(fx.key, "logdensity_value", "skipped", f"emit refused: {e}")
    exp = expected["logdensity_value"]
    d = abs(got - exp)
    if d <= VALUE_ATOL or (math.isinf(exp) and got == exp):
        return CheckResult(fx.key, "logdensity_value", "passed",
                           f"Δ={d:.2e} (got {got:.6f} vs {exp:.6f})", d)
    return CheckResult(fx.key, "logdensity_value", "failed",
                       f"Δ={d:.2e} > {VALUE_ATOL} (got {got:.6f} vs scipy {exp:.6f})", d)


def check_gradient(fx, expected) -> CheckResult:
    if not fx.grad_params:
        return CheckResult(fx.key, "logdensity_gradient", "skipped",
                           "no continuous parameters" if fx.key == "uniform"
                           else "gradient not executable (see notes)")
    argnums = [_arg_index(fx, n) for n in fx.grad_params]
    try:
        src = executor.emit(HERE / fx.key / f"{fx.key}.flatppl", "logdensity")
        got = executor.gradient(src, fx.param_values(), argnums)
    except executor.EmitRefused as e:
        return CheckResult(fx.key, "logdensity_gradient", "skipped", f"emit refused: {e}")
    except Exception as e:  # noqa: BLE001 — executor (Enzyme) autodiff limitation
        return CheckResult(fx.key, "logdensity_gradient", "skipped",
                           f"executor could not differentiate: {str(e).splitlines()[0][:80]}")
    worst = 0.0
    detail = []
    for name, g in zip(fx.grad_params, got):
        exp = expected["gradient"][name]
        gv = np.atleast_1d(np.asarray(g, dtype=float))
        ev = np.atleast_1d(np.asarray(exp, dtype=float))
        d = float(np.max(np.abs(gv - ev)))
        worst = max(worst, d)
        detail.append(f"{name}:Δ={d:.2e}")
    if worst <= GRAD_ATOL:
        return CheckResult(fx.key, "logdensity_gradient", "passed",
                           " ".join(detail), worst)
    return CheckResult(fx.key, "logdensity_gradient", "failed",
                       "worst Δ=%.2e > %s (%s)" % (worst, GRAD_ATOL, " ".join(detail)), worst)


def _draw(fx):
    # sample models live next to the logdensity model as <key>.sample.flatppl
    src = executor.emit(HERE / fx.key / f"{fx.key}.sample.flatppl", "sample")
    return executor.samples(src, N_SAMPLES, list(fx.sample_args))


def check_distribution(fx) -> CheckResult:
    if fx.sample_ref is None:
        return CheckResult(fx.key, "sample_distribution", "skipped", "not distributional-tested")
    try:
        xs = _draw(fx).reshape(-1)
    except executor.EmitRefused as e:
        return CheckResult(fx.key, "sample_distribution", "skipped", f"emit refused: {e}")
    ref = fx.sample_ref()
    emp_mean, emp_var = float(xs.mean()), float(xs.var())
    ref_mean, ref_var = float(ref.mean()), float(ref.var())
    mean_tol = max(MOMENT_RTOL * abs(ref_mean), 6.0 * math.sqrt(ref_var / len(xs)))
    dmean = abs(emp_mean - ref_mean)
    dvar_rel = abs(emp_var - ref_var) / ref_var if ref_var else abs(emp_var - ref_var)
    if fx.sample_discrete:
        ok = dmean <= mean_tol and dvar_rel <= 0.05
        stat = dmean
        detail = f"mean {emp_mean:.4f} vs {ref_mean:.4f} (tol {mean_tol:.4f}); var relΔ {dvar_rel:.3f}"
    else:
        from scipy.stats import kstest
        ks = float(kstest(xs, ref.cdf).statistic)
        ok = ks <= KS_MAX and dmean <= mean_tol and dvar_rel <= 0.05
        stat = ks
        detail = f"KS={ks:.4f} (max {KS_MAX}); mean {emp_mean:.4f} vs {ref_mean:.4f}; var relΔ {dvar_rel:.3f}"
    return CheckResult(fx.key, "sample_distribution",
                       "passed" if ok else "failed", detail, stat)


def _lag1_autocorr(x: np.ndarray) -> float:
    x = x - x.mean()
    denom = float((x * x).sum())
    if denom == 0:
        return 0.0
    return float((x[:-1] * x[1:]).sum() / denom)


def check_independence(fx) -> CheckResult:
    if fx.independence is None:
        return CheckResult(fx.key, "sample_independence", "skipped", "not an independence subject")
    try:
        xs = _draw(fx)
    except executor.EmitRefused as e:
        return CheckResult(fx.key, "sample_independence", "skipped", f"emit refused: {e}")

    if fx.independence == "beta":
        # Beta = X/(X+Y), X~Gamma, Y~Gamma from SEPARATE rng streams. A shared
        # stream would give X==Y => Beta ≡ 0.5 (zero variance). Independence:
        # spread present + successive draws uncorrelated (lag-1 autocorr ≈ 0).
        x = xs.reshape(-1)
        std = float(x.std())
        frac_half = float(np.mean(np.abs(x - 0.5) < 1e-4))
        ac = _lag1_autocorr(x)
        ok = std > 0.05 and frac_half < 0.01 and abs(ac) < CORR_ABS_MAX
        detail = (f"std={std:.4f} (not collapsed to 0.5; frac≈0.5={frac_half:.4f}); "
                  f"lag1 autocorr={ac:+.4f} (|·|<{CORR_ABS_MAX})")
        return CheckResult(fx.key, "sample_independence",
                           "passed" if ok else "failed", detail, abs(ac))

    # dirichlet: 3 components, one Gamma rng stream each. A shared stream would
    # give [1/3,1/3,1/3] every draw. Independence of the underlying streams is
    # observable as (a) non-degenerate components, (b) each marginal ~
    # Beta(a_i, a0-a_i), and (c) component correlations matching the THEORETICAL
    # (negative) Dirichlet values — a shared stream would give +1, not these.
    from scipy.stats import beta as beta_dist, kstest
    alpha = np.asarray(fx.sample_args[0], dtype=float)
    a0 = alpha.sum()
    spread = float(np.mean(xs.max(axis=1) - xs.min(axis=1)))
    # marginal KS per component
    ks_marg = []
    for i in range(xs.shape[1]):
        ref = beta_dist(alpha[i], a0 - alpha[i])
        ks_marg.append(float(kstest(xs[:, i], ref.cdf).statistic))
    worst_ks = max(ks_marg)
    # empirical vs theoretical component correlations
    def theo_corr(i, j):
        return -math.sqrt(alpha[i] * alpha[j] / ((a0 - alpha[i]) * (a0 - alpha[j])))
    corr = np.corrcoef(xs, rowvar=False)
    pairs = [(0, 1), (0, 2), (1, 2)]
    worst_corr = max(abs(corr[i, j] - theo_corr(i, j)) for i, j in pairs)
    # lag-1 autocorrelation across draws, per component (genuinely ≈ 0)
    worst_ac = max(abs(_lag1_autocorr(xs[:, i])) for i in range(xs.shape[1]))
    ok = (spread > 0.05 and worst_ks <= KS_MAX
          and worst_corr <= 0.05 and worst_ac < CORR_ABS_MAX)
    emp = {f"{i}{j}": round(float(corr[i, j]), 3) for i, j in pairs}
    theo = {f"{i}{j}": round(theo_corr(i, j), 3) for i, j in pairs}
    detail = (f"marginal KS≤{worst_ks:.4f}; comp-corr emp={emp} theo={theo} "
              f"(worstΔ={worst_corr:.3f}); lag1 autocorr≤{worst_ac:.4f}")
    return CheckResult(fx.key, "sample_independence",
                       "passed" if ok else "failed", detail, worst_corr)


# ---------------------------------------------------------------------------
# rng-threaded rand: key-ABI checks (reproducibility / advance / chaining /
# fan-out). None of these need the scipy oracle — they check the THREADING
# contract of `@sample(%key) -> (value, new_key)` itself.
# ---------------------------------------------------------------------------
def check_sample_key_reproducibility(fx) -> CheckResult:
    if not fx.sample_flatppl:
        return CheckResult(fx.key, "sample_key_reproducibility", "skipped",
                           "not @sample-tested")
    try:
        src = executor.emit(HERE / fx.key / f"{fx.key}.sample.flatppl", "sample")
        v1, k1 = executor.sample_call(src, executor.DEFAULT_KEY, list(fx.sample_args))
        v2, k2 = executor.sample_call(src, executor.DEFAULT_KEY, list(fx.sample_args))
    except executor.EmitRefused as e:
        return CheckResult(fx.key, "sample_key_reproducibility", "skipped", f"emit refused: {e}")
    ok = np.array_equal(v1, v2) and np.array_equal(k1, k2)
    detail = f"same key -> value bit-identical={np.array_equal(v1, v2)}, new_key identical={np.array_equal(k1, k2)}"
    return CheckResult(fx.key, "sample_key_reproducibility", "passed" if ok else "failed", detail)


def check_sample_key_advance(fx) -> CheckResult:
    if not fx.sample_flatppl:
        return CheckResult(fx.key, "sample_key_advance", "skipped", "not @sample-tested")
    try:
        src = executor.emit(HERE / fx.key / f"{fx.key}.sample.flatppl", "sample")
        keys = [np.asarray(executor.DEFAULT_KEY, dtype=np.uint64)]
        cur = executor.DEFAULT_KEY
        for _ in range(5):
            _, cur = executor.sample_call(src, cur, list(fx.sample_args))
            keys.append(np.asarray(cur))
    except executor.EmitRefused as e:
        return CheckResult(fx.key, "sample_key_advance", "skipped", f"emit refused: {e}")
    # A real counter-based rng_bit_generator advance is never expected to
    # repeat within a short chain — a purely MECHANICAL property of the key,
    # independent of whether the DECODED draw happens to coincide (which is
    # common and not a bug for a low-cardinality discrete distribution, e.g.
    # Bernoulli(0.3) repeats its outcome ~58% of the time by chance).
    seen = {tuple(int(x) for x in k) for k in keys}
    ok = len(seen) == len(keys)
    detail = f"{len(keys)} chained keys, {len(seen)} distinct (want all distinct)"
    return CheckResult(fx.key, "sample_key_advance", "passed" if ok else "failed", detail)


def check_fanout_distribution(fx) -> CheckResult:
    if not fx.fanout_flatppl:
        return CheckResult(fx.key, "sample_fanout_distribution", "skipped",
                           "no Tier-1 fan-out lowering for this kernel")
    try:
        src = executor.emit(HERE / fx.key / f"{fx.key}.iid.sample.flatppl", "sample")
        xs = executor.samples_fanned(src, FANOUT_N, list(fx.sample_args))
    except executor.EmitRefused as e:
        return CheckResult(fx.key, "sample_fanout_distribution", "skipped", f"emit refused: {e}")
    ref = fx.sample_ref()
    emp_mean, emp_var = float(xs.mean()), float(xs.var())
    ref_mean, ref_var = float(ref.mean()), float(ref.var())
    mean_tol = max(MOMENT_RTOL * abs(ref_mean), 6.0 * math.sqrt(ref_var / len(xs)))
    dmean = abs(emp_mean - ref_mean)
    dvar_rel = abs(emp_var - ref_var) / ref_var if ref_var else abs(emp_var - ref_var)
    from scipy.stats import kstest
    ks = float(kstest(xs, ref.cdf).statistic)
    ok = ks <= KS_MAX and dmean <= mean_tol and dvar_rel <= 0.05
    detail = (f"n={len(xs)} (batch {fx.fanout_n}, one rng_bit_generator/call); "
              f"KS={ks:.4f} (max {KS_MAX}); mean {emp_mean:.4f} vs {ref_mean:.4f}; "
              f"var relΔ {dvar_rel:.3f}")
    return CheckResult(fx.key, "sample_fanout_distribution",
                       "passed" if ok else "failed", detail, ks)


def check_mvnormal_fanout_distribution(fx) -> CheckResult:
    """Tier-2 MULTIVARIATE fan-out (MvNormal, `iid(K, n)` -> `[n, d]`): unlike
    the scalar `check_fanout_distribution`, there is no 1-d `sample_ref.cdf`
    to KS-test against — the correctness signal is the per-component mean AND
    the full sample covariance of the `[n, d]` draw vs the fixture's mu/cov.
    This is the check that would catch a wrong `dot_general` contraction
    (e.g. `z @ L` instead of `z @ L^T`): that bug leaves the marginal means
    and even the per-component variances alone (L and L^T share a diagonal
    and the same row/column norms up to a permutation for this 2x2 case) but
    corrupts the off-diagonal covariance entries, which THIS check reads
    directly from `np.cov`, not from a KS test on a flattened 1-d sample."""
    if not fx.fanout_flatppl:
        return CheckResult(fx.key, "sample_fanout_distribution", "skipped",
                           "no Tier-2 multivariate fan-out lowering for this kernel")
    d = fx.fanout_dim
    try:
        src = executor.emit(HERE / fx.key / f"{fx.key}.iid.sample.flatppl", "sample")
        xs = executor.samples_fanned_multivariate(src, fx.fanout_n, d, fx.param_values())
    except executor.EmitRefused as e:
        return CheckResult(fx.key, "sample_fanout_distribution", "skipped", f"emit refused: {e}")
    mu = np.asarray(fx.params["mu"], dtype=float)
    cov = np.asarray(fx.params["cov"], dtype=float)
    n = len(xs)
    emp_mean = xs.mean(axis=0)
    emp_cov = np.cov(xs, rowvar=False)
    # 6-sigma bands from the standard asymptotic sampling distributions: the
    # mean estimator has var Var(X_i)/n; the covariance estimator (for a
    # jointly-normal X) has var (Var(X_i)*Var(X_j) + Cov(X_i,X_j)^2)/n. Not
    # loosened ad hoc — these are the textbook standard errors, same style as
    # `check_distribution`'s `6.0 * sqrt(ref_var/len(xs))`.
    mean_se = np.sqrt(np.diag(cov) / n)
    cov_se = np.sqrt((np.outer(np.diag(cov), np.diag(cov)) + cov**2) / n)
    mean_z = np.abs(emp_mean - mu) / np.maximum(mean_se, 1e-12)
    cov_z = np.abs(emp_cov - cov) / np.maximum(cov_se, 1e-12)
    worst_z = float(max(mean_z.max(), cov_z.max()))
    ok = worst_z <= 6.0
    detail = (
        f"n={n} [n,{d}] fanned draws; mean {emp_mean.tolist()} vs {mu.tolist()} "
        f"(max {mean_z.max():.2f}sigma); cov {emp_cov.tolist()} vs {cov.tolist()} "
        f"(max {cov_z.max():.2f}sigma); worst {worst_z:.2f}sigma (max 6.00sigma)"
    )
    return CheckResult(fx.key, "sample_fanout_distribution",
                       "passed" if ok else "failed", detail, worst_z)


def check_chaining_independent_draws() -> CheckResult:
    """Two SEPARATE destructured `rand`s drawing from the IDENTICAL
    Normal(0,1) kernel (`corpora/stablehlo/chaining/{d1,d2}.sample.flatppl`,
    otherwise byte-identical source): `d1,s2=rand(k,lawof(x)); d2,s3=rand(s2,
    lawof(y))`. Not a distribution check — since the kernel is shared, the
    ONLY way `d1` and `d2` can differ is if the second draw actually consumed
    the first's ADVANCED key rather than re-reading the source key or
    somehow reusing the first draw's bits (a threading bug would show up as
    d1 == d2 for every key, since both would be the same op sequence over the
    same random bits)."""
    d1_path = HERE / "chaining" / "d1.sample.flatppl"
    d2_path = HERE / "chaining" / "d2.sample.flatppl"
    try:
        src1 = executor.emit(d1_path, "sample")
        src2 = executor.emit(d2_path, "sample")
        v1a, k1a = executor.sample_call(src1, executor.DEFAULT_KEY)
        v1b, k1b = executor.sample_call(src1, executor.DEFAULT_KEY)
        v2a, k2a = executor.sample_call(src2, executor.DEFAULT_KEY)
        v2b, k2b = executor.sample_call(src2, executor.DEFAULT_KEY)
    except executor.EmitRefused as e:
        return CheckResult("chaining", "chaining_independent_draws", "skipped", f"emit refused: {e}")
    d1_repro = bool(np.array_equal(v1a, v1b) and np.array_equal(k1a, k1b))
    d2_repro = bool(np.array_equal(v2a, v2b) and np.array_equal(k2a, k2b))
    independent = not np.array_equal(v1a, v2a)
    ok = d1_repro and d2_repro and independent
    detail = (f"d1 reproducible={d1_repro}; d2 reproducible={d2_repro}; "
              f"d1={float(v1a):.6f} vs d2={float(v2a):.6f} (independent={independent})")
    return CheckResult("chaining", "chaining_independent_draws",
                       "passed" if ok else "failed", detail)


def run() -> list[CheckResult]:
    results: list[CheckResult] = []
    for fx in oracle.FIXTURES:
        expected = json.loads((HERE / fx.key / "expected.json").read_text())
        results.append(check_value(fx, expected))
        results.append(check_gradient(fx, expected))
        results.append(check_distribution(fx))
        results.append(check_independence(fx))
        results.append(check_sample_key_reproducibility(fx))
        results.append(check_sample_key_advance(fx))
        results.append(
            check_mvnormal_fanout_distribution(fx) if fx.fanout_dim
            else check_fanout_distribution(fx)
        )
    results.append(check_chaining_independent_draws())
    return results


_OUTCOME = {"passed": "PASS", "skipped": "SKIP", "failed": "MISMATCH"}


def render(results) -> str:
    labels = [f"{r.test_id}::{r.check_id}" for r in results]
    width = max((len(x) for x in labels), default=8)
    lines = [
        "=" * 96,
        "STABLEHLO NUMERIC-EXECUTION GATE — emitted StableHLO under Enzyme-JAX vs scipy oracle",
        "=" * 96, "",
        f"  {'test_id :: check':<{width}}  outcome    detail",
        f"  {'-' * width}  -------    ------",
    ]
    for r, label in zip(results, labels):
        detail = r.message.splitlines()[0] if r.message else ""
        if len(detail) > 108:
            detail = detail[:105] + "..."
        lines.append(f"  {label:<{width}}  {_OUTCOME.get(r.status, r.status):<7}    {detail}")
    n_pass = sum(r.status == "passed" for r in results)
    n_skip = sum(r.status == "skipped" for r in results)
    n_bad = sum(r.status == "failed" for r in results)
    lines += ["", f"  {n_pass} PASS, {n_skip} SKIP, {n_bad} MISMATCH (of {len(results)} checks)"]
    return "\n".join(lines)


def main() -> int:
    if not executor.binary_supports_stablehlo():
        print(f"FLATPPL_BIN ({executor.flatppl_bin()}) has no `stablehlo` subcommand — "
              "point FLATPPL_BIN at a binary built with the `stablehlo` feature.")
        return 2
    if not executor.executor_available():
        print("Enzyme-JAX not importable — run under `pixi run -e stablehlo`.")
        return 2
    results = run()
    print(render(results))
    return 1 if any(r.status == "failed" for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())
