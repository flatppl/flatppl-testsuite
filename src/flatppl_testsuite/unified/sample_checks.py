"""Generic, dist-agnostic `@sample` checks for the unified (sample, stablehlo)
test_type — a faithful port of `corpora/stablehlo/gate.py`'s five sample
checks (`check_distribution`, `check_independence`,
`check_sample_key_reproducibility`, `check_sample_key_advance`,
`check_fanout_distribution` / `_discrete` / `_mvnormal` / `_dirichlet`) onto
the per-test-dir harness. The other two checks in that gate's docstring
(`logdensity_value`, `logdensity_gradient`) are NOT here — they already have a
home in `runners/logdensity_stablehlo.py`.

Each check takes already-drawn samples (the runner owns emitting +
executing) plus a frozen `stat` dict (written by a sample `test.py`'s
`stat()` via `regen.py`) and returns a `CheckResult`. `stat['distribution']`
is a `{"family": <scipy.stats name>, "kwargs": {...}}` recipe — NOT a frozen
number — because a KS test needs a live `.cdf`; reconstructing a scipy frozen
distribution from a frozen recipe is a deterministic library call, not an
oracle computation (the actual oracle code — `test.py`'s `stat()` — still
runs only under `regen.py`, same convention as `oracle(point)`).

Tolerances mirror `corpora/stablehlo/manifest.json` / `gate.py`'s module
constants exactly (not invented): `KS_STAT_MAX = 0.02`,
`MOMENT_REL_TOL = 0.03`, `INDEPENDENCE_ABS_CORR_MAX = 0.05`.
"""
from __future__ import annotations

import math

import numpy as np

from flatppl_testsuite.scoring.result import CheckResult, NUMERIC_MISMATCH

KS_STAT_MAX = 0.02
MOMENT_REL_TOL = 0.03
INDEPENDENCE_ABS_CORR_MAX = 0.05

DEFAULT_TOLERANCES = {
    "ks_stat_max": KS_STAT_MAX,
    "moment_rel_tol": MOMENT_REL_TOL,
    "independence_abs_corr_max": INDEPENDENCE_ABS_CORR_MAX,
}


def _build_ref(spec: dict):
    """A live scipy frozen distribution from a `{"family", "kwargs"}` recipe."""
    from scipy import stats

    return getattr(stats, spec["family"])(**spec.get("kwargs", {}))


def _lag1_autocorr(x: np.ndarray) -> float:
    """Ported verbatim from `gate.py`'s `_lag1_autocorr`."""
    x = x - x.mean()
    denom = float((x * x).sum())
    if denom == 0:
        return 0.0
    return float((x[:-1] * x[1:]).sum() / denom)


def dirichlet_theo_corr(alpha: np.ndarray, a0: float, i: int, j: int) -> float:
    """Ported verbatim from `oracle.py`'s `dirichlet_theo_corr`: the
    closed-form Dirichlet component correlation (always negative)."""
    return -math.sqrt(alpha[i] * alpha[j] / ((a0 - alpha[i]) * (a0 - alpha[j])))


def _binned_chisquare(xs: np.ndarray, dist, k_min: int, k_max: int, min_expected: float = 5.0):
    """Ported verbatim from `gate.py`'s `_binned_chisquare`."""
    from scipy.stats import chisquare

    n = len(xs)
    ks = np.arange(k_min, k_max)
    expected = n * dist.pmf(ks)
    observed = np.array([np.sum(xs == k) for k in ks], dtype=float)
    expected = np.append(expected, n * dist.sf(k_max - 1))
    observed = np.append(observed, np.sum(xs >= k_max))

    merged_obs: list[float] = []
    merged_exp: list[float] = []
    obs_acc = exp_acc = 0.0
    for o, e in zip(observed, expected):
        obs_acc += o
        exp_acc += e
        if exp_acc >= min_expected:
            merged_obs.append(obs_acc)
            merged_exp.append(exp_acc)
            obs_acc = exp_acc = 0.0
    if exp_acc > 0:
        if merged_obs:
            merged_obs[-1] += obs_acc
            merged_exp[-1] += exp_acc
        else:
            merged_obs.append(obs_acc)
            merged_exp.append(exp_acc)
    stat, pval = chisquare(merged_obs, merged_exp)
    return float(stat), float(pval), len(merged_obs) - 1


# ---------------------------------------------------------------------------
# distribution — ports gate.py's check_distribution: empirical moments vs the
# frozen reference, plus KS (continuous) or nothing extra (discrete; a binned
# chi-square lives in fanout_distribution, mirroring the fact that gate.py's
# scalar check_distribution never runs chi-square either — only its fanned
# sibling check_fanout_distribution_discrete does).
# ---------------------------------------------------------------------------
def check_distribution(
    test_id: str, xs: np.ndarray, stat: dict,
    *, ks_stat_max: float = KS_STAT_MAX, moment_rel_tol: float = MOMENT_REL_TOL,
    **_ignored,
) -> CheckResult:
    xs = np.asarray(xs).reshape(-1)
    ref = _build_ref(stat["distribution"])
    discrete = bool(stat.get("discrete", False))
    emp_mean, emp_var = float(xs.mean()), float(xs.var())
    ref_mean, ref_var = float(ref.mean()), float(ref.var())
    mean_tol = max(moment_rel_tol * abs(ref_mean), 6.0 * math.sqrt(ref_var / len(xs)))
    dmean = abs(emp_mean - ref_mean)
    dvar_rel = abs(emp_var - ref_var) / ref_var if ref_var else abs(emp_var - ref_var)
    if discrete:
        ok = dmean <= mean_tol and dvar_rel <= 0.05
        detail = f"mean {emp_mean:.4f} vs {ref_mean:.4f} (tol {mean_tol:.4f}); var relΔ {dvar_rel:.3f}"
    else:
        from scipy.stats import kstest
        ks = float(kstest(xs, ref.cdf).statistic)
        ok = ks <= ks_stat_max and dmean <= mean_tol and dvar_rel <= 0.05
        detail = f"KS={ks:.4f} (max {ks_stat_max}); mean {emp_mean:.4f} vs {ref_mean:.4f}; var relΔ {dvar_rel:.3f}"
    return CheckResult(test_id, "distribution", "passed" if ok else "failed",
                        "" if ok else NUMERIC_MISMATCH, detail)


# ---------------------------------------------------------------------------
# independence — ports gate.py's check_independence (beta: X/(X+Y) from two
# separate Gamma rng streams; dirichlet: one Gamma stream per component).
# `params` supplies the fixture's free parameters (alpha for dirichlet).
# ---------------------------------------------------------------------------
def check_independence(
    test_id: str, xs: np.ndarray, stat: dict, params: dict,
    *, independence_abs_corr_max: float = INDEPENDENCE_ABS_CORR_MAX,
    ks_stat_max: float = KS_STAT_MAX,
    **_ignored,
) -> CheckResult:
    kind = stat.get("independence")
    if kind == "beta":
        x = np.asarray(xs).reshape(-1)
        std = float(x.std())
        frac_half = float(np.mean(np.abs(x - 0.5) < 1e-4))
        ac = _lag1_autocorr(x)
        ok = std > 0.05 and frac_half < 0.01 and abs(ac) < independence_abs_corr_max
        detail = (f"std={std:.4f} (not collapsed to 0.5; frac≈0.5={frac_half:.4f}); "
                  f"lag1 autocorr={ac:+.4f} (|·|<{independence_abs_corr_max})")
        return CheckResult(test_id, "independence", "passed" if ok else "failed",
                            "" if ok else NUMERIC_MISMATCH, detail)

    if kind == "dirichlet":
        from scipy.stats import beta as beta_dist, kstest
        alpha = np.asarray(params["alpha"], dtype=float)
        a0 = alpha.sum()
        spread = float(np.mean(xs.max(axis=1) - xs.min(axis=1)))
        ks_marg = []
        for i in range(xs.shape[1]):
            ref = beta_dist(alpha[i], a0 - alpha[i])
            ks_marg.append(float(kstest(xs[:, i], ref.cdf).statistic))
        worst_ks = max(ks_marg)
        corr = np.corrcoef(xs, rowvar=False)
        pairs = [(0, 1), (0, 2), (1, 2)]
        worst_corr = max(abs(corr[i, j] - dirichlet_theo_corr(alpha, a0, i, j)) for i, j in pairs)
        worst_ac = max(abs(_lag1_autocorr(xs[:, i])) for i in range(xs.shape[1]))
        ok = (spread > 0.05 and worst_ks <= ks_stat_max
              and worst_corr <= 0.05 and worst_ac < independence_abs_corr_max)
        emp = {f"{i}{j}": round(float(corr[i, j]), 3) for i, j in pairs}
        theo = {f"{i}{j}": round(dirichlet_theo_corr(alpha, a0, i, j), 3) for i, j in pairs}
        detail = (f"marginal KS≤{worst_ks:.4f}; comp-corr emp={emp} theo={theo} "
                  f"(worstΔ={worst_corr:.3f}); lag1 autocorr≤{worst_ac:.4f}")
        return CheckResult(test_id, "independence", "passed" if ok else "failed",
                            "" if ok else NUMERIC_MISMATCH, detail)

    return CheckResult(test_id, "independence", "skipped", "", "not an independence subject")


# ---------------------------------------------------------------------------
# key_reproducibility — ports gate.py's check_sample_key_reproducibility: same
# key -> bit-identical (value, new_key). No oracle/stat needed.
# ---------------------------------------------------------------------------
def check_key_reproducibility(
    test_id: str, value1: np.ndarray, key1: np.ndarray, value2: np.ndarray, key2: np.ndarray,
) -> CheckResult:
    value_eq = bool(np.array_equal(value1, value2))
    key_eq = bool(np.array_equal(key1, key2))
    ok = value_eq and key_eq
    detail = f"same key -> value bit-identical={value_eq}, new_key identical={key_eq}"
    return CheckResult(test_id, "key_reproducibility", "passed" if ok else "failed",
                        "" if ok else NUMERIC_MISMATCH, detail)


# ---------------------------------------------------------------------------
# key_advance — ports gate.py's check_sample_key_advance: chaining a short
# run of draws never repeats a key (mechanical rng_bit_generator property,
# independent of whether the DECODED value happens to coincide).
# ---------------------------------------------------------------------------
def check_key_advance(test_id: str, keys: list[np.ndarray]) -> CheckResult:
    seen = {tuple(int(x) for x in k) for k in keys}
    ok = len(seen) == len(keys)
    detail = f"{len(keys)} chained keys, {len(seen)} distinct (want all distinct)"
    return CheckResult(test_id, "key_advance", "passed" if ok else "failed",
                        "" if ok else NUMERIC_MISMATCH, detail)


# ---------------------------------------------------------------------------
# fanout_distribution — ports gate.py's check_fanout_distribution /
# _discrete / _mvnormal / _dirichlet, dispatched on `stat`'s
# fanout_dim/fanout_simplex/discrete flags (mirrors gate.py's `run()`
# if/elif chain over fx.fanout_simplex / fx.fanout_dim / fx.sample_discrete).
# `params` supplies mu/cov (mvnormal) or alpha (dirichlet) for the closed-form
# reference — the same free params the `.iid.sample.flatppl` query is called
# with.
# ---------------------------------------------------------------------------
def check_fanout_distribution(
    test_id: str, xs: np.ndarray, stat: dict, params: dict,
    *, ks_stat_max: float = KS_STAT_MAX, moment_rel_tol: float = MOMENT_REL_TOL,
    **_ignored,
) -> CheckResult:
    if stat.get("fanout_simplex"):
        return _check_dirichlet_fanout(test_id, xs, params, moment_rel_tol)
    if stat.get("fanout_dim"):
        return _check_mvnormal_fanout(test_id, xs, params)
    if stat.get("discrete"):
        return _check_fanout_discrete(test_id, xs, stat, moment_rel_tol)
    return _check_fanout_continuous(test_id, xs, stat, ks_stat_max, moment_rel_tol)


def _check_fanout_continuous(test_id, xs, stat, ks_stat_max, moment_rel_tol) -> CheckResult:
    xs = np.asarray(xs).reshape(-1)
    ref = _build_ref(stat["distribution"])
    emp_mean, emp_var = float(xs.mean()), float(xs.var())
    ref_mean, ref_var = float(ref.mean()), float(ref.var())
    mean_tol = max(moment_rel_tol * abs(ref_mean), 6.0 * math.sqrt(ref_var / len(xs)))
    dmean = abs(emp_mean - ref_mean)
    dvar_rel = abs(emp_var - ref_var) / ref_var if ref_var else abs(emp_var - ref_var)
    from scipy.stats import kstest
    ks = float(kstest(xs, ref.cdf).statistic)
    ok = ks <= ks_stat_max and dmean <= mean_tol and dvar_rel <= 0.05
    detail = (f"n={len(xs)}; KS={ks:.4f} (max {ks_stat_max}); "
              f"mean {emp_mean:.4f} vs {ref_mean:.4f}; var relΔ {dvar_rel:.3f}")
    return CheckResult(test_id, "fanout_distribution", "passed" if ok else "failed",
                        "" if ok else NUMERIC_MISMATCH, detail)


def _check_fanout_discrete(test_id, xs, stat, moment_rel_tol) -> CheckResult:
    xs = np.asarray(xs).reshape(-1)
    ref = _build_ref(stat["distribution"])
    emp_mean, emp_var = float(xs.mean()), float(xs.var())
    ref_mean, ref_var = float(ref.mean()), float(ref.var())
    mean_tol = max(moment_rel_tol * abs(ref_mean), 6.0 * math.sqrt(ref_var / len(xs)))
    dmean = abs(emp_mean - ref_mean)
    dvar_rel = abs(emp_var - ref_var) / ref_var if ref_var else abs(emp_var - ref_var)
    non_degenerate = emp_var > 1e-6

    k_min = int(stat.get("fanout_discrete_kmin", 0))
    k_max = k_min + max(20, int(math.ceil(ref_mean + 10.0 * math.sqrt(max(ref_var, 1.0)))))
    chi2_stat, pval, dof = _binned_chisquare(xs, ref, k_min, k_max)

    ok = non_degenerate and dmean <= mean_tol and dvar_rel <= 0.05 and pval >= 1e-4
    detail = (f"n={len(xs)}; mean {emp_mean:.4f} vs {ref_mean:.4f} (tol {mean_tol:.4f}); "
              f"var relΔ {dvar_rel:.3f}; non-degenerate={non_degenerate}; "
              f"chi2 GOF stat={chi2_stat:.2f} dof={dof} p={pval:.4g} (min 1e-4)")
    return CheckResult(test_id, "fanout_distribution", "passed" if ok else "failed",
                        "" if ok else NUMERIC_MISMATCH, detail)


def _check_mvnormal_fanout(test_id, xs, params) -> CheckResult:
    mu = np.asarray(params["mu"], dtype=float)
    cov = np.asarray(params["cov"], dtype=float)
    n = len(xs)
    emp_mean = xs.mean(axis=0)
    emp_cov = np.cov(xs, rowvar=False)
    mean_se = np.sqrt(np.diag(cov) / n)
    cov_se = np.sqrt((np.outer(np.diag(cov), np.diag(cov)) + cov**2) / n)
    mean_z = np.abs(emp_mean - mu) / np.maximum(mean_se, 1e-12)
    cov_z = np.abs(emp_cov - cov) / np.maximum(cov_se, 1e-12)
    worst_z = float(max(mean_z.max(), cov_z.max()))
    ok = worst_z <= 5.0
    detail = (
        f"n={n} [n,{len(mu)}] fanned draws; mean {emp_mean.tolist()} vs {mu.tolist()} "
        f"(max {mean_z.max():.2f}sigma); cov {emp_cov.tolist()} vs {cov.tolist()} "
        f"(max {cov_z.max():.2f}sigma); worst {worst_z:.2f}sigma (max 5.00sigma)"
    )
    return CheckResult(test_id, "fanout_distribution", "passed" if ok else "failed",
                        "" if ok else NUMERIC_MISMATCH, detail)


def _check_dirichlet_fanout(test_id, xs, params, moment_rel_tol) -> CheckResult:
    alpha = np.asarray(params["alpha"], dtype=float)
    a0 = float(alpha.sum())
    n = len(xs)
    d = xs.shape[1]

    row_sums = xs.sum(axis=1)
    worst_row = float(np.max(np.abs(row_sums - 1.0)))

    ref_mean = alpha / a0
    emp_mean = xs.mean(axis=0)
    ref_var = alpha * (a0 - alpha) / (a0 ** 2 * (a0 + 1))
    mean_tol = np.maximum(moment_rel_tol * ref_mean, 6.0 * np.sqrt(ref_var / n))
    mean_ok = bool(np.all(np.abs(emp_mean - ref_mean) <= mean_tol))

    emp_var = xs.var(axis=0)
    dvar_rel = np.abs(emp_var - ref_var) / ref_var
    worst_dvar_rel = float(dvar_rel.max())

    corr = np.corrcoef(xs, rowvar=False)
    pairs = [(i, j) for i in range(d) for j in range(i + 1, d)]
    worst_corr = max(abs(corr[i, j] - dirichlet_theo_corr(alpha, a0, i, j)) for i, j in pairs)

    ok = worst_row <= 1e-4 and mean_ok and worst_dvar_rel <= 0.05 and worst_corr <= 0.05
    detail = (f"n={n} [n,{d}] fanned draws; simplex row-sum maxΔ={worst_row:.2e}; "
              f"mean {emp_mean.tolist()} vs {ref_mean.tolist()}; "
              f"var relΔ max={worst_dvar_rel:.3f}; comp-corr worstΔ={worst_corr:.3f}")
    return CheckResult(test_id, "fanout_distribution", "passed" if ok else "failed",
                        "" if ok else NUMERIC_MISMATCH, detail)
