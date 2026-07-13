#!/usr/bin/env python3
"""Generate ``corpora/examples/<test_id>/expected.json`` from an INDEPENDENT
oracle (scipy / Julia Distributions.jl — never the sibling FlatPPL engine),
one per ``status: "lowers"`` entry in ``corpora/examples/manifest.json``.

Each flatppl-examples model already ends in ``posterior =
bayesupdate(L, prior)`` with no query, so the oracle here must reproduce the
SAME posterior log-density this corpus's manifest constructs — the prior
log-density plus the likelihood log-density, at each point in that entry's
theta grid — exactly like
``corpora/bayesian_inference/gen_expected.py``'s ``oracle_bi_posterior``/
``oracle_eight_schools``, just parameterized over one oracle function per
example instead of writing the model out again.

Per spec (``flatppl-design/docs/06-measure-algebra.md``, "Posterior
construction"): ``logdensityof(bayesupdate(L, prior), theta) =
logdensityof(L, theta) + logdensityof(prior, theta)``, i.e. the prior draws'
log-densities at theta, plus the forward-model's log-density at the observed
data (``likelihoodof``/``kernelof``). No integration is needed. Distribution
conventions (``08-distributions.md``) and combinators
(``06-measure-algebra.md``) used below:

- ``Exponential(rate)``: ``scipy.stats.expon.logpdf(x, scale=1/rate)``.
- ``Gamma(shape, rate)``: shape-rate, ``scipy.stats.gamma.logpdf(x, a=shape,
  scale=1/rate)``.
- ``Uniform(interval(a, b))``: ``scipy.stats.uniform.logpdf(x, loc=a,
  scale=b-a)``.
- ``Beta(alpha, beta)``: ``scipy.stats.beta.logpdf(x, a=alpha, b=beta)``.
- ``Binomial(n, p)``: ``scipy.stats.binom.logpmf(k, n, p)``.
- ``Bernoulli(p)``: ``scipy.stats.bernoulli.logpmf(y, p)``; ``invlogit(z) =
  1/(1+exp(-z))``, the plain sigmoid (no alternate-scale convention).
- ``Poisson(rate)``: ``scipy.stats.poisson.logpmf(k, mu=rate)``.
- ``locscale(BaseDist(...), shift, scale)`` (measure-algebra §"Transformation
  and projection"): affine pushforward ``y = shift + scale*x``, matching
  scipy's ``loc``/``scale`` kwargs directly, e.g.
  ``scipy.stats.t.logpdf(y, df=nu, loc=mu, scale=sigma)`` for
  ``locscale(StudentT(nu), mu, sigma)``.
- Half-Cauchy (``normalize(truncate(Cauchy(x0, gamma), interval(0, inf)))``):
  ``truncate`` restricts the density to the positive half-line (``-inf``
  outside); ``normalize`` divides by the total mass of that half
  (``totalmass = 1/2`` by symmetry of Cauchy about ``x0=0``), so
  ``logdensityof(..., tau) = log(2) + cauchy.logpdf(tau, x0, gamma)`` for
  ``tau > 0`` — the same convention already frozen in
  ``corpora/bayesian_inference/gen_expected.py``'s ``oracle_eight_schools``.

Not on the default test path (``pixi run test`` does not import this
module). Run it manually to verify / regenerate:

    pixi run python corpora/examples/gen_expected.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Callable

from scipy.stats import bernoulli, beta as beta_dist, binom, cauchy, expon
from scipy.stats import gamma as gamma_dist
from scipy.stats import norm, poisson, t as t_dist, uniform

HERE = Path(__file__).resolve().parent


def _invlogit(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))


def oracle_bayesian_inference(theta_grid: list[dict]) -> list[float]:
    """``bayesian_inference_1``/``_2``: ``theta1 ~ Normal(0,1)``, ``theta2 ~
    Exponential(rate=1)``, ``a = 5*theta2``, ``b = abs(theta1)*theta2``,
    ``obs ~ iid(Normal(mu=a, sigma=b), 10)`` against the fixed 10-point
    ``observed_data``. Same model as ``corpora/bayesian_inference``'s
    ``bi1_posterior``/``bi2_posterior``, generalized over the theta grid
    instead of one fixed point."""
    data = [1.2, 3.4, 5.1, 2.8, 4.0, 3.7, 5.5, 2.1, 4.3, 3.9]
    values = []
    for th in theta_grid:
        t1, t2 = th["theta1"], th["theta2"]
        a = 5 * t2
        b = abs(t1) * t2
        lp = norm.logpdf(t1, 0, 1) + expon.logpdf(t2, scale=1.0)
        lp += sum(norm.logpdf(x, a, b) for x in data)
        values.append(lp)
    return values


def oracle_best_estimation(theta_grid: list[dict]) -> list[float]:
    """BEST (Kruschke 2013): ``mu1, mu2 ~ Normal(100, 20)``, ``sigma1,
    sigma2 ~ Uniform(interval(0.1, 20))``, ``nu ~ Exponential(rate=1/29)``;
    ``y1 ~ iid(locscale(StudentT(nu), mu1, sigma1), 8)``, ``y2`` likewise
    with group 2's params, against the fixed placeholder data."""
    y1_data = [101.0, 100.0, 102.0, 104.0, 100.0, 103.0, 99.0, 105.0]
    y2_data = [99.0, 101.0, 100.0, 98.0, 100.0, 97.0, 102.0, 100.0]
    values = []
    for th in theta_grid:
        mu1, mu2 = th["mu1"], th["mu2"]
        sigma1, sigma2 = th["sigma1"], th["sigma2"]
        nu = th["nu"]
        lp = (
            norm.logpdf(mu1, 100.0, 20.0)
            + norm.logpdf(mu2, 100.0, 20.0)
            + uniform.logpdf(sigma1, loc=0.1, scale=19.9)
            + uniform.logpdf(sigma2, loc=0.1, scale=19.9)
            + expon.logpdf(nu, scale=29.0)
        )
        lp += sum(t_dist.logpdf(y, nu, mu1, sigma1) for y in y1_data)
        lp += sum(t_dist.logpdf(y, nu, mu2, sigma2) for y in y2_data)
        values.append(lp)
    return values


def oracle_capture_recapture(theta_grid: list[dict]) -> list[float]:
    """``rcp ~ Uniform(interval(0, rcp_max))`` with ``rcp_max = M_obs /
    (C_obs - R_obs + M_obs) = 0.5``; ``R ~ Binomial(C_obs, rcp)`` against the
    fixed ``R_obs = 5`` out of ``C_obs = 15`` trials."""
    m_obs, c_obs, r_obs = 10, 15, 5
    rcp_max = m_obs / (c_obs - r_obs + m_obs)
    values = []
    for th in theta_grid:
        rcp = th["rcp"]
        lp = uniform.logpdf(rcp, loc=0.0, scale=rcp_max) + binom.logpmf(r_obs, c_obs, rcp)
        values.append(lp)
    return values


def oracle_eight_schools(theta_grid: list[dict]) -> list[float]:
    """Rubin's eight-schools model: ``mu ~ Normal(0,5)``, ``tau ~
    half-Cauchy(0,5)``, ``theta ~ iid(Normal(mu, tau), 8)``, ``y | theta ~
    Normal(theta, std_errs)`` against the fixed ``y_data``/``std_errs_data``.
    Same model as ``corpora/bayesian_inference``'s
    ``eight_schools_posterior``, generalized over the theta grid."""
    y = [28, 8, -3, 7, -1, 1, 18, 12]
    se = [15, 10, 16, 11, 9, 11, 10, 18]
    values = []
    for th in theta_grid:
        mu, tau = th["mu"], th["tau"]
        theta = th["theta"]
        lp = norm.logpdf(mu, 0, 5) + (math.log(2) + cauchy.logpdf(tau, 0, 5))
        lp += sum(norm.logpdf(t, mu, tau) for t in theta)
        lp += sum(norm.logpdf(yi, ti, sei) for yi, ti, sei in zip(y, theta, se))
        values.append(lp)
    return values


def oracle_gamma_reparam(theta_grid: list[dict]) -> list[float]:
    """``mu ~ Normal(0, 5)``; ``sigma ~ Gamma(gamma_shape_rate(2.0, 1.0))`` —
    the reparameterization is applied to LITERAL constants (mean=2.0,
    sd=1.0), so it resolves to the static prior ``Gamma(shape=4, rate=2)``,
    not a function of the draws; ``y ~ iid(Normal(mu, sigma), 5)`` against
    the fixed ``y_data``."""
    y_data = [1.8, 2.3, 1.1, 2.9, 2.0]
    shape, rate = 2.0**2 / 1.0**2, 2.0 / 1.0**2  # gamma_shape_rate(2.0, 1.0)
    values = []
    for th in theta_grid:
        mu, sigma = th["mu"], th["sigma"]
        lp = norm.logpdf(mu, 0.0, 5.0) + gamma_dist.logpdf(sigma, a=shape, scale=1.0 / rate)
        lp += sum(norm.logpdf(y, mu, sigma) for y in y_data)
        values.append(lp)
    return values


def oracle_hierarchical_logistic(theta_grid: list[dict]) -> list[float]:
    """``mu_a ~ Normal(0,1)``, ``sigma_a ~ Gamma(shape=4, rate=2)``, ``a ~
    iid(Normal(mu_a, sigma_a), 3)``, ``b ~ locscale(StudentT(3), 0, 2.5)``;
    ``eta_i = a[group_i] + b*x_i``, ``y_i ~ Bernoulli(invlogit(eta_i))``
    against the fixed placeholder data (``group_data`` is 1-based)."""
    x_data = [-1.2, 0.4, 1.1, -0.3, 0.8, 2.0]
    group_data = [1, 2, 3, 1, 2, 3]
    y_data = [0, 1, 1, 0, 1, 1]
    values = []
    for th in theta_grid:
        mu_a, sigma_a, b = th["mu_a"], th["sigma_a"], th["b"]
        a = th["a"]
        lp = (
            norm.logpdf(mu_a, 0.0, 1.0)
            + gamma_dist.logpdf(sigma_a, a=4.0, scale=1.0 / 2.0)
            + sum(norm.logpdf(ag, mu_a, sigma_a) for ag in a)
            + t_dist.logpdf(b, 3.0, 0.0, 2.5)
        )
        for xi, gi, yi in zip(x_data, group_data, y_data):
            eta = a[gi - 1] + b * xi
            lp += bernoulli.logpmf(yi, _invlogit(eta))
        values.append(lp)
    return values


def oracle_partial_pooling(theta_grid: list[dict]) -> list[float]:
    """``phi ~ Uniform(interval(0,1))``, ``kappa ~ Gamma(shape=2,
    rate=0.05)``; ``alpha = phi*kappa``, ``beta = (1-phi)*kappa``; ``theta ~
    iid(Beta(alpha, beta), 8)``; ``y ~ Binomial.(at_bats_data, theta)``
    against the fixed placeholder hits/at-bats data."""
    at_bats = [45, 45, 45, 60, 20, 100, 15, 250]
    hits = [12, 15, 10, 18, 8, 29, 6, 71]
    values = []
    for th in theta_grid:
        phi, kappa = th["phi"], th["kappa"]
        theta = th["theta"]
        alpha = phi * kappa
        beta_param = (1.0 - phi) * kappa
        lp = uniform.logpdf(phi, loc=0.0, scale=1.0) + gamma_dist.logpdf(
            kappa, a=2.0, scale=1.0 / 0.05
        )
        lp += sum(beta_dist.logpdf(ti, alpha, beta_param) for ti in theta)
        lp += sum(binom.logpmf(h, n, ti) for h, n, ti in zip(hits, at_bats, theta))
        values.append(lp)
    return values


def oracle_poisson_glm_link(theta_grid: list[dict]) -> list[float]:
    """``intercept, slope ~ Normal(0,1)``; ``eta_i = intercept +
    slope*x_i``, ``mu_i = exp(eta_i)``, ``y_i ~ Poisson(mu_i)`` against the
    fixed placeholder covariate/count data."""
    x_data = [-1.0, 0.2, 0.5, 1.3, 2.1]
    y_data = [0, 1, 2, 3, 8]
    values = []
    for th in theta_grid:
        intercept, slope = th["intercept"], th["slope"]
        lp = norm.logpdf(intercept, 0.0, 1.0) + norm.logpdf(slope, 0.0, 1.0)
        for xi, yi in zip(x_data, y_data):
            eta = intercept + slope * xi
            mu = math.exp(eta)
            lp += poisson.logpmf(yi, mu)
        values.append(lp)
    return values


def oracle_poisson_model(theta_grid: list[dict]) -> list[float]:
    """Conjugate Poisson-Gamma: ``lambda ~ Gamma(shape=2, rate=1)``; ``y ~
    iid(Poisson(lambda), 5)`` against the fixed ``counts_data``."""
    counts = [2, 3, 7, 6, 4]
    values = []
    for th in theta_grid:
        lam = th["lambda"]
        lp = gamma_dist.logpdf(lam, a=2.0, scale=1.0)
        lp += sum(poisson.logpmf(c, lam) for c in counts)
        values.append(lp)
    return values


def oracle_rasch_1pl(theta_grid: list[dict]) -> list[float]:
    """Rasch 1PL: ``theta ~ iid(Normal(0, 1.5), 4)`` (abilities), ``b ~
    iid(Normal(0, 1.5), 5)`` (difficulties); ``prob_k =
    invlogit(theta[person_k] - b[item_k])``, ``y_k ~ Bernoulli(prob_k)``
    against the fixed long-format placeholder responses (``person``/``item``
    are 1-based)."""
    person = [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4]
    item = [1, 2, 3, 4, 5] * 4
    y_data = [
        True, True, False, False, False,
        True, True, True, False, False,
        True, False, True, True, False,
        True, True, True, True, True,
    ]
    values = []
    for th in theta_grid:
        theta = th["theta"]
        b = th["b"]
        lp = sum(norm.logpdf(tp, 0.0, 1.5) for tp in theta) + sum(
            norm.logpdf(bi, 0.0, 1.5) for bi in b
        )
        for p, i, y in zip(person, item, y_data):
            eta = theta[p - 1] - b[i - 1]
            lp += bernoulli.logpmf(int(y), _invlogit(eta))
        values.append(lp)
    return values


# test_id -> (oracle_fn, list-of-frozen-values). oracle_fn takes the
# manifest entry's theta grid (list[dict]) and returns a list[float] of
# log-densities, one per theta point, in grid order. FROZEN pins the actual
# computed values (mirroring corpora/bayesian_inference/gen_expected.py's
# FROZEN dict) so re-running this script is a regression check on the
# oracle functions above, not just a re-derivation.
ORACLES: dict[str, Callable[[list[dict]], list[float]]] = {
    "ex_bayesian_inference_1": oracle_bayesian_inference,
    "ex_bayesian_inference_2": oracle_bayesian_inference,
    "ex_best_estimation": oracle_best_estimation,
    "ex_capture_recapture": oracle_capture_recapture,
    "ex_eight_schools": oracle_eight_schools,
    "ex_gamma_reparam": oracle_gamma_reparam,
    "ex_hierarchical_logistic": oracle_hierarchical_logistic,
    "ex_partial_pooling": oracle_partial_pooling,
    "ex_poisson_glm_link": oracle_poisson_glm_link,
    "ex_poisson_model": oracle_poisson_model,
    "ex_rasch_1pl": oracle_rasch_1pl,
}

FROZEN: dict[str, list[float]] = {
    "ex_bayesian_inference_1": [-74.10185205965193, -19.724594988746496],
    "ex_bayesian_inference_2": [-74.10185205965193, -19.724594988746496],
    "ex_best_estimation": [-53.82562036678366, -50.908393725676376],
    "ex_capture_recapture": [-1.1077782815739887, -0.9891956484874084],
    "ex_eight_schools": [-43.43563727714813, -59.67053378866518],
    "ex_gamma_reparam": [-19.21723985865041, -12.4172842197703],
    "ex_hierarchical_logistic": [-10.770987544664301, -11.164337828046966],
    "ex_partial_pooling": [-80.35566358637037, -32.17783165600062],
    "ex_poisson_glm_link": [-19.9273866189426, -11.504337313999388],
    "ex_poisson_model": [-26.76737305321146, -13.499290413844939],
    "ex_rasch_1pl": [-25.782576383014415, -30.926401673418287],
}


def gen(test_id: str, model: str, values: list[float],
        tolerance: dict[str, float] | None = None) -> None:
    tolerance = tolerance or {"atol": 1e-9, "rtol": 1e-9}
    doc = {
        "schema_version": 1,
        "test_id": test_id,
        "model": model,
        "reference_backend": "scipy 1.18",
        "checks": [
            {
                "id": f"theta_{i}",
                "kind": "logdensity_value",
                "index": i,
                "binding": "posterior",
                "expected": value,
                "tolerance": tolerance,
            }
            for i, value in enumerate(values)
        ],
    }
    out_dir = HERE / test_id
    out_dir.mkdir(exist_ok=True)
    (out_dir / "expected.json").write_text(json.dumps(doc, indent=2) + "\n")
    print(f"{test_id}: {len(values)} value(s) written")


def main() -> None:
    manifest = json.loads((HERE / "manifest.json").read_text())
    for ex in manifest.get("examples", []):
        if ex["status"] != "lowers":
            continue
        test_id = ex["test_id"]
        oracle_fn = ORACLES[test_id]
        values = oracle_fn(ex["theta"])
        frozen = FROZEN[test_id]
        for i, (value, froz) in enumerate(zip(values, frozen)):
            diff = abs(value - froz)
            assert diff <= 1e-12, (
                f"{test_id}[{i}]: scipy={value!r} frozen={froz!r} "
                f"diff={diff!r} > 1e-12"
            )
        gen(test_id, ex["model"], values)


if __name__ == "__main__":
    main()
