"""Independent normalized densities and analytic gradients for the brms cases.

Inputs are captured brms design matrices and documented prior parameters. No
FlatPPL source, converter recipe, frozen score, or autodiff system is consulted.
"""
import math

import numpy as np
from scipy import special, stats


def _prior(prior, value, parameters):
    kind, args = prior["kind"], list(prior["args"])
    location = prior.get("location_parameter")
    if location:
        args[1 if kind == "student" else 0] = parameters[location]
    if kind == "normal":
        mean, scale = args
        distribution = stats.norm(mean, scale)
        derivative = -(value - mean) / scale**2
    elif kind == "cauchy":
        mean, scale = args
        distribution = stats.cauchy(mean, scale)
        derivative = -2 * (value - mean) / (scale**2 + (value - mean)**2)
    elif kind == "student":
        df, mean, scale = args
        distribution = stats.t(df, mean, scale)
        derivative = -(df + 1) * (value - mean) / (df * scale**2 + (value - mean)**2)
    elif kind == "gamma":
        shape, rate = args
        distribution = stats.gamma(shape, scale=1 / rate)
        derivative = (shape - 1) / value - rate
    elif kind == "invgamma":
        shape, scale = args
        distribution = stats.invgamma(shape, scale=scale)
        derivative = -(shape + 1) / value + scale / value**2
    else:
        raise ValueError(f"unknown oracle prior {kind}")
    score = distribution.logpdf(value)
    location_derivative = -derivative if location else 0.0
    if "lower" in prior:
        logmass = distribution.logsf(prior["lower"])
        score -= logmass
        if location:
            location_derivative -= math.exp(distribution.logpdf(prior["lower"]) - logmass)
    return score, derivative, location_derivative


def evaluate(reference, point):
    if reference["family"] == "cumulative":
        return _cumulative(reference, point)
    names = reference["coordinates"]
    positions = {name: i for i, name in enumerate(names)}
    values = np.asarray(point["theta"], dtype=float)
    p = dict(zip(names, values))
    p.update(reference.get("constants", {}))
    data = reference["standata"]
    x = np.asarray(data["X"], dtype=float)[:, 1:]
    x -= x.mean(axis=0)
    beta = np.array([p[f"b[{i + 1}]"] for i in range(x.shape[1])])
    eta = p["temp_Intercept"] + x @ beta + np.asarray(data.get("offsets", 0))
    jacobian = np.zeros((len(eta), len(names)))
    jacobian[:, positions["temp_Intercept"]] = 1
    for i in range(x.shape[1]):
        jacobian[:, positions[f"b[{i + 1}]"]] = x[:, i]

    score, gradient = 0.0, np.zeros(len(names))
    for name, prior in reference["priors"].items():
        contribution, derivative, location_derivative = _prior(prior, p[name], p)
        score += contribution
        gradient[positions[name]] += derivative
        if "location_parameter" in prior:
            gradient[positions[prior["location_parameter"]]] += location_derivative

    for group in reference.get("groups", []):
        # Both Stan representations have public z[coefficient, level] semantics.
        # For dimension two, the LKJ rho density is (1-rho^2)^(eta-1)/B(1/2,eta).
        name, index, rho_name = group["name"], group["index"], group.get("rho")
        rho = p[rho_name] if rho_name else 0.0
        root = math.sqrt(1 - rho**2)
        if rho_name:
            shape = group["lkj_shape"]
            score += (shape - 1) * math.log1p(-rho**2) - special.betaln(0.5, shape)
            gradient[positions[rho_name]] -= 2 * (shape - 1) * rho / (1 - rho**2)
        for i, level in enumerate(data[f"J_{index}"]):
            z0, z1 = p[f"z_{name}[1,{level}]"], p[f"z_{name}[2,{level}]"]
            sd0, sd1 = p[f"sd_{name}[1]"], p[f"sd_{name}[2]"]
            a, b = data[f"Z_{index}_1"][i], data[f"Z_{index}_2"][i]
            eta[i] += a * sd0 * z0 + b * sd1 * (rho * z0 + root * z1)
            jacobian[i, positions[f"sd_{name}[1]"]] = a * z0
            jacobian[i, positions[f"sd_{name}[2]"]] = b * (rho * z0 + root * z1)
            jacobian[i, positions[f"z_{name}[1,{level}]"]] = a * sd0 + b * sd1 * rho
            jacobian[i, positions[f"z_{name}[2,{level}]"]] = b * sd1 * root
            if rho_name:
                jacobian[i, positions[rho_name]] = b * sd1 * (z0 - rho * z1 / root)

    y = np.asarray(data["Y"], dtype=float)
    weights = np.asarray(data.get("weights", 1), dtype=float)

    def sum_rows(values):
        return np.sum(weights * values)

    family, link = reference["family"], reference["link"]
    if link == "identity":
        mu, derivative = eta, np.ones_like(eta)
    elif link == "log":
        mu = np.exp(eta)
        derivative = mu
    elif link == "logit":
        mu = special.expit(eta)
        derivative = mu * (1 - mu)
    elif link == "softit":
        softplus = np.logaddexp(0, eta)
        mu = softplus / (1 + softplus)
        derivative = special.expit(eta) / (1 + softplus)**2
    elif link == "tan_half":
        mu = 2 * np.arctan(eta)
        derivative = 2 / (1 + eta**2)
    else:
        raise ValueError(f"unknown oracle link {link}")

    if family in ("gaussian", "student"):
        sigma = p.get("sigma", 0.0)
        scale = np.sqrt(sigma**2 + np.asarray(data.get("se", 0))**2)
    if family == "gaussian":
        residual = y - mu
        score += sum_rows(stats.norm.logpdf(y, mu, scale))
        dmu = residual / scale**2
        if "sigma" in positions:
            gradient[positions["sigma"]] += sum_rows(
                (residual**2 / scale**3 - 1 / scale) * sigma / scale)
    elif family == "lognormal":
        sigma = p["sigma"]
        residual = np.log(y) - mu
        score += stats.lognorm.logpdf(y, sigma, scale=np.exp(mu)).sum()
        dmu = residual / sigma**2
        gradient[positions["sigma"]] += np.sum(residual**2 / sigma**3 - 1 / sigma)
    elif family == "student":
        nu = p["nu"]
        residual = y - mu
        square = (residual / scale)**2
        score += sum_rows(stats.t.logpdf(y, nu, mu, scale))
        dmu = (nu + 1) * residual / (nu * scale**2 + residual**2)
        if "sigma" in positions:
            gradient[positions["sigma"]] += sum_rows(
                ((nu + 1) * square / (nu + square) - 1) * sigma / scale**2)
        if "nu" in positions:
            gradient[positions["nu"]] += sum_rows(
                (special.digamma((nu + 1) / 2) - special.digamma(nu / 2) - 1 / nu
                 - np.log1p(square / nu) + (nu + 1) * square / (nu * (nu + square))) / 2)
    elif family in ("bernoulli", "binomial"):
        trials = np.asarray(data["trials"]) if family == "binomial" else 1
        score += sum_rows(stats.binom.logpmf(y, trials, mu))
        dmu = (y - trials * mu) / (mu * (1 - mu))
    elif family == "poisson":
        score += sum_rows(stats.poisson.logpmf(y, mu))
        dmu = y / mu - 1
    elif family == "geometric":
        score += stats.geom.logpmf(y + 1, 1 / (1 + mu)).sum()
        dmu = y / mu - (y + 1) / (1 + mu)
    elif family == "negbinomial":
        shape = p["shape"]
        score += sum_rows(stats.nbinom.logpmf(y, shape, shape / (shape + mu)))
        dmu = y / mu - (shape + y) / (shape + mu)
        gradient[positions["shape"]] += sum_rows(
            special.digamma(y + shape) - special.digamma(shape) + np.log(shape / (shape + mu))
            + 1 - (shape + y) / (shape + mu))
    elif family == "gamma":
        shape = p["shape"]
        score += sum_rows(stats.gamma.logpdf(y, shape, scale=mu / shape))
        dmu = shape * (y - mu) / mu**2
        gradient[positions["shape"]] += sum_rows(
            np.log(shape / mu) + 1 - special.digamma(shape) + np.log(y) - y / mu)
    elif family == "exponential":
        score += sum_rows(stats.expon.logpdf(y, scale=mu))
        dmu = (y - mu) / mu**2
    elif family == "weibull":
        shape = p["shape"]
        scale = mu / special.gamma(1 + 1 / shape)
        log_ratio = np.log(y / scale)
        power = np.exp(shape * log_ratio)
        score += stats.weibull_min.logpdf(y, shape, scale=scale).sum()
        dmu = shape * (power - 1) / mu
        gradient[positions["shape"]] += np.sum(
            1 / shape + (1 - power) *
            (log_ratio - special.digamma(1 + 1 / shape) / shape))
    elif family == "beta":
        phi = p["phi"]
        a, b = mu * phi, (1 - mu) * phi
        da, db = np.log(y) - special.digamma(a), np.log1p(-y) - special.digamma(b)
        score += stats.beta.logpdf(y, a, b).sum()
        dmu = phi * (da - db)
        gradient[positions["phi"]] += np.sum(mu * da + (1 - mu) * db + special.digamma(phi))
    elif family == "von_mises":
        kappa = p["kappa"]
        score += stats.vonmises.logpdf(y, kappa, loc=mu).sum()
        dmu = kappa * np.sin(y - mu)
        gradient[positions["kappa"]] += np.sum(
            np.cos(y - mu) - special.i1e(kappa) / special.i0e(kappa))
    else:
        raise ValueError(f"unknown oracle family {family}")
    gradient += jacobian.T @ (weights * dmu * derivative)
    return float(score), {"theta": gradient.tolist()}


def _cumulative(reference, point):
    names = reference["coordinates"]
    positions = {name: i for i, name in enumerate(names)}
    p = dict(zip(names, point["theta"]))
    data = reference["standata"]
    x = np.asarray(data["X"], dtype=float)
    x -= x.mean(axis=0)
    beta = np.array([p[f"b[{i + 1}]"] for i in range(x.shape[1])])
    cuts = np.array([p[f"cuts[{i + 1}]"] for i in range(data["nthres"])])
    score, gradient = math.lgamma(len(cuts) + 1), np.zeros(len(names))
    for name, prior in reference["priors"].items():
        contribution, derivative, _ = _prior(prior, p[name], p)
        score += contribution
        gradient[positions[name]] += derivative
    for row, eta, category in zip(x, x @ beta, data["Y"], strict=True):
        if category == 1:
            upper = cuts[0] - eta
            score += special.log_expit(upper)
            upper_derivative = special.expit(-upper)
            gradient[positions["cuts[1]"]] += upper_derivative
            eta_derivative = -upper_derivative
        elif category == len(cuts) + 1:
            lower = cuts[-1] - eta
            score += special.log_expit(-lower)
            lower_derivative = -special.expit(lower)
            gradient[positions[f"cuts[{len(cuts)}]"]] += lower_derivative
            eta_derivative = -lower_derivative
        else:
            lower, upper = cuts[category - 2:category] - eta
            score += (special.log_expit(upper) + special.log_expit(-lower)
                      + math.log(-math.expm1(lower - upper)))
            gap_ratio = 1 / math.expm1(upper - lower)
            gradient[positions[f"cuts[{category - 1}]"]] -= special.expit(lower) + gap_ratio
            gradient[positions[f"cuts[{category}]"]] += special.expit(-upper) + gap_ratio
            eta_derivative = special.expit(lower) - special.expit(-upper)
        for i, value in enumerate(row):
            gradient[positions[f"b[{i + 1}]"]] += value * eta_derivative
    return float(score), {"theta": gradient.tolist()}
