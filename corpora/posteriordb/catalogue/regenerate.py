"""Recompute the catalogue fixtures with BridgeStan and the pinned posteriorDB data."""

import argparse
import hashlib
import json
import math
import zipfile
from pathlib import Path

import numpy as np


def density_offset(provenance, point):
    count = sum(np.size(point[name]) for name in provenance["normalized_half_priors"])
    offset = count * math.log(2)
    if "ifr_noise" in point:
        # Positive Normal(1, 0.1) has mass Phi(10).
        offset -= np.size(point["ifr_noise"]) * math.log1p(
            -0.5 * math.erfc(10 / math.sqrt(2))
        )
    return float(offset)


def score_points(model, points, provenance):
    """Convert Stan's AD results to derivatives with respect to the record fields."""
    if model.model_version() != (2, 9, 0):
        raise ValueError(f"expected BridgeStan 2.9.0, got {model.model_version()}")
    names = model.param_names()
    if names != model.param_unc_names():
        raise ValueError("these fixtures require scalar or elementwise Stan charts")
    indices = []
    for name in names:
        field, *subscripts = name.split(".")
        indices.append((field, tuple(int(i) - 1 for i in subscripts)))
    positive = np.array(
        [field in provenance["positive_parameters"] for field, _ in indices]
    )
    bounded = np.array(
        [field in provenance["unit_interval_parameters"] for field, _ in indices]
    )
    raw, ad_scores, gradients = [], [], []
    for point in points:
        record = {k: np.asarray(v, dtype=np.float64) for k, v in point.items()}
        x = np.array([record[field][idx] for field, idx in indices])
        free = model.param_unconstrain_json(json.dumps(point))
        expected_free = x.copy()
        expected_free[positive] = np.log(x[positive])
        expected_free[bounded] = np.log(x[bounded]) - np.log1p(-x[bounded])
        np.testing.assert_allclose(free, expected_free, rtol=2e-14, atol=2e-14)
        np.testing.assert_allclose(model.param_constrain(free), x, rtol=2e-14, atol=0)

        score = model.log_density(free, propto=False, jacobian=False)
        ad_score, h = model.log_density_gradient(free, propto=False, jacobian=False)
        dx_du = np.ones_like(x)
        dx_du[positive] = x[positive]
        dx_du[bounded] = x[bounded] * (1 - x[bounded])
        gradient = {name: np.full_like(value, np.nan) for name, value in record.items()}
        for (field, idx), value in zip(indices, h / dx_du, strict=True):
            gradient[field][idx] = value
        if not (
            math.isfinite(score)
            and math.isfinite(ad_score)
            and all(np.isfinite(v).all() for v in gradient.values())
        ):
            raise ValueError("incomplete or nonfinite Stan value or gradient")
        raw.append(float(score))
        ad_scores.append(float(ad_score))
        gradients.append({k: v.tolist() for k, v in gradient.items()})
    return raw, ad_scores, gradients


def verified_digest(data, expected):
    digest = hashlib.sha256(data).hexdigest()
    if digest != expected:
        raise ValueError(f"SHA-256 {digest} differs from {expected}")


def verify_points(points, hashes):
    for point, expected in zip(points, hashes, strict=True):
        if point.keys() != expected.keys():
            raise ValueError("record fields differ from the matched point")
        for name, value in point.items():
            x = np.asarray(value, dtype="<f8")
            verified_digest(
                str((x.dtype.str, x.shape)).encode() + x.tobytes(), expected[name]
            )


def soil_solver(source, relative, absolute, max_steps=1000000):
    original = (
        b"integrate_ode_rk45(two_pool_feedback, C_t0, t0, ts, theta, x_r,\n"
        b"                               x_i)"
    )
    if source.count(original) != 1:
        raise ValueError("could not locate the pinned soil RK45 call")
    return source.replace(
        original, original[:-1] + f", {relative}, {absolute}, {max_steps})".encode()
    )


def original_data(database, provenance):
    name = provenance["dataset"] + ".json"
    with zipfile.ZipFile(database / "data/data" / (name + ".zip")) as archive:
        (member,) = [p for p in archive.namelist() if Path(p).name == name]
        data = archive.read(member)
    verified_digest(data, provenance["data_sha256"])
    return data.decode()


def evaluate(stan, data, spec, build):
    import bridgestan as bs

    build = build.resolve() / hashlib.sha256(stan).hexdigest()
    build.mkdir(parents=True, exist_ok=True)
    source = build / "reference.stan"
    if not source.exists() or source.read_bytes() != stan:
        source.write_bytes(stan)
    library = bs.compile_model(str(source))
    model = bs.StanModel(library, data, warn=False)
    raw, ad_scores, gradients = score_points(model, spec["points"], spec["provenance"])
    return {
        "stan_log_density": raw,
        "stan_log_density_with_gradient": ad_scores,
        "expected_grad": gradients,
    }


def check(actual, frozen):
    for name in ("stan_log_density", "stan_log_density_with_gradient"):
        np.testing.assert_allclose(actual[name], frozen[name], rtol=1e-8, atol=1e-8)
    for point, reference in zip(
        actual["expected_grad"], frozen["expected_grad"], strict=True
    ):
        if point.keys() != reference.keys():
            raise ValueError("gradient fields differ from the frozen result")
        for name, value in point.items():
            np.testing.assert_allclose(
                value, reference[name], rtol=1e-8, atol=1e-8, strict=True
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "database",
        type=Path,
        help="posteriorDB checkout's posterior_database directory",
    )
    parser.add_argument(
        "build", type=Path, help="directory for compiled original Stan models"
    )
    parser.add_argument("fixtures", type=Path, nargs="+")
    parser.add_argument(
        "--write", action="store_true", help="replace the frozen scores and gradients"
    )
    args = parser.parse_args()
    for fixture in args.fixtures:
        spec = json.loads(fixture.read_text())
        p = spec["provenance"]
        verify_points(spec["points"], p["point_hashes"])
        stan = (args.database / "models/stan" / (p["model"] + ".stan")).read_bytes()
        verified_digest(stan, p["stan_sha256"])
        data = original_data(args.database, p)
        result = evaluate(stan, data, spec, args.build)
        if solver := p.get("ode_rk45"):
            if args.write:
                p["original_solver"] = result
            else:
                check(result, p["original_solver"])
            stan = soil_solver(
                stan,
                solver["relative_tolerance"],
                solver["absolute_tolerance"],
                solver["max_steps"],
            )
            verified_digest(stan, solver["source_sha256"])
            result = evaluate(stan, data, spec, args.build)
        offset = density_offset(p, spec["points"][0])
        scores = [v + offset for v in result["stan_log_density"]]
        if args.write:
            spec.update(expected=scores, expected_grad=result.pop("expected_grad"))
            p.update(result, log_density_offset=offset)
            fixture.write_text(json.dumps(spec, indent=2, allow_nan=False) + "\n")
        else:
            check(result, dict(p, expected_grad=spec["expected_grad"]))
            np.testing.assert_allclose(scores, spec["expected"], rtol=1e-8, atol=1e-8)
        print(fixture.stem)


if __name__ == "__main__":
    main()
